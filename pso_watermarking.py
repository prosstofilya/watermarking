from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .arnold import arnold_transform, inverse_arnold_transform
from .metrics import ncc


try:
    import pywt
except Exception as e:  # pragma: no cover
    pywt = None
    _pywt_import_error = e
else:
    _pywt_import_error = None


def _require_pywt() -> None:
    if pywt is None:  # pragma: no cover
        raise RuntimeError(
        ) from _pywt_import_error


def _dwt2(img: np.ndarray, wavelet: str = "haar") -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    _require_pywt()
    coeffs2 = pywt.dwt2(img, wavelet)
    ll, (lh, hl, hh) = coeffs2
    return ll, (lh, hl, hh)


def _idwt2(ll: np.ndarray, bands: Tuple[np.ndarray, np.ndarray, np.ndarray], wavelet: str = "haar") -> np.ndarray:
    _require_pywt()
    lh, hl, hh = bands
    return pywt.idwt2((ll, (lh, hl, hh)), wavelet)


def _svd(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # full_matrices=False gives compact form; OK for reconstruction.
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return u, s, vt


def _as_gray_f64(img_u8: np.ndarray) -> np.ndarray:
    a = img_u8.astype(np.float64)
    return a


def _clip_u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _binarize01(a: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (a > thr).astype(np.float64)


@dataclass(frozen=True)
class PSOParams:
    particles: int = 5
    iterations: int = 8
    c1: float = 2.0
    c2: float = 2.0
    inertia_w: float = 0.7
    # Для практики запрещаем нулевой масштаб встраивания,
    # чтобы watermark всегда реально встраивался.
    alpha_min: float = 1.0
    alpha_max: float = 150.0
    v_max: float = 150.0


@dataclass(frozen=True)
class EmbedConfig:
    wavelet: str = "haar"
    use_arnold: bool = True
    arnold_iters: int = 10
    pso: PSOParams = PSOParams()
    attacks_for_fitness: Optional[List[str]] = None


@dataclass(frozen=True)
class EmbedArtifacts:
    alpha: float
    u_ll: np.ndarray
    s_ll: np.ndarray
    vt_ll: np.ndarray
    arnold_iters: int
    use_arnold: bool
    wavelet: str


def _embed_with_alpha(
    cover_u8: np.ndarray,
    watermark_01: np.ndarray,
    alpha: float,
    *,
    wavelet: str,
    use_arnold: bool,
    arnold_iters: int,
) -> Tuple[np.ndarray, EmbedArtifacts]:
    c = _as_gray_f64(cover_u8)
    ll, bands = _dwt2(c, wavelet=wavelet)

    u_ll, s_ll_vec, vt_ll = _svd(ll)
    s_ll = np.diag(s_ll_vec)

    w = watermark_01
    if use_arnold:
        w = arnold_transform(w, iterations=arnold_iters)

    # S_LLM = S_LL + alpha * W
    # We embed watermark into the top-left block of S_LL (sizes may differ).
    s_mod = s_ll.copy()
    h = min(s_mod.shape[0], w.shape[0])
    w_ = min(s_mod.shape[1], w.shape[1])
    s_mod[:h, :w_] = s_mod[:h, :w_] + float(alpha) * w[:h, :w_]

    # SVD(S_LLM) = U_LLMA * S_LLMA * V_LLMA^T
    u_ma, s_ma_vec, vt_ma = _svd(s_mod)
    s_ma = np.diag(s_ma_vec)

    #  LL' = U_LL * S_LLMA * V_LL^T
    ll_prime = u_ll @ s_ma @ vt_ll
    cw = _idwt2(ll_prime, bands, wavelet=wavelet)
    cw_u8 = _clip_u8(cw)

    artifacts = EmbedArtifacts(
        alpha=float(alpha),
        u_ll=u_ll,
        s_ll=s_ll,
        vt_ll=vt_ll,
        arnold_iters=int(arnold_iters),
        use_arnold=bool(use_arnold),
        wavelet=str(wavelet),
    )
    return cw_u8, artifacts


def extract_watermark(
    attacked_watermarked_u8: np.ndarray,
    artifacts: EmbedArtifacts,
    *,
    watermark_shape: Tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Extraction per Section 4.2, Eq.(11)-(13) + inverse Arnold + threshold 0.5.
    Returns binary watermark as uint8 values {0,1}.
    """
    cw_star = _as_gray_f64(attacked_watermarked_u8)
    ll_star, _bands = _dwt2(cw_star, wavelet=artifacts.wavelet)

    # SVD(LL*') = U'_LL * S'_LLMA * V'^T
    u_p, s_p_vec, vt_p = _svd(ll_star)
    s_llma_p = np.diag(s_p_vec)

    # S'_LLM = U'_LLMA * S'_LLMA * V'^T_LLMA (paper uses MA subscripts; this is the LL* SVD)
    s_llm_p = u_p @ s_llma_p @ vt_p

    # W' = (S'_LLM - S_LL) / alpha
    alpha = artifacts.alpha if artifacts.alpha != 0 else 1e-9
    w_scr = (s_llm_p - artifacts.s_ll) / alpha

    # Сохранить только размер водяного знака (верхний левый блок), затем обратное преобразование Арнольда, затем порог 0.5.
    h, w = watermark_shape
    w_scr = w_scr[:h, :w]
    if artifacts.use_arnold:
        w_est = inverse_arnold_transform(w_scr, iterations=artifacts.arnold_iters)
    else:
        w_est = w_scr

    w_bin = (w_est > threshold).astype(np.uint8)
    return w_bin


def _fitness(
    cover_u8: np.ndarray,
    watermark_01: np.ndarray,
    alpha: float,
    *,
    wavelet: str,
    use_arnold: bool,
    arnold_iters: int,
    attack_fns: Iterable,
) -> float:
    """
    Максимизировать (1/R) * sum_i corr(w, w_i') + corr(c, c_w)
    Здесь корреляция реализована как NCC.
    """
    cw_u8, artifacts = _embed_with_alpha(
        cover_u8,
        watermark_01,
        alpha,
        wavelet=wavelet,
        use_arnold=use_arnold,
        arnold_iters=arnold_iters,
    )

    corr_c = ncc(cover_u8.astype(np.float64), cw_u8.astype(np.float64))

    corr_sum = 0.0
    r = 0
    base_img = Image.fromarray(cw_u8, mode="L")
    for attack in attack_fns:
        r += 1
        # attack может быть либо функцией, либо объектом с полем .fn (как Attack).
        fn = attack.fn if hasattr(attack, "fn") else attack
        attacked_img = fn(base_img)
        attacked = np.array(attacked_img.convert("L"), dtype=np.uint8)
        w_hat = extract_watermark(attacked, artifacts, watermark_shape=watermark_01.shape, threshold=0.5)
        corr_sum += ncc(watermark_01, w_hat.astype(np.float64))
    if r == 0:
        return corr_c
    return float((corr_sum / r) + corr_c)


def optimize_alpha_pso(
    cover_u8: np.ndarray,
    watermark_01: np.ndarray,
    *,
    config: EmbedConfig,
    attack_fns: Dict[str, callable],
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    PSO as described in Section 4.2 (params below Eq.(13)).
    We optimize a single scalar alpha.
    """
    p = config.pso
    if rng is None:
        rng = np.random.default_rng()

    names = config.attacks_for_fitness
    if names is None:
        # По умолчанию используем все атаки в fitness, если не указано.
        use_attacks = list(attack_fns.values())
    else:
        use_attacks = [attack_fns[n].fn if hasattr(attack_fns[n], "fn") else attack_fns[n] for n in names]

    # Инициализировать частицы (масштаб не меньше alpha_min)
    x = rng.uniform(p.alpha_min, p.alpha_max, size=(p.particles,))
    v = rng.uniform(-p.v_max, p.v_max, size=(p.particles,))
    pbest_x = x.copy()
    pbest_f = np.full((p.particles,), -np.inf, dtype=np.float64)

    def eval_f(alpha_val: float) -> float:
        return _fitness(
            cover_u8,
            watermark_01,
            float(alpha_val),
            wavelet=config.wavelet,
            use_arnold=config.use_arnold,
            arnold_iters=config.arnold_iters,
            attack_fns=use_attacks,
        )

    gbest_x = float(x[0])
    gbest_f = -np.inf

    for _it in range(p.iterations):
        # Оценить каждую частицу
        for i in range(p.particles):
            fi = eval_f(float(x[i]))
            if fi > pbest_f[i]:
                pbest_f[i] = fi
                pbest_x[i] = x[i]
            if fi > gbest_f:
                gbest_f = fi
                gbest_x = float(x[i])

        # Обновить скорости и позиции
        r1 = rng.random(size=(p.particles,))
        r2 = rng.random(size=(p.particles,))
        v = (
            p.inertia_w * v
            + p.c1 * r1 * (pbest_x - x)
            + p.c2 * r2 * (gbest_x - x)
        )
        v = np.clip(v, -p.v_max, p.v_max)
        x = x + v
        x = np.clip(x, p.alpha_min, p.alpha_max)

    # Гарантируем, что итоговый alpha не меньше alpha_min
    return float(max(gbest_x, p.alpha_min))


def embed_watermark(
    cover_u8: np.ndarray,
    watermark_u8_or01: np.ndarray,
    *,
    config: EmbedConfig,
    attack_fns: Dict[str, callable],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, EmbedArtifacts]:
    """
    Full embedding: pick alpha via PSO, then embed using that alpha.
    """
    w = watermark_u8_or01.astype(np.float64)
    if w.max() > 1.0:
        w = w / 255.0
    w01 = _binarize01(w, thr=0.5)

    # Автоматически отключаем Arnold, если watermark не квадратный,
    # чтобы не падать с ошибкой при произвольном логотипе.
    local_config = config
    h_wm, w_wm = w01.shape
    if h_wm != w_wm and config.use_arnold:
        local_config = EmbedConfig(
            wavelet=config.wavelet,
            use_arnold=False,
            arnold_iters=config.arnold_iters,
            pso=config.pso,
            attacks_for_fitness=config.attacks_for_fitness,
        )

    alpha = optimize_alpha_pso(cover_u8, w01, config=local_config, attack_fns=attack_fns, rng=rng)
    return _embed_with_alpha(
        cover_u8,
        w01,
        alpha,
        wavelet=local_config.wavelet,
        use_arnold=local_config.use_arnold,
        arnold_iters=local_config.arnold_iters,
    )

