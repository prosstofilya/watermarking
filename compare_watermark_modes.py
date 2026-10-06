from __future__ import annotations

"""
Скрипт для сравнения двух режимов: PSO и AT-PSO на одних и тех же изображениях.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from praktos_watermarking.attacks import default_attacks
from praktos_watermarking.metrics import ncc, nsr, psnr
from praktos_watermarking.pso_watermarking import EmbedConfig, embed_watermark, extract_watermark


def _load_gray_u8(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im, dtype=np.uint8)


def run_for_mode(mode: str, cover: np.ndarray, watermark: np.ndarray) -> None:
    attacks = default_attacks()
    use_arnold = mode == "at-pso"

    cfg = EmbedConfig(
        wavelet="haar",
        use_arnold=use_arnold,
        arnold_iters=10,
        attacks_for_fitness=None,
    )

    cw_u8, artifacts = embed_watermark(cover, watermark, config=cfg, attack_fns=attacks)

    print(f"\n=== Режим: {mode.upper()} (use_arnold={artifacts.use_arnold}) ===")
    print(f"alpha = {artifacts.alpha:.6f}")
    print(f"PSNR(cover, watermarked) = {psnr(cover, cw_u8):.4f} dB")
    print(f"NCC(cover, watermarked) = {ncc(cover.astype(np.float64), cw_u8.astype(np.float64)):.6f}")
    print("")

    wm01 = (watermark > 127).astype(np.uint8)
    for key, atk in attacks.items():
        attacked_img = atk.fn(Image.fromarray(cw_u8, mode="L"))
        attacked_u8 = np.array(attacked_img.convert("L"), dtype=np.uint8)

        w_hat = extract_watermark(attacked_u8, artifacts, watermark_shape=wm01.shape, threshold=0.5)

        print(
            f"{key:>14} | "
            f"PSNR(attacked)={psnr(cover, attacked_u8):7.3f} dB | "
            f"NCC={ncc(wm01.astype(np.float64), w_hat.astype(np.float64)):.6f} | "
            f"NSR={nsr(wm01, w_hat):.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", type=str, required=True)
    ap.add_argument("--watermark", type=str, required=True)
    args = ap.parse_args()

    cover_path = Path(args.cover)
    wm_path = Path(args.watermark)

    cover = _load_gray_u8(cover_path)
    watermark = _load_gray_u8(wm_path)

    # Режим 1: только PSO (без Arnold)
    run_for_mode("pso", cover, watermark)

    # Режим 2: AT-PSO (Arnold+PSO). Если watermark не квадратный,
    # внутри embed_watermark Arnold автоматически отключится.
    run_for_mode("at-pso", cover, watermark)


if __name__ == "__main__":
    main()

