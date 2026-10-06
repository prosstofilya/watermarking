from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .attacks import default_attacks
from .metrics import ncc, nsr, psnr
from .pso_watermarking import EmbedConfig, embed_watermark, extract_watermark


def _load_gray_u8(path: Path, size=None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize(size, resample=Image.BICUBIC)
    return np.array(im, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", type=str, required=True, help="Path to cover image (e.g., 512x512).")
    ap.add_argument("--watermark", type=str, required=True, help="Path to watermark image (e.g., 256x256, binary/logo).")
    ap.add_argument(
        "--mode",
        type=str,
        choices=["pso", "at-pso"],
        default="at-pso",
        help="Embedding mode: 'pso' (без Arnold) или 'at-pso' (с Arnold). По умолчанию at-pso.",
    )
    ap.add_argument("--arnold-iters", type=int, default=10)
    ap.add_argument("--wavelet", type=str, default="haar")
    args = ap.parse_args()

    cover_path = Path(args.cover)
    wm_path = Path(args.watermark)

    cover = _load_gray_u8(cover_path)
    watermark = _load_gray_u8(wm_path)

    attacks = default_attacks()

    use_arnold = args.mode == "at-pso"

    cfg = EmbedConfig(
        wavelet=args.wavelet,
        use_arnold=use_arnold,
        arnold_iters=args.arnold_iters,
        # По умолчанию используем все атаки в fitness, если не указано.
        attacks_for_fitness=None,
    )

    cw_u8, artifacts = embed_watermark(cover, watermark, config=cfg, attack_fns=attacks)

    out_dir = Path.cwd() / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cw_u8, mode="L").save(out_dir / "watermarked.png")

    print(f"alpha = {artifacts.alpha:.6f}")
    print(f"PSNR(cover, watermarked) = {psnr(cover, cw_u8):.4f} dB")
    print(f"NCC(cover, watermarked) = {ncc(cover.astype(np.float64), cw_u8.astype(np.float64)):.6f}")
    print("")

    wm01 = (watermark > 127).astype(np.uint8)
    for key, atk in attacks.items():
        attacked_img = atk.fn(Image.fromarray(cw_u8, mode="L"))
        attacked_u8 = np.array(attacked_img.convert("L"), dtype=np.uint8)
        Image.fromarray(attacked_u8, mode="L").save(out_dir / f"attacked_{key}.png")

        w_hat = extract_watermark(attacked_u8, artifacts, watermark_shape=wm01.shape, threshold=0.5)
        Image.fromarray((w_hat * 255).astype(np.uint8), mode="L").save(out_dir / f"wm_extracted_{key}.png")

        print(
            f"{key:>14} | "
            f"PSNR(attacked)={psnr(cover, attacked_u8):7.3f} dB | "
            f"NCC={ncc(wm01.astype(np.float64), w_hat.astype(np.float64)):.6f} | "
            f"NSR={nsr(wm01, w_hat):.6f}"
        )


if __name__ == "__main__":
    main()

