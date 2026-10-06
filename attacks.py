from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _to_uint8_gray(img: Image.Image, size: Tuple[int, int] | None = None) -> Image.Image:
    g = img.convert("L")
    if size is not None:
        g = g.resize(size, resample=Image.BICUBIC)
    return g


def gaussian_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def scaling_50(img: Image.Image) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, w // 2), max(1, h // 2)), resample=Image.BICUBIC)
    return small.resize((w, h), resample=Image.BICUBIC)


def histogram_equalization(img: Image.Image) -> Image.Image:
    return ImageOps.equalize(img)


def rotation_20(img: Image.Image) -> Image.Image:
    return img.rotate(20, resample=Image.BICUBIC, expand=False, fillcolor=0)


def jpeg_q25(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=25, optimize=True)
    buf.seek(0)
    return Image.open(buf).copy()


def cropping_12(img: Image.Image) -> Image.Image:
    w, h = img.size
    dx = int(round(w * 0.06))
    dy = int(round(h * 0.06))
    cropped = img.crop((dx, dy, max(dx + 1, w - dx), max(dy + 1, h - dy)))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def salt_and_pepper(img: Image.Image, amount: float = 0.01, salt_vs_pepper: float = 0.5) -> Image.Image:
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    n = int(round(amount * h * w))
    if n <= 0:
        return img.copy()
    rng = np.random.default_rng()
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    salt_n = int(round(n * salt_vs_pepper))
    pepper_n = n - salt_n
    if salt_n:
        arr[ys[:salt_n], xs[:salt_n]] = 255
    if pepper_n:
        arr[ys[salt_n:], xs[salt_n:]] = 0
    return Image.fromarray(arr, mode=img.mode)


@dataclass(frozen=True)
class Attack:
    name: str
    fn: Callable[[Image.Image], Image.Image]


def default_attacks() -> Dict[str, Attack]:
    return {
        "gaussian_blur": Attack("gaussian_blur", lambda im: gaussian_blur(im, radius=1.0)),
        "scaling_50": Attack("scaling_50", scaling_50),
        "hist_eq": Attack("hist_eq", histogram_equalization),
        "rotation_20": Attack("rotation_20", rotation_20),
        "jpeg_q25": Attack("jpeg_q25", jpeg_q25),
        "cropping_12": Attack("cropping_12", cropping_12),
        "salt_pepper": Attack("salt_pepper", lambda im: salt_and_pepper(im, amount=0.01)),
    }

