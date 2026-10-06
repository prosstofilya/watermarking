from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    # Eq.(17) in the paper: correlation coefficient between watermark arrays.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    am = a.mean()
    bm = b.mean()
    num = np.sum((a - am) * (b - bm))
    den = np.sqrt(np.sum((a - am) ** 2) * np.sum((b - bm) ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


def sr_binary(w: np.ndarray, w_hat: np.ndarray) -> float:
    """
    Similarity Ratio (SR) для бинарной водяного знака с уже примененным порогом.
    """
    w = (w > 0).astype(np.uint8)
    w_hat = (w_hat > 0).astype(np.uint8)
    s = int(np.sum(w == w_hat))
    d = int(w.size - s)
    if s + d == 0:
        return 0.0
    return float(s / (s + d))


def nsr(w: np.ndarray, w_hat: np.ndarray) -> float:
    """
    NSR = (SR - min(SR)) / (1 - min(SR))
    Для бинарной водяного знака, min(SR) происходит для изображения всех нулей или всех единиц.
    """
    sr = sr_binary(w, w_hat)
    w = (w > 0).astype(np.uint8)
    ones = np.ones_like(w)
    zeros = np.zeros_like(w)
    min_sr = min(sr_binary(w, zeros), sr_binary(w, ones))
    if 1.0 - min_sr == 0:
        return 0.0
    return float((sr - min_sr) / (1.0 - min_sr))

