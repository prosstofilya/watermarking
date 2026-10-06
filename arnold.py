from __future__ import annotations

import numpy as np


def arnold_transform(img: np.ndarray, iterations: int = 10) -> np.ndarray:
    f"""
    Преобразование Арнольда для массива NxN.
    """
    a = np.asarray(img)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError()
    n = a.shape[0]
    out = a.copy()
    for _ in range(int(iterations)):
        tmp = np.empty_like(out)
        for x in range(n):
            for y in range(n):
                xp = (x + y) % n
                yp = (x + 2 * y) % n
                tmp[xp, yp] = out[x, y]
        out = tmp
    return out


def inverse_arnold_transform(img: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Обратное преобразование Арнольда соответствующее матрице [[1,1],[1,2]] mod N.
    Обратная матрица [[2,-1],[-1,1]] mod N.
    """
    a = np.asarray(img)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError()
    n = a.shape[0]
    out = a.copy()
    for _ in range(int(iterations)):
        tmp = np.empty_like(out)
        for x in range(n):
            for y in range(n):
                xp = (2 * x - y) % n
                yp = (-x + y) % n
                tmp[xp, yp] = out[x, y]
        out = tmp
    return out

