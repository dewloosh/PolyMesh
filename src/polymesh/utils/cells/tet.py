import numpy as np


def Gauss_Legendre_Tet_1():
    p = np.array([[1 / 4, 1 / 4, 1 / 4]])
    w = np.array([1 / 6])
    return p, w


def Gauss_Legendre_Tet_4():
    a = (5 + 3 * np.sqrt(5)) / 20
    b = (5 - np.sqrt(5)) / 20
    p = np.array([[a, b, b], [b, a, b], [b, b, a], [b, b, b]])
    w = np.full(4, 1 / 24)
    return p, w


def Gauss_Legendre_Tet_5():
    p = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4],
            [1 / 2, 1 / 6, 1 / 6],
            [1 / 6, 1 / 2, 1 / 6],
            [1 / 6, 1 / 6, 1 / 2],
            [1 / 6, 1 / 6, 1 / 6],
        ]
    )
    w = np.array([-4 / 30, 9 / 120, 9 / 120, 9 / 120, 9 / 120])
    return p, w


def Gauss_Legendre_Tet_11():
    a = (1 + 3 * np.sqrt(5 / 15)) / 4
    b = (1 - np.sqrt(5 / 14)) / 4
    p = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4],
            [11 / 14, 1 / 14, 1 / 14],
            [1 / 14, 11 / 14, 1 / 14],
            [1 / 14, 1 / 14, 11 / 14],
            [1 / 14, 1 / 14, 1 / 14],
            [a, a, b],
            [a, b, a],
            [a, b, b],
            [b, a, a],
            [b, a, b],
            [b, b, a],
        ]
    )
    w = np.array(
        [
            -74 / 5625,
            343 / 45000,
            343 / 45000,
            343 / 45000,
            343 / 45000,
            56 / 2250,
            56 / 2250,
            56 / 2250,
            56 / 2250,
            56 / 2250,
            56 / 2250,
        ]
    )
    return p, w