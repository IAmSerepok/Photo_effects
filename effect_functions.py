from numba import njit
from numpy import array, ndarray
from typing import List


@njit(fastmath=True)
def box_blur(img: ndarray, x: int, y: int) -> List[int, int, int]:
    """Применяет box blur (размытие по квадрату 3x3) к пикселю изображения.

    Args:
        img (ndarray): Входное изображение в формате массива numpy (H, W, 3).
        x (int): Координата X целевого пикселя (1 <= x < H-1).
        y (int): Координата Y целевого пикселя (1 <= y < W-1).

    Returns:
        color (List[int, int, int]): Новое значение пикселя [R, G, B] после размытия (0-255).

    Note:
        Использует простое усреднение 8 соседних пикселей (ядро 3x3).
    """

    r, g, b = 0, 0, 0
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            r0, g0, b0 = img[i, j]
            r, g, b = array([r, g, b]) + array([r0, g0, b0])
    
    r, g, b = array([r, g, b]) // 9

    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return [r, g, b]


@njit(fastmath=True)
def gaussian_blur(img: ndarray, x: int, y: int) -> List[int, int, int]:
    """Применяет приближенное Gaussian blur (размытие по Гауссу) к пикселю.

    Args:
        img (ndarray): Входное изображение в формате массива numpy (H, W, 3).
        x (int): Координата X целевого пикселя (1 <= x < H-1).
        y (int): Координата Y целевого пикселя (1 <= y < W-1).

    Returns:
        color (List[int, int, int]): Новое значение пикселя [R, G, B] после размытия (0-255).

    Note:
        Использует упрощенное ядро Гаусса с весами:
        1/16 | 2/16 | 1/16
        ------------------
        2/16 | 4/16 | 2/16
        ------------------
        1/16 | 2/16 | 1/16
    """
    r, g, b = 4 * img[x, y]

    for dx, dy in [
        (-1, 0), (1, 0), (0, -1), (0, 1)
    ]: 

        r0, g0, b0 = 2 * img[x + dx, y + dy]
        r, g, b = array([r, g, b]) + array([r0, g0, b0])
    

    for dx, dy in [
        (-1, -1), (1, 1), (1, -1), (-1, 1)
    ]: 
        r0, g0, b0 = img[x + dx, y + dy]
        r, g, b = array([r, g, b]) + array([r0, g0, b0])

    r, g, b = array([r, g, b]) // 16

    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return [r, g, b]


@njit(fastmath=True)
def clarifier(img: ndarray, x: int, y: int) -> List[int, int, int]:
    """Усиливает доминирующий канал пикселя.

    Args:
        img (ndarray): Входное изображение в формате массива numpy (H, W, 3).
        x (int): Координата X целевого пикселя.
        y (int): Координата Y целевого пикселя.

    Returns:
        color (List[int, int, int]): Новое значение пикселя [R, G, B] (0-255).

    Note:
        Увеличивает на 10 значение канала (R, G или B), который был максимальным.
    """
    r, g, b = img[x, y]
    if max(r, g, b) == r:
        r += 10
    elif max(r, g, b) == g:
        g += 10
    elif max(r, g, b) == b:
        b += 10

    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return [r, g, b]


@njit(fastmath=True)
def dimmer(img: ndarray, x: int, y: int) -> List[int, int, int]:
    """Осветляет пиксель, добавляя фиксированное значение ко всем каналам.

    Args:
        img (ndarray): Входное изображение в формате массива numpy (H, W, 3).
        x (int): Координата X целевого пикселя.
        y (int): Координата Y целевого пикселя.

    Returns:
        List[int]: Новое значение пикселя [R, G, B] (0-255).

    Note:
        Добавляет 10 к каждому цветовому каналу.
    """
    r, g, b = img[x, y]
    r, g, b = array([r, g, b]) + array([10, 10, 10])

    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return [r, g, b]
