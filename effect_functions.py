from numba import njit


@njit(fastmath=True)
def box_blur(img, x, y):

    r, g, b = 0, 0, 0
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            r0, g0, b0 = img[i, j]
            r += r0
            g += g0
            b += b0
    r //= 9
    g //= 9
    b //= 9
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return [r, g, b]


@njit(fastmath=True)
def gaussian_blur(img, x, y):
    r, g, b = 4 * img[x, y]

    r0, g0, b0 = 2 * img[x - 1, y]
    r += r0
    g += g0
    b += b0
    r0, g0, b0 = 2 * img[x + 1, y]
    r += r0
    g += g0
    b += b0
    r0, g0, b0 = 2 * img[x, y - 1]
    r += r0
    g += g0
    b += b0
    r0, g0, b0 = 2 * img[x, y + 1]
    r += r0
    g += g0
    b += b0

    r0, g0, b0 = img[x - 1, y - 1]
    r += r0
    g += g0
    b += b0
    r0, g0, b0 = img[x + 1, y + 1]
    r += r0
    g += g0
    b += b0
    r0, g0, b0 = img[x + 1, y - 1]
    r += r0
    g += g0
    b += b0
    r0, g0, b0 = img[x - 1, y + 1]
    r += r0
    g += g0
    b += b0

    r //= 16
    g //= 16
    b //= 16

    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return [r, g, b]


@njit(fastmath=True)
def clarifier(img, x, y):
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
def dimmer(img, x, y):
    r, g, b = img[x, y]

    r += 10
    g += 10
    b += 10

    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return [r, g, b]
