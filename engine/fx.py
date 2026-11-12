"""Full-frame numpy post effects: the glitch / pixel / print vocabulary."""
import numpy as np

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], np.float32) / 64.0


def rgb_split(img, dx, dy=0):
    if dx == 0 and dy == 0:
        return img
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], (dy, dx), axis=(0, 1))
    out[..., 2] = np.roll(img[..., 2], (-dy, -dx), axis=(0, 1))
    return out


def slice_shift(img, rng, n=12, max_h=80, max_shift=240, min_h=2):
    """Horizontal tear: random bands displaced sideways."""
    out = img.copy()
    h = img.shape[0]
    for _ in range(n):
        bh = int(rng.integers(min_h, max_h))
        y = int(rng.integers(0, max(1, h - bh)))
        s = int(rng.integers(-max_shift, max_shift))
        out[y:y + bh] = np.roll(img[y:y + bh], s, axis=1)
    return out


def block_smear(img, rng, n=8, max_w=500, max_h=200):
    """Datamosh-ish: copy a 1px column/row of a block across the block."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        bw = int(rng.integers(40, max_w))
        bh = int(rng.integers(10, max_h))
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        if rng.random() < 0.5:
            out[y:y + bh, x:x + bw] = img[y:y + bh, x:x + 1]
        else:
            out[y:y + bh, x:x + bw] = img[y:y + 1, x:x + bw]
    return out


def block_displace(img, rng, n=10, cell=32, max_cells=8, max_shift=12):
    """Copy random cell-aligned blocks from elsewhere — chunky pixel glitch."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        bw = int(rng.integers(1, max_cells + 1)) * cell
        bh = int(rng.integers(1, max_cells // 2 + 2)) * cell
        x = int(rng.integers(0, max(1, (w - bw) // cell))) * cell
        y = int(rng.integers(0, max(1, (h - bh) // cell))) * cell
        sx = int(np.clip(x + rng.integers(-max_shift, max_shift + 1) * cell, 0, w - bw))
        sy = int(np.clip(y + rng.integers(-max_shift // 2, max_shift // 2 + 1) * cell, 0, h - bh))
        out[y:y + bh, x:x + bw] = img[sy:sy + bh, sx:sx + bw]
    return out


def pixelate(img, cell):
    if cell <= 1:
        return img
    h, w = img.shape[:2]
    hh, ww = h // cell, w // cell
    small = img[:hh * cell, :ww * cell].reshape(hh, cell, ww, cell, -1).mean(axis=(1, 3))
    big = np.repeat(np.repeat(small, cell, 0), cell, 1).astype(img.dtype)
    out = img.copy()
    out[:hh * cell, :ww * cell] = big
    return out


def pixelate_region(img, x, y, w, h, cell):
    out = img.copy()
    out[y:y + h, x:x + w] = pixelate(img[y:y + h, x:x + w], cell)
    return out


def pixel_blocks(img, rng, n, cell, colors):
    """Hard flat squares sprinkled on a cell grid (Marathon's pixel debris)."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        x = int(rng.integers(0, w // cell)) * cell
        y = int(rng.integers(0, h // cell)) * cell
        k = int(rng.integers(1, 4))
        out[y:y + cell * k, x:x + cell * int(rng.integers(1, 5))] = colors[int(rng.integers(0, len(colors)))]
    return out


def scanlines(img, strength=0.12, period=3):
    out = img.astype(np.float32)
    out[::period] *= (1 - strength)
    return out.clip(0, 255).astype(np.uint8)


def grain(img, rng, amount=6.0):
    n = rng.normal(0, amount, img.shape[:2]).astype(np.float32)[..., None]
    return (img.astype(np.float32) + n).clip(0, 255).astype(np.uint8)


def invert_band(img, y0, y1):
    out = img.copy()
    out[y0:y1] = 255 - out[y0:y1]
    return out


def channel_swap(img, order=(2, 1, 0)):
    return img[..., list(order)]


def threshold_tint(img, color, thresh=128):
    lum = img.mean(axis=2)
    out = np.zeros_like(img)
    out[lum > thresh] = color
    return out


def glitch_burst(img, rng, amt):
    """Composite glitch scaled by amt in [0,1]."""
    if amt <= 0:
        return img
    out = img
    if amt > 0.15:
        out = block_displace(out, rng, n=int(4 + 18 * amt), cell=24 if amt < 0.6 else 48)
    out = slice_shift(out, rng, n=int(3 + 20 * amt), max_h=int(10 + 90 * amt), max_shift=int(30 + 400 * amt))
    if amt > 0.4:
        out = block_smear(out, rng, n=int(2 + 8 * amt))
    out = rgb_split(out, int(2 + 14 * amt * (1 if rng.random() < 0.5 else -1)))
    return out


def vignette(img, strength=0.25):
    h, w = img.shape[:2]
    y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    m = 1 - strength * (x * x * 0.6 + y * y)
    return (img * m[..., None]).clip(0, 255).astype(np.uint8)
