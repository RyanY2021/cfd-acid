"""Turn scalar fields into flat, posterised, dithered graphics."""
import numpy as np
import contourpy

from .fx import BAYER4, BAYER8


def norm(f, lo=None, hi=None):
    lo = np.nanmin(f) if lo is None else lo
    hi = np.nanmax(f) if hi is None else hi
    return np.clip((f - lo) / (hi - lo + 1e-12), 0, 1)


def palette_map(x, colors, cell=1, dither=True, bayer=BAYER4, gamma=1.0):
    """Map x in [0,1] to a stepped palette with ordered dithering between steps.
    cell>1 makes chunky dither (field is assumed to be at final pixel resolution)."""
    colors = np.asarray(colors, np.uint8)
    n = len(colors)
    v = np.clip(x, 0, 1) ** gamma * (n - 1)
    if dither:
        h, w = x.shape
        b = bayer
        if cell > 1:
            b = np.repeat(np.repeat(bayer, cell, 0), cell, 1)
        bh, bw = b.shape
        tile = np.tile(b, (h // bh + 1, w // bw + 1))[:h, :w]
        idx = np.clip(np.floor(v + tile).astype(np.int32), 0, n - 1)
    else:
        idx = np.clip(np.round(v).astype(np.int32), 0, n - 1)
    return colors[idx]


def diverging_map(f, scale, neg_colors, pos_colors, zero_color, cell=1, dither=True, bayer=BAYER4,
                  gamma=0.8):
    """Signed field -> palette: zero in the middle, ramps out to each side."""
    neg = list(neg_colors)[::-1]
    cols = neg + [zero_color] + list(pos_colors)
    k = len(neg)
    x = np.clip(f / (scale + 1e-12), -1, 1)
    s = np.sign(x) * np.abs(x) ** gamma
    v = (s * k + k) / (2 * k)   # -> [0,1] with zero at centre
    return palette_map(v, cols, cell=cell, dither=dither, bayer=bayer)


def upscale(img, k):
    if k == 1:
        return img
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def contours(field, levels):
    gen = contourpy.contour_generator(z=field, line_type=contourpy.LineType.Separate)
    out = []
    for lv in np.atleast_1d(levels):
        out.append(gen.lines(lv))
    return out   # list (per level) of list of (N,2) arrays (x=col, y=row)


def bilinear(field, x, y):
    """Sample field (ny,nx) at float coords."""
    ny, nx = field.shape
    x = np.clip(x, 0, nx - 1.001)
    y = np.clip(y, 0, ny - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    a = field[y0, x0]
    b = field[y0, x0 + 1]
    c = field[y0 + 1, x0]
    d = field[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def splat_points(h, w, x, y, colors, size=2, img=None):
    """Write square points into an RGB image (painter order = array order)."""
    if img is None:
        img = np.zeros((h, w, 3), np.uint8)
    xi = np.round(x).astype(np.int32)
    yi = np.round(y).astype(np.int32)
    m = (xi >= 0) & (yi >= 0) & (xi < w - size) & (yi < h - size)
    xi, yi = xi[m], yi[m]
    cs = colors[m] if np.ndim(colors) == 2 else np.broadcast_to(np.asarray(colors, np.uint8), (len(xi), 3))
    for dy in range(size):
        for dx in range(size):
            img[yi + dy, xi + dx] = cs
    return img


class GlyphAtlas:
    """Pre-rendered monospace glyph ramp for ASCII-field rendering."""

    def __init__(self, chars, fname, size, cw, chh, color=(255, 255, 255)):
        from .core import Frame
        self.cw, self.ch = cw, chh
        self.n = len(chars)
        self.tiles = np.zeros((self.n, chh, cw), np.float32)
        for i, c in enumerate(chars):
            fr = Frame(cw, chh, bg=(0, 0, 0))
            fr.text(c, cw / 2, chh / 2, fname, size, (255, 255, 255), align="center", anchor="middle")
            self.tiles[i] = fr.array()[..., 0] / 255.0

    def render(self, levels01, color, bg=None):
        """levels01: (rows, cols) in [0,1] -> RGB image."""
        idx = np.clip((levels01 * (self.n - 1)).round().astype(np.int32), 0, self.n - 1)
        r, c = idx.shape
        tiles = self.tiles[idx]                       # r,c,ch,cw
        mask = tiles.transpose(0, 2, 1, 3).reshape(r * self.ch, c * self.cw)
        col = np.asarray(color, np.float32)
        base = np.zeros((r * self.ch, c * self.cw, 3), np.float32) if bg is None else \
            np.broadcast_to(np.asarray(bg, np.float32), (r * self.ch, c * self.cw, 3)).copy()
        out = base * (1 - mask[..., None]) + col * mask[..., None]
        return out.astype(np.uint8), mask
