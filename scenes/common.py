"""Shared HUD vocabulary + sim-field display helpers for all scenes."""
import functools
import math

import numpy as np

from engine.core import *  # noqa: F401,F403
from engine.core import CACHE
from engine import fx, fieldviz as fv


# ----------------------------------------------------------------- simulation data access
@functools.lru_cache(maxsize=None)
def sim(name, field):
    """Memory-mapped (frames, ny, nx) float16 stack."""
    return np.load(CACHE / f"{name}_{field}.npy", mmap_mode="r")


def sim_frame(name, field, i):
    a = sim(name, field)
    i = int(np.clip(i, 0, a.shape[0] - 1))
    return np.asarray(a[i], np.float32)


def sim_parts(name, i):
    a = sim(name, "parts")
    i = int(np.clip(i, 0, a.shape[0] - 1))
    p = np.asarray(a[i], np.float32)
    return p[np.isfinite(p[:, 0])]


class Resampler:
    """Bilinear crop+resample of a sim field to an output pixel grid (cached geometry)."""

    def __init__(self, src_shape, rect, out_w, out_h):
        ny, nx = src_shape
        x0, y0, x1, y1 = rect
        xs = x0 + (np.arange(out_w) + 0.5) * (x1 - x0) / out_w - 0.5
        ys = y0 + (np.arange(out_h) + 0.5) * (y1 - y0) / out_h - 0.5
        xs = np.clip(xs, 0, nx - 1.001)
        ys = np.clip(ys, 0, ny - 1.001)
        self.xi = xs.astype(np.int32)
        self.yi = ys.astype(np.int32)
        self.fx = (xs - self.xi).astype(np.float32)[None, :]
        self.fy = (ys - self.yi).astype(np.float32)[:, None]
        self.rect = rect
        self.out = (out_w, out_h)

    def __call__(self, f):
        a = f[self.yi[:, None], self.xi[None, :]]
        b = f[self.yi[:, None], self.xi[None, :] + 1]
        c = f[self.yi[:, None] + 1, self.xi[None, :]]
        d = f[self.yi[:, None] + 1, self.xi[None, :] + 1]
        return (a * (1 - self.fx) + b * self.fx) * (1 - self.fy) + (c * (1 - self.fx) + d * self.fx) * self.fy

    def to_out(self, x, y):
        x0, y0, x1, y1 = self.rect
        ow, oh = self.out
        return (x - x0 + 0.5) * ow / (x1 - x0), (y - y0 + 0.5) * oh / (y1 - y0)


def splat(img, x, y, color, size=2, alpha=1.0):
    h, w = img.shape[:2]
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    m = (xi >= 0) & (yi >= 0) & (xi < w - size) & (yi < h - size)
    xi, yi = xi[m], yi[m]
    col = np.asarray(color, np.float32)
    for dy in range(size):
        for dx in range(size):
            if alpha >= 1:
                img[yi + dy, xi + dx] = col
            else:
                img[yi + dy, xi + dx] = (img[yi + dy, xi + dx] * (1 - alpha) + col * alpha)
    return img


def vort_image(w, scale, cell=2, neg=None, pos=None, zero=BLACK, gamma=0.75, dead=0.0):
    neg = neg or [BLUE_D, BLUE, (90, 120, 255)]
    pos = pos or [(120, 10, 20), RED, (255, 140, 60)]
    if dead > 0:
        d = dead * scale
        w = np.sign(w) * np.maximum(np.abs(w) - d, 0) * (scale / (scale - d))
    return fv.diverging_map(w, scale, neg, pos, zero, cell=cell, gamma=gamma)


# ----------------------------------------------------------------- HUD vocabulary
def seg_ring(fr, cx, cy, r0, r1, t, seed, color, n=14, fill=1.0, speed=20.0, gap=2.5, step_t=1 / 15,
             alpha=1.0, jitter=True, min_frac=0.25):
    """Chunky segmented ring.  fill reveals segments progressively.  Stepped rotation."""
    rng = np.random.default_rng(seed)
    widths = rng.uniform(min_frac, 1.0, n)
    present = rng.random(n) < 0.82
    widths = widths / widths.sum() * 360
    rot = speed * step(t, step_t)
    a = rot + rng.uniform(0, 360)
    k_on = int(math.ceil(clamp(fill) * n))
    for i in range(n):
        wdt = widths[i]
        if i < k_on and present[i]:
            rr0, rr1 = r0, r1
            if jitter:
                rr1 = r0 + (r1 - r0) * (0.55 + 0.45 * rng.random())
            fr.ring_segment(cx, cy, rr0, rr1, a + gap / 2, a + wdt - gap / 2, color, alpha)
        a += wdt


def ticks(fr, cx, cy, r, n, length, color, width=1.5, alpha=1.0, a0=0.0):
    for i in range(n):
        a = math.radians(a0 + 360 * i / n)
        fr.line(cx + r * math.cos(a), cy + r * math.sin(a), cx + (r + length) * math.cos(a),
                cy + (r + length) * math.sin(a), color, width, alpha)


def corner_hud(fr, t, color=GREY, alpha=0.9, tl="CFD//SOLVER.4", tr=None, bl="QR EXC FF+", br=None,
               size=13, m=48, boxed=False, box=BLACK):
    tr = tr or f"T+{t:06.2f}"
    br = br or f"FRM {int(t * FPS):05d}"
    ch = fr.cap_height("mono", size)
    for s, x, y, al in ((tl, m, m, "left"), (tr, W - m, m, "right"), (bl, m, H - m - ch, "left"),
                        (br, W - m, H - m - ch, "right")):
        if not s:
            continue
        if boxed:
            label_box(fr, s, x, y, "mono", size, color, box, pad=(8, 5), align=al, alpha=alpha, tracking=0.08)
        else:
            fr.text(s, x, y, "mono", size, color, alpha, align=al, anchor="top", tracking=0.08)


def plus_marks(fr, pts, s=10, color=WHITE, width=1.5, alpha=1.0):
    for x, y in pts:
        crosshair(fr, x, y, s, color, width, alpha)


def banner(fr, word, y, size, color, t, speed=120.0, fname="wide", fill_every=0, fill_phase=1,
           outline_w=1.6, alpha=1.0, sep="_", fill_color=None, x0=0.0, step_t=1 / 15, solid=False):
    """Repeating WORD_WORD_ across the frame.  Outlined, except every `fill_every`-th unit
    (or all when solid=True) which is filled — the Marathon PASSPHRASE device."""
    unit = word + sep
    uw = fr.text_width(unit, fname, size)
    shift = speed * step(t, step_t) + x0
    i0 = int(math.floor(shift / uw))
    x = -(shift - i0 * uw) - uw
    i = i0 - 1
    while x < W + uw:
        filled = solid or (fill_every and i % fill_every == fill_phase)
        if filled:
            fr.text(unit, x, y, fname, size, fill_color or color, alpha, anchor="middle")
        else:
            fr.text(unit, x, y, fname, size, color, alpha, anchor="middle", stroke=outline_w)
        x += uw
        i += 1


def label_box(fr, s, x, y, fname, size, fg, bg, pad=(10, 6), align="left", alpha=1.0, tracking=0.0):
    wdt = fr.text_width(s, fname, size, tracking)
    ch = fr.cap_height(fname, size)
    bx = x - (wdt if align == "right" else wdt / 2 if align == "center" else 0) - pad[0]
    fr.rect(bx, y - pad[1], wdt + 2 * pad[0], ch + 2 * pad[1], bg, alpha)
    fr.text(s, bx + pad[0], y, fname, size, fg, alpha, anchor="top", tracking=tracking)
    return wdt + 2 * pad[0]


def seg_bar(fr, x, y, w, h, p, color, n=24, gap=3, dim=None, alpha=1.0):
    cw = (w - gap * (n - 1)) / n
    k = int(round(clamp(p) * n))
    for i in range(n):
        if i < k:
            fr.rect(x + i * (cw + gap), y, cw, h, color, alpha)
        elif dim is not None:
            fr.rect(x + i * (cw + gap), y, cw, h, dim, alpha)


def data_noise(fr, rng, x, y, w, h, cell, color, density=0.08, alpha=1.0):
    """Dim pixel-chatter texture (the dark blue data fields in the refs)."""
    nx, ny = int(w // cell), int(h // cell)
    m = rng.random((ny, nx)) < density
    lens = rng.integers(1, 6, (ny, nx))
    import skia
    path = skia.Path()
    for j, i in zip(*np.nonzero(m)):
        path.addRect(skia.Rect.MakeXYWH(x + i * cell, y + j * cell, cell * lens[j, i], cell - 1))
    fr.path(path, color, alpha)


def glyph_row(fr, x, y, cell, color, seeds, gap=None, n=4, alpha=1.0):
    gap = gap if gap is not None else cell * 2
    for k, s in enumerate(seeds):
        checker_glyph(fr, x + k * (n * cell + gap), y, cell, color, seed=s, n=n, alpha=alpha)


def counter(v, digits=6):
    return f"{int(v):0{digits}d}"


def frame_rng(t, salt=0):
    return np.random.default_rng((int(round(t * FPS)) * 7919 + salt) & 0x7FFFFFFF)


def cut_flash(t, at, dur=1 / 30):
    return at <= t < at + dur


def beat_index(t, t0=0.0):
    return int(math.floor((t - t0) / BEAT))
