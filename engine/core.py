"""Core drawing layer: a Skia-backed frame with a small motion-design vocabulary."""
import math
import pathlib

import numpy as np
import skia

ROOT = pathlib.Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "assets" / "fonts"
CACHE = ROOT / "cache"

W, H = 1920, 1080
FPS = 30
BPM = 120
BEAT = 60.0 / BPM          # seconds
BAR = 4 * BEAT

# ---------------------------------------------------------------- palette
BLACK = (4, 4, 6)
NAVY = (6, 8, 42)
BLUE = (20, 28, 255)
BLUE_D = (10, 14, 120)
GREEN = (40, 255, 60)
LIME = (190, 255, 0)
YELLOW = (235, 255, 20)
RED = (255, 30, 40)
ORANGE = (255, 60, 20)
MAGENTA = (236, 0, 95)
PAPER = (245, 245, 242)
WHITE = (250, 250, 250)
GREY = (120, 120, 130)
TEAL = (30, 220, 190)
CRIMSON = (200, 8, 30)
CRIMSON_D = (70, 0, 10)
BONE = (196, 196, 192)

# ---------------------------------------------------------------- fonts
FONTS = {
    "stencil": "BigShouldersStencil-900.ttf",
    "stencil7": "BigShouldersStencil-700.ttf",
    "cond": "BigShoulders-900.ttf",
    "cond7": "BigShoulders-700.ttf",
    "anton": "Anton-400.ttf",
    "wide": "ArchivoExp-800.ttf",
    "wide9": "ArchivoExp-900.ttf",
    "inter": "Inter-500.ttf",
    "inter4": "Inter-400.ttf",
    "inter7": "Inter-700.ttf",
    "inter9": "Inter-900.ttf",
    "tight": "InterTight-700.ttf",
    "tight9": "InterTight-900.ttf",
    "tek": "Tektur-700.ttf",
    "tek9": "Tektur-900.ttf",
    "tek5": "Tektur-500.ttf",
    "mono": "JetBrainsMono-400.ttf",
    "mono7": "JetBrainsMono-700.ttf",
    "mono8": "JetBrainsMono-800.ttf",
    "martian": "MartianMono-400.ttf",
    "martian7": "MartianMono-700.ttf",
    "pixel": "Silkscreen-400.ttf",
    "pixel7": "Silkscreen-700.ttf",
    "doto": "Doto-900.ttf",
    "chakra": "ChakraPetch-700.ttf",
    "space": "SpaceGrotesk-700.ttf",
    "serif": "Cinzel-500.ttf",
    "serif7": "Cinzel-700.ttf",
    "serif9": "Cinzel-900.ttf",
    "csc": "CormorantSC-500.ttf",
}
_TF = {}
_FONT = {}


def typeface(name):
    if name not in _TF:
        _TF[name] = skia.Typeface.MakeFromFile(str(FONT_DIR / FONTS[name]))
    return _TF[name]


def font(name, size):
    key = (name, round(size * 4) / 4)
    f = _FONT.get(key)
    if f is None:
        f = skia.Font(typeface(name), key[1])
        f.setEdging(skia.Font.Edging.kAntiAlias)
        f.setSubpixel(True)
        f.setHinting(skia.FontHinting.kNone)
        _FONT[key] = f
    return f


def col(rgb, a=1.0):
    return skia.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(max(0, min(1, a)) * 255))


# ---------------------------------------------------------------- easing / timing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lin(t, t0, t1):
    if t1 == t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))


def ease_out(x, p=3):
    return 1 - (1 - clamp(x)) ** p


def ease_in(x, p=3):
    return clamp(x) ** p


def ease_io(x, p=3):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 and p == 3 else (1 - (-2 * x + 2) ** p / 2 if x >= 0.5 else (2 * x) ** p / 2)


def expo_out(x):
    x = clamp(x)
    return 1 if x >= 1 else 1 - 2 ** (-10 * x)


def step(t, period):
    """Quantised time — gives the stuttery, frame-held motion of the refs."""
    return math.floor(t / period) * period


def hash01(*args):
    """Deterministic pseudo-random in [0,1) from integers/floats."""
    h = 0x345678
    for a in args:
        h = (h * 1000003) ^ (int(a * 1000) & 0xFFFFFFFF)
        h &= 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5bd1e995) & 0xFFFFFFFF
    h ^= h >> 15
    return (h & 0xFFFFFF) / float(0x1000000)


def flicker(t, rate=24, seed=0, p=0.5):
    return hash01(math.floor(t * rate), seed) < p


def typed(s, x):
    """Reveal string s progressively for x in [0,1] (typewriter)."""
    n = int(round(clamp(x) * len(s)))
    return s[:n]


def scramble(s, x, t, seed=0, charset="#/\\_<>[]=+-01XQ%$"):
    """Decode-in effect: characters resolve left to right, unresolved ones flicker."""
    n = len(s)
    k = int(clamp(x) * (n + 4))
    out = []
    for i, ch in enumerate(s):
        if i < k - 4 or ch == " ":
            out.append(ch)
        elif i < k:
            out.append(charset[int(hash01(i, math.floor(t * 30), seed) * len(charset))])
        else:
            out.append(" ")
    return "".join(out).rstrip()


# ---------------------------------------------------------------- frame
class Frame:
    def __init__(self, w=W, h=H, bg=BLACK, alpha_bg=1.0):
        self.w, self.h = w, h
        info = skia.ImageInfo.Make(w, h, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        self.surface = skia.Surface.MakeRaster(info)
        self.c = self.surface.getCanvas()
        if bg is None:
            self.c.clear(skia.Color(0, 0, 0, 0))
        else:
            self.c.clear(col(bg, alpha_bg))

    # -- state
    def save(self):
        self.c.save()

    def restore(self):
        self.c.restore()

    def translate(self, x, y):
        self.c.translate(x, y)

    def rotate(self, deg, cx=0, cy=0):
        self.c.rotate(deg, cx, cy)

    def scale(self, sx, sy=None):
        self.c.scale(sx, sx if sy is None else sy)

    def clip(self, x, y, w, h):
        self.c.clipRect(skia.Rect.MakeXYWH(x, y, w, h))

    def fill(self, color, alpha=1.0):
        self.c.drawPaint(self._paint(color, alpha))

    # -- paint
    @staticmethod
    def _paint(color, alpha=1.0, stroke=None, blend=None, cap=None, dash=None, aa=True):
        p = skia.Paint(AntiAlias=aa, Color=col(color, alpha))
        if stroke is not None:
            p.setStyle(skia.Paint.kStroke_Style)
            p.setStrokeWidth(stroke)
            if cap == "round":
                p.setStrokeCap(skia.Paint.kRound_Cap)
            elif cap == "square":
                p.setStrokeCap(skia.Paint.kSquare_Cap)
            p.setStrokeJoin(skia.Paint.kMiter_Join)
        if dash is not None:
            p.setPathEffect(skia.DashPathEffect.Make(list(dash[0]), dash[1]))
        if blend is not None:
            p.setBlendMode(getattr(skia.BlendMode, "k" + blend.capitalize()))
        return p

    # -- primitives
    def rect(self, x, y, w, h, color, alpha=1.0, stroke=None, blend=None):
        self.c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), self._paint(color, alpha, stroke, blend))

    def line(self, x0, y0, x1, y1, color, width=1.0, alpha=1.0, cap=None, dash=None, blend=None):
        self.c.drawLine(x0, y0, x1, y1, self._paint(color, alpha, width, blend, cap, dash))

    def circle(self, x, y, r, color, alpha=1.0, stroke=None, blend=None, dash=None):
        self.c.drawCircle(x, y, r, self._paint(color, alpha, stroke, blend, dash=dash))

    def polyline(self, pts, color, width=1.0, alpha=1.0, closed=False, dash=None, blend=None,
                 fill=False, cap=None):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return
        path = skia.Path()
        path.addPoly([skia.Point(float(a), float(b)) for a, b in pts], closed)
        if fill:
            self.c.drawPath(path, self._paint(color, alpha, None, blend))
        else:
            self.c.drawPath(path, self._paint(color, alpha, width, blend, cap, dash))

    def polylines(self, plist, color, width=1.0, alpha=1.0, dash=None, blend=None):
        """Many polylines in a single path (fast)."""
        path = skia.Path()
        for pts in plist:
            if len(pts) < 2:
                continue
            path.addPoly([skia.Point(float(a), float(b)) for a, b in pts], False)
        self.c.drawPath(path, self._paint(color, alpha, width, blend, None, dash))

    def segments(self, xy0, xy1, color, width=1.0, alpha=1.0, blend=None):
        path = skia.Path()
        for (a, b), (c, d) in zip(xy0, xy1):
            path.moveTo(float(a), float(b))
            path.lineTo(float(c), float(d))
        self.c.drawPath(path, self._paint(color, alpha, width, blend))

    def squares(self, xy, size, color, alpha=1.0, blend=None):
        """Draw many axis-aligned squares centred at xy."""
        path = skia.Path()
        hs = size / 2
        for a, b in xy:
            path.addRect(skia.Rect.MakeXYWH(float(a) - hs, float(b) - hs, size, size))
        self.c.drawPath(path, self._paint(color, alpha, None, blend))

    def arc(self, cx, cy, r, a0, a1, width, color, alpha=1.0, blend=None):
        """Band arc between angles a0..a1 (degrees, clockwise from +x) with butt caps."""
        if abs(a1 - a0) < 1e-3:
            return
        path = skia.Path()
        path.addArc(skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), a0, a1 - a0)
        self.c.drawPath(path, self._paint(color, alpha, width, blend))

    def ring_segment(self, cx, cy, r0, r1, a0, a1, color, alpha=1.0, blend=None):
        """Filled annular sector — crisp chunky ring blocks."""
        if abs(a1 - a0) < 1e-3:
            return
        path = skia.Path()
        path.arcTo(skia.Rect.MakeLTRB(cx - r1, cy - r1, cx + r1, cy + r1), a0, a1 - a0, True)
        path.arcTo(skia.Rect.MakeLTRB(cx - r0, cy - r0, cx + r0, cy + r0), a1, a0 - a1, False)
        path.close()
        self.c.drawPath(path, self._paint(color, alpha, None, blend))

    def path(self, path, color, alpha=1.0, stroke=None, blend=None):
        self.c.drawPath(path, self._paint(color, alpha, stroke, blend))

    # -- text
    def text_width(self, s, fname, size, tracking=0.0):
        f = font(fname, size)
        return f.measureText(s) + tracking * size * max(0, len(s) - 1)

    def cap_height(self, fname, size):
        return font(fname, size).getMetrics().fCapHeight

    def text(self, s, x, y, fname, size, color, alpha=1.0, align="left", anchor="baseline",
             tracking=0.0, stroke=None, blend=None, rot=0.0):
        """Draw text. anchor: baseline | top (cap top) | middle (cap middle) | bottom.
        tracking is in em units. rot rotates around (x,y)."""
        if not s:
            return 0.0
        f = font(fname, size)
        wdt = self.text_width(s, fname, size, tracking)
        ch = f.getMetrics().fCapHeight
        if rot:
            self.c.save()
            self.c.rotate(rot, x, y)
        ox = x - (wdt if align == "right" else wdt / 2 if align == "center" else 0)
        oy = y + (ch if anchor == "top" else ch / 2 if anchor == "middle" else 0)
        p = self._paint(color, alpha, stroke, blend)
        if tracking == 0:
            self.c.drawString(s, ox, oy, f, p)
        else:
            glyphs = f.textToGlyphs(s)
            widths = f.getWidths(glyphs)
            xs = []
            cx = ox
            for wv in widths:
                xs.append(cx)
                cx += wv + tracking * size
            blob = skia.TextBlob.MakeFromPosTextH(s, xs, oy, f)
            self.c.drawTextBlob(blob, 0, 0, p)
        if rot:
            self.c.restore()
        return wdt

    def vtext(self, s, x, y, fname, size, color, alpha=1.0, tracking=0.0, up=True, align="left"):
        """Vertical text reading bottom-to-top (up) or top-to-bottom."""
        self.c.save()
        self.c.translate(x, y)
        self.c.rotate(-90 if up else 90)
        self.text(s, 0, 0, fname, size, color, alpha, align=align, anchor="top", tracking=tracking)
        self.c.restore()

    # -- images
    def image(self, arr, x=0, y=0, w=None, h=None, smooth=False, alpha=1.0, blend=None):
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[2] == 3:
            a = np.full(arr.shape[:2] + (1,), 255, np.uint8)
            arr = np.ascontiguousarray(np.concatenate([arr, a], axis=2))
        img = skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType)
        w = arr.shape[1] if w is None else w
        h = arr.shape[0] if h is None else h
        samp = skia.SamplingOptions(skia.FilterMode.kLinear) if smooth else skia.SamplingOptions(skia.FilterMode.kNearest)
        p = skia.Paint(AntiAlias=False)
        p.setAlphaf(alpha)
        if blend:
            p.setBlendMode(getattr(skia.BlendMode, "k" + blend.capitalize()))
        self.c.drawImageRect(img, skia.Rect.MakeXYWH(x, y, w, h), samp, p)

    def array(self):
        return self.surface.makeImageSnapshot().toarray()[..., :3].copy()

    def rgba(self):
        return self.surface.makeImageSnapshot().toarray()

    def set_array(self, rgb):
        self.c.clear(skia.Color(0, 0, 0, 255))
        self.image(rgb)


# ---------------------------------------------------------------- reusable glyph-y graphic devices
def crosshair(fr, x, y, s, color, width=2, alpha=1.0):
    fr.line(x - s, y, x + s, y, color, width, alpha)
    fr.line(x, y - s, x, y + s, color, width, alpha)


def crop_marks(fr, x, y, w, h, s, color, width=2, alpha=1.0, gap=0):
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        fr.line(px - dx * gap, py, px + dx * s, py, color, width, alpha)
        fr.line(px, py - dy * gap, px, py + dy * s, color, width, alpha)


def brackets(fr, x, y, w, h, s, color, width=2, alpha=1.0):
    crop_marks(fr, x, y, w, h, s, color, width, alpha)


def barcode(fr, x, y, w, h, color, seed=0, alpha=1.0, vertical=False):
    rng = np.random.default_rng(seed)
    pos = 0.0
    while pos < w:
        bw = rng.choice([1, 1, 2, 3, 4])
        if rng.random() < 0.6:
            if vertical:
                fr.rect(x, y + pos, h, bw, color, alpha)
            else:
                fr.rect(x + pos, y, bw, h, color, alpha)
        pos += bw + rng.choice([1, 2, 2, 3])


def checker_glyph(fr, x, y, cell, color, seed=0, n=4, alpha=1.0, density=0.5):
    """Small pixel glyph like the Marathon iconography."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, (n + 1) // 2)) < density
    m = np.concatenate([m, m[:, ::-1][:, n % 2:]], axis=1)  # mirror -> symbol-like
    for j in range(n):
        for i in range(n):
            if m[j, i]:
                fr.rect(x + i * cell, y + j * cell, cell, cell, color, alpha)


def checkerboard(fr, x, y, w, h, cell, c1, c2=None, alpha=1.0, phase=0):
    if c2 is not None:
        fr.rect(x, y, w, h, c2, alpha)
    path = skia.Path()
    nx, ny = int(math.ceil(w / cell)), int(math.ceil(h / cell))
    for j in range(ny):
        for i in range(nx):
            if (i + j + phase) % 2 == 0:
                path.addRect(skia.Rect.MakeXYWH(x + i * cell, y + j * cell,
                                                min(cell, w - i * cell), min(cell, h - j * cell)))
    fr.path(path, c1, alpha)


def hazard(fr, x, y, w, h, color, stripe=20, alpha=1.0, phase=0.0):
    fr.save()
    fr.clip(x, y, w, h)
    path = skia.Path()
    k = -h + (phase % (2 * stripe)) - 2 * stripe
    while k < w + h:
        path.moveTo(x + k, y + h)
        path.lineTo(x + k + stripe, y + h)
        path.lineTo(x + k + stripe + h, y)
        path.lineTo(x + k + h, y)
        path.close()
        k += 2 * stripe
    fr.path(path, color, alpha)
    fr.restore()


def nabla(fr, x, y, size, color, alpha=1.0, weight=0.16):
    """Draw a ∇ glyph whose cap-height box starts at baseline y (x = left)."""
    w = size * 0.78
    h = size * 0.72
    t = size * weight
    outer = skia.Path()
    outer.moveTo(x, y - h)
    outer.lineTo(x + w, y - h)
    outer.lineTo(x + w / 2, y)
    outer.close()
    # inner triangle (hole)
    k = t * 1.9
    inner = skia.Path()
    inner.moveTo(x + k, y - h + t)
    inner.lineTo(x + w - k, y - h + t)
    inner.lineTo(x + w / 2, y - t * 2.1)
    inner.close()
    outer.setFillType(skia.PathFillType.kEvenOdd)
    outer.addPath(inner)
    fr.path(outer, color, alpha)
    return w


def micro_block(fr, x, y, lines, color, size=13, fname="mono", alpha=1.0, lh=1.35, tracking=0.04):
    for i, s in enumerate(lines):
        fr.text(s, x, y + i * size * lh, fname, size, color, alpha, anchor="top", tracking=tracking)
