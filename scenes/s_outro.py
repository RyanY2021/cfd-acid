"""I — OUTRO (13 s): SIMULATE THE [corrupt] -> logo card becomes a CRT -> the old computer crashes, sparks,
smoke (Blender: computer/build_computer.py) -> SIMULATION EXHAUSTED."""
import functools

from PIL import Image

from scenes.common import *

CUT_GLITCH = True
CRT_DIR = ROOT / "render" / "crt"
T_CRT = 3.0             # logo card starts turning into a CRT
T_3D = 4.5              # cut to the Blender computer
N_3D = 240              # 8 s of Blender frames
T_END = T_3D + N_3D / FPS
F_EXHAUST = 168         # Blender frame where SIMULATION EXHAUSTED starts
JUNK = "#%&@$*!?<>/\\|=+~^{}[]0123456789ABCDEFXQZ_"


def junk(n, t, seed=0, rate=15):
    k = int(t * rate)
    return "".join(JUNK[int(hash01(i, k, seed) * len(JUNK))] for i in range(n))


def streaks_bg(t, alpha=0.35):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = BLACK
    p = sim_parts("karman", 60 + t * FPS)
    p = p[::2]
    x = (p[:, 0] - 300) * 2.25
    y = p[:, 1] * 2.25
    splat(img, x, y, (90, 90, 110), size=2, alpha=alpha)
    return img


def wind(img, rng, amt, t):
    """Blow pixels rightward: per-row shift growing with amt, streaked."""
    if amt <= 0:
        return img
    out = img.copy()
    h = img.shape[0]
    rows = np.arange(h)
    shift = (amt ** 2 * 1400 * (0.5 + 0.5 * np.sin(rows / 37.0 + t * 3) * np.sin(rows / 11.0))).astype(int)
    shift = (shift // 8) * 8
    for y in range(0, h, 4):
        s = shift[y]
        if s > 0:
            out[y:y + 4, s:] = img[y:y + 4, :-s]
            out[y:y + 4, :s] = img[y:y + 4, :1]
    return out


def logo(fr, t):
    size = 430
    fr.text("CFD", W / 2, 640, "stencil", size, LIME, align="center", tracking=0.02)
    plus_marks(fr, [(580, 466), (1340, 466)], 14, LIME, 2)
    fr.circle(1292, 300, 11, LIME, stroke=5)
    y = 720
    x0 = W / 2 - 330
    barcode(fr, x0, y - 8, 70, 18, LIME, seed=3)
    fr.text("X", x0 + 86, y + 1, "mono7", 18, LIME, anchor="middle")
    fr.circle(x0 + 120, y + 1, 8, LIME, stroke=2.5)
    fr.rect(x0 + 117, y - 2, 6, 6, LIME)
    fr.text("20 26", x0 + 142, y + 1, "tek", 22, LIME, anchor="middle")
    fr.text("COMPUTATIONAL FLUID DYNAMICS", x0 + 230, y - 6, "mono", 12, LIME, anchor="middle", tracking=0.08)
    fr.text('"SIMULATE THE ' + junk(10, t, 3, 12) + '"', x0 + 230, y + 10, "mono", 12, LIME, anchor="middle",
            tracking=0.08)


def logo_card(t):
    fr = Frame()
    fr.image(streaks_bg(t, 0.18))
    logo(fr, t)
    return fr


# ------------------------------------------------------------------ CRT look (2D, before the Blender cut)
@functools.lru_cache(maxsize=1)
def _grid():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)


TW, TH = 1920, 1440        # screen texture: the 16:9 card padded to 4:3; frame 0 shows its middle 1:1
PAD = (TH - H) // 2
DISSOLVE = 0.4             # seconds of 2D -> Blender match-dissolve (Blender camera holds still)


def scanline_mask():
    """Same profile as the Blender screen shader: 0.78 + 0.22 sin(720 pi v), v over the 1440-px texture."""
    v = (np.arange(H) + PAD + 0.5) / TH
    return (0.78 + 0.22 * np.sin(720 * np.pi * v)).astype(np.float32)[:, None, None]


def crt(img, k, rng):
    """Turn the flat card into what the CRT will show at frame 0 (no zoom, same scale): scanlines,
    the glass bulge (centre ~3% larger) and the tube's dark corners.  k in 0..1."""
    if k <= 0:
        return img
    u, v = _grid()
    r2 = np.minimum(u * u + v * v * 0.56, 1.0)
    m = 1 + 0.033 * k * (1 - r2)
    sx = np.clip(u / m * W / 2 + W / 2, 0, W - 1).astype(np.int32)
    sy = np.clip(v / m * H / 2 + H / 2, 0, H - 1).astype(np.int32)
    out = img[sy, sx].astype(np.float32)
    out *= 1 + 0.3 * k * (scanline_mask() * 1.28 - 1)     # Blender's denoiser softens its scanlines
    vign = np.clip(1.0 - 0.35 * k * (u * u + v * v) ** 1.5, 0, 1)[..., None]
    out *= vign * (1 - 0.04 * k * rng.random())
    return np.clip(out, 0, 255).astype(np.uint8)


def halation(img, amt=0.35):
    """Mild bloom shared by the 2D CRT phase and every Blender computer frame."""
    f = img.astype(np.float32)
    hi = np.clip(f - 150, 0, None)
    small = hi[::8, ::8]
    for _ in range(2):
        small = (small + np.roll(small, 1, 0) + np.roll(small, -1, 0) + np.roll(small, 1, 1) + np.roll(small, -1, 1)) / 5
    blur = np.repeat(np.repeat(small, 8, 0), 8, 1)[:img.shape[0], :img.shape[1]]
    return np.clip(f + blur * amt, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ CRT screen texture for Blender (4:3)
def padded_card(t):
    card = logo_card(t).array()
    return np.pad(card, ((PAD, PAD), (0, 0), (0, 0)), mode="reflect")


def screen_texture(f):
    """What the computer's CRT shows on Blender frame f (TW x TH)."""
    rng = np.random.default_rng(f * 13 + 1)
    t = T_3D + f / FPS
    if f < 100:
        img = padded_card(t)
        if f >= 72:
            k = (f - 72) / 28
            img = fx.slice_shift(img, rng, int(4 + 30 * k), int(25 + 160 * k), int(50 + 650 * k))
            img = np.roll(img, int((k ** 2) * 2900) % TH, axis=0)                # losing vertical hold
            img = fx.rgb_split(img, int(4 + 40 * k))
            if k > 0.35:
                img = fx.block_displace(img, rng, n=int(10 + 40 * k), cell=52, max_cells=8, max_shift=8)
            if k > 0.5:
                fr = Frame(TW, TH, bg=None)
                lines = ["FATAL // VELOCITY FIELD = NaN", "CFL 4.2E+07 > 1.0", "SOLVER DIVERGED AT STEP 5000001",
                         "CORE DUMP " + junk(18, f / 30.0, 9, 30)]
                for i, s_ in enumerate(lines[:1 + int((k - 0.5) * 8)]):
                    label_box(fr, s_, 330, 420 + i * 120, "mono7", 58, WHITE if i % 2 else BLACK,
                              RED if i % 2 == 0 else BLACK, pad=(24, 16))
                img = _over(img, fr.rgba())
            if k > 0.75:
                for _ in range(6):
                    y = int(rng.integers(0, TH - 80))
                    img[y:y + int(rng.integers(10, 80))] = rng.choice([LIME, RED, WHITE, BLUE])
    elif f < 102:
        img = np.full((TH, TW, 3), 255, np.uint8)
    else:
        img = np.zeros((TH, TW, 3), np.uint8)
        k = (f - 102) / 4.0
        cy, cx = TH // 2, TW // 2
        if k < 1:                                    # collapse to a horizontal line
            hh = int(TH * (1 - k) ** 2 / 2) + 4
            img[cy - hh:cy + hh, 80:TW - 80] = 235
        elif f < 116:                                # then to a dot
            k2 = (f - 106) / 10.0
            w2 = int(880 * (1 - k2) ** 3) + 8
            hgt = 4 if k2 < 0.4 else 8
            img[cy - hgt:cy + hgt, cx - w2:cx + w2] = int(255 * (1 - 0.6 * k2))
            if k2 > 0.4:
                r = int(18 * (1 - k2)) + 5
                img[cy - r:cy + r, cx - r:cx + r] = int(255 * (1 - k2))
    return img


def _over(rgb, rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgb * (1 - a) + rgba[..., :3].astype(np.float32)).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------ final card
def exhausted(fr, lt, dim=True):
    if dim:
        fr.rect(0, 0, W, H, BLACK, 0.35 * clamp(lt / 0.4))
    y = 470
    if lt > 0.0:
        label_box(fr, "PROCESS 0xCFD // TERMINATED", 160, y - 150, "mono7", 18, WHITE, RED, pad=(12, 8))
    if lt > 0.25:
        fr.text(scramble("SIMULATION", lin(lt, 0.25, 0.7), lt, 11), 152, y, "wide9", 150, WHITE, anchor="middle")
    if lt > 0.75:
        s = scramble("EXHAUSTED", lin(lt, 0.75, 1.3), lt, 12)
        if lt > 1.3 and flicker(lt, 12, 5, 0.12):
            s = junk(9, lt, 4, 24)
        fr.text(s, 152, y + 170, "wide9", 150, RED, anchor="middle")
    if lt > 1.2:
        lines = ["RESIDUAL ......... NaN", "TIME STEP ........ 5,000,001", "CFL .............. 4.2E+07",
                 "STATUS ........... DIVERGED", "HARDWARE ......... OFFLINE"]
        for i, s in enumerate(lines):
            if lt > 1.2 + i * 0.12:
                fr.text(s, 160, y + 300 + i * 26, "mono7", 16, (200, 200, 205), anchor="top", tracking=0.08)
        fr.rect(152, y + 280, 760, 3, RED)


@functools.lru_cache(maxsize=2)
def crt_frame(f):
    f = int(np.clip(f, 0, N_3D - 1))
    return np.asarray(Image.open(CRT_DIR / f"crt_{f:03d}.png").convert("RGB"))


def render(t, T):
    rng = frame_rng(T)
    if t < T_CRT:
        fr = Frame()
        fr.image(streaks_bg(t))
        tl = Frame(bg=None)
        if t >= 0:
            tl.text("SIMULATE", 120, 330, "wide9", 190, WHITE, anchor="middle")
        if t >= BEAT:
            tl.text("THE", 120, 530, "wide9", 190, WHITE, anchor="middle")
            # the voice is stuck on "the": echo copies step out like a jammed buffer
            n_echo = int(clamp((t - 1.0) / 0.09, 0, 9)) if t < 1.95 else 0
            tw = tl.text_width("THE", "wide9", 190)
            for e in range(1, n_echo + 1):
                tl.text("THE", 120 + e * (tw * 0.18), 530 + (e % 2) * 6, "wide9", 190,
                        LIME if e == n_echo else WHITE, 0.85 - e * 0.07, anchor="middle", stroke=2)
        if t >= 2 * BEAT:
            s = junk(10, t, 7, 15)
            tl.text(s, 120, 730, "wide9", 190, LIME, anchor="middle")
        layer = tl.rgba()
        amt = lin(t, 1.95, 3.0)
        comp = wind(layer, rng, amt, t)
        fr.image(comp)
        img = fr.array()
        if 1.0 <= t < 1.95:
            band = (int(730 - 120), int(730 + 120))
            seg = img[band[0]:band[1]]
            img[band[0]:band[1]] = fx.glitch_burst(seg, rng, 0.5 + 0.4 * rng.random())
        if amt > 0.3:
            img = fx.pixel_blocks(img, rng, int(80 * amt), 16, [LIME, WHITE])
        return img
    if t < T_3D + DISSOLVE:
        img = logo_card(t).array()
        if T_CRT <= t < T_CRT + 2 / 30:
            img = 255 - img
        k = ease_io(lin(t, T_CRT + 0.25, T_3D))
        img = halation(crt(img, k, rng), 0.35 * k)
        if t >= T_3D:                                  # match-dissolve into the Blender frame (camera holding)
            a = ease_io(lin(t, T_3D, T_3D + DISSOLVE))
            b = halation(crt_frame(int(round((t - T_3D) * FPS))))
            img = (img * (1 - a) + b * a).astype(np.uint8)
        return img
    if t < T_END:
        f = int(round((t - T_3D) * FPS))
        img = halation(crt_frame(f))
        if f >= F_EXHAUST:
            fr = Frame()
            fr.image(img)
            exhausted(fr, (f - F_EXHAUST) / FPS)
            img = fr.array()
            if f == F_EXHAUST or flicker(t, 10, 21, 0.04):
                img = fx.slice_shift(img, rng, 6, 30, 140)
        return img
    fr = Frame()
    lt = t - T_END
    if lt < 0.3 or flicker(t, 10, 3, 0.5):
        exhausted(fr, 3.0, dim=False)
    return fr.array()
