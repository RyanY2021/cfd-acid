"""0 — INTRO (0–6 s): aerial surf (Blender, ocean/build_ocean.py) graded + survey HUD, dissolving into the palette."""
import functools

from PIL import Image

from scenes.common import *

CUT_GLITCH = False
GRAIN = 4.0
OCEAN = ROOT / "render" / "ocean"
N_SRC = 180
LENS_K = 35 / 36 * W            # px per (metre / altitude)
TEAL = (20, 200, 180)
SHORE, LAND_X = 26.0, 30.5          # keep in sync with ocean/build_ocean.py
PAL = [BLACK, NAVY, BLUE_D, BLUE, TEAL, LIME, WHITE]


@functools.lru_cache(maxsize=4)
def src(f):
    f = int(np.clip(f, 0, N_SRC - 1))
    return np.asarray(Image.open(OCEAN / f"surf_{f:03d}.png").convert("RGB"), np.float32) / 255


def cam(t):
    k = t / 6.0
    return 8 + 3 * k, -10 + 16 * k, 168 - 34 * k, math.radians(1.5 * k)


def world_to_screen(t, X, Y, Z=0.0):
    cx, cy, alt, th = cam(t)
    dx, dy = X - cx, Y - cy
    lx = dx * math.cos(th) + dy * math.sin(th)
    ly = -dx * math.sin(th) + dy * math.cos(th)
    s = LENS_K / (alt - Z)
    return W / 2 + lx * s, H / 2 - ly * s


def grade(img):
    """Marathon reveal look: teal shadows, warm-neutral highlights, lifted & crushed."""
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    sh = np.clip(1 - lum * 1.6, 0, 1)[..., None]
    hi = np.clip(lum * 1.4 - 0.5, 0, 1)[..., None]
    out = img + sh * np.array([-0.05, 0.03, 0.035], np.float32) + hi * np.array([0.03, 0.015, -0.03], np.float32)
    grey = lum[..., None]
    out = grey + (out - grey) * 1.12
    out = out * np.array([0.93, 0.98, 0.96], np.float32)
    out = np.clip((out - 0.035) / 0.95, 0, 1) ** 1.16
    return out


def breaker_x(t, n):
    """x (m) of breaking front n at time t — mirrors the water shader's phase."""
    return SHORE + 6.2 * (n + 0.95 + t / 4.2) - 40.0


def snap_to_foam(t, x_pred, y):
    """Lock onto the nearest bright foam front actually present in the rendered frame."""
    img = src(t * FPS)
    yi = int(np.clip(y, 12, H - 12))
    shore, _ = world_to_screen(t, SHORE, 0.0)
    prof = img[yi - 10:yi + 10, :int(shore) - 20].mean(axis=(0, 2))
    prof = np.convolve(prof, np.ones(9) / 9, mode="same")
    pk = np.flatnonzero((prof[1:-1] > prof[:-2]) & (prof[1:-1] >= prof[2:]) & (prof[1:-1] > 0.62)) + 1
    if len(pk) == 0:
        return x_pred
    return float(pk[np.argmin(np.abs(pk - x_pred))])


def hud(fr, t):
    t_real = t
    on = lambda a: t_real >= a
    # registration grid of '+'
    if on(0.3):
        for gx in (0.18, 0.5, 0.82):
            for gy in (0.22, 0.78):
                if flicker(t + gx, 3, int(gx * 10 + gy * 7), 0.92) or t > 1.0:
                    crosshair(fr, W * gx, H * gy, 9, LIME, 2)
    # survey header
    if on(0.2):
        _, _, alt, _ = cam(t)
        micro_block(fr, 64, 64, [typed("COASTAL SURVEY // SITE 07", lin(t, 0.2, 0.8)),
                                 typed(f"ALT {alt:6.1f} M   //   DESCENT", lin(t, 0.5, 1.1)),
                                 typed("SURF ZONE  //  Re ~ 1E7  //  TURBULENT", lin(t, 0.8, 1.5))],
                    LIME, 14, "mono7", lh=1.5)
    # vertical locator stack on the beach (after the reference's coordinate column)
    bx, by = world_to_screen(t, SHORE - 9.0, 6.0)
    if on(0.9):
        fr.vtext("07 · 3 ·", bx - 30, by - 150, "mono7", 15, LIME, tracking=0.2)
        fr.vtext(scramble("1822", lin(t, 0.9, 1.5), t, 4), bx - 34, by + 60, "tek9", 58, LIME, tracking=0.0)
        barcode(fr, bx - 30, by + 110, 18, 60, LIME, seed=5, vertical=True)
        fr.text("1", bx + 70, by - 10, "mono7", 22, LIME, anchor="middle")
    # tracker on a breaking wave front
    if on(1.6):
        t = step(t, 0.1)
        n = 4
        while breaker_x(t, n) > SHORE - 6:
            n -= 1
        X = breaker_x(t, n)
        x, y = world_to_screen(t, X, 18.0)
        x = snap_to_foam(t, x, y)
        s = 54
        brackets(fr, x - s, y - s, 2 * s, 2 * s, 14, WHITE, 2)
        crosshair(fr, x, y, 7, WHITE, 1.5)
        label_box(fr, f"BREAKER_{n + 7:02d}   c 1.48 M/S", x + s + 10, y - s, "mono7", 13, BLACK, WHITE,
                  pad=(6, 4))
        fr.line(x + s, y - s + 9, x + s + 10, y - s + 9, WHITE, 1.5)
    t = t_real
    # compound labels
    if on(2.2):
        for i, (X, Y, lab) in enumerate(((60, 12, "WIND TUNNEL 02"), (70, 38, "HPC CLUSTER"),
                                         (62, -40, "KELVIN HYDRODYNAMICS LABORATORY"))):
            if t < 2.2 + i * 0.25:
                continue
            x, y = world_to_screen(t, X, Y, 4.0)
            if 0 < x < W and 0 < y < H:
                fr.rect(x - 5, y - 5, 10, 10, YELLOW)
                dx = 40 if len(lab) < 20 else -40
                fr.line(x, y, x + dx, y - 40, YELLOW, 1.5)
                label_box(fr, lab, x + dx + (4 if dx > 0 else -4), y - 52, "mono7", 12, BLACK, YELLOW,
                          pad=(5, 3), align="left" if dx > 0 else "right")
    # bottom rail
    if on(1.2):
        fr.text("FLUID IN MOTION // OBSERVE", 64, H - 64, "mono7", 14, LIME, anchor="baseline", tracking=0.2)
        glyph_row(fr, W - 250, 64, 5, LIME, [3, 9, 17, 21], n=4)


def divider_x(t):
    return world_to_screen(t, LAND_X, 0.0)[0]


def vscroll(fr, word, x, size, color, t, speed, alpha, stroke=None):
    """A column of WORD_WORD_ reading bottom-to-top, sliding (stepped like the HUD)."""
    unit = word + "_"
    uw = fr.text_width(unit, "wide9", size)
    off = (speed * step(t, 1 / 15)) % uw
    fr.save()
    fr.translate(x, H)
    fr.rotate(-90)
    pos = -uw + off
    while pos < H + uw:
        fr.text(unit, pos, 0, "wide9", size, color, alpha, anchor="middle", stroke=stroke)
        pos += uw
    fr.restore()


def loading(fr, t, x0):
    fr.save()
    fr.clip(x0, 0, W - x0, H)
    span = W - x0
    cols = [(0.17, LIME, None, 140, 0.30), (0.5, WHITE, 2.0, -110, 0.45), (0.83, LIME, 2.0, 170, 0.40)]
    for fx_, col_, stroke, speed, alpha in cols:
        vscroll(fr, "LOADING", x0 + span * fx_, 150, col_, t, speed, alpha, stroke)
    pct = int(100 * ease_in(lin(t, 0.4, 5.0), 1.5))
    label_box(fr, f"LOADING  {pct:03d}%", x0 + 20, H / 2 + 26, "mono7", 13, BLACK, LIME, pad=(6, 4))
    seg_bar(fr, x0 + 14, H / 2 + 56, 200, 7, pct / 100, LIME, n=20, dim=(40, 50, 30))
    fr.restore()


def render(t, T):
    rng = frame_rng(T)
    f = src(t * FPS)
    img = grade(f)
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    rgb = (img * 255).astype(np.uint8)
    xd = int(divider_x(t))
    # land side: darken so the type reads and the blocky set recedes
    rgb[:, xd:] = (rgb[:, xd:] * 0.72).astype(np.uint8)
    # end of shot: the divider sweeps right and the whole frame becomes a simulation field
    sx = xd + int((W - xd) * ease_io(lin(t, 4.4, 4.95)))
    k = lin(t, 4.95, 5.4)
    post = fv.palette_map(np.clip(lum[:, :sx] * 1.15, 0, 1), PAL, cell=1 + int(3 * k))
    cell = int(4 + 28 * k) // 4 * 4 if k > 0 else 0
    if cell >= 4:
        post = fx.pixelate(post, cell)
    rgb[:, :sx] = post
    fr = Frame()
    fr.image(rgb)
    if sx < W:
        loading(fr, t, sx)
    if t < 5.3:
        hud(fr, t)
    if 0 < sx < W:
        fr.rect(sx - 2, 0, 4, H, LIME)
        fr.rect(sx + 6, 0, 1, H, LIME, 0.5)
        label_box(fr, "RESAMPLE // FIELD", sx + 14, H / 2 - 10, "mono7", 13, BLACK, LIME, pad=(6, 4))
    # opening: black hold, then slit reveal
    if t < 0.45:
        kk = ease_out(lin(t, 0.05, 0.45))
        hh = H / 2 * kk
        fr.rect(0, 0, W, H / 2 - hh, BLACK)
        fr.rect(0, H / 2 + hh, W, H / 2 - hh, BLACK)
        fr.line(0, H / 2 - hh, W, H / 2 - hh, LIME, 2)
        fr.line(0, H / 2 + hh, W, H / 2 + hh, LIME, 2)
    out = fr.array()
    if 4.95 < t < 5.6:
        out = fx.glitch_burst(out, rng, lin(t, 4.95, 5.6))
        out = fx.pixel_blocks(out, rng, int(40 * lin(t, 5.1, 5.6)), 24, [LIME, BLUE, WHITE])
    if t >= 5.6:
        fr = Frame()
        if t < 5.75:
            fr.fill(LIME if t < 5.6 + 1 / 30 else BLACK)
        crosshair(fr, W / 2, H / 2, 14, LIME, 2, 0.6)
        out = fr.array()
    return out
