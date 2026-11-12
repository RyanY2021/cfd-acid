"""G4 — KELVIN–HELMHOLTZ (66–71 s): pseudo-spectral shear layer in NuCaloric magenta/Swiss dress."""
from scenes.common import *

SY0, SY1 = 330, 810                     # strip on screen
PINK_L = (250, 196, 218)
DARK_M = (120, 0, 48)
_RS = None


def strip(t):
    global _RS
    i = 34 + t * FPS * 1.05
    c = sim_frame("kh", "c", i)
    if _RS is None:
        _RS = Resampler(c.shape, (0, 32, 1024, 288), W, SY1 - SY0)
    f = _RS(c)
    img = fv.palette_map(np.clip(f, 0, 1), [PAPER, PINK_L, MAGENTA, MAGENTA], cell=2)
    return img, f


def render(t, T):
    rng = frame_rng(T)
    fr = Frame(bg=PAPER)
    img, f = strip(t)
    fr.image(img, 0, SY0)
    ls = fv.contours(f[::2, ::2], [0.5])[0]
    fr.polylines([ln * 2 + [0, SY0] for ln in ls if len(ln) > 6], BLACK, 1.6)
    crop_marks(fr, 70, SY0 - 26, W - 140, SY1 - SY0 + 52, 34, BLACK, 2, gap=-14)

    # roll crosshair
    cx, cy, r = 1210, (SY0 + SY1) / 2 - 10, 150
    fr.circle(cx, cy, r, BLACK, stroke=2)
    crosshair(fr, cx, cy, 34, BLACK, 2)
    fr.rect(cx - r - 40, cy - r - 30, 2 * r + 80, 2 * r + 60, BLACK, stroke=2)
    micro_block(fr, cx - r - 30, cy - r - 18, ["ROLL_03", "9D0E28"], BLACK, 15, "inter", lh=1.25, tracking=0)
    micro_block(fr, cx + r - 40, cy + r - 20, ["24SD949"], BLACK, 15, "inter", lh=1.2, tracking=0)

    # Swiss typography
    fr.text("Kelvin–Helmholtz", 110, 70, "inter", 96, BLACK, anchor="top", tracking=-0.02)
    fr.text("Instability", 110, 180, "inter", 96, MAGENTA, anchor="top", tracking=-0.02)
    fr.text("KH", W - 100, 300, "tek9", 260, MAGENTA, align="right", anchor="baseline")
    fr.text("Shear layer roll-up", W - 102, 64, "inter", 22, BLACK, align="right", anchor="top")

    # side rail
    fr.rect(0, 0, 46, H, MAGENTA)
    fr.vtext("Shear layer // Sponsored by fluid dynamics", 12, 720, "inter", 20, WHITE)
    fr.line(23, 820, 23, 900, WHITE, 3)
    fr.polyline([(11, 832), (23, 820), (35, 832)], WHITE, 3)

    # bottom data
    fr.text("ΔU 2.0   δ 0.07   Ri < 1/4", 110, 900, "mono7", 18, BLACK, anchor="top", tracking=0.06)
    fr.text("KH-1024x512 // PSEUDO-SPECTRAL // RK3", 110, 930, "mono", 15, BLACK, anchor="top", tracking=0.06)
    fr.text("YZ01-23AB-45CD   C33", 110, 1000, "mono", 15, BLACK, anchor="top", tracking=0.06)
    barcode(fr, W - 230, 880, 140, 120, BLACK, seed=9, vertical=True)
    checker_glyph(fr, W - 300, 1010, 9, BLACK, seed=33, n=4)
    fr.text("Two layers slide past each other. The interface curls.", 700, 900, "inter", 26, BLACK,
            anchor="top")
    fr.text("Shear becomes vortices.", 700, 940, "inter", 26, MAGENTA, anchor="top")

    img = fr.array()
    k = 1 - ease_out(lin(t, 0.0, 0.45))
    if k > 0.05:
        img = fx.pixelate(img, max(2, int(96 * k)))
    if t > 4.5:
        a = lin(t, 4.5, 5.0)
        img = fx.pixelate(img, max(2, int(64 * a)))
        img = fx.block_smear(img, rng, n=int(4 + 10 * a))
    return img
