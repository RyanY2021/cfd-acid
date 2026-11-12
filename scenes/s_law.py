"""D — THE LAW (24–36 s): Navier–Stokes assembled term by term; UNSOLVED; so we compute."""
from scenes.common import *

S = 134                       # equation size
REG, BOLD = "inter4", "inter9"
BLUE_T = (70, 100, 255)


# ---------------------------------------------------------------- tiny math layout
def atom_w(fr, a, s):
    k = a[0]
    if k in ("t", "b"):
        return fr.text_width(a[1], REG if k == "t" else BOLD, s)
    if k == "n":
        return s * 0.78
    if k == "sup":
        return fr.text_width(a[1], REG, s * 0.55)
    if k == "sp":
        return s * a[1]
    if k == "frac":
        return max(row_w(fr, a[1], s * 0.8), row_w(fr, a[2], s * 0.8)) + s * 0.16
    return 0


def row_w(fr, atoms, s):
    return sum(atom_w(fr, a, s) for a in atoms)


def draw_row(fr, atoms, x, y, s, color):
    for a in atoms:
        k = a[0]
        if k in ("t", "b"):
            fr.text(a[1], x, y, REG if k == "t" else BOLD, s, color)
        elif k == "n":
            nabla(fr, x + s * 0.02, y, s * 0.92, color, weight=0.13)
        elif k == "sup":
            fr.text(a[1], x, y - s * 0.42, REG, s * 0.55, color)
        elif k == "frac":
            fs = s * 0.8
            wd = atom_w(fr, a, s)
            axis = y - s * 0.30
            fr.rect(x + s * 0.04, axis - s * 0.035, wd - s * 0.08, s * 0.07, color)
            nw, dw = row_w(fr, a[1], fs), row_w(fr, a[2], fs)
            draw_row(fr, a[1], x + (wd - nw) / 2, axis - s * 0.16, fs, color)
            draw_row(fr, a[2], x + (wd - dw) / 2, axis + s * 0.16 + fs * 0.72, fs, color)
        x += atom_w(fr, a, s)
    return x


U = ("b", "u")
TERMS = [  # atoms, colour, label, appear time
    ([("frac", [("t", "∂"), U], [("t", "∂"), ("t", "t")])], WHITE, "UNSTEADY", 0.0),
    ([("t", "+")], GREY, None, 0.5),
    ([("t", "("), U, ("t", "·"), ("n",), ("t", ")"), U], LIME, "ADVECTION", 0.5),
    ([("t", "=")], GREY, None, 1.0),
    ([("t", "–"), ("frac", [("n",), ("t", "p")], [("t", "ρ")])], BLUE_T, "PRESSURE", 1.5),
    ([("t", "+")], GREY, None, 2.0),
    ([("t", "ν"), ("n",), ("sup", "2"), U], RED, "VISCOSITY", 2.0),
    ([("t", "+")], GREY, None, 2.5),
    ([("b", "f")], YELLOW, "FORCE", 2.5),
]
GAP = 0.22


def equation(fr, t, y=500):
    widths = [row_w(fr, a, S) for a, *_ in TERMS]
    total = sum(widths) + GAP * S * (len(TERMS) - 1)
    x = W / 2 - total / 2
    hi = int((t - 4.0) / BEAT) if 4.0 <= t < 6.0 else -1
    hi_terms = [0, 2, 4, 6, 8, 2, 4, 6][hi] if hi >= 0 else -1
    for i, ((atoms, color, label, t0), wd) in enumerate(zip(TERMS, widths)):
        if t >= t0:
            lt = t - t0
            c = color
            if lt < 2 / 30 and label:
                fr.rect(x - 14, y - S * 1.05, wd + 28, S * 1.5, color)
                c = BLACK
            if i == hi_terms:
                fr.rect(x - 14, y - S * 1.05, wd + 28, S * 1.5, color)
                c = BLACK
            draw_row(fr, atoms, x, y, S, c)
            if label:
                k = ease_out(lin(lt, 0, 0.25))
                fr.rect(x, y + S * 0.62, wd * k, 6, color)
                fr.text(label, x, y + S * 0.62 + 22, "mono7", 16, color, anchor="top", tracking=0.18)
                fr.text(f"{(i // 2) + 1:02d}", x, y - S * 1.25, "mono7", 14, color, anchor="top", tracking=0.1)
        x += wd + GAP * S


def continuity(fr, t, y=830):
    if t < 3.0:
        return
    s = 76
    atoms = [("n",), ("t", "·"), U, ("sp", 0.25), ("t", "="), ("sp", 0.25), ("t", "0")]
    wd = row_w(fr, atoms, s)
    x = W / 2 - wd / 2
    c = WHITE if t > 3.0 + 2 / 30 else BLACK
    if t < 3.0 + 2 / 30:
        fr.rect(x - 16, y - s * 1.0, wd + 32, s * 1.35, WHITE)
    draw_row(fr, atoms, x, y, s, c)
    fr.text("MASS CONSERVATION  //  INCOMPRESSIBLE", W / 2, y + 40, "mono7", 15, GREY, align="center",
            anchor="top", tracking=0.18)


def unsolved(fr, t, rng):
    fr.fill(ORANGE)
    # giant stencil word cropped by frame
    fr.text("UNSOLVED", W / 2 - 40 * step(t, 1 / 10), 760, "stencil", 600, BLACK, align="center",
            tracking=0.0)
    fr.rect(0, 430, W, 120, ORANGE)
    banner(fr, "NAVIER_STOKES", 490, 96, BLUE, t, speed=700, fname="wide9", sep="_", solid=True)
    for (x, y) in ((70, 70), (W - 130, 70), (70, H - 130), (W - 130, H - 130)):
        checker_glyph(fr, x, y, 15, BLACK, seed=int(x + y), n=4, density=0.6)
    fr.rect(W - 560, 160, 470, 210, BLACK)
    micro_block(fr, W - 540, 178, ["MILLENNIUM PRIZE PROBLEM", "CLAY MATHEMATICS INSTITUTE // 2000",
                                   "EXISTENCE & SMOOTHNESS (3D)", "REWARD ......... $1,000,000",
                                   "STATUS ......... OPEN", "SOLUTIONS ...... 0"], ORANGE, 16, "mono7", lh=1.45)
    fr.text("ASSERT", 200, 200, "wide9", 40, BLACK, anchor="top")
    fr.vtext("WARNING // ANALYTIC_FAILURE", 120, 980, "mono7", 15, BLACK, tracking=0.2)


def prize(fr, t):
    fr.fill(BLUE)
    s = "$1,000,000"
    fr.text(s, W / 2, 560, "tek9", 300, WHITE, align="center", anchor="middle")
    label_box(fr, "UNCLAIMED", W / 2, 750, "wide9", 40, BLUE, WHITE, pad=(18, 12), align="center")
    hazard(fr, 0, 0, W, 34, WHITE, 22, phase=t * 200)
    hazard(fr, 0, H - 34, W, 34, WHITE, 22, phase=-t * 200)
    fr.text("PRIZE FOR A PROOF THAT SMOOTH SOLUTIONS ALWAYS EXIST", W / 2, 300, "mono7", 18, WHITE,
            align="center", anchor="top", tracking=0.14)


def render(t, T):
    rng = frame_rng(T)
    fr = Frame()
    if t < 6.0:
        # faint grid
        for x in range(0, W, 96):
            fr.line(x, 0, x, H, NAVY, 1)
        for y in range(0, H, 96):
            fr.line(0, y, W, y, NAVY, 1)
        fr.text("NAVIER-STOKES", 64, 64, "wide9", 44, WHITE, anchor="top")
        fr.text("EQUATIONS OF MOTION // 1822 NAVIER  1845 STOKES", 64, 124, "mono7", 15, GREY, anchor="top",
                tracking=0.15)
        seg_ring(fr, W - 170, 150, 70, 92, t, 31, BLUE, n=11, fill=lin(t, 0, 2), speed=30)
        fr.text(f"{min(5, int(t / BEAT) + 1)}/5", W - 170, 150, "tek9", 34, WHITE, align="center", anchor="middle")
        equation(fr, t)
        continuity(fr, t)
        if t > 4.0:
            label_box(fr, "NONLINEAR  //  COUPLED  //  CHAOTIC", W / 2, 960, "mono7", 16, BLACK, LIME,
                      align="center", pad=(14, 8))
        img = fr.array()
        if 5.85 < t:
            img = fx.glitch_burst(img, rng, lin(t, 5.85, 6.0))
        return img
    if t < 8.5:
        unsolved(fr, t - 6.0, rng)
        img = fr.array()
        if beat_index(t) != beat_index(t - 1 / 30):
            img = fx.glitch_burst(img, rng, 0.6)
        elif flicker(t, 15, 3, 0.15):
            img = fx.slice_shift(img, rng, 8, 50, 200)
        return img
    if t < 10.0:
        prize(fr, t - 8.5)
        img = fr.array()
        if t < 8.5 + 2 / 30:
            img = fx.glitch_burst(img, rng, 0.6)
        return img
    lt = t - 10.0
    if lt < 1.0:
        fr.text(typed("NO GENERAL SOLUTION.", lin(lt, 0.05, 0.5)), W / 2, H / 2, "wide9", 64, WHITE,
                align="center", anchor="middle")
    else:
        fr.fill(LIME if lt < 1.0 + 2 / 30 else BLACK)
        fr.text("SO WE COMPUTE.", W / 2, H / 2, "wide9", 110, BLACK if lt < 1.0 + 2 / 30 else LIME,
                align="center", anchor="middle")
        fr.text("DISCRETIZE  >  ITERATE  >  CONVERGE", W / 2, H / 2 + 110, "mono7", 18, WHITE, align="center",
                anchor="top", tracking=0.25)
    img = fr.array()
    if lt > 1.8:
        img = fx.glitch_burst(img, rng, lin(lt, 1.8, 2.0))
    return img
