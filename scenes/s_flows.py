"""C — EVERYTHING FLOWS (16–24 s): beat-cut material cards, then molecules -> continuum."""
from scenes.common import *

CARDS = [  # word, bg, fg, data lines
    ("AIR", BLUE, LIME, ["RHO 1.225 KG/M3", "MU 1.8E-5 PA.S"]),
    ("WATER", LIME, BLACK, ["RHO 998 KG/M3", "MU 1.0E-3 PA.S"]),
    ("BLOOD", RED, BLACK, ["RHO 1060 KG/M3", "NON-NEWTONIAN"]),
    ("MAGMA", ORANGE, BLACK, ["RHO 2700 KG/M3", "MU 1E2..1E5 PA.S"]),
    ("PLASMA", MAGENTA, WHITE, ["IONISED GAS", "MAGNETO-HYDRODYNAMICS"]),
    ("SMOKE", BLACK, WHITE, ["BUOYANT PLUME", "PASSIVE SCALAR"]),
    ("HONEY", YELLOW, BLACK, ["RHO 1420 KG/M3", "MU 10 PA.S"]),
    ("STARS", BLUE, WHITE, ["STELLAR INTERIOR", "CONVECTION ZONE"]),
]


def card(fr, t, k):
    word, bg, fg, data = CARDS[k]
    fr.fill(bg)
    # ghost banner of the word behind
    banner(fr, word, 180, 120, fg, t + k, speed=600 * (1 if k % 2 else -1), fname="wide9", outline_w=1.5,
           alpha=0.55)
    banner(fr, word, 905, 120, fg, t + k, speed=-600 * (1 if k % 2 else -1), fname="wide9", outline_w=1.5,
           alpha=0.55)
    # giant word, slight stepped slide
    sz = 330 if len(word) <= 5 else 270
    dx = -60 * step(t, 1 / 15) * (1 if k % 2 else -1)
    fr.text(word, W / 2 + dx, 545, "wide9", sz, fg, align="center", anchor="middle", tracking=-0.02)
    # data block + index
    fr.rect(0, 1080 - 92, W, 92, bg)
    fr.text(f"{k + 1:02d}/08", 64, 1018, "tek9", 42, fg, anchor="middle")
    micro_block(fr, 210, 998, data, fg, 15, "mono7")
    barcode(fr, W - 340, 1000, 160, 34, fg, seed=k)
    checker_glyph(fr, W - 150, 996, 10, fg, seed=20 + k, n=4)
    fr.text("ASSERT: FLUID", 64, 60, "mono7", 15, fg, anchor="top", tracking=0.2)
    fr.text("STATE // CONTINUUM", W - 64, 60, "mono7", 15, fg, align="right", anchor="top", tracking=0.2)


_MOL = None


def molecules():
    global _MOL
    if _MOL is None:
        rng = np.random.default_rng(5)
        n = 2600
        _MOL = (rng.uniform(0, W, n), rng.uniform(0, H, n), rng.normal(0, 1, (n, 2)), rng.uniform(0, 1e3, n))
    return _MOL


_RS = None


def continuum_field(t):
    global _RS
    w = sim_frame("turb", "w", 240 + t * 14)
    if _RS is None:
        _RS = Resampler(w.shape, (0, 64, 512, 64 + 288), W, H)
    return _RS(w)


def render(t, T):
    rng = frame_rng(T)
    fr = Frame()
    if t < 4.0:
        k = int(t / BEAT)
        lt = t - k * BEAT
        card(fr, lt, k)
        img = fr.array()
        if lt < 2 / 30:
            img = fx.glitch_burst(img, rng, 0.5)
        return img

    lt = t - 4.0
    # --- molecules -> continuum
    x, y, v, ph = molecules()
    jit = 6.0
    px = x + jit * np.sin(ph + lt * 9 * v[:, 0]) + 30 * v[:, 0] * lt
    py = y + jit * np.cos(ph * 1.3 + lt * 9 * v[:, 1]) + 30 * v[:, 1] * lt
    blend = ease_io(lin(lt, 1.4, 2.8))
    base = np.zeros((H, W, 3), np.uint8)
    if blend > 0:
        f = continuum_field(lt)
        cimg = vort_image(f, 7.0, cell=3, neg=[BLUE_D, BLUE, (110, 140, 255)], pos=[(0, 90, 30), GREEN, LIME],
                          dead=0.18)
        base = (cimg.astype(np.float32) * blend).astype(np.uint8)
    size = 6 - int(4 * blend)
    a = 1.0 - 0.85 * blend
    splat(base, px, py, WHITE, size=max(2, size), alpha=a)
    fr.image(base)

    # lens / zoom ring
    if lt < 1.6:
        r = 330
        fr.circle(W / 2, H / 2, r, LIME, stroke=2)
        ticks(fr, W / 2, H / 2, r + 8, 120, 8, LIME, 1, 0.8)
        fr.text("10^25 MOLECULES / M3", W / 2, H / 2 + r + 40, "wide", 26, LIME, align="center", anchor="top",
                tracking=0.06)
        label_box(fr, "SCALE 1E-9 M", W / 2, H / 2 - r - 46, "mono7", 14, BLACK, LIME, align="center")
    # labels
    if lt < 1.4:
        fr.text(typed("TOO MANY TO TRACK.", lin(lt, 0.2, 0.8)), 96, 120, "wide9", 54, WHITE, anchor="top")
    elif lt < 2.9:
        fr.text("ZOOM OUT", 96, 120, "wide9", 54, WHITE, anchor="top")
        fr.text(f"SCALE 1E-{9 - int(8 * blend)} M", 96, 190, "mono7", 18, LIME, anchor="top", tracking=0.1)
    else:
        lab = "CONTINUUM_HYPOTHESIS"
        wd = fr.text_width(lab, "wide9", 60)
        fr.rect(90, 104, wd + 40, 92, LIME)
        fr.text(scramble(lab, lin(lt, 2.9, 3.3), lt, 4), 110, 150, "wide9", 60, BLACK, anchor="middle")
        items = [("RHO(x,t)", "DENSITY"), ("u(x,t)", "VELOCITY"), ("p(x,t)", "PRESSURE")]
        for i, (a1, b1) in enumerate(items):
            if lt > 3.1 + i * 0.2:
                y0 = 760 + i * 70
                fr.rect(96, y0, 330, 56, BLACK)
                fr.text(a1, 112, y0 + 28, "tek", 34, WHITE, anchor="middle")
                fr.text(b1, 300, y0 + 28, "mono7", 15, LIME, anchor="middle", tracking=0.15)
        fr.text("A FLUID IS A FIELD.", W - 96, 1000, "wide9", 40, WHITE, align="right", anchor="baseline")
    corner_hud(fr, t, (160, 160, 170), tl="", tr="", bl="ASSERT: CONTINUUM  //  KN << 1", br=f"T+{T:06.2f}",
               boxed=True)
    img = fr.array()
    if 3.85 < lt:
        img = fx.glitch_burst(img, rng, lin(lt, 3.85, 4.0))
    return img
