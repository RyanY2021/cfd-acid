"""B — TITLE (8–16 s): stencil CFD, then a lattice-Boltzmann wind sheds vortices off the letters."""
from scenes.common import *
from sims import geometry as G

_RS = None


def letters(fr, color, alpha=1.0):
    fr.text(G.TITLE_TEXT, W / 2, G.TITLE_BASE_Y, G.TITLE_FONT, G.TITLE_SIZE, color, alpha,
            align="center", tracking=G.TITLE_TRACK)


def flow_layer(t, rng):
    global _RS
    i = (t - 0.6) * FPS
    w = sim_frame("title", "vort", i)
    if _RS is None:
        _RS = Resampler(w.shape, (0, 0, w.shape[1], w.shape[0]), W, H)
    img = vort_image(_RS(w), 0.022, cell=2, dead=0.14)
    p = sim_parts("title", i)[::2]
    fade = np.clip(1.25 - p[:, 0] / 960, 0.25, 1.0)      # streaklines thin out downstream
    keep = np.random.default_rng(int(i)).random(len(p)) < fade
    p = p[keep]
    splat(img, p[:, 0] * 2, p[:, 1] * 2, WHITE, size=2, alpha=0.6)
    return img, w


def tracker(fr, w, t):
    # follow the strongest vortex core in the far wake (stepped at 10 Hz)
    sub = np.abs(w[260:500:4, 790:930:4])
    j, i = np.unravel_index(np.argmax(sub), sub.shape)
    x, y = (790 + i * 4) * 2, (260 + j * 4) * 2
    s = 46
    brackets(fr, x - s, y - s, 2 * s, 2 * s, 12, YELLOW, 2)
    sign = "+" if w[260 + j * 4, 790 + i * 4] > 0 else "-"
    label_box(fr, f"CORE_{int(t * 10) % 97:02d}  w{sign}", x - s, y + s + 10, "mono7", 13, BLACK, YELLOW, pad=(6, 4))


def render(t, T):
    rng = frame_rng(T)
    fr = Frame()
    if t >= 0.6:
        img, w = flow_layer(t, rng)
        a = ease_out(lin(t, 0.6, 1.4))
        if a < 1:
            img = (img * a).astype(np.uint8)
        fr.image(img)
    else:
        w = None

    # letters: inverted flash, then pixel resolve
    if t < 2 / 30:
        fr.fill(LIME)
        letters(fr, BLACK)
    else:
        lf = Frame(bg=None)
        letters(lf, LIME)
        cell = int(64 * (1 - ease_out(lin(t, 0.05, 0.5)))) // 4 * 4
        if cell >= 4:
            la = lf.rgba()
            la = fx.pixelate(la, cell)
            la[..., 3] = np.where(la[..., 3] > 90, 255, 0)
            la[..., :3] = np.where(la[..., 3:] > 0, np.array(LIME, np.uint8), 0)
            fr.image(la)
        else:
            letters(fr, LIME)

    # registration marks + ring mark
    if t > 0.3:
        plus_marks(fr, [(330, 470), (1590, 470)], 14, LIME, 2)
        fr.circle(1500, 222, 15, LIME, stroke=6)
        fr.rect(1497, 236, 6, 14, LIME)

    # subtitle + micro line
    if t > 1.0:
        s = scramble("COMPUTATIONAL FLUID DYNAMICS", lin(t, 1.0, 2.2), t, 9)
        wd = fr.text_width("COMPUTATIONAL FLUID DYNAMICS", "wide", 38, 0.12)
        fr.rect(W / 2 - wd / 2 - 18, 862, wd + 36, 66, BLACK)
        fr.text(s, W / 2 - wd / 2, 895, "wide", 38, LIME, anchor="middle", tracking=0.12)
    if t > 1.8:
        y = 975
        x0 = W / 2 - 330
        fr.rect(x0 - 10, y - 18, 670, 40, BLACK)
        barcode(fr, x0, y - 8, 70, 18, LIME, seed=3)
        fr.text("X", x0 + 86, y + 1, "mono7", 18, LIME, anchor="middle")
        fr.circle(x0 + 120, y + 1, 8, LIME, stroke=2.5)
        fr.rect(x0 + 117, y - 2, 6, 6, LIME)
        fr.text("20 26", x0 + 142, y + 1, "tek", 22, LIME, anchor="middle")
        fr.text("CFD / NAVIER-STOKES / LBM-D2Q9", x0 + 230, y - 6, "mono", 12, LIME, anchor="middle", tracking=0.08)
        fr.text('"SIMULATE THE INVISIBLE"', x0 + 230, y + 10, "mono", 12, LIME, anchor="middle", tracking=0.08)

    # HUD
    if t > 0.8:
        steps = int(max(0, t - 0.6) * FPS * 62)
        corner_hud(fr, t, (170, 170, 185), 0.9, tl="Re 5.8E3  //  Ma 0.13  //  INFLOW >>>",
                   tr=f"STEP {steps:06d}", bl="LBM D2Q9  //  LES SMAGORINSKY C=0.17",
                   br="960x540 -> 1920x1080", boxed=True)
    if w is not None and 3.5 < t < 7.4:
        tracker(fr, w, step(t, 0.1))

    img = fr.array()
    if 7.45 < t:
        amt = lin(t, 7.45, 8.0)
        img = fx.glitch_burst(img, rng, 0.2 + 0.8 * amt)
    elif flicker(t, 6, 77, 0.06) and t > 2:
        img = fx.slice_shift(img, rng, 6, 30, 120)
    return img
