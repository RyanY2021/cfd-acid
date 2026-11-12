"""G3 — REYNOLDS SWEEP (62–66 s): the same cylinder at four Reynolds numbers, 2x2."""
from scenes.common import *

PANELS = [  # sim, label, regime, scale
    ("re20", "20", "STEADY // ATTACHED TWIN VORTICES"),
    ("re100", "100", "PERIODIC // LAMINAR SHEDDING"),
    ("re1000", "10³", "UNSTEADY // TRANSITIONAL WAKE"),
    ("re20000", "2x10⁴", "CHAOTIC // TURBULENT WAKE"),
]
GAP = 6
PW, PH = (W - GAP) // 2, (H - GAP) // 2
_SC = {}


def vscale(name):
    if name not in _SC:
        a = sim(name, "vort")
        _SC[name] = float(np.percentile(np.abs(np.asarray(a[-1], np.float32)), 99.3))
    return _SC[name]


_RS = None


def panel(name, i):
    global _RS
    w = sim_frame(name, "vort", i)
    if _RS is None:
        _RS = Resampler(w.shape, (40, 0, 40 + w.shape[1] * 0.88, w.shape[0]), PW, PH)
    img = vort_image(_RS(w), vscale(name), cell=2, dead=0.1)
    p = sim_parts(name, i)
    px, py = _RS.to_out(p[:, 0], p[:, 1])
    splat(img, px, py, WHITE, size=2, alpha=0.3 if name == "re20000" else 0.5)
    return img


def render(t, T):
    rng = frame_rng(T)
    fr = Frame(bg=(20, 20, 26))
    for k, (name, lab, regime) in enumerate(PANELS):
        t0 = k * BEAT * 0.5
        if t < t0:
            continue
        x = (k % 2) * (PW + GAP)
        y = (k // 2) * (PH + GAP)
        img = panel(name, (t - t0) * FPS + 20)
        if t - t0 < 2 / 30:
            img = 255 - img
        fr.image(img, x, y)
        # cylinder mark (cell 150,163 in sims; crop starts at 40, scale PW/(576*.88))
        s = PW / (576 * 0.88)
        cx, cy, r = x + (150 - 40 + 0.5) * s, y + (163 + 0.5) * PH / 324, 15 * s
        fr.circle(cx, cy, r, LIME)
        ly = y + 24 if k < 2 else y + PH - 120
        fr.rect(x + 24, ly, 420, 96, BLACK)
        fr.text("Re", x + 40, ly + 16, "mono7", 18, LIME, anchor="top", tracking=0.1)
        fr.text(lab, x + 84, ly + 12, "tek9", 64, WHITE, anchor="top")
        fr.text(regime, x + 40, ly + 76, "mono7", 13, LIME, anchor="baseline", tracking=0.12)
        fr.text(f"0{k + 1}", x + PW - 24, y + 24, "mono7", 14, WHITE, align="right", anchor="top")
    # centre formula
    if t > 1.0:
        k = ease_out(lin(t, 1.0, 1.2))
        bw, bh = 560 * k, 150
        fr.rect(W / 2 - bw / 2, H / 2 - bh / 2, bw, bh, LIME)
        if k > 0.9:
            fr.text("Re = ρUD / μ", W / 2, H / 2 - 18, "inter9", 64, BLACK, align="center", anchor="middle")
            fr.text("INERTIA  /  VISCOSITY", W / 2, H / 2 + 42, "mono7", 16, BLACK, align="center", anchor="middle",
                    tracking=0.3)
    img = fr.array()
    if t > 3.85:
        img = fx.glitch_burst(img, rng, lin(t, 3.85, 4.0))
    return img
