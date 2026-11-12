"""G1 — AIRFOIL (52–56 s): exact potential flow (Joukowski + Kutta), streamlines, tracers, Cp field."""
from scenes.common import *
from sims import geometry as G

SC = 300.0                       # px per unit (chord ~4 -> ~1200 px)
OX, OY = W / 2 - 40, H / 2 + 30   # screen position of z = 0
U0 = 1.0
BETA = -np.angle(1 - G.MU)


def alpha_at(t):
    k = int(clamp(t / BEAT, 0, 7))
    return math.radians(2.0 + k * 1.1)


def gamma(al):
    return 4 * math.pi * U0 * G.R_CIRC * math.sin(al + BETA)


def zeta_of(z):
    s = np.sqrt(z * z - 4 + 0j)
    z1 = (z + s) / 2
    z2 = (z - s) / 2
    out = np.where(np.abs(z1 - G.MU) >= np.abs(z2 - G.MU), z1, z2)
    inside = np.abs(out - G.MU) < G.R_CIRC * 0.999
    return out, inside


def screen_to_body(x, y, al):
    zs = ((x - OX) - 1j * (y - OY)) / SC
    return zs * np.exp(1j * al)          # rotate into airfoil frame (airfoil pitched up by al)


def body_to_screen(z, al):
    zs = z * np.exp(-1j * al)
    return OX + zs.real * SC, OY - zs.imag * SC


_GRID = {}


def field(al):
    key = round(al, 5)
    if key not in _GRID:
        gx, gy = np.meshgrid(np.linspace(0, W, 640), np.linspace(0, H, 360))
        zb = screen_to_body(gx, gy, al)
        # the uniform-stream part must be along body x rotated: use psi of stream at angle al in body frame
        zt, inside = zeta_of(zb)
        zp = zt - G.MU
        g = gamma(al)
        Wc = U0 * (zp * np.exp(-1j * al) + G.R_CIRC ** 2 * np.exp(1j * al) / zp) + 1j * g / (2 * np.pi) * np.log(zp)
        dW = (U0 * (np.exp(-1j * al) - G.R_CIRC ** 2 * np.exp(1j * al) / zp ** 2) + 1j * g / (2 * np.pi * zp)) \
            / (1 - 1 / zt ** 2)
        psi = np.where(inside, np.nan, Wc.imag)
        cp = np.where(inside, 0, 1 - np.abs(dW) ** 2 / U0 ** 2)
        _GRID[key] = (psi, np.clip(np.nan_to_num(cp), -4, 1), inside)
    return _GRID[key]


def vel(x, y, al):
    zb = screen_to_body(x, y, al)
    zt, inside = zeta_of(zb)
    zp = zt - G.MU
    g = gamma(al)
    dW = (U0 * (np.exp(-1j * al) - G.R_CIRC ** 2 * np.exp(1j * al) / zp ** 2) + 1j * g / (2 * np.pi * zp)) \
        / (1 - 1 / zt ** 2)
    v = np.conj(dW) * np.exp(-1j * al)
    v = np.where(inside | ~np.isfinite(v), 0, v)
    return v.real * SC, -v.imag * SC


_TRAJ = None


def trajectories():
    """Precompute tracer paths for the whole shot (deterministic, per-process cache)."""
    global _TRAJ
    if _TRAJ is None:
        n_fr = int(4 * FPS) + 2
        rows = np.linspace(-60, H + 60, 70)
        cols = np.linspace(-1200, W, 46)
        X, Y = np.meshgrid(cols, rows)
        x = X.ravel() + np.random.default_rng(1).uniform(0, 30, X.size)
        y = Y.ravel().astype(np.float64)
        out = np.zeros((n_fr, x.size, 2), np.float32)
        dt = 1 / FPS * 0.9
        for f in range(n_fr):
            out[f, :, 0], out[f, :, 1] = x, y
            al = alpha_at(f / FPS)
            for _ in range(3):
                u1, v1 = vel(x, y, al)
                u2, v2 = vel(x + u1 * dt / 6, y + v1 * dt / 6, al)
                x = x + u2 * dt / 3
                y = y + v2 * dt / 3
            wrap = x > W + 40
            x = np.where(wrap, x - (W + 1300), x)
        _TRAJ = out
    return _TRAJ


def render(t, T):
    rng = frame_rng(T)
    al = alpha_at(t)
    psi, cp, inside = field(al)
    # Cp background: suction blue, stagnation red, posterised + dithered
    rs = Resampler(cp.shape, (0, 0, cp.shape[1], cp.shape[0]), W, H)
    img = fv.diverging_map(-rs(cp), 2.2, [(110, 10, 20), RED, ORANGE], [BLUE_D, BLUE, (100, 130, 255)],
                           BLACK, cell=3, gamma=0.9)
    # tracers as short streaks
    tr = trajectories()
    f = int(clamp(t * FPS, 2, len(tr) - 1))
    p1, p0 = tr[f], tr[f - 2]
    m = (np.abs(p1[:, 0] - p0[:, 0]) < 200)
    fr = Frame()
    fr.image(img)
    fr.segments(p0[m], p1[m], WHITE, 2.0, 0.85)
    # streamlines
    levels = np.linspace(-3.2, 3.2, 41)
    lines = fv.contours(psi, levels)
    sx, sy = W / (psi.shape[1] - 1), H / (psi.shape[0] - 1)
    pl = [ln * [sx, sy] for lv in lines for ln in lv if len(ln) > 3]
    fr.polylines(pl, LIME, 1.4, 0.9)
    # body
    zb = G.airfoil_outline(400)
    bx, by = body_to_screen(zb, al)
    fr.polyline(np.stack([bx, by], 1), BLACK, fill=True, closed=True)
    fr.polyline(np.stack([bx, by], 1), WHITE, 3, closed=True)
    # callouts
    i_te = np.argmax(zb.real)
    i_le = np.argmin(zb.real)
    fr.circle(bx[i_le], by[i_le], 14, RED, stroke=3)
    label_box(fr, "STAGNATION", bx[i_le] - 30, by[i_le] + 40, "mono7", 13, BLACK, RED, align="right", pad=(6, 4))
    crosshair(fr, bx[i_te], by[i_te], 16, YELLOW, 2)
    label_box(fr, "KUTTA CONDITION", bx[i_te] + 24, by[i_te] + 24, "mono7", 13, BLACK, YELLOW, pad=(6, 4))
    # lift arrow + numbers
    g = gamma(al)
    chord = zb.real.max() - zb.real.min()
    cl = 2 * g / (U0 * chord)
    ax = OX + 60
    fr.line(ax, OY - 120, ax, OY - 120 - 160 * cl, LIME, 8)
    fr.polyline([(ax - 22, OY - 100 - 160 * cl), (ax, OY - 126 - 160 * cl), (ax + 22, OY - 100 - 160 * cl)], LIME, 8)
    label_box(fr, f"LIFT  CL {cl:4.2f}", ax + 26, OY - 150 - 160 * cl, "wide9", 30, BLACK, LIME, pad=(12, 8))
    fr.text("AIRFOIL // POTENTIAL FLOW", 64, 64, "wide9", 44, WHITE, anchor="top")
    fr.text("JOUKOWSKI MAP  z = ζ + 1/ζ   //   KUTTA-JOUKOWSKI  L' = ρ U Γ", 64, 124, "mono7", 15, LIME,
            anchor="top", tracking=0.08)
    fr.text("ANGLE OF ATTACK", W - 64, 64, "mono7", 15, LIME, align="right", anchor="top", tracking=0.2)
    fr.text(f"{math.degrees(al):4.1f}°", W - 64, 92, "tek9", 96, WHITE, align="right", anchor="top")
    corner_hud(fr, t, (170, 170, 185), tl="", tr="", bl="Cp FIELD  //  BLUE SUCTION  //  RED STAGNATION",
               br=f"Γ {g:5.3f}", boxed=True)
    img = fr.array()
    if beat_index(t) != beat_index(t - 1 / 30) and t > 0.1:
        img = fx.glitch_burst(img, rng, 0.3)
    if t > 3.85:
        img = fx.glitch_burst(img, rng, lin(t, 3.85, 4.0))
    return img
