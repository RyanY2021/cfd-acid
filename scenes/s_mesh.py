"""E — DISCRETIZE (36–44 s): Joukowski airfoil, body-fitted O-grid draws itself, zoom into one cell."""
from scenes.common import *
from sims import geometry as G

NT, NR, GROW = 176, 64, 0.047
AOA = math.radians(6)
th = np.linspace(0, 2 * np.pi, NT + 1)
rr = G.R_CIRC * np.exp(GROW * np.arange(NR + 1))
ZETA = G.MU + rr[:, None] * np.exp(1j * th[None, :])          # (NR+1, NT+1)
Z = G.joukowski(ZETA) * np.exp(-1j * AOA)
FOCUS_J, FOCUS_K = 4, NT // 2                                   # leading-edge cell
FOCUS = Z[FOCUS_J:FOCUS_J + 2, FOCUS_K:FOCUS_K + 2].mean()


def cam(t):
    z = ease_io(lin(t, 4.3, 6.6))
    zoom = math.exp(math.log(46) * z)
    c0 = complex(0.0, 0.0)
    c = FOCUS - (FOCUS - c0) * (1 - z) / zoom      # focus point glides to centre, never off-screen
    return zoom, c


def to_screen(zz, zoom, c, s=215.0):
    q = (zz - c) * zoom * s
    return np.stack([W / 2 + q.real, H / 2 - q.imag + 20], -1)


def render(t, T):
    rng = frame_rng(T)
    fr = Frame(bg=BLUE)
    zoom, c = cam(t)
    lw = 1.3

    # rings (grow outward)
    nj = int(clamp((t - 0.9) / 2.2) * NR)
    rings = [to_screen(Z[j], zoom, c) for j in range(0, nj + 1)]
    fr.polylines([r for j, r in enumerate(rings) if j % 6], WHITE, lw, 0.75)
    fr.polylines([r for j, r in enumerate(rings) if j % 6 == 0 and j], LIME, lw * 1.6, 1.0)
    # radial lines sweep around
    nk = int(clamp((t - 1.6) / 1.8) * NT)
    if nk:
        rad = []
        for k in range(0, nk):
            jmax = max(2, int(NR * clamp((t - 1.6 - k / NT * 1.8) / 0.6)))
            rad.append(to_screen(Z[:jmax + 1, k], zoom, c))
        fr.polylines([r for k, r in enumerate(rad) if k % 8], WHITE, lw, 0.75)
        fr.polylines([r for k, r in enumerate(rad) if k % 8 == 0], LIME, lw * 1.6, 1.0)

    # computing cells flicker (pixel debris on the mesh)
    if 2.8 < t < 4.6:
        r2 = np.random.default_rng(int(t * 15))
        for _ in range(26):
            j = int(r2.integers(0, 14))
            k = int(r2.integers(0, NT))
            quad = np.array([Z[j, k], Z[j, k + 1], Z[j + 1, k + 1], Z[j + 1, k]])
            fr.polyline(to_screen(quad, zoom, c), LIME if r2.random() < 0.7 else YELLOW, fill=True, closed=True)

    # airfoil body
    body = to_screen(Z[0], zoom, c)
    k = ease_out(lin(t, 0.1, 0.9))
    if k >= 1:
        fr.polyline(body, BLACK, fill=True, closed=True)
    fr.polyline(body[: max(2, int(len(body) * k))], WHITE, 3.0)

    # focus cell + flux arrows
    if t > 5.4:
        quad = np.array([Z[FOCUS_J, FOCUS_K], Z[FOCUS_J, FOCUS_K + 1], Z[FOCUS_J + 1, FOCUS_K + 1],
                         Z[FOCUS_J + 1, FOCUS_K]])
        q = to_screen(quad, zoom, c)
        fl = flicker(t, 10, 2, 0.8) or t > 6.2
        fr.polyline(q, YELLOW if fl else LIME, fill=True, closed=True)
        fr.polyline(q, BLACK, 3, closed=True)
        cen = q.mean(0)
        if t > 6.0:
            for i in range(4):
                a, b = q[i], q[(i + 1) % 4]
                mid = (a + b) / 2
                n = mid - cen
                n = n / (np.linalg.norm(n) + 1e-9)
                e = mid + n * 110 * ease_out(lin(t, 6.0 + i * 0.08, 6.4 + i * 0.08))
                fr.line(mid[0], mid[1], e[0], e[1], BLACK, 5)
                ang = math.atan2(n[1], n[0])
                tip = [e + 18 * np.array([math.cos(ang + s), math.sin(ang + s)]) * -1 for s in (0.5, -0.5)]
                fr.polyline([tip[0], e, tip[1]], BLACK, 5)
            label_box(fr, "CELL 048213", cen[0] + 150, cen[1] - 150, "wide9", 34, BLUE, YELLOW, pad=(14, 10))
            fr.rect(cen[0] + 136, cen[1] - 92, 420, 120, BLACK)
            micro_block(fr, cen[0] + 152, cen[1] - 80, ["FINITE VOLUME // CONSERVATIVE", "SUM  F.n A  =  0",
                                                          "FLUX IN = FLUX OUT", "UNKNOWNS  u v p"],
                        YELLOW, 16, "mono7", lh=1.45)

    # typography
    fr.text("MESH_GENERATION", 64, 64, "wide9", 46, WHITE, anchor="top")
    fr.text("STEP 01 // DISCRETIZE THE DOMAIN", 64, 126, "mono7", 15, LIME, anchor="top", tracking=0.18)
    cells = int(1048576 * ease_in(lin(t, 1.0, 4.0), 2))
    fr.text("CELLS", W - 64, 64, "mono7", 15, LIME, align="right", anchor="top", tracking=0.2)
    fr.text(f"{cells:,}", W - 64, 92, "tek9", 72, WHITE, align="right", anchor="top")
    fr.vtext("O-GRID // BODY-FITTED // STRUCTURED", W - 48, 900, "mono7", 15, WHITE, tracking=0.2)
    fr.rect(0, H - 70, W, 70, BLACK)
    micro_block(fr, 64, H - 52, [f"NODES {NT} x {NR}   //   GROWTH 1.{int(GROW * 1000):03d}   //   y+ < 1   //   "
                                 f"ZOOM x{zoom:5.1f}"], LIME, 15, "mono7")
    glyph_row(fr, W - 300, H - 52, 6, WHITE, [1, 2, 3, 4, 5], n=4)
    img = fr.array()
    if t > 7.7:
        img = fx.glitch_burst(img, rng, lin(t, 7.7, 8.0))
    return img
