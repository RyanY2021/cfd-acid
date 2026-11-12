"""G2 — VON KÁRMÁN (56–62 s): cylinder wake, three render modes, vortex-core trackers."""
from scenes.common import *

NX, NY = 1440, 480
CYL = (260, 241.5, 23)
SCALE = H / NY                       # 2.25 px per cell
VS = 0.012


def view_x0(t):
    return 40 + 360 * ease_io(lin(t, 0.0, 6.0))


def mode(t):
    if 2.75 <= t < 4.5:
        return "wire"
    return "color"


def trackers(fr, w, x0, t):
    xs = slice(int(x0) + 330, int(x0) + 820, 3)
    sub = w[60:420:3, xs]
    out = []
    for sgn, col_, lab in ((1, RED, "+"), (-1, (90, 120, 255), "-")):
        j, i = np.unravel_index(np.argmax(sgn * sub), sub.shape)
        cx = (int(x0) + 330 + i * 3 - x0) * SCALE
        cy = (60 + j * 3) * SCALE
        s = 52
        brackets(fr, cx - s, cy - s, 2 * s, 2 * s, 14, WHITE, 2)
        crosshair(fr, cx, cy, 8, WHITE, 1.5)
        gam = float(np.abs(sub[max(0, j - 4):j + 5, max(0, i - 4):i + 5]).sum()) * 9 * 0.5
        label_box(fr, f"VORTEX {lab}  G {gam:4.2f}", cx - s, cy + s + 10, "mono7", 13, BLACK, WHITE, pad=(6, 4))
        out.append((cx, cy))
    return out


def render(t, T):
    rng = frame_rng(T)
    i = t * FPS
    w = sim_frame("karman", "vort", i)
    x0 = view_x0(t)
    x1 = x0 + W / SCALE
    rs = Resampler(w.shape, (x0, 0, x1, NY), W, H)
    fr = Frame()
    m = mode(t)
    if m == "color":
        img = vort_image(rs(w), VS, cell=2, dead=0.12)
        p = sim_parts("karman", i)
        p = p[np.abs(p[:, 1] - 241) < 105]
        px, py = rs.to_out(p[:, 0], p[:, 1])
        splat(img, px, py, WHITE, size=2, alpha=0.55)
        fr.image(img)
    else:
        f = rs(w)[::2, ::2]
        lv_pos = [VS * k for k in (0.05, 0.1, 0.18, 0.3, 0.45, 0.65, 0.85)]
        lv_neg = [-v for v in lv_pos]
        for lv, c in [(v, LIME) for v in lv_pos] + [(v, (90, 120, 255)) for v in lv_neg]:
            ls = fv.contours(f, [lv])[0]
            fr.polylines([ln * 2 for ln in ls if len(ln) > 4], c, 1.6)
        for x in range(0, W, 120):
            fr.line(x, 0, x, H, (25, 25, 40), 1)
    # cylinder
    cx, cy = (CYL[0] - x0) * SCALE, CYL[1] * SCALE
    r = CYL[2] * SCALE
    fr.circle(cx, cy, r, LIME)
    crosshair(fr, cx, cy, r + 26, WHITE, 1.5)
    fr.line(cx - r, cy + r + 40, cx + r, cy + r + 40, WHITE, 2)
    fr.text("D", cx, cy + r + 52, "mono7", 16, WHITE, align="center", anchor="top")
    if t > 0.6:
        trackers(fr, w, x0, step(t, 1 / 10))
    # typography
    fr.rect(0, 0, W, 150, BLACK, 0.0)
    banner(fr, "VON_KARMAN_VORTEX_STREET", 92, 64, WHITE, t, speed=220, fname="wide9", fill_every=3,
           fill_phase=0, fill_color=LIME, outline_w=1.5)
    fr.rect(64, H - 230, 560, 168, BLACK)
    fr.text("Re 220", 84, H - 210, "tek9", 72, LIME, anchor="top")
    micro_block(fr, 86, H - 120, ["STROUHAL  St = fD/U = 0.20", "VORTICITY  w = curl u",
                                  "ALTERNATING SHEDDING // 2S MODE"], WHITE, 15, "mono7", lh=1.4)
    fr.vtext("VORTEX_SHEDDING", W - 52, H - 80, "wide9", 26, WHITE, tracking=0.08)
    mode_lab = "RENDER: VORTICITY // POSTERISED" if m == "color" else "RENDER: ISO-VORTICITY // WIREFRAME"
    label_box(fr, mode_lab, W - 100, H - 60, "mono7", 13, BLACK, LIME if m == "color" else WHITE, align="right",
              pad=(8, 5))
    img = fr.array()
    if beat_index(t) != beat_index(t - 1 / 30) and t > 0.1:
        img = fx.glitch_burst(img, rng, 0.35)
    if 4.5 <= t < 4.5 + 2 / 30:
        img[:] = np.array(RED, np.uint8)
        img = fx.slice_shift(img, rng, 4, 40, 100)
    if t > 5.85:
        img = fx.glitch_burst(img, rng, lin(t, 5.85, 6.0))
    return img
