"""F — SOLVE (44–52 s): residuals fall, iterations climb, ring to 100 %, CONVERGED, build to the drop."""
from scenes.common import *

N_IT = 5000
PX, PY, PW, PH = 120, 230, 980, 640           # plot box
CURVES = [("CONTINUITY", LIME), ("X-MOMENTUM", (70, 100, 255)), ("Y-MOMENTUM", RED), ("TURB-K", WHITE)]


def histories():
    rng = np.random.default_rng(21)
    n = np.arange(N_IT + 1)
    out = []
    for i in range(len(CURVES)):
        rate = rng.uniform(6.6, 7.8)
        walk = np.convolve(rng.normal(0, 0.012, n.size).cumsum(), np.ones(80) / 80, mode="same")
        walk -= np.linspace(walk[0], walk[-1], n.size)
        bump = 0.9 * np.exp(-((n - rng.uniform(250, 700)) / 180) ** 2)
        wig = 0.18 * np.sin(n / rng.uniform(25, 60)) * np.exp(-n / 1800) + 0.05 * np.sin(n / 9.0)
        lg = -rate * (1 - np.exp(-n / 1500)) / (1 - math.exp(-N_IT / 1500)) - 0.3 * i + bump + walk + wig
        lg = np.clip(lg - 0.2, -8.2, 0.2)
        out.append(lg)
    return out


HIST = histories()


def to_plot(n, lg):
    return PX + n / N_IT * PW, PY + (-lg) / 8.5 * PH


def render(t, T):
    rng = frame_rng(T)
    fr = Frame()
    prog = ease_io(lin(t, 0.3, 5.8))
    it = int(N_IT * prog)

    # --- plot frame
    fr.text("RESIDUALS", PX, 120, "wide9", 46, WHITE, anchor="top")
    fr.text("STEP 02 // ITERATE UNTIL THE EQUATIONS BALANCE", PX, 182, "mono7", 15, LIME, anchor="top",
            tracking=0.16)
    fr.line(PX, PY, PX, PY + PH, GREY, 1.5)
    fr.line(PX, PY + PH, PX + PW, PY + PH, GREY, 1.5)
    for e in range(0, 9, 2):
        y = PY + e / 8.5 * PH
        fr.line(PX - 8, y, PX + PW, y, (40, 40, 52), 1)
        fr.text(f"1E-{e}" if e else "1E0", PX - 16, y, "mono", 12, GREY, align="right", anchor="middle")
    for k in range(0, 6):
        x = PX + k / 5 * PW
        fr.line(x, PY + PH, x, PY + PH + 8, GREY, 1)
        fr.text(f"{k * 1000}", x, PY + PH + 18, "mono", 12, GREY, align="center", anchor="top")
    ty = PY + 6 / 8.5 * PH
    fr.line(PX, ty, PX + PW, ty, YELLOW, 1.5, dash=([10, 8], -t * 60))
    label_box(fr, "TOL 1E-6", PX + PW - 4, ty - 30, "mono7", 13, BLACK, YELLOW, align="right", pad=(6, 4))

    # --- curves
    if it > 2:
        for (name, c), lg in zip(CURVES, HIST):
            n = np.arange(0, it + 1, 6)
            x, y = to_plot(n, lg[n])
            fr.polyline(np.stack([x, y], 1), c, 2.0)
            hx, hy = to_plot(it, lg[it])
            fr.rect(hx - 5, hy - 5, 10, 10, c)
    # legend
    for i, (name, c) in enumerate(CURVES):
        fr.rect(PX + i * 240, PY + PH + 60, 26, 6, c)
        fr.text(name, PX + i * 240 + 36, PY + PH + 63, "mono7", 13, c, anchor="middle", tracking=0.1)

    # --- ring + counters
    cx, cy = 1470, 470
    pct = prog * 100
    rc = RED if pct < 85 else YELLOW if pct < 99.5 else LIME
    seg_ring(fr, cx, cy, 196, 238, t, 41, rc, n=19, fill=1.0, speed=-18)
    seg_ring(fr, cx, cy, 160, 186, t, 42, BLUE, n=13, fill=1.0, speed=26)
    fr.arc(cx, cy, 270, -90, -90 + 360 * prog, 5, rc)
    fr.circle(cx, cy, 270, (40, 40, 52), stroke=1)
    ticks(fr, cx, cy, 282, 60, 7, GREY, 1)
    fr.text(f"{int(pct):03d}", cx, cy - 6, "tek9", 120, WHITE, align="center", anchor="middle")
    fr.text("CONVERGENCE %", cx, cy + 70, "mono7", 14, rc, align="center", anchor="top", tracking=0.2)
    fr.text("ITER", 1250, 800, "mono7", 15, GREY, anchor="top", tracking=0.2)
    fr.text(f"{it:05d}", 1250, 826, "tek9", 92, rc if pct >= 99.5 else WHITE, anchor="top")
    micro_block(fr, 1580, 812, ["CFL ........ 0.42", "dt ......... 2.0E-4", f"WALL ....... 00:{int(prog * 41):02d}:16",
                                "SCHEME ..... SIMPLE / 2ND"], GREY, 14, "mono7")

    # --- converged stamp
    if t >= 6.0:
        k = ease_out(lin(t, 6.0, 6.15))
        fr.rect(0, 470 - 95 * k, W, 190 * k, LIME)
        if k > 0.6:
            fr.text("CONVERGED", W / 2, 470, "wide9", 150, BLACK, align="center", anchor="middle")
            fr.text("RESIDUAL < 1E-6  //  SOLUTION LOCKED", W / 2, 470 + 120, "mono7", 18, LIME, align="center",
                    anchor="top", tracking=0.25)
    img = fr.array()
    if t > 7.0:
        amt = lin(t, 7.0, 8.0)
        img = fx.block_displace(img, rng, n=int(8 + 60 * amt), cell=32, max_cells=10, max_shift=10)
        img = fx.pixel_blocks(img, rng, int(60 * amt), 32, [LIME, BLACK, BLUE])
        img = fx.rgb_split(img, int(4 + 26 * amt))
        if t > 7.9:
            img[:] = 0
    elif 5.9 < t < 6.05:
        img = fx.glitch_burst(img, rng, 0.6)
    return img
