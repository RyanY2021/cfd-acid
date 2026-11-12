"""A — BOOT (0–8 s): micro labels, segmented ring assembly, INITIATING, blowout."""
from scenes.common import *

CX, CY = 960, 540
CUT_GLITCH = False


def ring_stack(fr, t, fill, palette=(BLUE, GREEN), scale=1.0, alpha=1.0):
    s = scale
    seg_ring(fr, CX, CY, 330 * s, 392 * s, t, 11, palette[0], n=13, fill=fill, speed=-14, alpha=alpha)
    seg_ring(fr, CX, CY, 286 * s, 322 * s, t, 12, palette[1], n=17, fill=fill * 1.15 - 0.1, speed=22, alpha=alpha)
    seg_ring(fr, CX, CY, 404 * s, 414 * s, t, 13, palette[0], n=23, fill=fill * 1.3 - 0.3, speed=8,
             alpha=alpha, jitter=False, min_frac=0.1)
    fr.circle(CX, CY, 262 * s, BLUE, alpha * 0.9, stroke=1.5)
    ticks(fr, CX, CY, 440 * s, 72, 6 * s, BLUE, 1.2, alpha * 0.8, a0=step(t, 0.1) * 3)


def render(t, T):
    fr = Frame()
    rng = frame_rng(T)

    # --- dim data chatter in the corners (stepped at 10 fps)
    r2 = np.random.default_rng(int(t * 10))
    data_noise(fr, r2, 40, 120, 360, 300, 6, NAVY, density=0.05 * clamp(t / 2), alpha=1)
    data_noise(fr, r2, W - 400, H - 420, 360, 300, 6, NAVY, density=0.05 * clamp(t / 2), alpha=1)

    # --- boot text (top left), typed
    lines = ["SYS://FLOW_SOLVER  v4.07", "KERNEL ........ NAVIER-STOKES", "LATTICE ....... D2Q9 / FV",
             "PRECISION ..... FP32", "DOMAIN ........ 1920 x 1080", "STATUS ........ INITIATING"]
    for i, s in enumerate(lines):
        x = lin(t, 0.15 + i * 0.22, 0.55 + i * 0.22)
        if x > 0:
            c = GREEN if i == len(lines) - 1 else (150, 150, 165)
            fr.text(typed(s, x), 64, 64 + i * 20, "mono", 14, c, anchor="top", tracking=0.06)
    if t < 2.0:
        k = 0.5 + 0.5 * math.cos(t * 12)
        crosshair(fr, CX, CY, 14, LIME, 2, 0.4 + 0.6 * k)
        fr.circle(CX, CY, 40 + 30 * ease_out(lin(t, 0, 1.0)), LIME, 0.5 * (1 - lin(t, 1.0, 2.0)), stroke=1.2)
    if t < 1.4 and flicker(t, 4, 1, 0.5):
        fr.rect(64 + fr.text_width(typed(lines[0], lin(t, 0.15, 0.55)), "mono", 14, 0.06) + 4, 64, 9, 14, GREEN)

    # --- corner HUD
    if t > 0.6:
        corner_hud(fr, t, GREY, 0.8, tl="", tr=f"T+{t:06.2f}", bl="QR EXC FF+  //  SDR37 2445S6",
                   br=f"BUF {int(26 * clamp((t - 2) / 3.5)):03d}%")

    # --- ring assembly
    fill = ease_out(lin(t, 1.0, 4.6), 2)
    if t < 6.0:
        ring_stack(fr, t, fill)
        # registration crosses
        if t > 1.2:
            for a in (45, 135, 225, 315):
                ar = math.radians(a)
                crosshair(fr, CX + 470 * math.cos(ar), CY + 470 * math.sin(ar), 9, RED, 1.5)
            fr.line(CX - 520, CY, CX - 470, CY, RED, 1.5)
            fr.line(CX + 470, CY, CX + 520, CY, RED, 1.5)
        # vertical INITIATING
        if t > 1.8:
            s = scramble("INITIATING", lin(t, 1.8, 2.6), t, 3)
            fr.vtext(s, CX - 470, CY + 150, "wide", 30, GREEN, tracking=0.12)
        # label over the ring
        if t > 2.6:
            lab = scramble("FLOW_SOLVER", lin(t, 2.6, 3.2), t, 5)
            wdt = fr.text_width("FLOW_SOLVER", "wide", 30, 0.04)
            fr.rect(CX + 230, CY - 128, wdt + 70, 44, BLACK)
            checker_glyph(fr, CX + 240, CY - 120, 7, GREEN, seed=4, n=4)
            fr.text(lab, CX + 282, CY - 106, "wide", 30, GREEN, anchor="middle", tracking=0.04)
        # centre counter
        if t > 2.0:
            pct = int(100 * ease_in(lin(t, 2.0, 5.6), 2))
            fr.text(f"{pct:03d}", CX, CY + 10, "tek9", 150, WHITE, align="center", anchor="middle")
            fr.text("BUFFER  //  LOAD", CX, CY + 92, "mono7", 14, BLUE, align="center", anchor="top",
                    tracking=0.2)
            seg_bar(fr, CX - 120, CY - 100, 240, 8, pct / 100, GREEN, n=20, gap=3, dim=NAVY)

    # --- beat cards 6.0–8.0
    if 6.0 <= t < 6.5:
        fr.fill(BLUE)
        cell = 34
        for j in range(4):
            for i in range(3):
                checker_glyph(fr, 560 + i * 230, 190 + j * 180, cell, YELLOW, seed=100 + i + j * 3, n=4,
                              density=0.62)
        fr.vtext("SOLVER", 1300, 760, "wide9", 92, BLACK, tracking=0.05)
        for k, (x, y) in enumerate(((420, 270), (1450, 460), (420, 640))):
            label_box(fr, "LINK", x, y, "mono7", 14, BLUE, YELLOW, pad=(8, 5))
    elif 6.5 <= t < 7.0:
        fr.fill(YELLOW)
        for k in range(-2, 3):
            if k == 0:
                fr.rect(0, 540 - 62, W, 124, BLACK)
                banner(fr, "INITIATE", 540, 104, YELLOW, t, speed=-1400, fname="wide9", solid=True)
            else:
                banner(fr, "INITIATE", 540 + k * 124, 104, BLACK, t, speed=900 * (1 if k % 2 else -1),
                       fname="wide9", outline_w=2.2, x0=k * 333)
        micro_block(fr, 64, 64, ["CFD//SOLVER.4", "HANDSHAKE ........ OK", "LINK .............. 4/4"],
                    BLACK, 15, "mono7")
        label_box(fr, "anRTKL  PULL  LATCH  KL2 KL4", 960, 120, "mono7", 15, YELLOW, BLACK, align="center")
    elif 7.0 <= t < 7.6:
        fr.fill(BLACK)
        ring_stack(fr, t * 3, 1.0, palette=(RED, BLUE), scale=1.0 + 0.6 * ease_in(lin(t, 7.0, 7.6)))
    elif t >= 7.6:
        fr.fill(BLACK)
        if t < 7.8:
            fr.text("READY", CX, CY, "mono7", 22, LIME, align="center", anchor="middle", tracking=0.4)

    img = fr.array()
    # --- glitch accents
    if 4.9 < t < 5.05 or 5.45 < t < 5.55:
        img = fx.glitch_burst(img, rng, 0.35)
    if 7.0 <= t < 7.6:
        amt = lin(t, 7.0, 7.6)
        img = fx.block_displace(img, rng, n=int(10 + 50 * amt), cell=48, max_cells=6, max_shift=6)
        img = fx.pixel_blocks(img, rng, int(30 * amt), 24, [BLUE, RED, WHITE])
        img = fx.rgb_split(img, int(6 + 20 * amt))
    if 5.95 < t < 6.0 or 6.45 < t < 6.5:
        img = fx.slice_shift(img, rng, 20, 60, 300)
    return img
