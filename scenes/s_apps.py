"""H — APPLICATIONS (76–88 s): header banner + four numbered cards over live sims."""
import functools

from PIL import Image

from scenes.common import *
from sims import geometry as G

CARD_T = 2.75
_RS = {}


def rs_for(name, shape, rect, ow, oh):
    if name not in _RS:
        _RS[name] = Resampler(shape, rect, ow, oh)
    return _RS[name]


def header(fr, t):
    fr.fill(LIME)
    for k in range(-3, 4):
        if k == 0:
            fr.rect(0, 540 - 70, W, 140, BLACK)
            banner(fr, "APPLICATIONS", 540, 116, LIME, t, speed=-1300, fname="wide9", solid=True)
        else:
            banner(fr, "APPLICATIONS", 540 + k * 140, 116, BLACK, t, speed=700 * (1 if k % 2 else -1),
                   fname="wide9", outline_w=2.2, x0=k * 400)
    label_box(fr, "WHERE IT FLOWS // 04 SECTORS", 64, 64, "mono7", 16, LIME, BLACK, pad=(10, 6))


def title_block(fr, idx, word, lines, accent, fg=WHITE, bg=BLACK):
    fr.rect(64, 64, 900, 190, bg)
    fr.text(f"{idx:02d}", 84, 84, "tek9", 150, accent, anchor="top")
    fr.text(word, 330, 96, "wide9", 72, fg, anchor="top")
    micro_block(fr, 334, 186, lines, accent, 15, "mono7", lh=1.4)
    label_box(fr, f"APPLICATIONS {idx:02d}/04", W - 64, 64, "mono7", 15, bg, accent, align="right", pad=(10, 6))


def aero(fr, lt, T=0.0):
    name = "airfoil"
    w = sim_frame(name, "vort", 10 + lt * FPS)
    rs = rs_for(name, w.shape, (60, 0, 60 + 960 * 0.98, 420), W, int(420 * W / (960 * 0.98)))
    oh = rs.out[1]
    img = vort_image(rs(w), 0.03, cell=2, dead=0.12)
    p = sim_parts(name, 10 + lt * FPS)
    px, py = rs.to_out(p[:, 0], p[:, 1])
    splat(img, px, py, WHITE, size=2, alpha=0.5)
    y0 = (H - oh) // 2 + 60
    fr.image(img, 0, y0)
    zb = G.airfoil_outline(400) * np.exp(-1j * np.radians(17))
    s = 240 / 4.0
    ax, ay = rs.to_out(300 + zb.real * s, 210 - zb.imag * s)
    fr.polyline(np.stack([ax, ay + y0], 1), LIME, fill=True, closed=True)
    title_block(fr, 1, "AEROSPACE", ["WING STALL // ANGLE OF ATTACK 17°", "SEPARATED SHEAR LAYER // LES"], LIME)


VEH = ROOT / "render" / "vehicle"
INK = (14, 14, 18)
# hero camera — keep in sync with vehicle/build_vehicle.py (HERO_CAM)
HERO_LOC, HERO_TGT, HERO_LENS, HERO_SHIFT = (-8.4, -9.8, 2.8), (0.15, 0.0, 0.66), 60.0, -0.21
N_HERO = 40
N_SIDE = 30                  # side-view wheel loop: 12 deg/frame CCW (vehicle/build_vehicle.py)
BELT = 435.0                 # px/s ground speed = wheel surface speed (r 69 px, 360 deg/s)


@functools.lru_cache(maxsize=64)
def vimg(name):
    a = np.asarray(Image.open(VEH / name).convert("RGBA")).astype(np.float32)
    a[..., :3] *= a[..., 3:] / 255.0                    # premultiply for Skia
    return a.astype(np.uint8)


def hero_angle(k):
    return math.radians(-34 + 30 * (1 - (1 - k) ** 2))


def project(P):
    loc, tgt = np.array(HERO_LOC), np.array(HERO_TGT)
    d = tgt - loc
    d /= np.linalg.norm(d)
    r = np.cross(d, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, d)
    v = np.asarray(P) - loc
    x, y, z = v @ r, v @ u, v @ d
    f = HERO_LENS / 36.0
    return W * (0.5 + x / z * f - HERO_SHIFT), H / 2 - y / z * f * W


def car_point(local, ang):
    x, y, z = local
    c, s_ = math.cos(ang), math.sin(ang)
    return (x * c - y * s_, x * s_ + y * c, z)


def callout(fr, p, dx, dy, text, color=INK, bg=PAPER):
    x, y = p
    fr.circle(x, y, 7, color, stroke=2)
    fr.line(x + 7 * np.sign(dx), y + 7 * np.sign(dy) * 0, x + dx, y + dy, color, 1.5)
    fr.line(x + dx, y + dy, x + dx + 60 * np.sign(dx), y + dy, color, 1.5)
    tx = x + dx + 68 * np.sign(dx)
    fr.text(text, tx, y + dy, "mono7", 13, color, align="left" if dx > 0 else "right", anchor="middle",
            tracking=0.06)


def auto_product(fr, lt, T):
    k = clamp(lt / 1.375)
    fi = int(round(k * (N_HERO - 1)))
    hero = vimg(f"hero_{fi:03d}.png")
    fr.image(hero[..., :3])
    ang = hero_angle(fi / (N_HERO - 1))
    # Swiss product layout (after the NuCaloric spots)
    fr.rect(0, 0, 54, H, MAGENTA)
    checker_glyph(fr, 9, 14, 9, WHITE, seed=7, n=4, density=0.6)
    fr.vtext("CFD-02  Slipstream", 16, 520, "inter", 26, WHITE)
    fr.line(27, 640, 27, 720, WHITE, 3)
    fr.polyline([(15, 652), (27, 640), (39, 652)], WHITE, 3)
    fr.vtext("Validated by simulation", 16, 1040, "inter", 20, WHITE)
    fr.text("Automotive", 112, 62, "inter", 104, INK, anchor="top", tracking=-0.025)
    fr.text("02", 98, 200, "tek9", 430, MAGENTA, anchor="top", tracking=-0.02)
    micro_block(fr, 114, 668, ["SLIPSTREAM // LOW-DRAG SHELL", "Cd 0.24   A 2.1 M2   L 4.24 M",
                               "SIDE PROFILE SOLVED: LBM D2Q9 + LES"], INK, 15, "mono7", lh=1.5)
    fr.text("YZ02-23AB-45CD      C33", 114, H - 70, "mono7", 15, INK, anchor="baseline", tracking=0.08)
    barcode(fr, W - 104, 800, 150, 44, INK, seed=21, vertical=True)
    checker_glyph(fr, W - 100, 970, 9, INK, seed=40, n=4, density=0.55)
    checker_glyph(fr, W - 100, 1020, 9, INK, seed=44, n=4, density=0.55)
    fr.text("Automotive // aerodynamics", W - 70, 62, "inter", 22, INK, align="right", anchor="top")
    # callouts locked to the 3D model
    if lt > 0.25:
        callout(fr, project(car_point((1.55, 0.0, 1.46), ang)), 70, -70, "TOW // 2.0 T")
        callout(fr, project(car_point((-0.35, -0.5, 1.30), ang)), -60, -110, "CANOPY // ATTACHED FLOW")
        callout(fr, project(car_point((-1.33, -1.13, 0.445), ang)), -90, 120, "LAMINAR COMPOUND")
        callout(fr, project(car_point((1.92, -0.55, 0.62), ang)), -70, 230, "WAKE // SEPARATION LINE")


def auto_tunnel(fr, lt, T):
    fr.fill(PAPER)
    i = 25 + lt * FPS * 1.6
    w = sim_frame("car", "vort", i)
    rs = rs_for("car_full", w.shape, (0, 40, 960, 340), W, 600)
    img = vort_image(rs(w), 0.024, cell=2, dead=0.12, zero=PAPER,
                     neg=[(180, 190, 255), (90, 110, 255), BLUE], pos=[(255, 170, 200), (245, 70, 140), MAGENTA],
                     gamma=0.7)
    p = sim_parts("car", i)
    px, py = rs.to_out(p[:, 0], p[:, 1])
    splat(img, px, py, INK, size=2, alpha=0.42)
    y0 = 300
    fr.image(img, 0, y0)
    fr.image(vimg(f"side_{int(lt * FPS) % N_SIDE:02d}.png"), 0, y0 - 80)
    gy = y0 + (300 - 40) * 2
    fr.rect(0, gy, W, H - gy, INK)
    hazard(fr, 0, gy, W, 22, PAPER, 16, phase=lt * BELT)
    fr.rect(0, y0 - 4, W, 4, INK)
    micro_block(fr, 64, gy + 52, ["GROUND PLANE // NO-SLIP", "INFLOW >>> U = 0.07 LU"], PAPER, 14, "mono7", lh=1.5)
    # header
    fr.text("02", 64, 56, "tek9", 150, MAGENTA, anchor="top")
    fr.text("Wind tunnel", 300, 64, "inter", 72, INK, anchor="top", tracking=-0.02)
    micro_block(fr, 304, 160, ["SAME SHAPE, SOLVED. 960 x 360 LATTICE // Re 2.5E3 (LES)",
                               "BLUE / MAGENTA = VORTICITY   //   INK = STREAKLINES"], INK, 14, "mono7", lh=1.5)
    cd = 0.24 + 0.012 * math.sin(lt * 9)
    fr.text(f"{cd:4.3f}", W - 64, 56, "tek9", 110, INK, align="right", anchor="top")
    fr.text("DRAG COEFFICIENT  Cd", W - 64, 186, "mono7", 14, INK, align="right", anchor="top", tracking=0.12)
    seg_bar(fr, W - 330, 216, 266, 8, cd / 0.4, MAGENTA, n=16, dim=(220, 220, 215))
    # wake tracker
    sub = np.abs(w[120:262:3, 600:900:3])
    j, ii = np.unravel_index(np.argmax(sub), sub.shape)
    cx, cy = (600 + ii * 3) * 2, y0 + (120 + j * 3 - 40) * 2
    brackets(fr, cx - 56, cy - 56, 112, 112, 14, INK, 2)
    label_box(fr, "WAKE VORTEX", cx - 56, cy + 66, "mono7", 13, PAPER, INK, pad=(6, 4))
    label_box(fr, f"APPLICATIONS 02/04", W - 64, H - 64 - 18, "mono7", 14, PAPER, MAGENTA, align="right",
              pad=(10, 6))


def auto(fr, lt, T=0.0):
    if lt < 1.375:
        auto_product(fr, lt, T)
    else:
        auto_tunnel(fr, lt - 1.375, T)


def cardio(fr, lt, T=0.0):
    """Arachne-style: black, glossy crimson artery walls, topographic vorticity bands, bone streaklines."""
    name = "stenosis"
    i = 10 + lt * FPS
    w = sim_frame(name, "vort", i)
    y0, ph = 270, 540
    rs = rs_for(name, w.shape, (0, 0, 960, 270), W, ph)
    img = np.zeros((H, W, 3), np.float32)
    img[y0:y0 + ph] = topo_bands(rs(w), 0.03)
    p = sim_parts(name, i)[::2]
    px, py = rs.to_out(p[:, 0], p[:, 1])
    lay = np.zeros((H, W, 3), np.float32)
    splat(lay, px, py + y0, BONE, size=2)
    img = np.maximum(img, lay * 0.42)
    beat = math.exp(-((lt * 2) % 1.0 / 0.12) ** 2)          # 120 BPM pulse
    img = np.where(WALL_MASK()[..., None], wall_shade(lt, beat), img)
    fr.image(np.clip(img, 0, 255).astype(np.uint8))
    film_hud(fr, lt, title="CARDIOVASCULAR", sub="STENOSED ARTERY // PULSATILE JET // Re 900",
             status="WALL SHEAR STRESS // RISK MAP", bpm=72 + int(6 * beat), ecg_on=True, case="03")


_WALL = {}


def _walls():
    if not _WALL:
        tb = np.load(CACHE / "stenosis_walls.npy")
        xs = np.linspace(0, tb.shape[1] - 1, W)
        top = np.interp(xs, np.arange(tb.shape[1]), tb[0]) * 2 + 270
        bot = np.interp(xs, np.arange(tb.shape[1]), tb[1]) * 2 + 270
        yy = np.arange(H)[:, None].astype(np.float32)
        d_top = top[None, :] - yy
        d_bot = yy - bot[None, :]
        d = np.where(d_top > 0, d_top, np.where(d_bot > 0, d_bot, -1.0)).astype(np.float32)
        _WALL["d"] = d
        _WALL["mask"] = d >= 0
        _WALL["tex"] = np.random.default_rng(3).normal(0, 1, (H // 4 + 1, W // 4 + 1)).astype(np.float32)
    return _WALL


def WALL_MASK():
    return _walls()["mask"]


def wall_shade(lt, beat):
    """Glossy crimson tube walls: body falls into darkness, a sharp specular rim at the lumen."""
    wd = _walls()
    d = np.maximum(wd["d"], 0)
    x = np.arange(W, dtype=np.float32)[None, :]
    tex = np.repeat(np.repeat(wd["tex"], 4, 0), 4, 1)[:H, :W]
    body = (0.18 + 0.82 * np.exp(-d / 70.0)) * (1 + 0.06 * tex) * (1 + 0.12 * beat)
    sheen = 0.5 + 0.5 * np.sin(x / 150.0 + lt * 1.7 + 0.6 * np.sin(x / 47.0))
    spec = np.exp(-((d - 12) / 5.0) ** 2) * (0.35 + 0.65 * sheen) + 0.35 * np.exp(-((d - 48) / 14.0) ** 2) * sheen
    col = np.array(CRIMSON, np.float32) * body[..., None]
    col = col + np.array((255, 150, 160), np.float32) * (spec[..., None] * 0.6)
    return col


def topo_bands(field, scale, n=7):
    """Topographic contour bands (after the Arachne contour card): crimson stripes on black."""
    v = np.clip(np.abs(field) / scale, 0, 1.2) * n
    fr_ = v - np.floor(v)
    aa = 0.12
    stripe = np.clip((fr_ - 0.45) / aa, 0, 1) * np.clip((1.0 - fr_) / aa, 0, 1)
    stripe *= v > 0.6
    lvl = np.clip(np.floor(v) / n, 0, 1)
    tone = 0.45 + 0.55 * lvl
    neg = field < 0
    base = np.where(neg[..., None], np.array((150, 0, 16), np.float32), np.array(CRIMSON, np.float32))
    return base * (stripe * tone)[..., None]


def climate(fr, lt, T=0.0):
    w = sim_frame("turb", "w", 60 + lt * FPS * 2)
    rs = rs_for("turbc", w.shape, (0, 0, 512, 288), W, H)
    f = rs(w)
    img = fv.diverging_map(f, 12, [(0, 10, 60), BLUE, (110, 150, 255)], [(0, 10, 60), (180, 200, 255), WHITE],
                           (2, 4, 30), cell=3, gamma=0.8)
    fr.image(img)
    for x in range(0, W, 160):
        fr.line(x, 0, x, H, (60, 70, 160), 1, 0.6)
    for y in range(0, H, 160):
        fr.line(0, y, W, y, (60, 70, 160), 1, 0.6)
    sub = f[::8, ::8]
    j, i = np.unravel_index(np.argmax(sub), sub.shape)
    cx, cy = i * 8, j * 8
    for r in (40, 80, 120):
        fr.circle(cx, cy, r, YELLOW, stroke=2, dash=([10, 8], lt * 40))
    label_box(fr, "CYCLONE // L", cx + 130, cy - 20, "mono7", 14, BLACK, YELLOW, pad=(6, 4))
    title_block(fr, 4, "CLIMATE", ["ATMOSPHERE & OCEAN // 2D TURBULENCE", "VORTEX MERGERS // INVERSE CASCADE"],
                (110, 150, 255))


HAND = ROOT / "render" / "hand"
HAND_T = 3.0
BAR = 138                      # 2.39:1 letterbox
GREY = (88, 88, 92)
PUSH_C = (1166, 790)           # radial artery at the wrist in the last hand frame (push-in target)


@functools.lru_cache(maxsize=4)
def himg(f):
    return np.asarray(Image.open(HAND / f"hand_{int(np.clip(f, 0, 89)):03d}.png").convert("RGB"))


def ecg(fr, lt, x0, y0, w, color, amp=1.0):
    pts = []
    for i in range(0, w, 3):
        tt = lt - (w - i) / w * 1.2
        v = 0.0
        for b in np.arange(0.0, 6.0, 0.5):
            d = tt - b
            v += -38 * math.exp(-(d / 0.012) ** 2) + 14 * math.exp(-((d - 0.03) / 0.02) ** 2) \
                - 8 * math.exp(-((d - 0.16) / 0.05) ** 2)
        pts.append((x0 + i, y0 + v * amp))
    fr.polyline(pts, color, 2)


def film_hud(fr, lt, title, sub, status, bpm, ecg_on, case):
    """Letterbox bars + thin grey HUD + serif titling (Arachne layout)."""
    fr.rect(0, 0, W, BAR, BLACK)
    fr.rect(0, H - BAR, W, BAR, BLACK)
    # thin rails with ticks
    for y in (BAR - 18, H - BAR + 18):
        fr.line(64, y, W - 64, y, (40, 40, 44), 1)
        for x in range(64, W - 63, 120):
            fr.line(x, y - 4, x, y + 4, (60, 60, 64), 1)
    for (x, y, dx, dy) in ((64, BAR + 14, 1, 1), (W - 64, BAR + 14, -1, 1), (64, H - BAR - 14, 1, -1),
                           (W - 64, H - BAR - 14, -1, -1)):
        fr.line(x, y, x + 26 * dx, y, (110, 110, 114), 1.5)
        fr.line(x, y, x, y + 26 * dy, (110, 110, 114), 1.5)
    # top bar: serif title + glyph row + micro labels
    fr.text(title, 64, 34, "serif7", 30, CRIMSON, anchor="top", tracking=0.08)
    glyph_row(fr, 66, 78, 4, CRIMSON, [5, 9, 13, 17, 21, 25, 29], gap=6, n=4)
    fr.text(sub, 420, 40, "mono", 12, GREY, anchor="top", tracking=0.12)
    fr.text(f"CASE {case} // HAEMODYNAMICS", 420, 60, "mono", 12, GREY, anchor="top", tracking=0.12)
    fr.text(f"T+{lt:05.2f}", W - 64, 40, "mono", 12, GREY, align="right", anchor="top", tracking=0.12)
    # bottom bar: ECG, status, BPM in serif
    if ecg_on:
        ecg(fr, lt, 64, H - 40, W - 520, CRIMSON, 0.8)
    else:
        fr.line(64, H - 40, W - 456, H - 40, (60, 60, 64), 1.5)
    fr.text(status, 64, H - BAR + 34, "mono", 12, GREY, anchor="top", tracking=0.14)
    fr.text(f"{bpm:03d}", W - 150, H - BAR / 2 + 8, "serif9", 64, CRIMSON if bpm else GREY, align="right",
            anchor="middle")
    fr.text("BPM", W - 64, H - BAR / 2 + 8, "serif7", 20, GREY, align="right", anchor="middle", tracking=0.1)
    fr.text("OMNIA FLUUNT", W / 2, 44, "serif7", 13, GREY, align="center", anchor="top", tracking=0.4)


def film_grade(img, rng, amt=1.0):
    """Crushed blacks, crimson halation, vignette, coarse luminance-weighted film grain."""
    f = img.astype(np.float32) / 255.0
    f = np.clip((f - 0.015) / 0.97, 0, 1) ** 1.08
    hi = np.clip(f - 0.45, 0, None)
    small = hi[::6, ::6]
    for _ in range(3):
        small = (small + np.roll(small, 1, 0) + np.roll(small, -1, 0) + np.roll(small, 1, 1) + np.roll(small, -1, 1)) / 5
    halo = np.repeat(np.repeat(small, 6, 0), 6, 1)[:H, :W]
    f = f + halo * np.array([0.9, 0.18, 0.2], np.float32) * 0.7 * amt
    yy, xx = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
    f *= (1 - 0.28 * amt * (xx * xx * 0.7 + yy * yy * 1.1))[..., None].astype(np.float32)
    lum = f.mean(axis=2, keepdims=True)
    g = rng.normal(0, 1, (H // 2 + 1, W // 2 + 1)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[:H, :W, None] * 0.7 + rng.normal(0, 1, (H, W, 1)).astype(np.float32) * 0.3
    f = f + g * 0.075 * amt * (0.35 + 2.2 * lum * (1 - np.clip(lum, 0, 1)))
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def hand_card(fr, lt, T=0.0):
    f = lt * FPS
    alive = lt >= 1.5
    img = himg(f)
    if lt > 2.65:                                        # push into the artery (picture only, HUD stays)
        z = 1 + 5.0 * ease_in(lin(lt, 2.65, 3.0), 2)
        cx, cy = PUSH_C
        hh, ww = int(H / z), int(W / z)
        y0 = int(np.clip(cy - hh / 2, 0, H - hh))
        x0 = int(np.clip(cx - ww / 2, 0, W - ww))
        img = np.asarray(Image.fromarray(img[y0:y0 + hh, x0:x0 + ww]).resize((W, H), Image.BICUBIC))
    fr.image(img)
    bpm = 0 if lt < 1.0 else int(min(72, 72 * ease_out(lin(lt, 1.0, 2.2))))
    status = "SHELL // STATIC   EVALUATING" if lt < 1.0 else (
        f"PERFUSION {int(100 * min(1, (lt - 1.0) / 1.6)):03d}%" if lt < 2.6 else "VITAL SIGNS // NOMINAL")
    if alive and lt < 1.5 + 3 / 30:
        status = "STRINGS RELEASED"
    film_hud(fr, lt, title="CARDIOVASCULAR", sub="PERFUSION // MARIONETTE SHELL 07", status=status, bpm=bpm,
             ecg_on=lt >= 1.0, case="03")


CARDS = [(aero, 2.75, 1), (auto, 2.75, 2), (hand_card, HAND_T, 0), (cardio, 2.75, 3), (climate, 2.75, 4)]
APPS_T = 1.0 + sum(d for _, d, _ in CARDS)


def render(t, T):
    rng = frame_rng(T)
    fr = Frame()
    if t < 1.0:
        header(fr, t)
        img = fr.array()
        if t > 0.9:
            img = fx.glitch_burst(img, rng, lin(t, 0.9, 1.0))
        return img
    lt = t - 1.0
    for fn, dur, case in CARDS:
        if lt < dur or fn is CARDS[-1][0]:
            break
        lt -= dur
    fn(fr, lt, T)
    filmic = fn in (hand_card, cardio)
    if case not in (0, 2) and not filmic:
        corner_hud(fr, t, (170, 170, 185), tl="", tr="", bl=f"CASE {case:02d} // LIVE SOLVE", br=f"T+{T:06.2f}",
                   boxed=True)
    img = fr.array()
    if filmic:
        img = film_grade(img, rng)
    if lt < 2 / 30 and not filmic:
        img = 255 - img if lt < 1 / 30 else fx.glitch_burst(img, rng, 0.6)
    elif fn is hand_card and lt < 1 / 30:
        img = (img * 0.25).astype(np.uint8)                 # dip from the white car card
    elif fn is hand_card and abs(lt - 1.5) < 1 / 30:
        img = fx.slice_shift(img, rng, 6, 20, 60)           # it comes alive: the strings snap
    elif beat_index(t) != beat_index(t - 1 / 30) and not filmic:
        img = fx.slice_shift(img, rng, 8, 40, 160)
    if t > APPS_T - 0.15:
        img = fx.glitch_burst(img, rng, lin(t, APPS_T - 0.15, APPS_T))
    return img
