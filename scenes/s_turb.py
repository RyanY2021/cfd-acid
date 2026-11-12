"""G5 — TURBULENCE (71–76 s): vorticity of 2D turbulence as an orbiting 3D point cloud."""
from scenes.common import *

N = 256
_G = None


def grid():
    global _G
    if _G is None:
        u = np.linspace(-1.6, 1.6, N)
        X, Y = np.meshgrid(u, u)
        _G = (X.ravel(), Y.ravel())
    return _G


def project(x, y, z, yaw, pitch, dist=4.2, f=980):
    cy, sy = math.cos(yaw), math.sin(yaw)
    xr, yr = x * cy - y * sy, x * sy + y * cy
    cp, sp = math.cos(pitch), math.sin(pitch)
    yc = yr * cp - z * sp
    zc = yr * sp + z * cp
    depth = dist + yc
    sx = W / 2 + f * xr / depth
    syy = H / 2 + 20 - f * zc / depth
    return sx, syy, depth


def spectrum_inset(fr, x, y, w, h):
    fr.rect(x, y, w, h, BLACK)
    fr.rect(x, y, w, h, GREY, stroke=1)
    fr.text("E(k)", x + 12, y + 12, "mono7", 13, WHITE, anchor="top")
    fr.text("k", x + w - 18, y + h - 22, "mono7", 13, WHITE, anchor="top")
    pts = []
    for i in range(60):
        lk = i / 59
        le = 0.25 + 0.6 * lk if lk < 0.18 else 0.25 + 0.6 * 0.18 - (5 / 3) * 0.55 * (lk - 0.18)
        if lk > 0.8:
            le -= 3.0 * (lk - 0.8) ** 1.5
        le = min(max(le, 0.02), 1.0)
        pts.append((x + 20 + lk * (w - 40), y + h - 30 - le * (h - 50)))
    fr.save()
    fr.clip(x, y, w, h)
    fr.polyline(pts, LIME, 2.5)
    fr.restore()
    fr.text("k^-5/3", x + w * 0.55, y + h * 0.42, "mono7", 15, LIME, anchor="top")
    fr.text("INERTIAL RANGE", x + 20, y + h - 18, "mono", 11, GREY, anchor="baseline", tracking=0.1)


def render(t, T):
    rng = frame_rng(T)
    w = sim_frame("turb", "w", 210 + t * FPS * 1.2)[::2, ::2]
    sc = 14.0
    v = np.clip(w.ravel() / sc, -1.5, 1.5)
    X, Y = grid()
    Z = v * 0.32
    yaw = math.radians(25 + 38 * t / 5)
    pitch = math.radians(58 - 8 * t / 5)
    sx, sy, d = project(X, Y, Z, yaw, pitch)
    order = np.argsort(-d)
    sx, sy, v = sx[order], sy[order], v[order]
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = BLACK
    av = np.abs(v)
    # faint carrier sheet
    m0 = (av < 0.12)
    sel = np.flatnonzero(m0)[::3]
    splat(img, sx[sel], sy[sel], (30, 38, 70), size=2)
    for lo, hi, cpos, cneg, size in ((0.12, 0.35, (20, 120, 40), (20, 30, 150), 2),
                                     (0.35, 0.7, GREEN, BLUE, 2),
                                     (0.7, 9, LIME, (120, 150, 255), 3)):
        mpos = (v >= lo) & (v < hi)
        mneg = (v <= -lo) & (v > -hi)
        splat(img, sx[mpos], sy[mpos], cpos, size=size)
        splat(img, sx[mneg], sy[mneg], cneg, size=size)
    fr = Frame()
    fr.text("TURBULENCE", W / 2, 560, "wide9", 236, WHITE, 0.45, align="center", anchor="middle", stroke=2)
    fr.image(img, blend="screen")
    label_box(fr, "ENERGY_CASCADE", 64, 64, "wide9", 40, BLACK, LIME, pad=(14, 10))
    micro_block(fr, 64, 136, ["2D DECAYING TURBULENCE // 512 x 512", "PSEUDO-SPECTRAL // DEALIASED 2/3",
                              "HEIGHT = VORTICITY"], WHITE, 15, "mono7", lh=1.4)
    spectrum_inset(fr, W - 420, H - 330, 360, 260)
    fr.text("KOLMOGOROV 1941", W - 420, H - 352, "mono7", 14, LIME, anchor="baseline", tracking=0.18)
    fr.text("BIG WHIRLS HAVE LITTLE WHIRLS", 64, H - 96, "wide9", 34, WHITE, anchor="top")
    fr.text("THAT FEED ON THEIR VELOCITY  - L.F. RICHARDSON, 1922", 64, H - 52, "mono7", 15, GREY, anchor="top",
            tracking=0.1)
    img = fr.array()
    if beat_index(t) != beat_index(t - 1 / 30) and t > 0.1:
        img = fx.glitch_burst(img, rng, 0.3)
    if t > 4.8:
        img = fx.glitch_burst(img, rng, lin(t, 4.8, 5.0))
    return img
