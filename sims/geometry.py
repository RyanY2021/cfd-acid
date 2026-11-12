"""Shared geometry: title letters, Joukowski airfoil, car, stenosis — used by both sims and scenes."""
import numpy as np
import skia

from engine.core import Frame, W, H

# --------------------------------------------------------------- title "CFD"
TITLE_TEXT = "CFD"
TITLE_FONT = "stencil"
TITLE_SIZE = 760          # px at 1080p
TITLE_BASE_Y = 790        # baseline at 1080p
TITLE_TRACK = 0.02


def title_mask(scale=0.5):
    """Obstacle mask of the title at sim resolution (scale relative to 1080p)."""
    nx, ny = int(W * scale), int(H * scale)
    fr = Frame(nx, ny, bg=(0, 0, 0))
    fr.text(TITLE_TEXT, nx / 2, TITLE_BASE_Y * scale, TITLE_FONT, TITLE_SIZE * scale, (255, 255, 255),
            align="center", tracking=TITLE_TRACK)
    return fr.array()[..., 0] > 127


# --------------------------------------------------------------- Joukowski airfoil
MU = complex(-0.09, 0.07)          # circle centre in zeta plane
R_CIRC = abs(1 - MU)               # passes through zeta = 1 -> sharp trailing edge


def joukowski(zeta):
    return zeta + 1.0 / zeta


def airfoil_outline(n=400):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = joukowski(MU + R_CIRC * np.exp(1j * th))
    return z   # chord roughly [-2, 2]


def poly_mask(nx, ny, pts_xy):
    fr = Frame(nx, ny, bg=(0, 0, 0))
    fr.polyline(pts_xy, (255, 255, 255), fill=True, closed=True)
    return fr.array()[..., 0] > 127


def airfoil_mask(nx, ny, cx, cy, chord_cells, aoa_deg):
    z = airfoil_outline(600)
    z = z * np.exp(-1j * np.radians(aoa_deg))        # nose-up rotation in y-down grid
    s = chord_cells / 4.0
    pts = np.stack([cx + z.real * s, cy - z.imag * s], 1)
    return poly_mask(nx, ny, pts)


# --------------------------------------------------------------- car silhouette (unit length ~1)
CAR = np.array([
    (0.00, 0.30), (0.02, 0.22), (0.07, 0.17), (0.18, 0.15), (0.30, 0.13), (0.38, 0.06),
    (0.46, 0.01), (0.56, 0.00), (0.66, 0.02), (0.76, 0.08), (0.88, 0.12), (0.97, 0.13),
    (1.00, 0.18), (1.00, 0.30), (0.93, 0.33), (0.07, 0.33)])
CAR_WHEELS = [(0.19, 0.31, 0.075), (0.81, 0.31, 0.075)]


def car_mask(nx, ny, x0, ground_y, length):
    fr = Frame(nx, ny, bg=(0, 0, 0))
    hgt = 0.38 * length
    top = ground_y - hgt + 0.03 * length
    pts = np.stack([x0 + CAR[:, 0] * length, top + CAR[:, 1] * length], 1)
    fr.polyline(pts, (255, 255, 255), fill=True, closed=True)
    for wx, wy, wr in CAR_WHEELS:
        fr.circle(x0 + wx * length, top + wy * length, wr * length, (255, 255, 255))
    m = fr.array()[..., 0] > 127
    m[ground_y:, :] = True
    return m


def car_outline_1080(x0, ground_y, length):
    hgt = 0.38 * length
    top = ground_y - hgt + 0.03 * length
    return np.stack([x0 + CAR[:, 0] * length, top + CAR[:, 1] * length], 1), \
        [(x0 + wx * length, top + wy * length, wr * length) for wx, wy, wr in CAR_WHEELS]


# --------------------------------------------------------------- stenosed artery channel
def stenosis_walls(nx, ny, wall=0.18, sev=0.62, xc=0.32, width=0.07):
    """Returns (mask, top_profile, bottom_profile) in cells."""
    x = np.arange(nx)
    bump = sev * 0.5 * (1 - 2 * wall) * ny * np.exp(-((x / nx - xc) / width) ** 2)
    top = wall * ny + bump
    bot = (1 - wall) * ny - bump
    yy = np.arange(ny)[:, None]
    m = (yy < top[None, :]) | (yy > bot[None, :])
    return m, top, bot


# --------------------------------------------------------------- Blender CFD-02 vehicle (side render -> obstacle)
def car_mask_blender(nx=960, ny=360, ground=300):
    """Obstacle from the orthographic side render (1920x720 px = 960x360 cells)."""
    import pathlib
    from PIL import Image
    from scipy import ndimage
    p = pathlib.Path(__file__).resolve().parent.parent / "render" / "vehicle" / "side_00.png"
    a = np.asarray(Image.open(p))[..., 3].astype(np.float32) / 255
    a = a.reshape(ny, 2, nx, 2).mean(axis=(1, 3))
    m = ndimage.binary_opening(a > 0.5, np.ones((3, 3)))     # drop the antenna wire
    m[ground:, :] = True
    return m
