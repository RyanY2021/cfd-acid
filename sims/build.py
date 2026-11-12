"""Define and run every simulation used in the film.  python -m sims.build <name> [<name>...]"""
import math
import sys

import numpy as np

from sims import lbm, spectral, geometry as G


def disk(nx, ny, cx, cy, r):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def bands(ny, period, duty=0.5, offset=0, margin=0, step=0.6):
    ys = np.arange(margin, ny - margin, step, dtype=np.float32)
    keep = ((ys + offset) % period) < period * duty
    return ys[keep]


# ----------------------------------------------------------------------------------- LBM cases
def title():
    # flow past the stencil letters C F D, at half of 1080p
    m = G.title_mask(0.5)
    ny, nx = m.shape
    m[:, :2] = False
    ys = bands(ny, 18, 0.34, step=0.6)
    lbm.run("title", m, u0=0.075, nu=0.004, cs=0.17, steps_per_frame=62, frames=240,
            spinup=0, seed_ys=ys, seed_every=8, max_particles=1_400_000, store_particles=420_000,
            init_uniform=True, kick=0.0)


def karman():
    nx, ny = 1440, 480
    D = 46
    m = disk(nx, ny, 260, ny / 2 + 1.5, D / 2)
    ys = bands(ny, 16, 0.3, step=0.5)
    lbm.run("karman", m, u0=0.1, nu=0.1 * D / 220, cs=0.0, steps_per_frame=26, frames=200,
            spinup=28000, seed_ys=ys, seed_every=6, max_particles=1_600_000, store_particles=420_000)


def resweep(res=(20, 100, 1000, 20000)):
    nx, ny = 576, 324
    D = 30
    for re in res:
        m = disk(nx, ny, 150, ny / 2 + 1.0, D / 2)
        u0 = 0.08 if re < 1000 else 0.1
        nu = u0 * D / re
        cs = 0.0 if re <= 100 else 0.17
        ys = bands(ny, 12, 0.34, step=0.5)
        lbm.run(f"re{re}", m, u0=u0, nu=nu, cs=cs, steps_per_frame=24, frames=150,
                spinup=40000 if re == 100 else 16000, kick=0.6 if re == 100 else 0.02, seed_ys=ys, seed_every=6, max_particles=1_000_000, store_particles=260_000)


def airfoil():
    nx, ny = 960, 420
    m = G.airfoil_mask(nx, ny, 300, ny / 2, 240, 17)
    ys = bands(ny, 14, 0.34, step=0.5)
    lbm.run("airfoil", m, u0=0.09, nu=0.09 * 240 / 4000, cs=0.17, steps_per_frame=26, frames=120,
            spinup=14000, seed_ys=ys, seed_every=6, max_particles=1_200_000, store_particles=300_000)


def car():
    nx, ny = 960, 360
    m = G.car_mask_blender(nx, ny, 300)
    ys = bands(ny, 12, 0.34, step=0.5)
    lbm.run("car", m, u0=0.07, nu=0.07 * 330 / 2500, cs=0.2, steps_per_frame=26, frames=120,
            spinup=12000, seed_ys=ys, seed_every=6, max_particles=1_200_000, store_particles=300_000)


def stenosis():
    nx, ny = 960, 270
    m, top, bot = G.stenosis_walls(nx, ny)
    np.save(lbm.CACHE / "stenosis_walls.npy", np.stack([top, bot]))
    ys = bands(ny, 7, 0.4, margin=int(0.18 * ny) + 1, step=0.45)
    period = 2400

    def inflow(s):
        ph = (s % period) / period
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * ph) ** 7 if math.sin(2 * math.pi * ph) > 0 else 0.5
        return 0.05 * (0.6 + 0.9 * pulse), 0.0

    lbm.run("stenosis", m, u0=0.05, nu=0.05 * 170 / 900, cs=0.15, steps_per_frame=24, frames=120,
            spinup=9000, seed_ys=ys, seed_every=6, max_particles=1_000_000, store_particles=260_000,
            inflow=inflow, kick=0.0)


# ----------------------------------------------------------------------------------- spectral cases
def kh():
    nx, ny = 1024, 512
    lx, ly = 4 * np.pi, 2 * np.pi
    x = np.arange(nx) * lx / nx
    y = np.arange(ny) * ly / ny
    X, Y = np.meshgrid(x, y)
    d = 0.07
    u = np.where(Y <= np.pi, np.tanh((Y - np.pi / 2) / d), np.tanh((3 * np.pi / 2 - Y) / d))
    rng = np.random.default_rng(3)
    v = np.zeros_like(X)
    for k, a in ((2, 0.05), (3, 0.03), (5, 0.02), (7, 0.012)):
        v += a * np.sin(k * X + rng.uniform(0, 2 * np.pi)) * (np.exp(-((Y - np.pi / 2) / 0.4) ** 2)
                                                             + np.exp(-((Y - 3 * np.pi / 2) / 0.4) ** 2))
    S = spectral.Spectral2D(nx, ny, lx, ly, 0)
    w = S.ifft(1j * S.KX * S.fft(v)) - S.ifft(1j * S.KY * S.fft(u))
    c = 0.5 * (np.tanh((Y - np.pi / 2) / 0.03) - np.tanh((Y - 3 * np.pi / 2) / 0.03))
    spectral.run("kh", w, c, lx, ly, nu=6e-5, kappa=2e-5, frames=200, t_per_frame=0.045)


def turb():
    n = 512
    L = 2 * np.pi
    S = spectral.Spectral2D(n, n, L, L, 0)
    rng = np.random.default_rng(11)
    k = np.sqrt(S.K2)
    amp = k ** 4 * np.exp(-(k / 12.0) ** 2)
    ph = np.exp(2j * np.pi * rng.random(S.K2.shape))
    w = S.ifft(amp * ph * S.dealias)
    w = w / np.abs(w).std() * 12.0
    spectral.run("turb", w, None, L, L, nu=8e-5, kappa=0, frames=420, t_per_frame=0.03)


CASES = dict(re100=lambda: resweep((100,)), title=title, karman=karman, resweep=resweep, airfoil=airfoil, car=car,
             stenosis=stenosis, kh=kh, turb=turb)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        CASES[name]()
