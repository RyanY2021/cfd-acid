"""D2Q9 lattice-Boltzmann solver (BGK + Smagorinsky LES) with tracer particles.

Pull-streaming, half-way bounce-back on a boolean obstacle mask, equilibrium
inlet at x=0, zero-gradient outlet, periodic in y (use solid rows for walls).
Records vorticity, speed and tracer positions into memory-mapped .npy stacks.
"""
import math
import pathlib
import time

import numba as nb
import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache"
CACHE.mkdir(exist_ok=True)


@nb.njit(parallel=True, fastmath=True, cache=True)
def lbm_step(f, fn, solid, tau0, smag, uin, vin, ux, uy):
    _, ny, nx = f.shape
    w0, ws, wd = 4.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0
    for y in nb.prange(ny):
        yp = y + 1 if y < ny - 1 else 0
        ym = y - 1 if y > 0 else ny - 1
        for x in range(nx):
            if solid[y, x]:
                ux[y, x] = 0.0
                uy[y, x] = 0.0
                for i in range(9):
                    fn[i, y, x] = f[i, y, x]
                continue
            if x == 0:
                r = 1.0
                u = uin
                v = vin
                usq = u * u + v * v
                fn[0, y, x] = w0 * r * (1 - 1.5 * usq)
                fn[1, y, x] = ws * r * (1 + 3 * u + 4.5 * u * u - 1.5 * usq)
                fn[2, y, x] = ws * r * (1 + 3 * v + 4.5 * v * v - 1.5 * usq)
                fn[3, y, x] = ws * r * (1 - 3 * u + 4.5 * u * u - 1.5 * usq)
                fn[4, y, x] = ws * r * (1 - 3 * v + 4.5 * v * v - 1.5 * usq)
                c = u + v
                fn[5, y, x] = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
                c = -u + v
                fn[6, y, x] = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
                c = -u - v
                fn[7, y, x] = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
                c = u - v
                fn[8, y, x] = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
                ux[y, x] = u
                uy[y, x] = v
                continue
            xm = x - 1
            xp = x + 1 if x < nx - 1 else x
            f0 = f[0, y, x]
            f1 = f[3, y, x] if solid[y, xm] else f[1, y, xm]
            f2 = f[4, y, x] if solid[ym, x] else f[2, ym, x]
            f3 = f[1, y, x] if solid[y, xp] else f[3, y, xp]
            f4 = f[2, y, x] if solid[yp, x] else f[4, yp, x]
            f5 = f[7, y, x] if solid[ym, xm] else f[5, ym, xm]
            f6 = f[8, y, x] if solid[ym, xp] else f[6, ym, xp]
            f7 = f[5, y, x] if solid[yp, xp] else f[7, yp, xp]
            f8 = f[6, y, x] if solid[yp, xm] else f[8, yp, xm]
            r = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            u = (f1 - f3 + f5 - f6 - f7 + f8) / r
            v = (f2 - f4 + f5 + f6 - f7 - f8) / r
            usq = u * u + v * v
            e0 = w0 * r * (1 - 1.5 * usq)
            e1 = ws * r * (1 + 3 * u + 4.5 * u * u - 1.5 * usq)
            e2 = ws * r * (1 + 3 * v + 4.5 * v * v - 1.5 * usq)
            e3 = ws * r * (1 - 3 * u + 4.5 * u * u - 1.5 * usq)
            e4 = ws * r * (1 - 3 * v + 4.5 * v * v - 1.5 * usq)
            c = u + v
            e5 = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
            c = -u + v
            e6 = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
            c = -u - v
            e7 = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
            c = u - v
            e8 = wd * r * (1 + 3 * c + 4.5 * c * c - 1.5 * usq)
            d1 = f1 - e1
            d2 = f2 - e2
            d3 = f3 - e3
            d4 = f4 - e4
            d5 = f5 - e5
            d6 = f6 - e6
            d7 = f7 - e7
            d8 = f8 - e8
            pxx = d1 + d3 + d5 + d6 + d7 + d8
            pyy = d2 + d4 + d5 + d6 + d7 + d8
            pxy = d5 - d6 + d7 - d8
            q = math.sqrt(pxx * pxx + pyy * pyy + 2 * pxy * pxy)
            tau = 0.5 * (tau0 + math.sqrt(tau0 * tau0 + smag * q / r))
            k = 1.0 - 1.0 / tau
            # regularised BGK: keep only the 2nd-order Hermite part of f_neq
            a = (pxx + pyy) / 3.0
            fn[0, y, x] = e0 + k * w0 * 4.5 * (-a)
            q1 = k * ws * 4.5 * (2.0 * pxx / 3.0 - pyy / 3.0)
            q2 = k * ws * 4.5 * (2.0 * pyy / 3.0 - pxx / 3.0)
            q5 = k * wd * 4.5 * (2.0 * a + 2.0 * pxy)
            q6 = k * wd * 4.5 * (2.0 * a - 2.0 * pxy)
            fn[1, y, x] = e1 + q1
            fn[3, y, x] = e3 + q1
            fn[2, y, x] = e2 + q2
            fn[4, y, x] = e4 + q2
            fn[5, y, x] = e5 + q5
            fn[7, y, x] = e7 + q5
            fn[6, y, x] = e6 + q6
            fn[8, y, x] = e8 + q6
            ux[y, x] = u
            uy[y, x] = v
    for y in nb.prange(ny):
        if not solid[y, nx - 1]:
            for i in range(9):
                fn[i, y, nx - 1] = fn[i, y, nx - 2]


@nb.njit(inline="always")
def _samp(a, x, y):
    ny, nx = a.shape
    x0 = int(x)
    y0 = int(y)
    fx = x - x0
    fy = y - y0
    if x0 < 0:
        x0 = 0
    if x0 > nx - 1:
        x0 = nx - 1
    if y0 < 0:
        y0 = 0
    if y0 > ny - 1:
        y0 = ny - 1
    x1 = x0 + 1 if x0 + 1 < nx else x0
    y1 = y0 + 1 if y0 + 1 < ny else 0
    return ((a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy) +
            (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy)


@nb.njit(parallel=True, cache=True)
def advect(px, py, alive, ux, uy, solid, dt):
    ny, nx = ux.shape
    for k in nb.prange(px.size):
        if not alive[k]:
            continue
        x = px[k]
        y = py[k]
        u1 = _samp(ux, x, y)
        v1 = _samp(uy, x, y)
        if not (u1 == u1 and v1 == v1 and abs(u1) < 1.0 and abs(v1) < 1.0):
            alive[k] = False
            continue
        xm = x + 0.5 * dt * u1
        ym = y + 0.5 * dt * v1
        if ym < 0:
            ym += ny
        if ym >= ny:
            ym -= ny
        if xm < 0 or xm >= nx - 1:
            alive[k] = False
            continue
        u2 = _samp(ux, xm, ym)
        v2 = _samp(uy, xm, ym)
        if not (u2 == u2 and v2 == v2 and abs(u2) < 1.0 and abs(v2) < 1.0):
            alive[k] = False
            continue
        x += dt * u2
        y += dt * v2
        if y < 0:
            y += ny
        if y >= ny:
            y -= ny
        if x < 0 or x >= nx - 2:
            alive[k] = False
            continue
        yi = int(y)
        if yi > ny - 1:
            yi = ny - 1
            y = 0.0
        if solid[yi, int(x)]:
            alive[k] = False
            continue
        px[k] = x
        py[k] = min(y, ny - 0.01)


def equilibrium(rho, u, v):
    w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, np.float32)
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], np.float32)
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], np.float32)
    usq = u * u + v * v
    f = np.empty((9,) + np.shape(u), np.float32)
    for i in range(9):
        cu = cx[i] * u + cy[i] * v
        f[i] = w[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f


def vorticity(ux, uy):
    return (np.gradient(uy, axis=1) - np.gradient(ux, axis=0)).astype(np.float32)


def run(name, solid, u0, nu, cs=0.0, steps_per_frame=20, frames=150, spinup=0,
        seed_ys=None, seed_every=3, max_particles=300_000, store_particles=200_000,
        inflow=None, init_uniform=True, kick=0.02, log_every=50, seed_from=None):
    """Run and record. inflow(step)->(u,v) overrides constant u0 inflow."""
    solid = np.ascontiguousarray(solid.astype(np.bool_))
    ny, nx = solid.shape
    tau0 = 3 * nu + 0.5
    smag = 18 * math.sqrt(2) * cs * cs
    print(f"[{name}] {nx}x{ny} tau0={tau0:.4f} Re-ish/len={u0 / nu:.1f} cs={cs}")
    u = np.full((ny, nx), u0 if init_uniform else 0.0, np.float32)
    v = np.zeros((ny, nx), np.float32)
    # asymmetric kick to trigger shedding early
    yy = np.arange(ny)[:, None]
    v += (kick * u0 * np.sin(2 * np.pi * yy / ny * 3)).astype(np.float32) * np.ones((1, nx), np.float32)
    u[solid] = 0
    v[solid] = 0
    f = equilibrium(1.0, u, v)
    fn = f.copy()
    ux = np.zeros((ny, nx), np.float32)
    uy = np.zeros((ny, nx), np.float32)
    if inflow is None:
        inflow = lambda s: (u0, 0.0)

    px = np.zeros(max_particles, np.float32)
    py = np.zeros(max_particles, np.float32)
    alive = np.zeros(max_particles, np.bool_)
    head = 0

    vort_mm = np.lib.format.open_memmap(CACHE / f"{name}_vort.npy", "w+", np.float16, (frames, ny, nx))
    spd_mm = np.lib.format.open_memmap(CACHE / f"{name}_speed.npy", "w+", np.float16, (frames, ny, nx))
    if seed_ys is not None:
        par_mm = np.lib.format.open_memmap(CACHE / f"{name}_parts.npy", "w+", np.float16,
                                           (frames, store_particles, 2))
    total = spinup + steps_per_frame * frames
    seed_start = max(0, spinup - int(1.1 * nx / max(u0, 1e-3))) if seed_from is None else seed_from
    t0 = time.time()
    fi = 0
    s = 0
    while fi < frames:
        uin, vin = inflow(s)
        lbm_step(f, fn, solid, tau0, smag, uin, vin, ux, uy)
        f, fn = fn, f
        s += 1
        rec = s > spinup
        if seed_ys is not None and s > seed_start:
            if s % seed_every == 0:
                k = len(seed_ys)
                idx = (head + np.arange(k)) % max_particles
                px[idx] = 1.0 + np.random.rand(k).astype(np.float32) * 0.5
                py[idx] = seed_ys
                alive[idx] = True
                head = (head + k) % max_particles
            if s % 2 == 0:
                advect(px, py, alive, ux, uy, solid, 2.0)
        if rec and (s - spinup) % steps_per_frame == 0:
            if not np.isfinite(ux).all():
                raise RuntimeError(f"{name}: diverged at step {s}")
            vort_mm[fi] = vorticity(ux, uy)
            spd_mm[fi] = np.sqrt(ux * ux + uy * uy)
            if seed_ys is not None:
                a = np.flatnonzero(alive)
                if len(a) > store_particles:
                    a = a[np.linspace(0, len(a) - 1, store_particles).astype(np.int64)]
                buf = np.full((store_particles, 2), np.nan, np.float16)
                buf[:len(a), 0] = px[a]
                buf[:len(a), 1] = py[a]
                par_mm[fi] = buf
            fi += 1
        if s % (log_every * steps_per_frame) == 0:
            el = time.time() - t0
            print(f"[{name}] step {s}/{total}  {s * nx * ny / el / 1e6:.0f} MLUPS  "
                  f"umax={float(np.abs(ux).max()):.3f}", flush=True)
    vort_mm.flush()
    spd_mm.flush()
    print(f"[{name}] done in {time.time() - t0:.0f}s")
