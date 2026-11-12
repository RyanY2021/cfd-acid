"""2D incompressible Navier-Stokes, pseudo-spectral vorticity form (periodic box).

w_t + u.grad(w) = nu lap(w);  c_t + u.grad(c) = kappa lap(c)
Integrating-factor SSP-RK3, 2/3 dealiasing, float64 FFTs via scipy (multi-threaded).
"""
import pathlib
import time

import numpy as np
import scipy.fft as sfft

CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache"
WORKERS = -1


class Spectral2D:
    def __init__(self, nx, ny, lx, ly, nu, kappa=0.0):
        self.nx, self.ny, self.lx, self.ly = nx, ny, lx, ly
        kx = 2 * np.pi / lx * sfft.rfftfreq(nx, 1.0 / nx)
        ky = 2 * np.pi / ly * sfft.fftfreq(ny, 1.0 / ny)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2i = np.where(self.K2 == 0, 0, 1.0 / np.where(self.K2 == 0, 1, self.K2))
        kxmax, kymax = kx.max(), np.abs(ky).max()
        self.dealias = (np.abs(self.KX) < 2 / 3 * kxmax) & (np.abs(self.KY) < 2 / 3 * kymax)
        self.nu, self.kappa = nu, kappa
        self.dx = lx / nx

    def fft(self, a):
        return sfft.rfft2(a, workers=WORKERS)

    def ifft(self, a):
        return sfft.irfft2(a, s=(self.ny, self.nx), workers=WORKERS)

    def velocity(self, wh):
        psih = wh * self.K2i
        u = self.ifft(1j * self.KY * psih)
        v = self.ifft(-1j * self.KX * psih)
        return u, v

    def rhs(self, wh, ch):
        u, v = self.velocity(wh)
        wx = self.ifft(1j * self.KX * wh)
        wy = self.ifft(1j * self.KY * wh)
        nw = -self.fft(u * wx + v * wy) * self.dealias
        nc = None
        if ch is not None:
            cx = self.ifft(1j * self.KX * ch)
            cy = self.ifft(1j * self.KY * ch)
            nc = -self.fft(u * cx + v * cy) * self.dealias
        return nw, nc, max(np.abs(u).max(), np.abs(v).max())

    def step(self, wh, ch, dt):
        """Lawson integrating-factor SSP-RK3."""
        Ew, Ew2, Ewm = (np.exp(-self.nu * self.K2 * s) for s in (dt, dt / 2, -dt / 2))
        Ec, Ec2, Ecm = (np.exp(-self.kappa * self.K2 * s) for s in (dt, dt / 2, -dt / 2))
        n1w, n1c, umax = self.rhs(wh, ch)
        w1 = Ew * (wh + dt * n1w)
        c1 = Ec * (ch + dt * n1c) if ch is not None else None
        n2w, n2c, _ = self.rhs(w1, c1)
        w2 = 0.75 * Ew2 * wh + 0.25 * Ewm * (w1 + dt * n2w)
        c2 = 0.75 * Ec2 * ch + 0.25 * Ecm * (c1 + dt * n2c) if ch is not None else None
        n3w, n3c, _ = self.rhs(w2, c2)
        wn = (1 / 3) * Ew * wh + (2 / 3) * Ew2 * (w2 + dt * n3w)
        cn = (1 / 3) * Ec * ch + (2 / 3) * Ec2 * (c2 + dt * n3c) if ch is not None else None
        return wn, cn, umax


def run(name, w0, c0, lx, ly, nu, kappa, frames, t_per_frame, cfl=0.5, tmax_dt=None, log=10):
    ny, nx = w0.shape
    S = Spectral2D(nx, ny, lx, ly, nu, kappa)
    wh = S.fft(w0) * S.dealias
    ch = S.fft(c0) * S.dealias if c0 is not None else None
    w_mm = np.lib.format.open_memmap(CACHE / f"{name}_w.npy", "w+", np.float16, (frames, ny, nx))
    c_mm = None
    if c0 is not None:
        c_mm = np.lib.format.open_memmap(CACHE / f"{name}_c.npy", "w+", np.float16, (frames, ny, nx))
    t0 = time.time()
    umax = 1.0
    for fi in range(frames):
        w_mm[fi] = S.ifft(wh).astype(np.float16)
        if c_mm is not None:
            c_mm[fi] = S.ifft(ch).astype(np.float16)
        tt = 0.0
        while tt < t_per_frame - 1e-12:
            dt = min(cfl * S.dx / max(umax, 1e-6), t_per_frame - tt)
            if tmax_dt:
                dt = min(dt, tmax_dt)
            wh, ch, umax = S.step(wh, ch, dt)
            tt += dt
        if fi % log == 0:
            print(f"[{name}] frame {fi}/{frames} umax={umax:.3f} {time.time() - t0:.0f}s", flush=True)
    w_mm.flush()
    if c_mm is not None:
        c_mm.flush()
    print(f"[{name}] done {time.time() - t0:.0f}s")
