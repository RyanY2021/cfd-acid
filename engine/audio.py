"""Procedural soundtrack, sequenced on the film's 120 BPM grid.  python -m engine.audio"""
import math
import pathlib
import wave

import numba as nb
import numpy as np
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parent.parent
SR = 48000
DUR = 104.0
BEAT = 0.5
S16 = BEAT / 4
N = int(SR * (DUR + 3))
rng = np.random.default_rng(2026)


def hz(semi, base=82.41):        # semitones from E2
    return base * 2 ** (semi / 12)


def tt(d):
    return np.arange(int(d * SR)) / SR


def noise(d):
    return rng.uniform(-1, 1, int(d * SR))


def hp(x, f, o=2):
    return signal.sosfilt(signal.butter(o, f, "hp", fs=SR, output="sos"), x)


def lp(x, f, o=2):
    return signal.sosfilt(signal.butter(o, f, "lp", fs=SR, output="sos"), x)


def bp(x, f0, f1, o=2):
    return signal.sosfilt(signal.butter(o, [f0, f1], "bp", fs=SR, output="sos"), x)


@nb.njit(cache=True)
def svf(x, cutoff, res):
    """Resonant state-variable low-pass; cutoff array (Hz) per sample."""
    y = np.empty_like(x)
    low = 0.0
    band = 0.0
    q = 1.0 - res
    for i in range(x.size):
        f = 2.0 * math.sin(math.pi * min(cutoff[i], 12000.0) / 48000.0)
        high = x[i] - low - q * band
        band += f * high
        low += f * band
        y[i] = low
    return y


@nb.njit(cache=True)
def saw_blep(freq):
    out = np.empty(freq.size)
    ph = 0.0
    for i in range(freq.size):
        dt = freq[i] / 48000.0
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
        v = 2.0 * ph - 1.0
        if ph < dt:
            t = ph / dt
            v -= t + t - t * t - 1.0
        elif ph > 1.0 - dt:
            t = (ph - 1.0) / dt
            v -= t * t + t + t + 1.0
        out[i] = v
    return out


# ------------------------------------------------------------------ instruments
def kick(d=0.45, f0=150, f1=46, drive=1.6):
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t / 0.16)
    x[:200] += hp(noise(200 / SR), 3000) * np.linspace(0.6, 0, 200)
    return np.tanh(drive * x)


def clap(d=0.3):
    t = tt(d)
    env = np.exp(-t / 0.09)
    for k in (0.0, 0.011, 0.022):
        env += 0.7 * ((t >= k) & (t < k + 0.008))
    return bp(noise(d), 900, 3200) * env * 0.9


def hat(open_=False):
    d = 0.3 if open_ else 0.06
    t = tt(d)
    return hp(noise(d), 7500, 4) * np.exp(-t / (0.11 if open_ else 0.016))


def sub_note(f, d):
    t = tt(d)
    env = np.minimum(1, t / 0.01) * np.minimum(1, (d - t) / 0.03)
    return (np.sin(2 * np.pi * f * t) + 0.15 * np.sin(4 * np.pi * f * t)) * env


def stab(freqs, d=0.35, cut=2400):
    t = tt(d)
    x = sum(saw_blep(np.full(t.size, f * dt)) for f in freqs for dt in (0.995, 1.005))
    x = lp(x, cut, 2) * np.exp(-t / 0.12)
    return x / (2 * len(freqs))


def pad(freqs, d, cut=900, att=1.0):
    t = tt(d)
    x = sum(saw_blep(np.full(t.size, f * (1 + dtn))) for f in freqs for dtn in (-0.007, 0, 0.006))
    env = np.minimum(1, t / att) * np.minimum(1, (d - t) / 1.2)
    return lp(x, cut, 2) * env / (3 * len(freqs))


def blip(f, d=0.05, fm=0.0):
    t = tt(d)
    m = np.sin(2 * np.pi * f * 2.01 * t) * fm
    return np.sin(2 * np.pi * f * t + m) * np.exp(-t / (d / 4))


def impact(d=2.6):
    t = tt(d)
    f = 30 + 70 * np.exp(-t / 0.09)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.7)
    crash = lp(noise(d), 5000) * np.exp(-t / 0.45) * 0.5
    return np.tanh(1.8 * (boom + crash))


def riser(d, f0=300, f1=7000):
    t = tt(d)
    cut = f0 * (f1 / f0) ** (t / d)
    x = svf(noise(d), cut, 0.75) * (t / d) ** 2
    tone = np.sin(2 * np.pi * np.cumsum(80 * 2 ** (4 * t / d)) / SR) * (t / d) ** 3 * 0.4
    return x * 0.8 + tone


def whoosh(d, f0=200, f1=4000):
    t = tt(d)
    cut = f0 * (f1 / f0) ** np.sin(np.pi * t / d)
    return svf(noise(d), cut, 0.6) * np.sin(np.pi * t / d) ** 2


def alarm(d):
    t = tt(d)
    f = np.where((t * 4) % 1 < 0.5, 880, 660)
    x = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))
    return lp(x, 3000) * 0.25


def acid(steps, t0, cut_fn, res=0.9, step_d=S16, base=82.41):
    """steps: list of (semi or None, accent, slide)."""
    n = int(len(steps) * step_d * SR) + SR // 4
    freq = np.zeros(n)
    amp = np.zeros(n)
    fenv = np.zeros(n)
    last = None
    for k, (semi, acc, slide) in enumerate(steps):
        a = int(k * step_d * SR)
        b = int((k + 1) * step_d * SR)
        if semi is None:
            if last is not None:
                freq[a:b] = last
            continue
        f = base * 2 ** (semi / 12)
        seg = np.arange(b - a) / SR
        if slide and last is not None:
            g = np.minimum(1, seg / 0.06)
            freq[a:b] = last * (1 - g) + f * g
        else:
            freq[a:b] = f
        gate = 0.85 if not slide else 1.0
        e_len = int((b - a) * gate)
        amp[a:a + e_len] = 1.0 if acc else 0.7
        amp[a + e_len:b] = 0
        fenv[a:b] = np.exp(-seg / (0.09 if acc else 0.05)) * (1.6 if acc else 0.9)
        last = f
    freq[freq == 0] = base
    tt_abs = t0 + np.arange(n) / SR
    cutoff = cut_fn(tt_abs) * (1 + 2.6 * fenv)
    x = svf(saw_blep(freq), cutoff, res)
    amp = signal.lfilter([1 - 0.995], [1, -0.995], amp) * 1.0
    return np.tanh(2.2 * x * amp) * 0.55


def pattern(seq, acc="", sld=""):
    """seq: space-separated tokens, semitone ints or '.' for rest; acc/sld: step indices strings."""
    out = []
    for i, tok in enumerate(seq.split()):
        out.append((None if tok == "." else int(tok), str(i % 16) in acc.split(), str(i % 16) in sld.split()))
    return out


# ------------------------------------------------------------------ mixing helpers
class Bus:
    def __init__(self):
        self.buf = np.zeros((N, 2))

    def add(self, x, t, g=1.0, pan=0.0):
        a = int(t * SR)
        if a >= N:
            return
        x = x[: N - a]
        l = math.cos((pan + 1) * math.pi / 4) * math.sqrt(2)
        r = math.sin((pan + 1) * math.pi / 4) * math.sqrt(2)
        self.buf[a:a + len(x), 0] += x * g * l
        self.buf[a:a + len(x), 1] += x * g * r


def reverb(x, decay=2.4, mix=1.0):
    n = int(decay * SR)
    t = np.arange(n) / SR
    out = np.zeros((x.shape[0] + n - 1, 2))
    for ch in range(2):
        ir = rng.normal(0, 1, n) * np.exp(-t / (decay / 6.9) * 1.0)
        ir = lp(ir, 6000)
        ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
        out[:, ch] = signal.fftconvolve(x[:, ch], ir / np.sqrt((ir ** 2).sum()) * 0.6)
    return out[: x.shape[0]] * mix


def load_vo(i):
    with wave.open(str(ROOT / "cache" / "vo" / f"vo{i:02d}.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    loud = np.flatnonzero(np.abs(x) > 0.01)
    x = x[max(0, loud[0] - 600): loud[-1] + 2400]                 # trim silence
    # ship-AI treatment: slight pitch drop, band-limit, crush, octave shadow, chorus
    x = signal.resample(x, int(len(x) * 1.06))
    x = bp(x, 180, 6500)
    crushed = np.round(x * 512) / 512
    hold = np.repeat(crushed[::3], 3)[: len(x)]
    y = 0.75 * x + 0.35 * hold
    d1, d2 = int(0.011 * SR), int(0.017 * SR)
    ch = np.zeros_like(y)
    ch[d1:] += 0.35 * y[:-d1]
    ch[d2:] += 0.25 * y[:-d2]
    y = y + ch
    t = np.arange(len(y)) / SR
    y *= 1 + 0.12 * np.sin(2 * np.pi * 70 * t)            # faint ring-mod shimmer
    return np.tanh(1.4 * y) * 0.8


# ------------------------------------------------------------------ arrangement
E, F, G, A, B, D = 0, 1, 3, 5, 7, 10
EM = [hz(-12 + E), hz(-12 + G), hz(-12 + B), hz(E)]


def build():
    drums, bass, music, fxb, vox, send = Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
    kick_times = []

    def K(t, g=1.0, drive=1.6):
        drums.add(kick(drive=drive), t, g)
        kick_times.append(t)

    def hats(t0, t1, every=S16 * 2, g=0.22, open_on=None):
        k = 0
        t = t0
        while t < t1 - 1e-6:
            o = open_on is not None and (k % open_on == open_on - 1)
            drums.add(hat(o), t, g * (1.4 if o else (1.0 if k % 2 == 0 else 0.7)), pan=0.25 if k % 2 else -0.25)
            t += every
            k += 1

    # ---------------- A BOOT 0-8
    music.add(pad([hz(-12), hz(-5), hz(0)], 8.6, cut=450, att=3.0), 0, 0.45)
    for i in range(48):
        t = 0.15 + i * 0.11 + rng.uniform(0, 0.05)
        if t < 5.8:
            fxb.add(blip(rng.choice([1200, 1600, 2400, 3200]), 0.035), t, 0.10, pan=rng.uniform(-0.8, 0.8))
    for b in range(4, 12):
        drums.add(kick(0.3, 90, 40, 1.2), b * BEAT, 0.35 + 0.05 * b)
    fxb.add(riser(2.0, 200, 5000), 4.0, 0.35)
    for t in (6.0, 6.5):
        K(t, 0.9)
        fxb.add(impact(0.6), t, 0.4)
    fxb.add(hp(noise(0.6), 2000) * np.exp(-tt(0.6) / 0.15), 7.0, 0.3)

    # ---------------- B TITLE 8-16
    fxb.add(impact(), 8.0, 1.0)
    send.add(impact(), 8.0, 0.4)
    K(8.0, 1.0, 2.2)
    for t in np.arange(9.0, 15.5, 1.0):
        K(t, 0.85)
    hats(10.0, 15.5, S16 * 2, 0.18)
    bass.add(sub_note(hz(-12), 7.4), 8.0, 0.55)
    music.add(pad(EM, 8.0, cut=1200, att=0.6), 8.0, 0.5)
    send.add(pad(EM, 8.0, cut=1200, att=0.6), 8.0, 0.25)
    fxb.add(whoosh(6.5, 150, 2500), 9.0, 0.22, pan=-0.3)
    fxb.add(riser(0.8, 800, 9000), 15.2, 0.4)

    # ---------------- C FLOWS 16-24
    chords = [[E, G, B], [F, A, D], [G, B, D + 2], [E, G, B], [D, F + 2, A + 2], [F, A, D], [G, B, D + 2], [E, G, B + 12]]
    for k in range(8):
        t = 16 + k * BEAT
        K(t, 1.0)
        music.add(stab([hz(s) for s in chords[k]], 0.4, 2600), t, 0.5)
        send.add(stab([hz(s) for s in chords[k]], 0.4, 2600), t, 0.2)
        fxb.add(hp(noise(0.12), 3000) * np.exp(-tt(0.12) / 0.03), t, 0.25)
        bass.add(sub_note(hz(chords[k][0] - 12), 0.45), t, 0.5)
    for i in range(380):
        t = 20.0 + rng.uniform(0, 3.9)
        fxb.add(blip(rng.uniform(3000, 9000), 0.008), t, 0.05 * (1 - (t - 20) / 4) + 0.02, pan=rng.uniform(-1, 1))
    music.add(pad([hz(-12), hz(-5), hz(3)], 4.5, cut=1600, att=1.2), 20.0, 0.5)
    fxb.add(riser(2.4, 300, 8000), 21.6, 0.45)

    # ---------------- D LAW 24-36
    for k, t in enumerate([24.0, 24.5, 25.0, 25.5, 26.0, 26.5]):
        K(t, 1.0, 2.4)
        hit = bp(noise(0.5), 400 + 300 * k, 1800 + 300 * k) * np.exp(-tt(0.5) / 0.12)
        fxb.add(hit, t, 0.35)
        bass.add(sub_note(hz(-12 + [0, 0, 3, 1, 0, -2][k]), 0.48), t, 0.6)
    fxb.add(impact(1.5), 27.0, 0.6)
    for t in np.arange(27.0, 30.0, BEAT):
        K(t, 0.9)
    hats(27.0, 30.0, S16, 0.15)
    for t in np.arange(27.5, 30.0, 1.0):
        drums.add(clap(), t, 0.5)
    for t in np.arange(27.0, 30.0, S16 * 2):
        bass.add(sub_note(hz(-12), S16 * 1.8), t, 0.45)
    # unsolved
    fxb.add(impact(), 30.0, 0.9)
    for t in np.arange(30.0, 34.0, BEAT):
        K(t, 1.0, 3.0)
    hats(30.0, 34.0, S16, 0.16, open_on=4)
    fxb.add(alarm(2.4), 30.0, 0.35)
    for t in np.arange(30.0, 34.0, S16 * 2):
        bass.add(np.tanh(3 * sub_note(hz(-12 + 1), S16 * 1.7)), t, 0.4)
    music.add(stab([hz(E + 12), hz(G + 12), hz(B + 12)], 0.8, 4000), 32.5, 0.5)
    send.add(stab([hz(E + 12), hz(G + 12), hz(B + 12)], 0.8, 4000), 32.5, 0.4)
    for i in range(14):
        fxb.add(blip(2400, 0.02), 34.05 + i * 0.033, 0.08)
    fxb.add(impact(1.8), 35.0, 0.8)
    K(35.0, 1.0, 2.6)
    fxb.add(riser(0.95, 500, 9000), 35.05, 0.3)

    # ---------------- E MESH 36-44 / F SOLVE 44-52 : arpeggio build
    arp_notes = [E, G, B, D + 2, E + 12, D + 2, B, G]
    cut_arp = lambda t: 600 + 5000 * ((t - 36) / 16) ** 2
    t = 36.0
    k = 0
    while t < 51.875 - 1e-6:
        f = hz(arp_notes[k % 8] + (12 if t > 44 and (k // 8) % 2 else 0))
        d = S16 * 0.9
        x = np.sign(np.sin(2 * np.pi * f * tt(d))) * np.exp(-tt(d) / 0.06)
        music.add(lp(x, cut_arp(t)), t, 0.16, pan=0.4 * math.sin(k * 0.7))
        send.add(lp(x, cut_arp(t)), t, 0.08)
        t += S16
        k += 1
    for t in np.arange(36.0, 51.9, BEAT):
        K(t, 0.8 if t < 44 else 0.9)
        bass.add(sub_note(hz(-12), 0.42), t, 0.45)
    hats(38.0, 51.875, S16 * 2, 0.16)
    for t in (36.0, 44.0, 50.0):
        fxb.add(impact(1.6), t, 0.6)
    # snare roll build 48-52
    t = 48.0
    while t < 51.875:
        rate = BEAT / 2 if t < 49 else BEAT / 4 if t < 50.5 else BEAT / 8
        drums.add(clap(0.15), t, 0.25 + 0.35 * (t - 48) / 4)
        t += rate
    fxb.add(riser(3.8, 200, 10000), 48.0, 0.6)
    music.add(pad([hz(-12), hz(-5), hz(-12 + B), hz(3)], 8.0, cut=700, att=4), 44.0, 0.4)

    # ---------------- G DROP 52-76 and H APPS 76-88
    fxb.add(impact(3.0), 52.0, 1.1)
    send.add(impact(3.0), 52.0, 0.5)
    pa = pattern("0 0 12 0 3 0 1 0 0 10 0 7 0 3 5 0", acc="2 6 9 14", sld="7 13")
    pb = pattern("0 . 0 12 0 . 15 12 0 3 . 0 10 7 3 1", acc="3 6 10 15", sld="5 7 13")
    pc = pattern("0 0 0 12 0 0 7 0 0 3 0 12 15 12 10 7", acc="0 3 11 12", sld="12 13 14")

    def cut_drop(t):
        c = np.where(t < 62, 350 + 2000 * ((t - 52) / 10) ** 1.5,
                     np.where(t < 66, 800 + 2200 * (0.5 + 0.5 * np.sin(2 * np.pi * (t - 62) / 4 * 1.5)),
                              np.where(t < 76, 1800 + 1600 * np.sin(2 * np.pi * t / 7) ** 2,
                                       1200 + 1800 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 5)))))
        return c

    def drop_bar(t0, pat, beat=True, clapp=True, hat16=True, acid_on=True, res=0.9, beats=4):
        pat = pat[:beats * 4]
        for b in range(beats):
            if beat:
                K(t0 + b * BEAT, 1.0, 2.0)
            if clapp and b % 2 == 1:
                drums.add(clap(), t0 + b * BEAT, 0.55)
                send.add(clap(), t0 + b * BEAT, 0.15)
        if hat16:
            hats(t0, t0 + beats * BEAT, S16, 0.13, open_on=4)
        if acid_on:
            a_ = acid(pat, t0, cut_drop, res=res)
            music.add(a_, t0, 0.62)
            send.add(a_, t0, 0.12)
        bass.add(sub_note(hz(-12), beats * BEAT - 0.05), t0, 0.35)

    bar = 52.0
    while bar < 66.0 - 1e-6:
        p = [pa, pa, pb, pa, pa, pc, pb][int((bar - 52) / 2) % 7]
        drop_bar(bar, p)
        bar += 2.0
    for t in (56.0, 62.0):
        fxb.add(impact(1.4), t, 0.55)
    # 66-71 KH: half-time, plucks, no acid
    fxb.add(impact(1.2), 66.0, 0.6)
    for t in np.arange(66.0, 71.0, 1.0):
        K(t, 0.9)
        drums.add(clap(), t + 0.5, 0.45)
    hats(66.0, 71.0, S16 * 2, 0.12)
    plucks = [E + 12, B, G, E + 12, D + 12, B, A, G]
    for k, t in enumerate(np.arange(66.0, 71.0, BEAT / 2)):
        f = hz(plucks[k % 8])
        x = saw_blep(np.full(int(0.25 * SR), f)) * np.exp(-tt(0.25) / 0.05)
        music.add(lp(x, 3500), t, 0.22, pan=0.5 if k % 2 else -0.5)
        send.add(lp(x, 3500), t, 0.12)
    bass.add(sub_note(hz(-12), 5.0), 66.0, 0.4)
    # 71-76 turbulence: full + resonance
    fxb.add(impact(1.6), 71.0, 0.8)
    bar = 71.0
    while bar < 76.0 - 1e-6:
        drop_bar(bar, [pc, pb, pa][int((bar - 71) / 2) % 3], res=0.94)
        bar += 2.0
    # apps 76-91: header, aero, auto | hand breakdown 82.5-85.5 | cardio, climate
    fxb.add(impact(2.0), 76.0, 1.0)
    send.add(impact(2.0), 76.0, 0.4)
    for k, bar in enumerate((76.0, 78.0, 80.0)):
        drop_bar(bar, [pb, pa, pc][k], res=0.9)
    drop_bar(82.0, pa, res=0.9, beats=1)
    for t in (77.0, 79.75, 82.5, 85.5, 88.25):
        fxb.add(impact(1.2), t, 0.55)
        fxb.add(hp(noise(0.8), 4000) * np.exp(-tt(0.8) / 0.25), t, 0.3)
    # the still hand: music drops out, a cold drone, then the heart starts on the visual beats
    music.add(pad([hz(-12), hz(-5), hz(-1)], 3.2, cut=600, att=0.4), 82.5, 0.35)
    for i in range(8):
        fxb.add(blip(rng.choice([2400, 3200]), 0.03), 82.6 + i * 0.11, 0.05, pan=rng.uniform(-0.6, 0.6))
    for k, t in enumerate((83.5, 84.0, 84.5, 85.0)):
        g = 0.55 + 0.15 * k
        drums.add(kick(0.35, 70, 38, 1.4), t, g)                       # lub
        drums.add(kick(0.3, 60, 34, 1.2), t + 0.13, g * 0.7)          # dub
        kick_times.append(t)
        sw = whoosh(0.45, 150, 1800 + 600 * k)
        fxb.add(sw, t, 0.25 + 0.05 * k, pan=-0.3 + 0.2 * k)          # blood surging
        bass.add(sub_note(hz(-12), 0.4), t, 0.3 + 0.08 * k)
    fxb.add(impact(1.0), 84.0, 0.5)                                    # it comes alive
    music.add(acid(pattern("0 . 0 . 0 . 12 . 0 . 0 . 3 . 1 ."), 84.0, lambda tt_: 300 + 1500 * np.clip((tt_ - 84) / 1.5, 0, 1),
                   res=0.92), 84.0, 0.45)
    fxb.add(riser(0.9, 600, 9000), 84.6, 0.45)
    for k, bar in enumerate((85.5, 87.5)):
        drop_bar(bar, [pc, pb][k], res=0.92)
    drop_bar(89.5, pa, res=0.9, beats=3)
    fxb.add(riser(1.4, 400, 9000), 89.6, 0.5)

    # ---------------- I OUTRO 91-104
    for t in (91.0, 91.5):
        K(t, 1.0, 2.6)
        fxb.add(impact(1.2), t, 0.6)
        music.add(stab([hz(E), hz(G), hz(B), hz(E + 12)], 0.6, 3000), t, 0.45)
    # third hit never comes: a broken digital burst instead
    junkb = np.round(hp(noise(0.5), 1200) * np.exp(-tt(0.5) / 0.2) * 6) / 6
    fxb.add(junkb, 92.0, 0.45)
    for k in range(10):
        fxb.add(blip(rng.choice([900, 1800, 3600, 7200]), 0.02, fm=3.0), 92.0 + k * 0.09, 0.08,
                pan=rng.uniform(-0.8, 0.8))
    fxb.add(whoosh(1.1, 300, 7000), 92.95, 0.5)
    # logo card -> CRT: impact, decaying pad with tape wow, the tube's hum and line whine
    fxb.add(impact(2.4), 94.0, 0.8)
    send.add(impact(2.4), 94.0, 0.4)
    d = 4.9
    tp = tt(d)
    wow = 1 + 0.012 * np.sin(2 * np.pi * 0.7 * tp) * np.clip(tp / 2, 0, 1) - 0.06 * np.clip((tp - 3.0) / 1.9, 0, 1) ** 2
    padv = sum(saw_blep(f0 * wow) for f0 in (hz(-12), hz(-5), hz(-12 + B), hz(3)))
    padv = lp(padv, 1300) * np.minimum(1, tp / 0.05) * np.exp(-np.clip(tp - 3.6, 0, None) / 0.4) / 4
    music.add(padv, 94.0, 0.5)
    send.add(padv, 94.0, 0.3)
    hd = 4.85
    th = tt(hd)
    hum = (np.sin(2 * np.pi * 60 * th) + 0.5 * np.sin(2 * np.pi * 120 * th) + 0.25 * np.sign(np.sin(2 * np.pi * 180 * th)))
    hum = lp(hum, 900) * 0.10 + np.sin(2 * np.pi * 15734 * th) * 0.012
    hum *= np.clip(th / 1.0, 0, 1)
    fxb.add(hum, 94.0, 1.0)
    # crash build: tearing, crushed stutters, a rising whine (computer frames 72-100 -> 97.9-98.83)
    for k in range(14):
        t = 97.9 + k * 0.066
        b = np.round(hp(noise(0.05), 2000) * 5) / 5 * np.exp(-tt(0.05) / 0.02)
        fxb.add(b, t, 0.12 + 0.02 * k, pan=rng.uniform(-0.8, 0.8))
    wd = 0.95
    tw_ = tt(wd)
    whine = np.sin(2 * np.pi * np.cumsum(1500 + 5000 * (tw_ / wd) ** 2) / SR) * (tw_ / wd) ** 2 * 0.15
    fxb.add(whine, 97.88, 1.0)
    # the spark: zap + crackle, tube power-down thunk and falling whine
    z = 1.4
    tz = tt(z)
    gate = (rng.random(len(tz)) < np.exp(-tz / 0.25) * 0.08).astype(float)
    gate = signal.lfilter([1.0], [1, -0.995], gate)
    zap = hp(noise(z), 2500) * np.clip(gate, 0, 1) * 1.2 + hp(noise(z), 800) * np.exp(-tz / 0.05) * 1.5
    fxb.add(np.tanh(zap * 2) * 0.8, 98.83, 1.0)
    send.add(np.tanh(zap * 2) * 0.5, 98.83, 0.6)
    fxb.add(impact(1.6), 98.86, 0.7)
    fdn = 0.9
    tdn = tt(fdn)
    down = np.sin(2 * np.pi * np.cumsum(15734 * np.exp(-tdn / 0.12) + 60) / SR) * np.exp(-tdn / 0.3) * 0.08
    fxb.add(down, 98.9, 1.0)
    for t in (98.83 + 23 / 30, 98.83 + 41 / 30, 98.83 + 58 / 30):            # stray sparks
        c = hp(noise(0.25), 3000) * (rng.random(int(0.25 * SR)) < 0.03) * 1.5
        fxb.add(signal.lfilter([1.0], [1, -0.99], c) * 0.3, t, 0.6, pan=rng.uniform(-0.5, 0.5))
    # smoke: soft hiss, ember crackle, room tone; silence otherwise
    sd = 5.2
    ts_ = tt(sd)
    hiss = lp(noise(sd), 1400) * np.clip((ts_ - 0.1) / 1.5, 0, 1) * np.exp(-np.clip(ts_ - 3.5, 0, None) / 1.0) * 0.18
    crack = (rng.random(len(ts_)) < 0.0009 * np.exp(-ts_ / 3.0)).astype(float)
    crack = hp(signal.lfilter([1.0], [1, -0.97], crack), 1500) * 0.5
    fxb.add(hiss + crack, 98.95, 1.0, pan=0.1)
    fxb.add(lp(noise(5.0), 200) * 0.06, 98.95, 1.0)
    # SIMULATION EXHAUSTED
    fxb.add(blip(1800, 0.06), 101.1, 0.1)
    dr = tt(2.6)
    fxb.add(np.sin(2 * np.pi * 36 * dr) * np.exp(-dr / 1.2) * 0.35, 101.1, 1.0)

    # ---------------- VO
    vo_times = [(1, 1.2), (2, 9.4), (3, 20.4), (4, 26.6), (5, 30.7), (6, 35.05), (7, 37.0), (8, 50.15), (9, 62.05),
                (10, 76.3), (11, 90.78)]          # the end card is not read aloud
    for i, t in vo_times:
        v = load_vo(i)
        vox.add(v, t, 0.95)
        send.add(v, t, 0.35)
        fxb.add(blip(2600, 0.03), t - 0.08, 0.07)
    # the voice gets stuck on "the": a jammed buffer, sliding down and crushing (visual echoes at +0.09 s)
    v11 = load_vo(11)
    a = int(0.72 * SR)
    grain = v11[a:a + int(0.1 * SR)] * np.hanning(int(0.1 * SR)) ** 0.3
    for k in range(10):
        g = signal.resample(grain, int(len(grain) * (1 + 0.035 * k)))
        lv = 2 + 10 // (k + 1)
        g = np.round(g * lv * 4) / (lv * 4)
        vox.add(g, 92.0 + k * 0.09, 0.95 - 0.04 * k)
        send.add(g, 92.0 + k * 0.09, 0.25)

    # ---------------- sidechain & master
    duck = np.ones(N)
    for t in kick_times:
        a = int(t * SR)
        n = int(0.32 * SR)
        e = 1 - 0.65 * np.exp(-np.arange(n) / SR / 0.09)
        b = min(N, a + n)
        duck[a:b] = np.minimum(duck[a:b], e[: b - a])
    bass.buf *= duck[:, None]
    music.buf *= duck[:, None] ** 0.7
    voxduck = np.ones(N)
    for i, t in vo_times:
        a = int((t - 0.1) * SR)
        b = min(N, a + len(load_vo(i)) + int(0.2 * SR))
        voxduck[a:b] = 0.55
    voxduck = signal.lfilter([0.002], [1, -0.998], voxduck) / 1.0
    # section dynamics arc (dB), applied to everything but the voice
    pts = [(0, 0), (95.4, 0), (95.6, -3), (98.8, -2), (98.85, 0), (99.8, -3), (DUR, -5)]
    tx = np.arange(N) / SR
    arc = 10 ** (np.interp(tx, [a for a, _ in pts], [b for _, b in pts]) / 20)
    rv = reverb(send.buf, 2.8)
    bed = drums.buf * 0.9 + lp(bass.buf.T, 180).T * 1.0 + music.buf * voxduck[:, None] + fxb.buf * 0.8 + rv * 0.5
    mix = bed * arc[:, None] + vox.buf * 1.0
    mix = hp(mix.T, 25).T
    glitch_stutter(mix)
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    mix /= np.abs(mix[:int(76 * SR)]).max() / 0.93         # level set by the score; later hits are limited
    mix = 0.985 * np.tanh(mix / 0.985)
    fade = np.ones(N)
    end = int(DUR * SR)
    fade[end - int(0.4 * SR):end] = np.linspace(1, 0, int(0.4 * SR))
    fade[end:] = 0
    mix *= fade[:, None]
    return mix[:end]


def glitch_stutter(mix):
    """Buffer-repeat stutters at visual glitch moments (in place)."""
    pts = [7.0, 7.3, 15.6, 15.8, 23.85, 29.9, 35.85, 43.8, 51.5, 51.7, 55.9, 61.85, 65.9, 70.6, 75.85, 82.45,
           85.45, 88.2, 90.85]
    for t in pts:
        a = int(t * SR)
        grain = int(rng.choice([0.018, 0.025, 0.04]) * SR)
        reps = int(0.16 * SR) // grain
        g = mix[a:a + grain].copy()
        g *= np.hanning(grain)[:, None] ** 0.2
        for r in range(reps):
            s = a + r * grain
            mix[s:s + grain] = g * (1 - r / (reps * 1.5))
        crush = mix[a:a + reps * grain]
        mix[a:a + reps * grain] = np.round(crush * 16) / 16


INTRO = 6.0          # seconds of ocean intro before the main score (see scenes/s_intro.py)


def build_intro():
    """Surf ambience under the aerial intro; tail overlaps the start of the boot."""
    d = INTRO + 2.5
    n = int(d * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    # breaking waves: one crash per 4.2 s front period, staggered L/R
    crashes = [0.6, 2.7, 4.8, 6.9]
    for ch in range(2):
        nz = rng.normal(0, 1, n)
        body = lp(nz, 700, 2) * 0.9 + bp(nz, 1200, 5500) * 0.35
        env = 0.28 + 0.1 * np.sin(2 * np.pi * t / 7.3 + ch)
        for c in crashes:
            c = c + 0.18 * ch
            rise = np.clip((t - (c - 0.5)) / 0.5, 0, 1) ** 2
            fall = np.exp(-np.clip(t - c, 0, None) / 1.3)
            env = env + 0.85 * rise * fall
        hiss = hp(rng.normal(0, 1, n), 4000) * 0.12 * env
        out[:, ch] = body * env + hiss
    # wind
    wind = svf(noise(d), 500 + 350 * np.sin(2 * np.pi * t / 3.1) ** 2, 0.55)
    out += np.stack([wind, wind[::-1]], 1) * 0.25
    # low drone, fades in
    dr = (np.sin(2 * np.pi * 41.2 * t) + 0.5 * np.sin(2 * np.pi * 61.7 * t) + 0.25 * np.sin(2 * np.pi * 82.4 * t))
    dr *= np.clip(t / 3.0, 0, 1) * 0.18
    out += dr[:, None]
    # slit-reveal whoosh + HUD blips as labels appear
    w = whoosh(0.7, 300, 6000)
    out[:len(w)] += w[:, None] * 0.35
    for tb, f in ((0.25, 2400), (0.5, 1800), (0.85, 2400), (0.95, 3200), (1.25, 1600), (1.65, 2800),
                  (2.25, 3600), (2.5, 3600), (2.75, 3600)):
        b = blip(f, 0.04)
        a = int(tb * SR)
        out[a:a + len(b)] += b[:, None] * 0.09
    # dissolve into the palette: riser, then a crushed stutter at the cut
    r = riser(1.25, 300, 9000)
    a = int(4.35 * SR)
    out[a:a + len(r)] += r[:, None] * 0.4
    vo = load_vo(0)
    a = int(1.0 * SR)
    out[a:a + len(vo)] += vo[:, None] * 0.95
    fade = np.ones(n)
    a, b = int(5.6 * SR), int(6.4 * SR)
    fade[a:b] = np.linspace(1, 0.18, b - a)
    fade[b:] = np.linspace(0.18, 0, n - b)
    out *= fade[:, None]
    g = out[int(5.6 * SR):int(5.6 * SR) + int(0.03 * SR)].copy()
    for k in range(5):
        s0 = int(5.6 * SR) + k * len(g)
        out[s0:s0 + len(g)] = np.round(g * (1 - k / 6) * 12) / 12
    imp = impact(1.2)
    a = int(5.6 * SR)
    out[a:a + len(imp)] += imp[:, None] * 0.45
    return out


def write_wav(x, path):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


if __name__ == "__main__":
    import time
    t0 = time.time()
    m = build()
    intro = build_intro()
    full = np.zeros((int((INTRO + DUR) * SR), 2))
    full[int(INTRO * SR):] += m
    k = min(len(intro), len(full))
    full[:k] += intro[:k]
    pk = np.abs(full).max()
    if pk > 0.97:
        full *= 0.97 / pk
    out = ROOT / "render" / "soundtrack.wav"
    out.parent.mkdir(exist_ok=True)
    write_wav(full, out)
    print(out, f"{time.time() - t0:.0f}s", full.shape)
