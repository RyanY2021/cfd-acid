"""Render driver.

  python -m engine.render still 9.5 30.2 ...        -> render/stills/*.png
  python -m engine.render sheet <scene> [n]          -> render/sheet_<scene>.jpg (contact sheet)
  python -m engine.render video [workers]            -> render/video_noaudio.mp4
  python -m engine.render mux                         -> out/CFD_ACID.mp4 (video + audio)
"""
import importlib
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

from engine.core import W, H, FPS, ROOT
from engine import fx

RENDER = ROOT / "render"
OUT = ROOT / "out"

# (module, start, end) in seconds — see STORYBOARD.md
_SHOTS = [
    ("s_intro", 6),
    ("s_boot", 8),
    ("s_title", 8),
    ("s_flows", 8),
    ("s_law", 12),
    ("s_mesh", 8),
    ("s_solve", 8),
    ("s_airfoil", 4),
    ("s_karman", 6),
    ("s_resweep", 4),
    ("s_kh", 5),
    ("s_turb", 5),
    ("s_apps", 15),
    ("s_outro", 13),
]
TIMELINE = []
_t = 0
for _name, _d in _SHOTS:
    TIMELINE.append((_name, _t, _t + _d))
    _t += _d
DURATION = TIMELINE[-1][2]
NFRAMES = int(DURATION * FPS)
_MODS = {}
_GRAIN = None


def module(name):
    if name not in _MODS:
        _MODS[name] = importlib.import_module(f"scenes.{name}")
    return _MODS[name]


def locate(t):
    for name, t0, t1 in TIMELINE:
        if t0 <= t < t1:
            return name, t0, t1
    name, t0, t1 = TIMELINE[-1]
    return name, t0, t1


def grain_bank():
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(7)
        _GRAIN = [rng.normal(0, 1, (H, W)).astype(np.float32)[..., None] for _ in range(6)]
    return _GRAIN


def render_frame(fi):
    t = fi / FPS
    name, t0, t1 = locate(t)
    mod = module(name)
    img = mod.render(t - t0, t)
    if img.shape != (H, W, 3):
        raise ValueError(f"{name} returned {img.shape}")
    # hard-cut glitch on the first frames of every scene (scene may opt out)
    k = fi - int(round(t0 * FPS))
    if k < 2 and t0 > 0 and getattr(mod, "CUT_GLITCH", True):
        rng = np.random.default_rng(fi * 31 + 5)
        img = fx.glitch_burst(img, rng, 0.55 if k == 0 else 0.25)
    amt = getattr(mod, "GRAIN", 5.0)
    if amt:
        g = grain_bank()[fi % 6]
        img = np.clip(img.astype(np.float32) + g * amt, 0, 255).astype(np.uint8)
    return img


def save_png(img, path):
    from PIL import Image
    Image.fromarray(img).save(path)


def cmd_still(times):
    d = RENDER / "stills"
    d.mkdir(parents=True, exist_ok=True)
    for ts in times:
        fi = int(round(float(ts) * FPS))
        t0 = time.time()
        img = render_frame(fi)
        p = d / f"t{float(ts):06.2f}.png"
        save_png(img, p)
        print(f"{p}  ({time.time() - t0:.2f}s)")


def cmd_sheet(scene, n=12, cols=4, tw=480):
    from PIL import Image
    t0, t1 = [(a, b) for s, a, b in TIMELINE if s == scene][0]
    ts = np.linspace(t0, t1 - 1 / FPS, n)
    th = tw * H // W
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * 4, rows * th + (rows - 1) * 4), (90, 90, 90))
    tt = time.time()
    for i, t in enumerate(ts):
        img = Image.fromarray(render_frame(int(round(t * FPS)))).resize((tw, th), Image.LANCZOS)
        sheet.paste(img, ((i % cols) * (tw + 4), (i // cols) * (th + 4)))
    p = RENDER / f"sheet_{scene}.jpg"
    RENDER.mkdir(exist_ok=True)
    sheet.save(p, quality=90)
    print(p, f"{(time.time() - tt) / n:.2f}s/frame")


def ffmpeg_writer(path, threads=2, crf=13):
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-x264-params", f"threads={threads}", "-g", str(FPS), str(path)],
        stdin=subprocess.PIPE)


def render_chunk(args):
    a, b, path = args
    p = ffmpeg_writer(path)
    for fi in range(a, b):
        p.stdin.write(render_frame(fi).tobytes())
    p.stdin.close()
    p.wait()
    return path


def cmd_video(workers=8, a=0, b=None):
    import multiprocessing as mp
    b = NFRAMES if b is None else b
    cdir = RENDER / "chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    size = 60
    jobs = [(s, min(s + size, b), cdir / f"c{s:05d}.mp4") for s in range(a, b, size)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, pth in enumerate(pool.imap_unordered(render_chunk, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {pth.name}  {time.time() - t0:.0f}s", flush=True)
    lst = cdir / "list.txt"
    lst.write_text("".join(f"file '{j[2].as_posix()}'\n" for j in jobs), encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(RENDER / "video_noaudio.mp4")], check=True)
    print("video done", time.time() - t0)


def cmd_mux():
    OUT.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(RENDER / "video_noaudio.mp4"),
                    "-i", str(RENDER / "soundtrack.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-shortest", "-movflags", "+faststart", str(OUT / "CFD_ACID.mp4")], check=True)
    print(OUT / "CFD_ACID.mp4")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "still":
        cmd_still(sys.argv[2:])
    elif cmd == "sheet":
        cmd_sheet(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif cmd == "video":
        cmd_video(int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                  *(int(x) for x in sys.argv[3:5]))
    elif cmd == "mux":
        cmd_mux()
