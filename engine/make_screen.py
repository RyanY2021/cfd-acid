"""Write the CRT screen image sequence for the Blender computer: render/crt_screen/screen_0001.png ..."""
import sys

from PIL import Image

from engine.core import ROOT
from scenes import s_outro

OUT = ROOT / "render" / "crt_screen"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = (int(x) for x in sys.argv[1:3]) if len(sys.argv) > 2 else (0, s_outro.N_3D)
    for f in range(a, b):
        Image.fromarray(s_outro.screen_texture(f)).save(OUT / f"screen_{f + 1:04d}.jpg", quality=95)
    print("screen frames", a, b, "->", OUT)
