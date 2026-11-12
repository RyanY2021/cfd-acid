"""Livery decal sheets for the CFD-02 vehicle, drawn with the film's own type engine.
side: X in [-2.3, 2.3] m, Z in [0, 1.6] m at 1000 px/m (nose at left).  top: X in [-2.3,2.3], Y in [-1,1]."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine.core import *  # noqa

OUT = ROOT / "assets" / "vehicle"
OUT.mkdir(parents=True, exist_ok=True)
INK = (14, 14, 18)


def px(xm, zm):
    return (xm + 2.3) * 1000, (1.6 - zm) * 1000


def side():
    fr = Frame(4600, 1600, bg=None)
    # pinstripes along the shoulder
    x0, _ = px(-1.55, 0)
    x1, _ = px(1.75, 0)
    for z, wdt in ((0.66, 10), (0.635, 4)):
        _, y = px(0, z)
        fr.rect(x0, y, x1 - x0, wdt, INK)
    # big race number
    x, y = px(0.18, 0.69)
    fr.text("02", x, y, "tek9", 540, INK, anchor="baseline", tracking=-0.02)
    # checker block + tag
    cx, cy = px(1.05, 1.06)
    for j in range(3):
        for i in range(6):
            if (i + j) % 2 == 0:
                fr.rect(cx + i * 52, cy + j * 52, 52, 52, INK)
    fr.text("+02K", cx, cy - 26, "tek9", 92, MAGENTA, anchor="baseline")
    x, y = px(1.36, 1.0)
    fr.text("TOW", x, y - 110, "mono7", 34, INK, anchor="baseline", tracking=0.15)
    # micro type block
    x, y = px(-1.42, 1.02)
    for k, s in enumerate(["CFD-02  //  SLIPSTREAM", "Cd 0.24   A 2.1 M2", "YZ02-23AB-45CD   C33"]):
        fr.text(s, x, y + k * 56, "mono7", 40, INK, anchor="top", tracking=0.04)
    # intake arrow
    x, y = px(-1.55, 0.80)
    fr.line(x + 28, y + 80, x + 28, y - 70, INK, 10)
    fr.polyline([(x, y - 40), (x + 28, y - 76), (x + 56, y - 40)], INK, 10)
    fr.text("INTAKE", x + 80, y + 10, "inter7", 56, INK, anchor="middle")
    # barcode (vertical) near tail
    x, y = px(1.78, 1.02)
    barcode(fr, x, y, 300, 110, INK, seed=11, vertical=True)
    # on the magenta lower hull: white type + hazard tick
    x, y = px(-1.2, 0.47)
    fr.text("COMPUTATIONAL FLUID DYNAMICS", x, y, "inter7", 50, WHITE, anchor="middle", tracking=0.02)
    x, y = px(1.05, 0.47)
    hazard(fr, x, y - 28, 520, 56, WHITE, 22)
    x, y = px(-1.95, 0.47)
    fr.text("CFD", x, y, "stencil", 120, WHITE, anchor="middle")
    # panel seams, fuel cap, rivets (fine product detail)
    SEAM = (150, 150, 155)
    a, b = px(-0.62, 1.0)
    c, d = px(0.98, 0.60)
    path = skia.Path()
    path.addRoundRect(skia.Rect.MakeLTRB(a, b, c, d), 60, 60)
    fr.path(path, SEAM, stroke=5)
    x, y = px(1.62, 0.86)
    fr.circle(x, y, 46, SEAM, stroke=5)
    fr.text("H2", x, y, "mono7", 30, SEAM, align="center", anchor="middle")
    for xm in np.linspace(-1.5, 1.7, 17):
        x, y = px(xm, 0.605)
        fr.circle(x, y, 7, SEAM)
    x, y = px(-0.5, 0.93)
    fr.text("PRESS", x, y, "mono7", 24, SEAM, anchor="top", tracking=0.2)
    fr.surface.makeImageSnapshot().save(str(OUT / "side_decal.png"), skia.kPNG)


def top():
    fr = Frame(4600, 2000, bg=None)
    # hood checker flag
    for j in range(8):
        for i in range(6):
            if (i + j) % 2 == 0:
                fr.rect(380 + i * 70, 720 + j * 70, 70, 70, INK)
    fr.text("02", 1000, 1000, "tek9", 330, INK, anchor="middle")
    fr.rect(1500, 980, 900, 40, MAGENTA)
    fr.text("SLIPSTREAM", 1520, 950, "wide9", 70, INK, anchor="baseline")
    # roof-tail stripe
    fr.rect(3300, 960, 900, 80, LIME)
    fr.surface.makeImageSnapshot().save(str(OUT / "top_decal.png"), skia.kPNG)


def rim():
    fr = Frame(1024, 1024, bg=None)
    c = 512
    fr.circle(c, c, 500, (238, 238, 236))
    fr.circle(c, c, 430, MAGENTA)
    fr.circle(c, c, 330, (238, 238, 236))
    import skia as sk
    for k in range(5):
        a0 = k * 72
        fr.ring_segment(c, c, 120, 320, a0, a0 + 34, INK)
    fr.circle(c, c, 110, INK)
    fr.text("02", c, c, "tek9", 110, LIME, align="center", anchor="middle")
    for k in range(24):
        a = math.radians(k * 15)
        fr.circle(c + 465 * math.cos(a), c + 465 * math.sin(a), 9, INK)
    fr.surface.makeImageSnapshot().save(str(OUT / "rim.png"), skia.kPNG)


def tire():
    """Sidewall lettering ring, tire-local XZ plane, 0.9 m square."""
    fr = Frame(1024, 1024, bg=None)
    c = 512
    txt = "SLIPSTREAM  //  LAMINAR COMPOUND  //  02  //  SLIPSTREAM  //  LAMINAR COMPOUND  //  02  //  "
    f = font("wide9", 30)
    r = 410
    ang = -90.0
    for ch in txt:
        wv = f.measureText(ch)
        fr.save()
        fr.translate(c, c)
        fr.rotate(ang + 90)
        fr.text(ch, 0, -r + 14, "wide9", 30, (240, 240, 240), align="left")
        fr.restore()
        ang += math.degrees((wv + 1.5) / r)
        if ang > 268:
            break
    fr.surface.makeImageSnapshot().save(str(OUT / "tire.png"), skia.kPNG)


if __name__ == "__main__":
    side()
    top()
    rim()
    tire()
    print("decals ->", OUT)
