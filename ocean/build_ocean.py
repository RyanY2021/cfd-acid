"""Top-down aerial surf shot for the intro (after the Marathon Reveal beach).  Run with Blender:

  blender -b --factory-startup --python ocean/build_ocean.py -- <mode> [start end]
    modes: test (three stills) | frames (render frames start..end-1) | save
Layout (metres): ocean, surf zone up to the shoreline at SHORE, thin wet margin, land/compound x>LAND_X.
"""
import math
import pathlib
import sys

import bpy
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "render" / "ocean"
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["test"]
MODE = ARGS[0]
FPS = 30
N_FRAMES = 180
SHORE = 26.0        # shoreline x (m); water reaches almost to the land boundary
LAND_X = 30.5       # land / compound boundary (the HUD dividing line)


def lin(rgb):
    return tuple(((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92 for c in rgb) + (1.0,)


class G:
    """Tiny node-graph helper."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.N, self.L = self.nt.nodes, self.nt.links

    def node(self, kind, **kw):
        n = self.N.new(kind)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.L.new(a, b)

    def math(self, op, a, b=None, c=None):
        n = self.node("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                self.link(v, n.inputs[i])
        return n.outputs[0]

    def smooth(self, v, lo, hi, out_lo=0.0, out_hi=1.0):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        n.inputs["From Min"].default_value = lo
        n.inputs["From Max"].default_value = hi
        n.inputs["To Min"].default_value = out_lo
        n.inputs["To Max"].default_value = out_hi
        self.link(v, n.inputs["Value"])
        return n.outputs["Result"]

    def noise(self, vec, scale, detail=4.0, rough=0.55, distortion=0.0):
        n = self.node("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        self.link(vec, n.inputs["Vector"])
        return n.outputs["Fac"]

    def voronoi_edge(self, vec, scale):
        n = self.node("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
        n.inputs["Scale"].default_value = scale
        self.link(vec, n.inputs["Vector"])
        return n.outputs["Distance"]

    def mixc(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="RGBA")
        self.link(fac, n.inputs["Factor"]) if not isinstance(fac, float) else None
        if isinstance(fac, float):
            n.inputs["Factor"].default_value = fac
        for key, v in (("A", a), ("B", b)):
            if isinstance(v, tuple):
                n.inputs[key].default_value = v
            else:
                self.link(v, n.inputs[key])
        return n.outputs["Result"]


TIME = {}


# ------------------------------------------------------------------ materials
def water_material():
    m = bpy.data.materials.new("Water")
    g = G(m)
    bsdf = g.N["Principled BSDF"]
    geo = g.node("ShaderNodeNewGeometry")

    class _P:
        outputs = {"Object": geo.outputs["Position"]}
    tc = _P()
    sep = g.node("ShaderNodeSeparateXYZ")
    g.link(geo.outputs["Position"], sep.inputs[0])
    x, y = g.math("SUBTRACT", sep.outputs["X"], SHORE), sep.outputs["Y"]
    tval = g.node("ShaderNodeValue")
    tval.outputs[0].default_value = 0.0
    TIME["water"] = tval
    t = tval.outputs[0]

    # depth below a 5% sand slope (shoreline at x = 0)
    depth = g.math("MULTIPLY", x, -0.05)
    ramp = g.node("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = lin((96, 178, 160))
    cr.elements[1].position = 1.0
    cr.elements[1].color = lin((3, 26, 34))
    for pos, c in ((0.05, (52, 160, 150)), (0.15, (16, 108, 112)), (0.36, (7, 62, 72))):
        e = cr.elements.new(pos)
        e.color = lin(c)
    patches = g.math("MULTIPLY", g.math("SUBTRACT", g.noise(tc.outputs["Object"], 0.012, 3.0), 0.5), 0.5)
    g.link(g.math("ADD", g.math("DIVIDE", depth, 6.0), patches), ramp.inputs["Fac"])

    # breaking-wave fronts travelling toward the shore, wavy along y
    wob = g.math("ADD", g.math("MULTIPLY", g.math("SUBTRACT", g.noise(tc.outputs["Object"], 0.035, 2.0), 0.5), 2.2),
                 g.math("MULTIPLY", g.math("SUBTRACT", g.noise(tc.outputs["Object"], 0.16, 2.0), 0.5), 0.7))
    phase = g.math("ADD", g.math("DIVIDE", g.math("ADD", x, 40.0), 6.2), wob)
    phase = g.math("SUBTRACT", phase, g.math("DIVIDE", t, 4.2))
    f = g.math("FRACT", phase)
    front = g.math("MULTIPLY", g.smooth(f, 0.62, 0.93), g.smooth(f, 1.0, 0.972))
    chunk = g.smooth(g.noise(tc.outputs["Object"], 0.35, 5.0, 0.65), 0.38, 0.6)
    front = g.math("MULTIPLY", front, g.math("ADD", 0.12, g.math("MULTIPLY", chunk, 1.0)))
    tail = g.math("POWER", g.smooth(f, 0.05, 0.97), 1.7)
    breaking = g.smooth(x, -60.0, -32.0)
    nearshore = g.smooth(x, 2.0, -1.0)
    zone = g.math("MULTIPLY", breaking, nearshore)
    # lacy foam lattice
    warp = g.node("ShaderNodeVectorMath", operation="ADD")
    g.link(tc.outputs["Object"], warp.inputs[0])
    nvec = g.node("ShaderNodeTexNoise")
    nvec.inputs["Scale"].default_value = 0.22
    g.link(tc.outputs["Object"], nvec.inputs["Vector"])
    sc = g.node("ShaderNodeVectorMath", operation="SCALE")
    g.link(nvec.outputs["Color"], sc.inputs[0])
    sc.inputs["Scale"].default_value = 5.0
    g.link(sc.outputs[0], warp.inputs[1])
    lace = g.smooth(g.voronoi_edge(warp.outputs[0], 1.3), 0.05, 0.0)
    lace = g.math("MULTIPLY", lace, g.smooth(g.noise(warp.outputs[0], 0.5, 3.0), 0.4, 0.6))
    fine = g.noise(warp.outputs[0], 1.4, 10.0, 0.72, 1.5)
    lace = g.math("MAXIMUM", g.math("MULTIPLY", lace, 0.6), g.smooth(fine, 0.55, 0.7))
    foam_front = g.math("MULTIPLY", front, g.math("ADD", 0.7, g.math("MULTIPLY", lace, 0.3)))
    foam_tail = g.math("MULTIPLY", g.math("MULTIPLY", tail, lace), 0.7)
    surf = g.math("MULTIPLY", g.math("MAXIMUM", foam_front, foam_tail), zone)
    # swash line sliding up and down the beach
    swash_x = g.math("MULTIPLY", g.math("SINE", g.math("ADD", g.math("MULTIPLY", t, 2 * math.pi / 5.2),
                                                         g.math("MULTIPLY", y, 0.05))), 2.2)
    dist = g.math("ABSOLUTE", g.math("SUBTRACT", x, g.math("SUBTRACT", swash_x, 1.2)))
    swash = g.math("MULTIPLY", g.smooth(dist, 1.6, 0.2), g.math("ADD", 0.35, g.math("MULTIPLY", lace, 0.65)))
    # ocean-modifier whitecaps offshore
    attr = g.node("ShaderNodeAttribute", attribute_name="foam")
    caps = g.math("MULTIPLY", g.smooth(attr.outputs["Fac"], 0.35, 0.9), g.math("MULTIPLY", lace, 0.8))
    foam = g.math("MINIMUM", g.math("MAXIMUM", g.math("MAXIMUM", surf, swash), caps), 1.0)

    col = g.mixc(foam, ramp.outputs["Color"], lin((236, 242, 238)))
    g.link(col, bsdf.inputs["Base Color"])
    g.link(g.math("ADD", 0.035, g.math("MULTIPLY", foam, 0.75)), bsdf.inputs["Roughness"])
    bump = g.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    g.link(g.noise(tc.outputs["Object"], 0.9, 6.0, 0.6), bump.inputs["Height"])
    g.link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    # very shallow water turns transparent to show the sand below
    alpha = g.math("MAXIMUM", g.smooth(depth, -0.05, 0.55), foam)
    g.link(alpha, bsdf.inputs["Alpha"])
    return m


def sand_material():
    m = bpy.data.materials.new("Sand")
    g = G(m)
    bsdf = g.N["Principled BSDF"]
    tc = g.node("ShaderNodeTexCoord")
    P = tc.outputs["Object"]
    sep = g.node("ShaderNodeSeparateXYZ")
    g.link(P, sep.inputs[0])
    z, x, y = sep.outputs["Z"], sep.outputs["X"], sep.outputs["Y"]
    n1 = g.noise(P, 0.06, 6.0, 0.62)
    n2 = g.noise(P, 0.6, 8.0, 0.7)
    dry = g.mixc(n1, lin((150, 140, 98)), lin((196, 186, 138)))
    dry = g.mixc(g.math("MULTIPLY", n2, 0.35), dry, lin((120, 112, 80)))
    wet = g.mixc(n1, lin((70, 68, 52)), lin((98, 94, 70)))
    wetness = g.smooth(z, 0.6, 0.05)
    c = g.mixc(wetness, dry, wet)
    # inland: dark rocky scrub
    land = g.mixc(g.noise(P, 0.04, 6.0, 0.72), lin((40, 42, 34)), lin((92, 86, 64)))
    scrub = g.smooth(g.noise(P, 0.45, 6.0, 0.7), 0.56, 0.66)
    land = g.mixc(scrub, land, lin((30, 44, 30)))
    rocks = g.smooth(g.noise(P, 1.4, 4.0, 0.6), 0.66, 0.72)
    land = g.mixc(rocks, land, lin((120, 112, 92)))
    c = g.mixc(g.smooth(g.math("ADD", x, g.math("MULTIPLY", n1, 3.0)), LAND_X, LAND_X + 3.0), c, land)
    g.link(c, bsdf.inputs["Base Color"])
    g.link(g.math("ADD", 0.92, g.math("MULTIPLY", wetness, -0.72)), bsdf.inputs["Roughness"])
    bump = g.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    g.link(g.math("ADD", n2, g.math("MULTIPLY", rocks, 0.6)), bump.inputs["Height"])
    g.link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def flat(name, rgb, rough=0.6, emission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin(rgb)
    b.inputs["Roughness"].default_value = rough
    if emission:
        b.inputs["Emission Color"].default_value = lin(rgb)
        b.inputs["Emission Strength"].default_value = emission
    return m


# ------------------------------------------------------------------ geometry
def sand_mesh():
    xs = np.linspace(-14, 110, 500)
    ys = np.linspace(-90, 90, 600)
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(4)
    ph = rng.uniform(0, 6.28, 6)
    wig = (0.9 * np.sin(Y * 0.045 + ph[0]) + 0.5 * np.sin(Y * 0.11 + ph[1]) + 0.25 * np.sin(Y * 0.27 + ph[2]))
    Z = 0.05 * (X - SHORE - wig)
    Z = np.where(X - SHORE - wig < 0, 0.32 * (X - SHORE - wig), Z)
    Z = np.where(X > LAND_X - 1, Z + 0.5 * np.clip((X - LAND_X + 1) / 5, 0, 1), Z)              # berm
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    nx = len(xs)
    faces = []
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.append((a, a + 1, a + nx + 1, a + nx))
    me = bpy.data.meshes.new("sand")
    me.from_pydata(verts.tolist(), [], faces)
    o = bpy.data.objects.new("sand", me)
    bpy.context.collection.objects.link(o)
    for p in me.polygons:
        p.use_smooth = True
    o.data.materials.append(sand_material())


def ocean():
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    o = bpy.context.active_object
    o.name = "ocean"
    m = o.modifiers.new("ocean", "OCEAN")
    m.geometry_mode = "GENERATE"
    m.resolution = 16
    m.spatial_size = 50
    m.repeat_x = 4
    m.repeat_y = 5
    m.wave_scale = 0.75
    m.choppiness = 1.1
    m.wind_velocity = 11
    m.wave_alignment = 0.75
    m.wave_direction = 0.0
    m.use_foam = True
    m.foam_layer_name = "foam"
    m.foam_coverage = 0.25
    m.use_normals = True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = o.evaluated_get(dg)
    xs = [v.co.x for v in ev.data.vertices]
    ys = [v.co.y for v in ev.data.vertices]
    o.location = (SHORE + 6 - max(xs), -(min(ys) + max(ys)) / 2, 0)
    print("OCEAN bounds", min(xs), max(xs), min(ys), max(ys))
    bpy.ops.object.shade_smooth()
    o.data.materials.append(water_material())
    TIME["ocean"] = m
    return o


def compound():
    rng = np.random.default_rng(12)
    pal = [(235, 205, 20), (35, 60, 200), (232, 232, 226), (228, 100, 22), (236, 0, 95), (30, 190, 180),
           (110, 40, 190), (232, 232, 226), (235, 205, 20), (35, 60, 200), (200, 200, 196), (60, 60, 66)]
    mats = [flat(f"b{i}", c, 0.5) for i, c in enumerate(pal)]
    dark = flat("roofd", (34, 36, 42), 0.45)
    light = flat("roofl", (210, 210, 205), 0.6)
    road = flat("road", (82, 80, 76), 0.85)
    mark = flat("mark", (230, 230, 220), 0.6)
    solar = flat("solar", (14, 22, 60), 0.18)

    def box(x, y, z, sx, sy, sz, mat, bevel=0.08):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z + sz / 2))
        b = bpy.context.active_object
        b.scale = (sx, sy, sz)
        b.data.materials.append(mat)
        if bevel:
            bev = b.modifiers.new("b", "BEVEL")
            bev.width = bevel
            bev.segments = 2
        return b

    gz = 1.25
    for cl in range(7):
        cy = -78 + cl * 26 + 13
        for k in range(int(rng.integers(14, 22))):
            sx, sy = rng.uniform(1.6, 6.5), rng.uniform(1.6, 4.5)
            if rng.random() < 0.3:
                sx, sy = rng.uniform(8, 14), rng.uniform(2.2, 3.2)
            x = 54 + rng.uniform(0, 24)
            y = cy + rng.uniform(-10, 10)
            h = rng.uniform(1.5, 5.5)
            mat = mats[int(rng.integers(0, len(mats)))]
            box(x, y, gz, sx, sy, h, mat)
            top = gz + h
            if rng.random() < 0.55:
                box(x, y, top, sx * 0.92, sy * 0.3, 0.15, dark if rng.random() < 0.6 else light, 0.03)
            for u in range(int(rng.integers(0, 4))):
                box(x + rng.uniform(-sx / 3, sx / 3), y + rng.uniform(-sy / 3, sy / 3), top, 0.6, 0.6, 0.5,
                    light, 0.03)
        if rng.random() < 0.7:
            for k in range(int(rng.integers(1, 4))):
                bpy.ops.mesh.primitive_cylinder_add(radius=rng.uniform(1.4, 2.8), depth=3.5,
                                                    location=(44 + k * 5.5, cy - 6, gz + 1.75))
                bpy.context.active_object.data.materials.append(mats[2])
        for r in range(2):
            for k in range(5):
                box(41 + k * 2.2, cy + 6 + r * 3.2, gz + 0.3, 1.9, 2.8, 0.12, solar, 0.02)
        for k in range(int(rng.integers(3, 9))):
            box(52.6, cy + rng.uniform(-11, 11), gz, 2.0, 1.0, 0.8, mats[int(rng.integers(0, 6))], 0.15)
    for k in range(9):
        box(35.0 + rng.uniform(-1, 1), -60 + k * 15 + rng.uniform(-2, 2), 1.1, 2.4, 6.0, 2.5,
            mats[0 if k % 3 else 1])


def world_and_sun():
    sc = bpy.context.scene
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    for st in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING", "HOSEK_WILKIE"):
        try:
            sky.sky_type = st
            break
        except Exception:
            continue
    try:
        sky.sun_elevation = math.radians(48)
        sky.sun_rotation = math.radians(200)
    except Exception:
        pass
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.22
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy = 3.4
    sd.angle = math.radians(1.2)
    so = bpy.data.objects.new("sun", sd)
    so.rotation_euler = (math.radians(48), 0, math.radians(75))
    bpy.context.collection.objects.link(so)


def camera_at(t):
    k = t / (N_FRAMES / FPS)
    alt = 168 - 34 * k
    cam = bpy.context.scene.camera
    cam.location = (8 + 3 * k, -10 + 16 * k, alt)
    cam.rotation_euler = (0, 0, math.radians(1.5 * k))


def setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
    sc.cycles.samples = 48
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPTIX"
    sc.cycles.max_bounces = 6
    sc.cycles.transparent_max_bounces = 8
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.exposure = -0.7
    for look in ("AgX - Punchy", "Punchy", "AgX - Medium High Contrast"):
        try:
            sc.view_settings.look = look
            break
        except Exception:
            pass
    sc.render.image_settings.file_format = "PNG"
    sand_mesh()
    ocean()
    compound()
    world_and_sun()
    cd = bpy.data.cameras.new("cam")
    cd.lens = 35
    co = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(co)
    sc.camera = co


def set_time(t):
    TIME["water"].outputs[0].default_value = t
    TIME["ocean"].time = 1.0 + t
    camera_at(t)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    sc = bpy.context.scene
    if MODE == "test":
        for t in (0.0, 4.0):
            set_time(t)
            sc.render.filepath = str(OUT / f"test_{t:04.1f}.png")
            bpy.ops.render.render(write_still=True)
    elif MODE == "save":
        bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "ocean" / "surf.blend"))
    else:
        a, b = int(ARGS[1]), int(ARGS[2])
        for f in range(a, b):
            set_time(f / FPS)
            sc.render.filepath = str(OUT / f"surf_{f:03d}.png")
            bpy.ops.render.render(write_still=True)
