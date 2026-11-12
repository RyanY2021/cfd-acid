"""CFD-02 'SLIPSTREAM' — procedural vehicle, studio + renders.  Run with Blender:

  blender -b --factory-startup --python vehicle/build_vehicle.py -- <mode>
    modes: test (hero still, low samples) | hero (turntable frames) | side (orthographic side, RGBA loop) | save
"""
import math
import pathlib
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSET = ROOT / "assets" / "vehicle"
OUT = ROOT / "render" / "vehicle"
MODE = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "test"


def lin(rgb):
    return tuple(((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92 for c in rgb) + (1.0,)


WHITE = lin((232, 232, 230))
MAGENTA = lin((236, 0, 95))
LIME = lin((190, 255, 0))
INK = lin((14, 14, 18))
PAPER = lin((245, 245, 242))


# ------------------------------------------------------------------ geometry helpers
def spline(pts, s):
    xs, ys = np.array(pts).T
    # monotone-ish smooth interpolation: cubic Hermite with Catmull-Rom tangents
    s = np.clip(s, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, s) - 1, 0, len(xs) - 2)
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    m = np.gradient(ys, xs)
    t = (s - x0) / (x1 - x0)
    h = x1 - x0
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * y0 + h10 * h * m[i] + h01 * y1 + h11 * h * m[i + 1]


def loft(name, x0, x1, zt, zb, hw, n=3.0, ends=(0.08, 0.08), ns=160, na=96, yc=0.0, end_pow=2.2):
    """Superellipse-section loft along X.  zt/zb/hw: lists of (s, value) control points, s in [0,1]."""
    s = np.linspace(0, 1, ns)
    top, bot, half = spline(zt, s), spline(zb, s), spline(hw, s)
    a0, a1 = ends
    f = np.ones_like(s)
    m0 = s < a0
    f[m0] = (1 - (1 - s[m0] / a0) ** end_pow) ** (1 / end_pow)
    m1 = s > 1 - a1
    f[m1] = (1 - (1 - (1 - s[m1]) / a1) ** end_pow) ** (1 / end_pow)
    f = np.maximum(f, 1e-3)
    zc = (top + bot) / 2
    hh = (top - bot) / 2 * f
    ww = half * f
    th = np.linspace(0, 2 * np.pi, na, endpoint=False)
    c, sn = np.cos(th), np.sin(th)
    cy = np.sign(c) * np.abs(c) ** (2 / n)
    sz = np.sign(sn) * np.abs(sn) ** (2 / n)
    X = x0 + s * (x1 - x0)
    verts = []
    for k in range(ns):
        for j in range(na):
            verts.append((X[k], yc + ww[k] * cy[j], zc[k] + hh[k] * sz[j]))
    faces = []
    for k in range(ns - 1):
        for j in range(na):
            a = k * na + j
            b = k * na + (j + 1) % na
            faces.append((a, b, b + na, a + na))
    # caps (fans to centre points)
    verts.append((X[0], yc, zc[0]))
    verts.append((X[-1], yc, zc[-1]))
    c0, c1 = len(verts) - 2, len(verts) - 1
    for j in range(na):
        faces.append((c0, (j + 1) % na, j))
        faces.append((c1, (ns - 1) * na + j, (ns - 1) * na + (j + 1) % na))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True
    return obj


def revolve_tire(name, R=0.36, rh=0.085, rw=0.19, n=3.2, nu=128, nv=48):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    cv, sv = np.cos(v), np.sin(v)
    r_off = np.sign(cv) * np.abs(cv) ** (2 / n) * rh
    y_off = np.sign(sv) * np.abs(sv) ** (2 / n) * rw
    verts = []
    for a in u:
        for k in range(nv):
            r = R + r_off[k]
            verts.append((r * math.cos(a), y_off[k], r * math.sin(a)))
    faces = []
    for i in range(nu):
        for k in range(nv):
            a = i * nv + k
            b = i * nv + (k + 1) % nv
            c = ((i + 1) % nu) * nv + (k + 1) % nv
            d = ((i + 1) % nu) * nv + k
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True
    return obj


def cylinder(name, r, depth, loc, rot=(math.pi / 2, 0, 0), verts=96):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    bev = o.modifiers.new("bevel", "BEVEL")
    bev.width = min(r, depth) * 0.18
    bev.segments = 4
    bev.limit_method = "ANGLE"
    bpy.ops.object.shade_smooth()
    return o


def torus(name, R, r, loc, rot):
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=64, minor_segments=24,
                                     location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    return o


# ------------------------------------------------------------------ materials
def principled(name, color, rough=0.25, coat=1.0, coat_rough=0.04, emission=None, estr=0.0, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = coat_rough
    if emission is not None:
        b.inputs["Emission Color"].default_value = emission
        b.inputs["Emission Strength"].default_value = estr
    return m


def body_material():
    m = bpy.data.materials.new("Body")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    bsdf = N["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.22
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.035
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])

    def cmp(inp, thr, smooth=0.004):
        mr = N.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = thr - smooth
        mr.inputs["From Max"].default_value = thr + smooth
        L.new(inp, mr.inputs["Value"])
        return mr.outputs["Result"]

    def mix(fac, a, b):
        mx = N.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        L.new(fac, mx.inputs["Factor"])
        if isinstance(a, tuple):
            mx.inputs["A"].default_value = a
        else:
            L.new(a, mx.inputs["A"])
        if isinstance(b, tuple):
            mx.inputs["B"].default_value = b
        else:
            L.new(b, mx.inputs["B"])
        return mx.outputs["Result"]

    # colour blocking by height: magenta hull / lime keyline / white shell
    above_band = cmp(sep.outputs["Z"], 0.555)
    above_line = cmp(sep.outputs["Z"], 0.578)
    c = mix(above_band, MAGENTA, LIME)
    c = mix(above_line, c, WHITE)

    # object-space normal for decal masks
    geo = N.new("ShaderNodeNewGeometry")
    vt = N.new("ShaderNodeVectorTransform")
    vt.vector_type = "NORMAL"
    vt.convert_from = "WORLD"
    vt.convert_to = "OBJECT"
    L.new(geo.outputs["Normal"], vt.inputs["Vector"])
    nsep = N.new("ShaderNodeSeparateXYZ")
    L.new(vt.outputs["Vector"], nsep.inputs[0])

    def decal(path, ucomp, vcomp, uoff, uscale, voff, vscale, mirror=False):
        img = bpy.data.images.load(str(path))
        tex = N.new("ShaderNodeTexImage")
        tex.image = img
        tex.extension = "CLIP"
        tex.interpolation = "Cubic"
        u = N.new("ShaderNodeMath")
        u.operation = "MULTIPLY_ADD"
        L.new(ucomp, u.inputs[0])
        u.inputs[1].default_value = (-1 if mirror else 1) / uscale
        u.inputs[2].default_value = (1 - uoff / uscale) if mirror else uoff / uscale
        v = N.new("ShaderNodeMath")
        v.operation = "MULTIPLY_ADD"
        L.new(vcomp, v.inputs[0])
        v.inputs[1].default_value = 1 / vscale
        v.inputs[2].default_value = voff / vscale
        cmb = N.new("ShaderNodeCombineXYZ")
        L.new(u.outputs[0], cmb.inputs["X"])
        L.new(v.outputs[0], cmb.inputs["Y"])
        L.new(cmb.outputs[0], tex.inputs["Vector"])
        return tex

    # side decals: -Y side reads nose->tail, +Y side mirrored so text stays readable
    side_a = decal(ASSET / "side_decal.png", sep.outputs["X"], sep.outputs["Z"], 2.3, 4.6, 0.0, 1.6)
    side_b = decal(ASSET / "side_decal.png", sep.outputs["X"], sep.outputs["Z"], 2.3, 4.6, 0.0, 1.6, mirror=True)
    neg_y = N.new("ShaderNodeMath")
    neg_y.operation = "MULTIPLY"
    L.new(nsep.outputs["Y"], neg_y.inputs[0])
    neg_y.inputs[1].default_value = -1.0
    m_a = cmp(neg_y.outputs[0], 0.4, 0.12)
    m_b = cmp(nsep.outputs["Y"], 0.4, 0.12)

    def alpha_mask(tex, mask):
        mm = N.new("ShaderNodeMath")
        mm.operation = "MULTIPLY"
        L.new(tex.outputs["Alpha"], mm.inputs[0])
        L.new(mask, mm.inputs[1])
        return mm.outputs[0]

    c = mix(alpha_mask(side_a, m_a), c, side_a.outputs["Color"])
    c = mix(alpha_mask(side_b, m_b), c, side_b.outputs["Color"])
    top = decal(ASSET / "top_decal.png", sep.outputs["X"], sep.outputs["Y"], 2.3, 4.6, 1.0, 2.0)
    m_t = cmp(nsep.outputs["Z"], 0.7, 0.1)
    c = mix(alpha_mask(top, m_t), c, top.outputs["Color"])
    L.new(c, bsdf.inputs["Base Color"])
    return m


def rim_material():
    m = principled("Rim", WHITE, rough=0.3)
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    tex = N.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(ASSET / "rim.png"))
    tex.extension = "CLIP"
    # rim objects are cylinders rotated about X; in object space the disc lies in X-Y
    u = N.new("ShaderNodeMath")
    u.operation = "MULTIPLY_ADD"
    L.new(sep.outputs["X"], u.inputs[0])
    u.inputs[1].default_value = 1 / 0.62
    u.inputs[2].default_value = 0.5
    v = N.new("ShaderNodeMath")
    v.operation = "MULTIPLY_ADD"
    L.new(sep.outputs["Y"], v.inputs[0])
    v.inputs[1].default_value = 1 / 0.62
    v.inputs[2].default_value = 0.5
    cmb = N.new("ShaderNodeCombineXYZ")
    L.new(u.outputs[0], cmb.inputs["X"])
    L.new(v.outputs[0], cmb.inputs["Y"])
    L.new(cmb.outputs[0], tex.inputs["Vector"])
    mx = N.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    L.new(tex.outputs["Alpha"], mx.inputs["Factor"])
    mx.inputs["A"].default_value = INK
    L.new(tex.outputs["Color"], mx.inputs["B"])
    L.new(mx.outputs["Result"], N["Principled BSDF"].inputs["Base Color"])
    return m


def tire_material():
    m = principled("Rubber", lin((22, 22, 26)), rough=0.55, coat=0.0)
    N, L = m.node_tree.nodes, m.node_tree.links
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    tex = N.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(ASSET / "tire.png"))
    tex.extension = "CLIP"
    u = N.new("ShaderNodeMath")
    u.operation = "MULTIPLY_ADD"
    L.new(sep.outputs["X"], u.inputs[0])
    u.inputs[1].default_value = 1 / 0.9
    u.inputs[2].default_value = 0.5
    v = N.new("ShaderNodeMath")
    v.operation = "MULTIPLY_ADD"
    L.new(sep.outputs["Z"], v.inputs[0])
    v.inputs[1].default_value = 1 / 0.9
    v.inputs[2].default_value = 0.5
    cmb = N.new("ShaderNodeCombineXYZ")
    L.new(u.outputs[0], cmb.inputs["X"])
    L.new(v.outputs[0], cmb.inputs["Y"])
    L.new(cmb.outputs[0], tex.inputs["Vector"])
    geo = N.new("ShaderNodeNewGeometry")
    vt = N.new("ShaderNodeVectorTransform")
    vt.vector_type = "NORMAL"
    vt.convert_from = "WORLD"
    vt.convert_to = "OBJECT"
    L.new(geo.outputs["Normal"], vt.inputs["Vector"])
    ns = N.new("ShaderNodeSeparateXYZ")
    L.new(vt.outputs["Vector"], ns.inputs[0])
    ab = N.new("ShaderNodeMath")
    ab.operation = "ABSOLUTE"
    L.new(ns.outputs["Y"], ab.inputs[0])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.55
    mr.inputs["From Max"].default_value = 0.75
    L.new(ab.outputs[0], mr.inputs["Value"])
    mm = N.new("ShaderNodeMath")
    mm.operation = "MULTIPLY"
    L.new(tex.outputs["Alpha"], mm.inputs[0])
    L.new(mr.outputs["Result"], mm.inputs[1])
    mx = N.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    L.new(mm.outputs[0], mx.inputs["Factor"])
    mx.inputs["A"].default_value = lin((22, 22, 26))
    mx.inputs["B"].default_value = lin((235, 235, 235))
    L.new(mx.outputs["Result"], N["Principled BSDF"].inputs["Base Color"])
    return m


# ------------------------------------------------------------------ build the car
def build_car():
    car = bpy.data.objects.new("CAR", None)
    bpy.context.collection.objects.link(car)
    parts = []
    body = loft("body", -2.12, 2.12,
                zt=[(0, 0.66), (0.07, 0.80), (0.22, 0.93), (0.42, 1.06), (0.62, 1.09), (0.82, 1.03), (0.95, 0.94),
                    (1, 0.84)],
                zb=[(0, 0.50), (0.1, 0.36), (0.5, 0.30), (0.9, 0.35), (1, 0.48)],
                hw=[(0, 0.66), (0.14, 0.82), (0.45, 0.88), (0.8, 0.87), (1, 0.74)],
                n=3.2, ends=(0.10, 0.08))
    body.data.materials.append(body_material())
    parts.append(body)

    canopy = loft("canopy", -0.95, 0.95,
                  zt=[(0, 1.02), (0.25, 1.36), (0.55, 1.42), (0.85, 1.30), (1, 1.10)],
                  zb=[(0, 0.86), (1, 0.90)],
                  hw=[(0, 0.52), (0.4, 0.66), (1, 0.56)], n=2.6, ends=(0.32, 0.26))
    canopy.location.x = -0.05
    glass = principled("Glass", lin((10, 10, 14)), rough=0.04, coat=1.0, coat_rough=0.01)
    canopy.data.materials.append(glass)
    parts.append(canopy)

    white = principled("WhiteGloss", WHITE, rough=0.22)
    mag = principled("MagentaGloss", MAGENTA, rough=0.24)
    rubber = tire_material()
    rimm = rim_material()
    for xw in (-1.33, 1.36):
        for side in (-1, 1):
            yw = side * 0.93
            fender = loft(f"fender_{xw}_{side}", xw - 0.66, xw + 0.66,
                          zt=[(0, 0.62), (0.3, 0.92), (0.5, 0.97), (0.7, 0.92), (1, 0.62)],
                          zb=[(0, 0.56), (1, 0.56)],
                          hw=[(0, 0.18), (0.5, 0.25), (1, 0.18)], n=2.8, ends=(0.22, 0.22), yc=yw, ns=90, na=64)
            fender.data.materials.append(mag if xw > 0 else white)
            parts.append(fender)
            tire = revolve_tire(f"tire_{xw}_{side}")
            tire.location = (xw, yw, 0.445)
            tire.data.materials.append(rubber)
            parts.append(tire)
            rim = cylinder(f"rim_{xw}_{side}", 0.31, 0.30, (xw, yw, 0.445))
            rim.data.materials.append(rimm)
            parts.append(rim)
    # tow loop + base (NuCaloric spout nod)
    base = loft("loopbase", 1.38, 1.72, zt=[(0, 1.12), (0.5, 1.2), (1, 1.12)], zb=[(0, 0.95), (1, 0.95)],
                hw=[(0, 0.14), (1, 0.14)], n=3.0, ends=(0.3, 0.3), ns=40, na=48)
    base.data.materials.append(mag)
    loop = torus("loop", 0.13, 0.042, (1.55, 0, 1.33), (math.pi / 2, 0, 0))
    loop.data.materials.append(mag)
    parts += [base, loop]
    # light strips
    head = loft("headlight", -2.1, -2.02, zt=[(0, 0.70), (1, 0.70)], zb=[(0, 0.64), (1, 0.64)],
                hw=[(0, 0.46), (1, 0.46)], n=4, ends=(0.3, 0.3), ns=20, na=48)
    head.data.materials.append(principled("HeadLED", WHITE, emission=lin((230, 255, 160)), estr=14))
    tail = loft("taillight", 2.04, 2.12, zt=[(0, 0.83), (1, 0.83)], zb=[(0, 0.77), (1, 0.77)],
                hw=[(0, 0.5), (1, 0.5)], n=4, ends=(0.3, 0.3), ns=20, na=48)
    tail.data.materials.append(principled("TailLED", MAGENTA, emission=lin((255, 20, 90)), estr=12))
    parts += [head, tail]
    # sensor pods + antenna
    for side in (-1, 1):
        pod = loft(f"pod{side}", -0.75, -0.45, zt=[(0, 1.08), (1, 1.08)], zb=[(0, 0.98), (1, 0.98)],
                   hw=[(0, 0.07), (1, 0.07)], n=3, ends=(0.4, 0.4), yc=side * 0.86, ns=30, na=32)
        pod.data.materials.append(glass)
        parts.append(pod)
    ant = cylinder("antenna", 0.012, 0.55, (1.05, 0.42, 1.32), rot=(0, 0, 0), verts=16)
    ant.data.materials.append(rubber)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.045, location=(1.05, 0.42, 1.6))
    tip = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    tip.data.materials.append(principled("LimeTip", LIME, emission=LIME, estr=3))
    parts += [ant, tip]
    for p in parts:
        p.parent = car
    return car, parts


# ------------------------------------------------------------------ studio
def studio(backdrop=True):
    sc = bpy.context.scene
    world = bpy.data.worlds.new("W")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 0.16
    if backdrop:
        # cyclorama sweep
        prof = [(y, 0.0) for y in np.linspace(-30, 6, 30)]
        prof += [(6 + 5 * math.sin(a), 5 - 5 * math.cos(a)) for a in np.linspace(0.05, math.pi / 2, 24)]
        prof += [(11, z) for z in np.linspace(5.5, 20, 6)]
        verts, faces = [], []
        for x in (-40, 40):
            for y, z in prof:
                verts.append((x, y, z))
        n = len(prof)
        for i in range(n - 1):
            faces.append((i, i + 1, n + i + 1, n + i))
        me = bpy.data.meshes.new("cyc")
        me.from_pydata(verts, [], faces)
        cyc = bpy.data.objects.new("cyc", me)
        bpy.context.collection.objects.link(cyc)
        for p in me.polygons:
            p.use_smooth = True
        cyc.data.materials.append(principled("Paper", PAPER, rough=0.9, coat=0.0, emission=PAPER, estr=0.62))
        cyc.rotation_euler.z = math.radians(-35)
    for name, loc, size, energy, rot in (
            ("key", (-3.5, -4.5, 6.5), 5.0, 400, (math.radians(50), 0, math.radians(-40))),
            ("fill", (5.0, -6.0, 2.0), 7.0, 120, (math.radians(70), 0, math.radians(55))),
            ("rim", (3.0, 5.0, 4.0), 3.0, 650, (math.radians(-55), 0, math.radians(165))),
            ("top", (0, 0, 7.5), 8.0, 160, (0, 0, 0))):
        ld = bpy.data.lights.new(name, "AREA")
        ld.size = size
        ld.energy = energy
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = rot
        bpy.context.collection.objects.link(lo)


def setup_render(w, h, samples, transparent=False):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPTIX"
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = transparent
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"


def camera(loc, target, lens=60, ortho=None, shift_x=0.0):
    cd = bpy.data.cameras.new("cam")
    co = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(co)
    co.location = loc
    d = Vector(target) - Vector(loc)
    co.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
    else:
        cd.lens = lens
    cd.shift_x = shift_x
    bpy.context.scene.camera = co
    return co


# ------------------------------------------------------------------ main
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


# hero camera — scenes/s_apps.py projects callouts with the same values
HERO_CAM = ((-8.4, -9.8, 2.8), (0.15, 0.0, 0.66), 60.0, -0.21)

# side-view mapping shared with the LBM mask (sim grid 960x360, 1 cell = S metres, ground row 300)
S_CELL = 4.24 / 330
X0 = -2.12 - 210 * S_CELL
CAM_X = X0 + 480 * S_CELL
CAM_Z = (300 - 180) * S_CELL


if __name__ == "__main__":
    reset()
    OUT.mkdir(parents=True, exist_ok=True)
    car, parts = build_car()
    if MODE in ("test", "hero", "save"):
        studio(True)
        setup_render(1920, 1080, 64 if MODE == "test" else 160)
        camera(*HERO_CAM[:2], lens=HERO_CAM[2], shift_x=HERO_CAM[3])
        if MODE == "test":
            car.rotation_euler.z = math.radians(-12)
            bpy.context.scene.render.filepath = str(OUT / "test_hero.png")
            bpy.ops.render.render(write_still=True)
        elif MODE == "save":
            bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "vehicle" / "cfd02.blend"))
        else:
            sc = bpy.context.scene
            n = 40
            for f in range(n):
                k = f / (n - 1)
                car.rotation_euler.z = math.radians(-34 + 30 * (1 - (1 - k) ** 2))
                sc.render.filepath = str(OUT / f"hero_{f:03d}.png")
                bpy.ops.render.render(write_still=True)
    elif MODE == "side":
        studio(False)
        bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.42
        setup_render(1920, 720, 128, transparent=True)
        camera((CAM_X, -30.0, CAM_Z), (CAM_X, 0.0, CAM_Z), ortho=960 * S_CELL)
        rims = [p for p in parts if p.name.startswith("rim_")]
        tires = [p for p in parts if p.name.startswith("tire_")]
        sc = bpy.context.scene
        # the car faces -X (left on screen): rolling left = counter-clockwise seen from the camera (-Y)
        for f in range(30):
            a = math.radians(12 * f)
            for r in rims:
                r.rotation_euler = (math.pi / 2, 0, 0)
                r.rotation_euler.rotate_axis("Z", a)      # local Z points at the camera -> +a is CCW
            for tr in tires:
                tr.rotation_euler = (0, -a, 0)            # world +Y points away from the camera -> -a is CCW
            sc.render.filepath = str(OUT / f"side_{f:02d}.png")
            bpy.ops.render.render(write_still=True)
