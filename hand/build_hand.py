"""Cardiovascular transition: a still mannequin hand on strings, blood is pumped in, it comes alive.

  blender -b --factory-startup --python hand/build_hand.py -- test | frames [a b]
Right hand, palm toward camera (-Y), fingers +Z, thumb +X.  Metres.  90 frames @ 30 fps.
Look after the Marathon Arachne intro: black, crimson latex drapery, pale porcelain hands, hard top light.
"""
import math
import pathlib
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "render" / "hand"
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["test"]
MODE = ARGS[0]
N = 90
BEATS = (30, 45, 60, 75)          # heartbeat frames (120 BPM)


def lin(rgb):
    return tuple(((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92 for c in rgb) + (1.0,)


def smooth(x, a, b):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ skeleton + pose
FINGERS = {   # knuckle (x, z), splay deg, segment lengths, radii
    "index": ((0.029, 0.088), 13.0, (0.044, 0.026, 0.021), (0.0128, 0.0116, 0.0104)),
    "middle": ((0.009, 0.093), 2.0, (0.048, 0.029, 0.022), (0.0134, 0.0121, 0.0107)),
    "ring": ((-0.011, 0.089), -9.0, (0.044, 0.027, 0.021), (0.0126, 0.0114, 0.0102)),
    "pinky": ((-0.031, 0.079), -21.0, (0.034, 0.021, 0.018), (0.0112, 0.0102, 0.0092)),
}


def rot_about(v, axis, ang):
    return Matrix.Rotation(ang, 3, axis) @ v


def finger_chain(name, curl, spread, wiggle):
    (kx, kz), splay, lens, _ = FINGERS[name]
    base = Vector((kx, 0.0, kz))
    d = Vector((0, 0, 1))
    d = rot_about(d, Vector((0, 1, 0)), -math.radians(splay * (1 + spread)))
    lateral = d.cross(Vector((0, -1, 0))).normalized()
    pts = [base]
    flex = [0.30, 0.55, 0.40]
    p = base.copy()
    for i, L in enumerate(lens):
        ang = (0.06 + curl * flex[i] * 1.6 + wiggle * 0.15) if i else (0.04 + curl * 0.5 + wiggle * 0.2)
        d = rot_about(d, lateral, -ang)       # flex toward the palm (-Y)
        p = p + d * L
        pts.append(p.copy())
    return pts


def thumb_chain(curl, wiggle):
    pts = [Vector((0.020, -0.004, 0.014))]
    d = Vector((0.62, -0.32, 0.72)).normalized()
    lens = (0.040, 0.032, 0.026)
    axis = Vector((0.3, 0.4, -0.2)).normalized()
    p = pts[0]
    for i, L in enumerate(lens):
        d = rot_about(d, axis, -(0.12 + curl * 0.35 + wiggle * 0.12) * (1 if i else 0.4))
        p = p + d * L
        pts.append(p.copy())
    return pts


def pose(f):
    """Animation state at frame f."""
    alive = smooth(f, 40, 70)
    curl = 0.05 + 0.32 * alive + 0.05 * alive * math.sin((f - 40) / 9.0)
    spread = 0.25 * alive
    return dict(curl=curl, spread=spread, alive=alive, f=f)


# ------------------------------------------------------------------ build
def capsule(mb, a, b, r, stiff=2.0):
    e = mb.elements.new(type="CAPSULE")
    set_capsule(e, a, b, r, stiff)
    return e


def set_capsule(e, a, b, r, stiff=2.0):
    a, b = Vector(a), Vector(b)
    d = b - a
    e.co = (a + b) / 2
    e.size_x = max(d.length / 2, 1e-4)
    e.radius = r
    e.stiffness = stiff
    q = Vector((1, 0, 0)).rotation_difference(d.normalized())
    e.rotation = q


def ellipsoid(mb, c, sx, sy, sz, r, rot=None, stiff=2.0):
    e = mb.elements.new(type="ELLIPSOID")
    e.co = c
    e.size_x, e.size_y, e.size_z = sx, sy, sz
    e.radius = r
    e.stiffness = stiff
    if rot is not None:
        e.rotation = rot
    return e


class Hand:
    def __init__(self):
        mb = bpy.data.metaballs.new("hand")
        mb.resolution = 0.004
        mb.render_resolution = 0.0011
        mb.threshold = 0.6
        self.mb = mb
        self.obj = bpy.data.objects.new("hand", mb)
        bpy.context.collection.objects.link(self.obj)
        # static parts
        capsule(mb, (0, 0.006, -0.20), (0, 0.0, -0.012), 0.040, 2.2)                       # forearm
        ellipsoid(mb, (0.0, 0.0, 0.046), 0.040, 0.013, 0.042, 0.046, stiff=2.4)           # palm
        ellipsoid(mb, (0.024, -0.007, 0.024), 0.018, 0.012, 0.022, 0.026, stiff=1.8)     # thenar
        ellipsoid(mb, (-0.026, -0.005, 0.034), 0.013, 0.010, 0.028, 0.022, stiff=1.6)    # hypothenar
        for (kx, kz) in [v[0] for v in FINGERS.values()]:                                  # metacarpals
            capsule(mb, (kx * 0.45, 0.0, 0.010), (kx, 0.0, kz - 0.004), 0.0165, 1.6)
        self.fingers = {}
        for name in FINGERS:
            self.fingers[name] = [mb.elements.new(type="CAPSULE") for _ in range(3)]
        self.thumb = [mb.elements.new(type="CAPSULE") for _ in range(3)]
        self.vessels = []

    def apply(self, P):
        wig = lambda k: 0.5 * P["alive"] * math.sin(P["f"] / 7.0 + k * 1.3)
        chains = {}
        for k, (name, els) in enumerate(self.fingers.items()):
            pts = finger_chain(name, P["curl"] * (1 + 0.15 * k), P["spread"], wig(k))
            radii = FINGERS[name][3]
            for i, e in enumerate(els):
                set_capsule(e, pts[i], pts[i + 1], radii[i] * 1.05, 2.8)
            chains[name] = pts
        tp = thumb_chain(P["curl"], wig(5))
        for i, e in enumerate(self.thumb):
            set_capsule(e, tp[i], tp[i + 1], (0.0190, 0.0160, 0.0138)[i], 2.0)
        chains["thumb"] = tp
        return chains


def vessel_paths(chains):
    """Arterial tree on the palm side: radial + ulnar arteries, palmar arch, digital arteries."""
    off = lambda v, d=0.0125: Vector((v.x, v.y - d, v.z))
    paths = []
    radial = [Vector((0.016, -0.019, -0.15)), Vector((0.018, -0.020, -0.07)), Vector((0.020, -0.020, -0.01)),
              Vector((0.022, -0.018, 0.020))]
    ulnar = [Vector((-0.017, -0.019, -0.15)), Vector((-0.019, -0.020, -0.07)), Vector((-0.020, -0.019, -0.01)),
             Vector((-0.020, -0.016, 0.030))]
    arch = [Vector((0.022, -0.018, 0.020))]
    for a in np.linspace(0, 1, 8)[1:]:
        arch.append(Vector((0.022 - 0.042 * a, -0.0165 - 0.002 * math.sin(math.pi * a), 0.020 + 0.034 * math.sin(math.pi * a * 0.9))))
    paths += [radial, ulnar, arch]
    for name in ("index", "middle", "ring", "pinky"):
        pts = chains[name]
        kx = FINGERS[name][0][0]
        start = Vector((kx * 0.8, -0.016, 0.052))
        d = [start] + [off(p, FINGERS[name][3][min(i, 2)] * 1.02) for i, p in enumerate(pts[:-1])]
        d.append(off(pts[-1] - (pts[-1] - pts[-2]) * 0.25, FINGERS[name][3][2] * 0.95))
        paths.append(d)
    tp = chains["thumb"]
    paths.append([Vector((0.022, -0.018, 0.020))] + [off(p, 0.0118) for p in tp[1:-1]] + [off(tp[-1] - (tp[-1] - tp[-2]) * 0.3, 0.010)])
    return paths


def make_vessel_objects(n, mat):
    objs = []
    for i in range(n):
        cu = bpy.data.curves.new(f"v{i}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.0019 if i < 3 else 0.0014
        cu.bevel_resolution = 4
        cu.use_fill_caps = True
        sp = cu.splines.new("NURBS")
        sp.order_u = 3
        sp.use_endpoint_u = True
        o = bpy.data.objects.new(f"vessel{i}", cu)
        bpy.context.collection.objects.link(o)
        o.data.materials.append(mat)
        objs.append(o)
    return objs


def densify(P, n=24):
    P = np.asarray(P)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, seg[-1], n)
    return np.stack([np.interp(u, seg, P[:, k]) for k in range(3)], 1)


def project(paths, hand_obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    bvh = BVHTree.FromObject(hand_obj, dg)
    out = []
    for pts in paths:
        P = densify(np.array([tuple(p) for p in pts]))
        Q = []
        for p in P:
            o = Vector(p) + Vector((0, -0.12, 0))
            loc, nrm, _, _ = bvh.ray_cast(o, Vector((0, 1, 0)), 0.3)
            Q.append(tuple(loc + nrm * 0.0009) if loc is not None else tuple(p))
        out.append(Q)
    return out


def update_vessels(objs, paths):
    for o, pts in zip(objs, paths):
        P = np.array([tuple(p) for p in pts])
        sp = o.data.splines[0]
        need = len(P) - len(sp.points)
        if need > 0:
            sp.points.add(need)
        for i, p in enumerate(P):
            sp.points[i].co = (p[0], p[1], p[2], 1.0)


# ------------------------------------------------------------------ materials
class NT:
    def __init__(self, m):
        self.N, self.L = m.node_tree.nodes, m.node_tree.links

    def new(self, kind, **inputs):
        n = self.N.new(kind)
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(n, k[1:], v)
            else:
                n.inputs[k].default_value = v
        return n

    def math(self, op, a, b=None):
        n = self.N.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                self.L.new(v, n.inputs[i])
        return n.outputs[0]

    def mix(self, fac, a, b):
        n = self.N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        for key, v in (("Factor", fac), ("A", a), ("B", b)):
            if isinstance(v, (tuple, float, int)):
                n.inputs[key].default_value = v
            else:
                self.L.new(v, n.inputs[key])
        return n.outputs["Result"]


def skin_material():
    """Pale porcelain mannequin: pores, fine scratches, mottled grime; warms with a red subsurface when alive."""
    m = bpy.data.materials.new("Skin")
    g = NT(m)
    b = g.N["Principled BSDF"]
    alive = g.new("ShaderNodeValue")
    alive.name = "alive"
    tc = g.new("ShaderNodeTexCoord")
    P = tc.outputs["Object"]
    mott = g.new("ShaderNodeTexNoise", Scale=38.0, Detail=6.0, Roughness=0.6)
    g.L.new(P, mott.inputs["Vector"])
    grime = g.new("ShaderNodeTexNoise", Scale=140.0, Detail=8.0, Roughness=0.7)
    g.L.new(P, grime.inputs["Vector"])
    cold = g.mix(g.math("MULTIPLY", mott.outputs["Fac"], 0.5), lin((196, 195, 192)), lin((236, 235, 232)))
    cold = g.mix(g.math("MULTIPLY", g.math("GREATER_THAN", grime.outputs["Fac"], 0.7), 0.3), cold, lin((150, 148, 146)))
    warm = g.mix(g.math("MULTIPLY", mott.outputs["Fac"], 0.5), lin((222, 196, 194)), lin((244, 226, 222)))
    g.L.new(g.mix(alive.outputs[0], cold, warm), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.42
    b.inputs["Coat Weight"].default_value = 0.35
    b.inputs["Coat Roughness"].default_value = 0.22
    b.inputs["Subsurface Radius"].default_value = (0.004, 0.0012, 0.0008)
    b.inputs["Subsurface Scale"].default_value = 0.6
    g.L.new(g.math("MULTIPLY", alive.outputs[0], 0.35), b.inputs["Subsurface Weight"])
    # micro relief: pores + hairline scratches
    pores = g.new("ShaderNodeTexNoise", Scale=900.0, Detail=2.0)
    g.L.new(P, pores.inputs["Vector"])
    scr = g.new("ShaderNodeTexVoronoi", _feature="DISTANCE_TO_EDGE", Scale=90.0)
    g.L.new(P, scr.inputs["Vector"])
    scratch = g.math("LESS_THAN", scr.outputs["Distance"], 0.012)
    h = g.math("ADD", g.math("MULTIPLY", pores.outputs["Fac"], 0.6), g.math("MULTIPLY", scratch, -0.5))
    bump = g.new("ShaderNodeBump", Strength=0.12, Distance=0.0004)
    g.L.new(h, bump.inputs["Height"])
    g.L.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m, alive


def seam_material():
    m = bpy.data.materials.new("Seam")
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin((18, 18, 20))
    b.inputs["Roughness"].default_value = 0.5
    return m


def string_material():
    m = bpy.data.materials.new("String")
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin((200, 200, 200))
    b.inputs["Roughness"].default_value = 0.3
    return m


def vessel_material():
    """Raised pale vessels; blood fills outward from the wrist (crimson, glossy, pulsing faintly)."""
    m = bpy.data.materials.new("Vessel")
    g = NT(m)
    b = g.N["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.16
    b.inputs["Coat Weight"].default_value = 0.8
    b.inputs["Coat Roughness"].default_value = 0.05
    geo = g.new("ShaderNodeNewGeometry")
    tf = g.new("ShaderNodeVectorTransform", _convert_from="WORLD", _convert_to="OBJECT")
    g.L.new(geo.outputs["Position"], tf.inputs["Vector"])
    dist = g.new("ShaderNodeVectorMath", _operation="DISTANCE")
    g.L.new(tf.outputs["Vector"], dist.inputs[0])
    dist.inputs[1].default_value = (0.0, -0.02, -0.17)
    front = g.new("ShaderNodeValue")
    front.name = "front"
    phase = g.new("ShaderNodeValue")
    phase.name = "phase"
    d = dist.outputs["Value"]
    filled = g.math("GREATER_THAN", front.outputs[0], d)
    edge = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", front.outputs[0], d)), 0.010)
    pulse = g.math("ADD", 0.5, g.math("MULTIPLY", 0.5, g.math("SINE", g.math("SUBTRACT", g.math("MULTIPLY", d, 90.0),
                                                                                       phase.outputs[0]))))
    g.L.new(g.math("ADD", g.math("MULTIPLY", g.math("MULTIPLY", filled, pulse), 0.7), g.math("MULTIPLY", edge, 3.5)),
            b.inputs["Emission Strength"])
    b.inputs["Emission Color"].default_value = lin((255, 10, 36))
    g.L.new(g.mix(filled, lin((176, 174, 172)), lin((120, 0, 14))), b.inputs["Base Color"])
    return m, front, phase


def drape_material():
    """Crimson latex/silk: glossy coat, fine weave bump, darker in the folds."""
    m = bpy.data.materials.new("Drape")
    g = NT(m)
    b = g.N["Principled BSDF"]
    tc = g.new("ShaderNodeTexCoord")
    P = tc.outputs["Object"]
    n = g.new("ShaderNodeTexNoise", Scale=6.0, Detail=4.0)
    g.L.new(P, n.inputs["Vector"])
    g.L.new(g.mix(n.outputs["Fac"], lin((70, 0, 8)), lin((150, 4, 20))), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.28
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.08
    weave = g.new("ShaderNodeTexWave", Scale=420.0, _wave_type="BANDS")
    g.L.new(P, weave.inputs["Vector"])
    bump = g.new("ShaderNodeBump", Strength=0.05, Distance=0.0002)
    g.L.new(weave.outputs["Fac"], bump.inputs["Height"])
    g.L.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


# ------------------------------------------------------------------ set dressing
def drape(name, x0, x1, y, mat, seed):
    rng = np.random.default_rng(seed)
    nx, nz = 160, 120
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(-0.45, 0.6, nz)
    k = rng.uniform(0, 6.28, 4)
    verts = []
    for z in zs:
        for x in xs:
            g = 1 + 0.9 * (0.6 - z)                                   # folds gather toward the top
            fold = (0.045 * np.sin(x * 26 * g + k[0] + 0.6 * np.sin(z * 3 + k[1])) +
                    0.018 * np.sin(x * 61 * g + k[2] + 1.5 * z) + 0.010 * np.sin(x * 9 + z * 5 + k[3]))
            sag = 0.05 * np.exp(-((z - 0.3) / 0.3) ** 2) * np.sin(x * 7 + k[1])
            verts.append((x, y + fold + sag, z))
    faces = []
    for j in range(nz - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.append((a, a + 1, a + nx + 1, a + nx))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    for pl in me.polygons:
        pl.use_smooth = True
    o.data.materials.append(mat)
    return o


def ring_objects(n, mat):
    objs = []
    for i in range(n):
        cu = bpy.data.curves.new(f"seam{i}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.00045
        cu.bevel_resolution = 2
        sp = cu.splines.new("POLY")
        sp.points.add(23)
        sp.use_cyclic_u = True
        o = bpy.data.objects.new(f"seam{i}", cu)
        bpy.context.collection.objects.link(o)
        o.data.materials.append(mat)
        objs.append(o)
    return objs


def joint_frames(chains):
    """(point, axis) for every seam ring: finger PIP/DIP, thumb, knuckle line, wrist."""
    out = []
    for name in ("index", "middle", "ring", "pinky", "thumb"):
        pts = chains[name]
        for i in (1, 2):
            out.append((pts[i], (pts[i + 1] - pts[i - 1]).normalized()))
    out.append((Vector((0.0, 0.002, -0.022)), Vector((0, 0, 1))))
    out.append((Vector((0.0, 0.003, -0.085)), Vector((0, 0, 1))))
    return out


def update_rings(objs, frames, hand_obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    bvh = BVHTree.FromObject(hand_obj, dg)
    for o, (p, ax) in zip(objs, frames):
        u = ax.orthogonal().normalized()
        v = ax.cross(u).normalized()
        sp = o.data.splines[0]
        for k in range(24):
            a = 2 * math.pi * k / 24
            dvec = u * math.cos(a) + v * math.sin(a)
            hit, nrm, _, _ = bvh.ray_cast(p, dvec, 0.03)          # from inside the finger -> its own skin
            q = hit + dvec * 0.0002 if hit is not None else p + dvec * 0.012
            sp.points[k].co = (q.x, q.y, q.z, 1.0)


def string_objects(n, mat):
    objs = []
    for i in range(n):
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.00045, depth=1.0)
        o = bpy.context.active_object
        o.data.materials.append(mat)
        objs.append(o)
    return objs


def update_strings(objs, chains, visible):
    tips = [chains[n][-1] for n in ("index", "middle", "ring", "pinky", "thumb")] + [Vector((0.0, 0.004, -0.05))]
    for o, tpt in zip(objs, tips):
        top = Vector((tpt.x * 1.4, tpt.y + 0.02, 0.9))
        d = top - tpt
        o.location = (tpt + top) / 2
        o.scale = (1, 1, d.length)
        o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        o.hide_render = not visible


# ------------------------------------------------------------------ scene
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
    sc.cycles.samples = 128
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPTIX"
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "AgX - Punchy", "Punchy"):
        try:
            sc.view_settings.look = look
            break
        except Exception:
            pass
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0015, 0.0012, 0.0012, 1)
    lights = {}
    for name, kind, loc, aim, energy, color, size in (
            ("key", "SPOT", (0.04, -0.14, 0.66), (0.0, 0.0, 0.07), 20.0, (1.0, 0.97, 0.94), 0.015),
            ("drapeL", "SPOT", (-0.75, 0.02, 0.55), (-0.25, 0.22, 0.1), 16.0, (1.0, 0.92, 0.92), 0.04),
            ("drapeR", "SPOT", (0.85, -0.02, 0.40), (0.35, 0.25, 0.1), 12.0, (1.0, 0.92, 0.92), 0.04),
            ("rim", "AREA", (-0.30, 0.18, 0.30), (0.0, 0.0, 0.05), 2.0, (0.85, 0.9, 1.0), 0.06),
            ("bounce", "AREA", (0.10, -0.15, -0.30), (0.0, 0.0, 0.0), 0.8, (1.0, 0.12, 0.15), 0.4)):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = color
        if kind == "SPOT":
            ld.spot_size = math.radians(30 if name == "key" else 50)
            ld.spot_blend = 0.7
            ld.shadow_soft_size = size
        else:
            ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(lo)
        lights[name] = ld
    dm = drape_material()
    drape("drapeL", -0.55, -0.045, 0.20, dm, 1)
    drape("drapeR", 0.115, 0.6, 0.23, dm, 2)
    cd = bpy.data.cameras.new("cam")
    cd.lens = 50
    co = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(co)
    sc.camera = co
    return co, lights


def set_camera(co, f):
    k = f / (N - 1)
    ang = math.radians(-28 + 18 * k)
    dist = 0.52 - 0.15 * smooth(f, 62, 89)
    target = Vector((0.004, 0.0, 0.085 - 0.085 * smooth(f, 62, 89)))
    co.location = target + Vector((math.sin(ang) * dist, -math.cos(ang) * dist, -0.13 + 0.03 * k))
    co.rotation_euler = (target - co.location).to_track_quat("-Z", "Y").to_euler()
    co.rotation_euler.rotate_axis("Z", math.radians(-9))
    co.data.shift_x = -0.17


def blood_front(f):
    """Arterial front radius (m): surges on each heartbeat."""
    R = 0.12 * smooth(f, 24, 30)                 # forearm (off-screen) primes just before the first beat
    for b in BEATS:                               # wrist -> palm -> fingers -> fingertips
        R += 0.072 * smooth(f, b, b + 7)
    return R


def heartbeat(f):
    return sum(math.exp(-((f - b) / 2.5) ** 2) for b in BEATS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cam, L = setup()
    hand = Hand()
    skin, alive_v = skin_material()
    hand.obj.data.materials.append(skin)
    vmat, front, phase = vessel_material()
    chains = hand.apply(pose(0))
    vobjs = make_vessel_objects(len(vessel_paths(chains)), vmat)
    rings = ring_objects(len(joint_frames(chains)), seam_material())
    strings = string_objects(6, string_material())
    sc = bpy.context.scene
    frames = [0, 38, 56, 80] if MODE == "test" else range(int(ARGS[1]), int(ARGS[2]))
    for f in frames:
        P = pose(f)
        chains = hand.apply(P)
        update_vessels(vobjs, project(vessel_paths(chains), hand.obj))
        update_rings(rings, joint_frames(chains), hand.obj)
        update_strings(strings, chains, visible=f < 45)        # the strings snap as it comes alive
        front.outputs[0].default_value = blood_front(f)
        phase.outputs[0].default_value = f * 0.9
        alive_v.outputs[0].default_value = P["alive"]
        L["key"].energy = 20.0 * (1 + 0.2 * heartbeat(f))
        set_camera(cam, f)
        sc.render.filepath = str(OUT / (f"test_{f:03d}.png" if MODE == "test" else f"hand_{f:03d}.png"))
        bpy.ops.render.render(write_still=True)
