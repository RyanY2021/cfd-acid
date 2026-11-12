"""The ending: an old beige computer shows the CFD card, crashes, sparks, and smokes (Mantaflow).

  blender -b --factory-startup --python computer/build_computer.py -- test | frames
Screen content: render/crt_screen/screen_0001.png.. (engine/make_screen.py).  Output: render/crt/crt_000.png..
Units: metres.  Desk top at z=0, computer front faces -Y.  240 frames @ 30 fps; crash at frame 100.
"""
import math
import pathlib
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "render" / "crt"
SCREEN = ROOT / "render" / "crt_screen"
CACHE = ROOT / "cache" / "smoke"
FONT = ROOT / "assets" / "fonts"
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["test"]
MODE = ARGS[0]
N = 240
CRASH = 100


def lin(rgb):
    return tuple(((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92 for c in rgb) + (1.0,)


def smooth(x, a, b):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def mat(name, rgb, rough=0.45, emission=None, estr=0.0, bump=0.0, metallic=0.0):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin(rgb)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if emission is not None:
        b.inputs["Emission Color"].default_value = lin(emission)
        b.inputs["Emission Strength"].default_value = estr
    if bump:
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 900.0
        tex.inputs["Detail"].default_value = 2.0
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump
        nt.links.new(tex.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def box(name, center, size, material, bevel=0.0, seg=3):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        b = o.modifiers.new("bevel", "BEVEL")
        b.width = bevel
        b.segments = seg
        b.limit_method = "ANGLE"
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o


def text(body, loc, size, material, font="Inter-700.ttf", rot=(math.pi / 2, 0, 0), extrude=0.0004):
    cu = bpy.data.curves.new(body, "FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(str(FONT / font))
    cu.size = size
    cu.extrude = extrude
    cu.align_x = "LEFT"
    o = bpy.data.objects.new(body, cu)
    o.location = loc
    o.rotation_euler = rot
    bpy.context.collection.objects.link(o)
    o.data.materials.append(material)
    return o


# ------------------------------------------------------------------ model
P = dict(x0=-0.21, x1=0.21, yf=-0.12, yb=0.06, z0=0.025, z1=0.40)
SCR_C = Vector((0.0, -0.1215, 0.262))
SCR_W, SCR_H = 0.272, 0.204


def build_computer():
    beige = mat("Beige", (150, 142, 124), 0.45, bump=0.04)
    beige2 = mat("BeigeDark", (120, 112, 96), 0.5, bump=0.03)
    dark = mat("DarkSlot", (22, 21, 20), 0.6)
    keycap = mat("Key", (206, 196, 170), 0.5, bump=0.03)
    keydk = mat("KeyDark", (128, 120, 106), 0.5)
    ink = mat("Ink", (70, 64, 56), 0.6)
    vent_glow = mat("VentGlow", (30, 28, 26), 0.6, emission=(255, 110, 30), estr=0.0)
    led = mat("LED", (40, 255, 80), 0.3, emission=(60, 255, 90), estr=6.0)
    xm = (P["x0"] + P["x1"]) / 2
    # main housing + bezel step
    hw, hd, hh = P["x1"] - P["x0"], P["yb"] - P["yf"], P["z1"] - P["z0"]
    house = box("housing", (xm, (P["yf"] + P["yb"]) / 2, (P["z0"] + P["z1"]) / 2), (hw, hd, hh), beige, 0.018, 5)
    # CRT rear bulge (tapered)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        if v.co.y > 0:
            v.co.x *= 0.62
            v.co.z *= 0.6
            v.co.z += 0.03
    me = bpy.data.meshes.new("rear")
    bm.to_mesh(me)
    rear = bpy.data.objects.new("rear", me)
    bpy.context.collection.objects.link(rear)
    rear.scale = (0.36, 0.26, 0.30)
    rear.location = (xm, P["yb"] + 0.12, 0.235)
    b = rear.modifiers.new("bevel", "BEVEL")
    b.width, b.segments = 0.03, 5
    rear.data.materials.append(beige)
    for p in me.polygons:
        p.use_smooth = True
    # bezel recess frame around the tube
    box("bezel", (xm, P["yf"] + 0.002, SCR_C.z), (SCR_W + 0.05, 0.006, SCR_H + 0.05), beige2, 0.012, 4)
    # curved glass face (bulged grid)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=48, y_subdivisions=36, size=1)
    g = bpy.context.active_object
    g.name = "screen"
    for v in g.data.vertices:
        u, w = v.co.x * 2, v.co.y * 2
        r = 0.03
        ax = max(abs(u) * SCR_W / 2 - (SCR_W / 2 - r), 0)
        aw = max(abs(w) * SCR_H / 2 - (SCR_H / 2 - r), 0)
        if ax * ax + aw * aw > r * r:                     # round the corners
            s = r / math.sqrt(ax * ax + aw * aw)
            ax2, aw2 = ax * s, aw * s
            v.co.x = math.copysign((SCR_W / 2 - r + ax2) / SCR_W, u)
            v.co.y = math.copysign((SCR_H / 2 - r + aw2) / SCR_H, w)
        v.co.z = 0.012 * (1 - (2 * v.co.x) ** 2) * (1 - (2 * v.co.y) ** 2)
    g.scale = (SCR_W, SCR_H, 1.0)
    g.rotation_euler = (math.pi / 2, 0, 0)
    g.location = (SCR_C.x, SCR_C.y - 0.001, SCR_C.z)
    bpy.ops.object.shade_smooth()
    g.data.materials.append(screen_material())
    # floppy drive, badge, LED
    box("floppy", (0.11, P["yf"] - 0.002, 0.075), (0.11, 0.004, 0.012), dark, 0.002)
    box("floppy_btn", (0.18, P["yf"] - 0.003, 0.075), (0.016, 0.006, 0.01), beige2, 0.002)
    box("led", (-0.17, P["yf"] - 0.003, 0.075), (0.008, 0.004, 0.004), led, 0.0015)
    text("CFD-86", (-0.18, P["yf"] - 0.003, 0.11), 0.016, ink, "Tektur-900.ttf")
    text("FLOW STATION", (-0.18, P["yf"] - 0.003, 0.094), 0.0075, ink, "JetBrainsMono-700.ttf")
    # side and top vents
    for k in range(9):
        box(f"svent{k}", (P["x1"] + 0.001, -0.06 + k * 0.012, 0.30), (0.004, 0.005, 0.07), dark, 0.0015)
        box(f"svent_l{k}", (P["x0"] - 0.001, -0.06 + k * 0.012, 0.30), (0.004, 0.005, 0.07), dark, 0.0015)
    vents = []
    for k in range(7):
        vents.append(box(f"tvent{k}", (0.0, -0.035 + k * 0.012, P["z1"] + 0.0005), (0.17, 0.0055, 0.004),
                         vent_glow, 0.0015))
    # feet
    for sx in (-1, 1):
        for sy in (-1, 1):
            box("foot", (sx * 0.17, -0.03 + sy * 0.07, 0.012), (0.04, 0.03, 0.024), dark, 0.004)
    # keyboard
    kb = box("keyboard", (0.0, -0.36, 0.016), (0.44, 0.17, 0.03), beige, 0.01, 4)
    kb.rotation_euler.x = math.radians(4)
    rows = [(13, 0.0), (13, 0.006), (12, 0.012), (11, 0.02)]
    for r, (nk, off) in enumerate(rows):
        for i in range(nk):
            x = -0.19 + off + i * 0.0295
            y = -0.405 + r * 0.03
            dk = (i == 0 or i == nk - 1)
            box("key", (x + 0.012, y, 0.036 + r * 0.002), (0.025, 0.025, 0.012), keydk if dk else keycap, 0.003, 2)
    box("space", (0.0, -0.435, 0.034), (0.2, 0.025, 0.012), keycap, 0.003, 2)
    # cable
    pts = [(0.0, P["yb"] + 0.24, 0.12), (0.05, 0.45, 0.02), (0.25, 0.55, 0.005), (0.6, 0.62, 0.005)]
    cu = bpy.data.curves.new("cable", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.006
    sp = cu.splines.new("NURBS")
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (*p, 1)
    sp.use_endpoint_u = True
    sp.order_u = 3
    co = bpy.data.objects.new("cable", cu)
    bpy.context.collection.objects.link(co)
    co.data.materials.append(dark)
    return house, vents, led, vent_glow


def screen_material():
    m = bpy.data.materials.new("Screen")
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    out = N["Material Output"]
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin((6, 8, 8))
    b.inputs["Roughness"].default_value = 0.06
    b.inputs["Coat Weight"].default_value = 0.35
    img = bpy.data.images.load(str(SCREEN / "screen_0001.jpg"))
    img.source = "SEQUENCE"
    tex = N.new("ShaderNodeTexImage")
    tex.image = img
    tex.image_user.frame_duration = N_FRAMES
    tex.image_user.frame_start = 1
    tex.image_user.frame_offset = 0
    tex.image_user.use_auto_refresh = True
    tex.interpolation = "Linear"
    uv = N.new("ShaderNodeTexCoord")
    L.new(uv.outputs["UV"], tex.inputs["Vector"])
    # scanlines
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 720 * math.pi
    L.new(sep.outputs["Y"], mul.inputs[0])
    sn = N.new("ShaderNodeMath")
    sn.operation = "SINE"
    L.new(mul.outputs[0], sn.inputs[0])
    sq = N.new("ShaderNodeMath")
    sq.operation = "MULTIPLY_ADD"
    L.new(sn.outputs[0], sq.inputs[0])
    sq.inputs[1].default_value = 0.22
    sq.inputs[2].default_value = 0.78
    em = N.new("ShaderNodeVectorMath")
    em.operation = "SCALE"
    L.new(tex.outputs["Color"], em.inputs[0])
    L.new(sq.outputs[0], em.inputs["Scale"])
    L.new(em.outputs[0], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 1.28     # mean scanline factor 0.78 -> texture at 1:1
    return m


N_FRAMES = N


def smoke(house):
    """Mantaflow gas domain above the top vents; flow switched on at the crash."""
    CACHE.mkdir(parents=True, exist_ok=True)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.0, 0.05, 0.92))
    dom = bpy.context.active_object
    dom.name = "smoke_domain"
    dom.scale = (1.1, 1.0, 1.1)
    bpy.ops.object.transform_apply(scale=True)
    md = dom.modifiers.new("Fluid", "FLUID")
    md.fluid_type = "DOMAIN"
    ds = md.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = 120
    ds.cache_directory = str(CACHE)
    ds.cache_type = "REPLAY"
    ds.cache_frame_start = 1
    ds.cache_frame_end = N
    ds.vorticity = 0.55
    ds.alpha = 0.6
    ds.beta = 1.4
    ds.use_dissolve_smoke = True
    ds.dissolve_speed = 260
    ds.use_dissolve_smoke_log = True
    for side in ("top", "left", "right", "front", "back"):
        setattr(ds, f"use_collision_border_{side}", False)       # let the plume leave the box
    m = bpy.data.materials.new("Smoke")
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = lin((120, 118, 116))
    pv.inputs["Density"].default_value = 60.0
    pv.inputs["Anisotropy"].default_value = 0.35
    nt.links.new(pv.outputs["Volume"], nt.nodes["Material Output"].inputs["Volume"])
    dom.data.materials.append(m)
    # emitter just above the top vents
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.0, 0.001, P["z1"] + 0.012))
    fl = bpy.context.active_object
    fl.name = "smoke_flow"
    fl.scale = (0.17, 0.09, 0.016)
    bpy.ops.object.transform_apply(scale=True)
    fm = fl.modifiers.new("Fluid", "FLUID")
    fm.fluid_type = "FLOW"
    fs = fm.flow_settings
    fs.flow_type = "SMOKE"
    fs.flow_behavior = "INFLOW"
    fs.flow_source = "MESH"
    fs.surface_distance = 1.5
    fs.smoke_color = (0.2, 0.2, 0.2)
    fs.temperature = 3.0
    fs.density = 1.0
    fs.use_initial_velocity = True
    fs.velocity_coord = (0.0, 0.0, 0.6)
    fl.hide_render = True
    em = house.modifiers.new("Fluid", "FLUID")
    em.fluid_type = "EFFECTOR"
    return dom, fs


def sparks():
    spark = bpy.data.objects.new("spark_src", bpy.data.meshes.new("spark"))
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.0022)
    bm.to_mesh(spark.data)
    bpy.context.collection.objects.link(spark)
    spark.data.materials.append(mat("Spark", (255, 220, 160), 0.3, emission=(255, 190, 110), estr=180.0))
    spark.location = (0, 0, -5)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0.0, 0.0, P["z1"] + 0.004))
    em = bpy.context.active_object
    em.name = "spark_emitter"
    em.scale = (0.15, 0.08, 1)
    bpy.ops.object.transform_apply(scale=True)
    systems = []
    for name, count, a, b, speed, life in (("burst", 520, CRASH, CRASH + 3, 3.2, 16),
                                            ("tail", 90, CRASH + 4, CRASH + 70, 1.6, 12)):
        mod = em.modifiers.new(name, "PARTICLE_SYSTEM")
        st = mod.particle_system.settings
        st.count = count
        st.frame_start, st.frame_end = a, b
        st.lifetime = life
        st.lifetime_random = 0.6
        st.emit_from = "FACE"
        st.normal_factor = speed
        st.factor_random = speed * 0.9
        st.render_type = "OBJECT"
        st.instance_object = spark
        st.particle_size = 1.0
        st.size_random = 0.7
        st.effector_weights.gravity = 1.0
        systems.append(st)
    em.show_instancer_for_render = False
    return em


def lights():
    sc = bpy.context.scene
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.006, 0.012, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    made = {}
    for name, kind, loc, energy, color, size in (
            ("rim", "AREA", (-0.9, 0.9, 1.1), 3.0, (0.45, 0.55, 1.0), 0.5),
            ("back", "SPOT", (0.2, 0.9, 1.9), 60.0, (0.4, 0.55, 1.0), 0.05),
            ("flash", "POINT", (0.0, 0.0, P["z1"] + 0.06), 0.0, (1.0, 0.75, 0.45), 0.02),
            ("ember", "POINT", (0.0, 0.0, P["z1"] + 0.03), 0.0, (1.0, 0.45, 0.15), 0.06)):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = color
        if kind == "AREA":
            ld.size = size
        else:
            ld.shadow_soft_size = size
        if kind == "SPOT":
            ld.spot_size = math.radians(16)
            ld.spot_blend = 0.6
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        if kind in ("AREA", "SPOT"):
            aim = Vector((0, 0.05, 0.95)) if name == "back" else Vector((0, 0, 0.3))
            lo.rotation_euler = (aim - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(lo)
        made[name] = ld
    return made


def scene_set():
    # desk + wall
    desk = box("desk", (0.0, 0.0, -0.02), (3.2, 2.2, 0.04), mat("Desk", (18, 16, 15), 0.42, bump=0.02))
    wall = box("wall", (0.0, 1.05, 0.9), (4.0, 0.04, 2.4), mat("Wall", (10, 10, 12), 0.9))
    return desk, wall


def setup_render(samples):
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
    sc.cycles.max_bounces = 6
    sc.cycles.volume_bounces = 1
    sc.cycles.volume_step_rate = 2.0
    sc.render.use_motion_blur = True
    sc.cycles.film_exposure = 1.0
    sc.render.motion_blur_shutter = 0.6
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.frame_start, sc.frame_end = 1, N
    sc.render.fps = 30


# ------------------------------------------------------------------ camera
def ease(x):
    x = min(1, max(0, x))
    return x * x * x * (x * (6 * x - 15) + 10)


def camera_keys(f):
    """f is 0-based.  Screen-filling close-up -> 3/4 reveal -> slow rise with the smoke."""
    edge_y = SCR_C.y - 0.001                       # plane of the screen's rim
    d0 = SCR_W / (36.0 / 50.0)                     # frame width == screen width
    k0 = (Vector((0.0, edge_y - d0, SCR_C.z)), Vector((0.0, SCR_C.y + 1.0, SCR_C.z)))
    k1 = (Vector((0.5, -1.10, 0.50)), Vector((0.0, -0.06, 0.26)))
    k2 = (Vector((0.70, -1.38, 0.72)), Vector((0.0, 0.03, 0.62)))
    a = ease((f - 10) / 82.0)                      # frames 0-10 hold for the 2D match-dissolve
    r = ease((f - 22) / 70.0)                      # rotation lags the dolly: starts as a straight pull-back
    loc = k0[0].lerp(k1[0], a)
    tgt = k0[1].lerp(k1[1], r)
    b = ease((f - 105) / 130.0)
    loc = loc.lerp(k2[0], b)
    tgt = tgt.lerp(k2[1], b)
    return loc, tgt


def set_camera(cam, f):
    loc, tgt = camera_keys(f)
    shake = 0.0
    if CRASH <= f < CRASH + 12:
        shake = 0.004 * (1 - (f - CRASH) / 12)
    rng = np.random.default_rng(f)
    cam.location = loc + Vector(rng.normal(0, shake, 3))
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()


def animate(f, L, led, vent_glow, fs):
    """Per-frame state (f 0-based)."""
    rng = np.random.default_rng(1000 + f)
    # spark flash
    if CRASH <= f < CRASH + 6:
        L["flash"].energy = float(260 * (1 - (f - CRASH) / 6) * (0.6 + 0.4 * rng.random()))
    elif f in (CRASH + 23, CRASH + 41, CRASH + 58):
        L["flash"].energy = 40.0
    else:
        L["flash"].energy = 0.0
    # embers in the vents, flickering and slowly dying
    if f >= CRASH:
        e = math.exp(-(f - CRASH) / 90) * (0.65 + 0.35 * rng.random())
        vent_glow.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 22 * e
        L["ember"].energy = float(6 * e)
    led_b = led.node_tree.nodes["Principled BSDF"]
    if f < CRASH - 25:
        led_b.inputs["Emission Strength"].default_value = 6.0
    elif f < CRASH:
        led_b.inputs["Emission Strength"].default_value = 6.0 if (f // 3) % 2 else 0.4
        led_b.inputs["Emission Color"].default_value = lin((255, 40, 30))
    else:
        led_b.inputs["Emission Strength"].default_value = 0.0
    fs.density = 0.0 if f < CRASH + 1 else 1.0
    fs.temperature = 0.0 if f < CRASH + 1 else 3.0


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    OUT.mkdir(parents=True, exist_ok=True)
    setup_render(48 if MODE == "test" else 64)
    scene_set()
    house, vents, led, vent_glow = build_computer()
    dom, fs = smoke(house)
    sparks()
    L = lights()
    cd = bpy.data.cameras.new("cam")
    cd.lens = 50
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(cam)
    sc = bpy.context.scene
    sc.camera = cam
    led_mat = led
    want = {0, 40, 70} if MODE == "test" else set(range(N))
    last = max(want)
    if MODE == "test":
        sc.cycles.samples = 32
    for f in range(0, last + 1):
        animate(f, L, led_mat, vent_glow, fs)
        sc.frame_set(f + 1)
        set_camera(cam, f)
        if f in want:
            sc.render.filepath = str(OUT / (f"test_{f:03d}.png" if MODE == "test" else f"crt_{f:03d}.png"))
            bpy.ops.render.render(write_still=True)
    if MODE == "test":
        bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "computer" / "computer.blend"))
