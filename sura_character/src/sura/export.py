"""Demo actions, LODs and file export (GLB / FBX / .blend)."""
import math

import bpy

from .butil import set_active

FPS = 30


def _key_rot(pb, frame, x=0.0, y=0.0, z=0.0):
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (math.radians(x), math.radians(y), math.radians(z))
    pb.keyframe_insert("rotation_euler", frame=frame)


def _new_action(arm, name):
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
    return act


FINGER_CURL = {1: 75, 2: 85, 3: 60}
_THUMB = {}


def solve_thumb(arm, side):
    """Thumb rotations for a closed fist, solved once with a temporary IK constraint."""
    if side in _THUMB:
        return _THUMB[side]
    pb = arm.pose.bones
    saved = {b.name: b.rotation_euler.copy() for b in pb}
    for f in ("index", "middle", "ring", "pinky"):
        for k in (1, 2, 3):
            b = pb[f"{f}_{k:02d}_{side}"]
            b.rotation_mode = "XYZ"
            b.rotation_euler = (math.radians(FINGER_CURL[k]), 0, 0)
    for k in (1, 2, 3):
        pb[f"thumb_{k:02d}_{side}"].rotation_mode = "XYZ"
        pb[f"thumb_{k:02d}_{side}"].rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    mw = arm.matrix_world
    i2 = pb[f"index_02_{side}"]
    m2 = pb[f"middle_02_{side}"]
    target = (i2.head * 0.65 + m2.head * 0.35)
    palm = (pb[f"middle_metacarpal_{side}"].head + pb[f"index_metacarpal_{side}"].head) * 0.5
    out = (target - palm).normalized()
    target = target + out * 0.012
    emp = bpy.data.objects.new("_thumb_target", None)
    bpy.context.scene.collection.objects.link(emp)
    emp.location = mw @ target
    c = pb[f"thumb_03_{side}"].constraints.new("IK")
    c.target = emp
    c.chain_count = 3
    c.use_tail = True
    bpy.context.view_layer.update()
    res = []
    for k in (1, 2, 3):
        b = pb[f"thumb_{k:02d}_{side}"]
        loc = arm.convert_space(pose_bone=b, matrix=b.matrix, from_space="POSE", to_space="LOCAL")
        res.append(tuple(math.degrees(a) for a in loc.to_euler("XYZ")))
    pb[f"thumb_03_{side}"].constraints.remove(c)
    bpy.data.objects.remove(emp)
    for b in pb:
        b.rotation_euler = saved[b.name]
    bpy.context.view_layer.update()
    _THUMB[side] = res
    return res


def fist(arm, frame, side, amount):
    pb = arm.pose.bones
    for f in ("index", "middle", "ring", "pinky"):
        for k in (1, 2, 3):
            _key_rot(pb[f"{f}_{k:02d}_{side}"], frame, x=amount * FINGER_CURL[k])
    for k, rot in enumerate(solve_thumb(arm, side), 1):
        _key_rot(pb[f"thumb_{k:02d}_{side}"], frame, *(amount * a for a in rot))


def make_actions(arm):
    """Idle (breathing, demon claws flexing, hair sway) and a Shaolin ready stance."""
    pb = arm.pose.bones
    acts = []
    # --- Idle: 4 s loop ---------------------------------------------------------------
    act = _new_action(arm, "Idle")
    n = 4 * FPS
    for f in range(0, n + 1, 5):
        t = f / n * 2 * math.pi
        br = math.sin(t)
        _key_rot(pb["spine_03"], f, x=1.2 * br)
        _key_rot(pb["spine_05"], f, x=-0.8 * br)
        _key_rot(pb["neck_01"], f, x=-0.5 * br)
        _key_rot(pb["head"], f, x=0.6 * math.sin(t + 0.6), z=1.5 * math.sin(t * 0.5))
        _key_rot(pb["clavicle_l"], f, z=0.8 * br)
        _key_rot(pb["clavicle_r"], f, z=-0.8 * br)
        _key_rot(pb["upperarm_l"], f, x=-2 * br)
        _key_rot(pb["upperarm_r"], f, x=-2 * br)
        flex = 0.5 + 0.5 * math.sin(t * 2)
        fist(arm, f, "l", 0.15 + 0.35 * flex)
        fist(arm, f, "r", 0.25)
        for chain in ("hair_l", "hair_c", "hair_r"):
            for k in range(1, 5):
                _key_rot(pb[f"{chain}_{k:02d}"], f, x=2.0 * k * math.sin(t - k * 0.5), z=1.2 * k * math.sin(t * 0.5 - k * 0.4))
    acts.append(act)
    # --- Stance: arms down, fists clenched, weight settled (like the reference photo) ------
    act = _new_action(arm, "Stance")
    for f in (0, 30):
        _key_rot(pb["upperarm_l"], f, x=8, z=38)
        _key_rot(pb["upperarm_r"], f, x=8, z=-38)
        _key_rot(pb["lowerarm_l"], f, x=-14)
        _key_rot(pb["lowerarm_r"], f, x=-14)
        _key_rot(pb["hand_l"], f, z=0)
        fist(arm, f, "l", 0.55)
        fist(arm, f, "r", 1.0)
        _key_rot(pb["spine_04"], f, x=2)
        _key_rot(pb["head"], f, x=-3)
        _key_rot(pb["thigh_l"], f, z=-3)
        _key_rot(pb["thigh_r"], f, z=3)
    acts.append(act)
    # rest
    arm.animation_data.action = None
    for p in pb:
        p.rotation_euler = (0, 0, 0)
    for a in acts:
        tr = arm.animation_data.nla_tracks.new()
        tr.name = a.name
        tr.strips.new(a.name, 0, a)
        tr.mute = True
    return acts


def select_export(objs):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def export_glb(path, objs, image_format="AUTO", jpeg_quality=90, animations=True):
    select_export(objs)
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True,
        export_image_format=image_format, export_jpeg_quality=jpeg_quality,
        export_texcoords=True, export_normals=True, export_tangents=False,
        export_skins=True, export_all_influences=False, export_morph=True, export_morph_normal=False,
        export_try_sparse_sk=True, export_animations=animations, export_animation_mode="ACTIONS",
        export_yup=True, export_apply=False, export_extras=False)
    return path


def export_fbx(path, objs, animations=True):
    select_export(objs)
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"ARMATURE", "MESH"},
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE", add_leaf_bones=False,
        primary_bone_axis="Y", secondary_bone_axis="X", armature_nodetype="NULL",
        mesh_smooth_type="FACE", use_tspace=True, use_mesh_modifiers=False,
        bake_anim=animations, bake_anim_use_all_actions=animations, bake_anim_use_nla_strips=False,
        path_mode="RELATIVE", embed_textures=False)
    return path


def make_lod(objs, arm, ratio, suffix):
    """Decimated copies (no blendshapes) for distance LODs."""
    out = []
    for o in objs:
        c = o.copy()
        c.data = o.data.copy()
        c.name = o.name + suffix
        bpy.context.scene.collection.objects.link(c)
        if c.data.shape_keys:
            set_active(c)
            bpy.ops.object.shape_key_remove(all=True)
        r = ratio if "Hair" not in o.name else max(ratio, 0.5)
        dec = c.modifiers.new("dec", "DECIMATE")
        dec.ratio = r
        dec.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = c
        c.modifiers.move(len(c.modifiers) - 1, 0)
        set_active(c)
        bpy.ops.object.modifier_apply(modifier="dec")
        c.parent = arm
        out.append(c)
    return out


def tri_count(objs):
    t = 0
    for o in objs:
        o.data.calc_loop_triangles()
        t += len(o.data.loop_triangles)
    return t
