"""Tiny helper DSL for building Blender shader node trees from Python."""
import bpy


class Tree:
    def __init__(self, mat):
        if mat.node_tree is None:
            mat.use_nodes = True
        self.mat = mat
        self.t = mat.node_tree
        self.t.nodes.clear()
        self.x = 0

    def node(self, kind, **props):
        n = self.t.nodes.new(kind)
        n.location = (self.x, 0)
        self.x += 40
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def _set(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.t.links.new(val, sock)
        elif val is not None:
            if isinstance(val, (int, float)) and hasattr(sock, "default_value") and hasattr(sock.default_value, "__len__"):
                n = len(sock.default_value)
                sock.default_value = [val] * n if n != 4 else [val, val, val, 1.0]
            elif isinstance(val, (tuple, list)) and hasattr(sock.default_value, "__len__") and len(sock.default_value) == 4 and len(val) == 3:
                sock.default_value = (*val, 1.0)
            else:
                sock.default_value = val

    def link(self, a, b):
        self.t.links.new(a, b)

    # --- inputs ---------------------------------------------------------------------
    def obj_coords(self):
        return self.node("ShaderNodeTexCoord").outputs["Object"]

    def uv(self):
        return self.node("ShaderNodeTexCoord").outputs["UV"]

    def attr(self, name):
        n = self.node("ShaderNodeAttribute", attribute_name=name)
        return n

    def sep(self, col):
        n = self.node("ShaderNodeSeparateColor")
        self._set(n.inputs[0], col)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def sepxyz(self, vec):
        n = self.node("ShaderNodeSeparateXYZ")
        self._set(n.inputs[0], vec)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def image(self, img, vec=None, non_color=False):
        n = self.node("ShaderNodeTexImage", image=img)
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        if vec is not None:
            self._set(n.inputs["Vector"], vec)
        return n

    # --- math -----------------------------------------------------------------------
    def m(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._set(n.inputs[0], a)
        if b is not None:
            self._set(n.inputs[1], b)
        if c is not None:
            self._set(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b):
        return self.m("ADD", a, b)

    def mul(self, a, b, clamp=False):
        return self.m("MULTIPLY", a, b, clamp=clamp)

    def sub(self, a, b):
        return self.m("SUBTRACT", a, b)

    def inv(self, a):
        return self.m("SUBTRACT", 1.0, a)

    def smooth(self, e0, e1, x):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self._set(n.inputs["Value"], x)
        self._set(n.inputs["From Min"], e0)
        self._set(n.inputs["From Max"], e1)
        return n.outputs["Result"]

    def remap(self, x, a0, a1, b0, b1, clamp=True):
        n = self.node("ShaderNodeMapRange", clamp=clamp)
        self._set(n.inputs["Value"], x)
        self._set(n.inputs["From Min"], a0)
        self._set(n.inputs["From Max"], a1)
        self._set(n.inputs["To Min"], b0)
        self._set(n.inputs["To Max"], b1)
        return n.outputs["Result"]

    def vmul(self, v, s):
        n = self.node("ShaderNodeVectorMath", operation="MULTIPLY")
        self._set(n.inputs[0], v)
        self._set(n.inputs[1], s)
        return n.outputs[0]

    def vadd(self, v, s):
        n = self.node("ShaderNodeVectorMath", operation="ADD")
        self._set(n.inputs[0], v)
        self._set(n.inputs[1], s)
        return n.outputs[0]

    # --- colour -----------------------------------------------------------------------
    def mix(self, fac, a, b, blend="MIX", clamp=True):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend, clamp_factor=clamp)
        self._set(n.inputs[0], fac)
        self._set(n.inputs[6], a)
        self._set(n.inputs[7], b)
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT", clamp_factor=True)
        self._set(n.inputs[0], fac)
        self._set(n.inputs[2], a)
        self._set(n.inputs[3], b)
        return n.outputs[0]

    def rgb(self, c):
        n = self.node("ShaderNodeRGB")
        n.outputs[0].default_value = (*c, 1.0)
        return n.outputs[0]

    # --- textures -------------------------------------------------------------------
    def noise(self, vec, scale, detail=2.0, rough=0.5, distortion=0.0, dims="3D", w=0.0):
        n = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        self._set(n.inputs["Vector"], vec)
        self._set(n.inputs["Scale"], scale)
        self._set(n.inputs["Detail"], detail)
        self._set(n.inputs["Roughness"], rough)
        self._set(n.inputs["Distortion"], distortion)
        if dims == "4D":
            self._set(n.inputs["W"], w)
        return n.outputs["Fac"]

    def voronoi(self, vec, scale, feature="F1", randomness=1.0, out="Distance"):
        n = self.node("ShaderNodeTexVoronoi", feature=feature)
        self._set(n.inputs["Vector"], vec)
        self._set(n.inputs["Scale"], scale)
        self._set(n.inputs["Randomness"], randomness)
        return n.outputs[out]

    def wave(self, vec, scale, kind="BANDS", direction="X", distortion=0.0, detail=0.0, profile="SIN"):
        n = self.node("ShaderNodeTexWave", wave_type=kind, bands_direction=direction, wave_profile=profile)
        self._set(n.inputs["Vector"], vec)
        self._set(n.inputs["Scale"], scale)
        self._set(n.inputs["Distortion"], distortion)
        self._set(n.inputs["Detail"], detail)
        return n.outputs["Fac"]

    def ridged(self, vec, scale, detail=3.0, sharp=6.0, distortion=0.0):
        """Thin ridge lines (veins / creases) from noise zero crossings, 0..1."""
        f = self.noise(vec, scale, detail, 0.55, distortion)
        r = self.m("ABSOLUTE", self.sub(f, 0.5))
        r = self.inv(self.mul(r, 2.0, clamp=True))
        return self.m("POWER", r, sharp)

    # --- outputs ------------------------------------------------------------------------
    def emission_out(self, col):
        em = self.node("ShaderNodeEmission")
        self._set(em.inputs["Color"], col)
        out = self.node("ShaderNodeOutputMaterial")
        self.link(em.outputs[0], out.inputs["Surface"])
        return out

    def bump_out(self, height, distance, strength=1.0):
        b = self.node("ShaderNodeBump")
        self._set(b.inputs["Height"], height)
        self._set(b.inputs["Distance"], distance)
        self._set(b.inputs["Strength"], strength)
        p = self.node("ShaderNodeBsdfPrincipled")
        self.link(b.outputs["Normal"], p.inputs["Normal"])
        out = self.node("ShaderNodeOutputMaterial")
        self.link(p.outputs[0], out.inputs["Surface"])
        return out
