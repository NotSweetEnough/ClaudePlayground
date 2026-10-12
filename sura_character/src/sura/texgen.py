"""Hand-authored-style textures generated with numpy/PIL: hair strand atlas, horn/claw keratin, eyes."""
import colorsys
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .hair import N_COLS


def _lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def hair_atlas(path_color, path_normal, size=2048, seed=3):
    """RGBA strand atlas. Columns 0-3: hair card variants (soft edges), 4: lashes/brows, 5: opaque scalp.
    Roots at the top of the image."""
    from .hair import CAP_COL, LASH_COL
    rng = np.random.RandomState(seed)
    ss = 2
    W = H = size * ss
    col_w = W / N_COLS
    rgb = Image.new("RGB", (W, H), (96, 95, 97))
    alpha = Image.new("L", (W, H), 0)
    dr, da = ImageDraw.Draw(rgb), ImageDraw.Draw(alpha)

    def strand(x, L, w, lum, amp, fr, ph, drift, root_dark=0.6, a_tip=0.35, alpha_on=True, fade=True):
        y0 = rng.uniform(0, 0.07) * H if fade else 0.0            # staggered roots: no straight card edge
        pts = [(x + amp * math.sin(t * fr + ph) + drift * t, y0 + t * (L - y0)) for t in np.linspace(0, 1, 40)]
        tint = np.array([0.985, 0.99, 1.0])
        for k in range(len(pts) - 1):
            t = k / (len(pts) - 1)
            root = min(1.0, root_dark + t * 3)
            c = tuple(int(255 * v) for v in np.clip(tint * lum * root, 0, 1))
            ww = max(1, int(round(w * (1 - 0.6 * t ** 2))))
            dr.line([pts[k], pts[k + 1]], fill=c, width=ww)
            if alpha_on:
                fade_in = min(1.0, (t + 0.01) / 0.05) if fade else 1.0       # cards fade in at the root
                da.line([pts[k], pts[k + 1]], fill=int(255 * fade_in * (1 - a_tip * t ** 3)), width=ww)

    variants = [  # strands, min length, clumps, width range
        (260, 0.85, 0, (1.6, 3.0)),
        (190, 0.70, 0, (1.4, 2.8)),
        (220, 0.78, 8, (1.4, 2.8)),
        (110, 0.55, 0, (1.2, 2.2)),
    ]
    for c, (n, lmin, clumps, wr) in enumerate(variants):
        cx = (c + 0.5) * col_w
        half = col_w * 0.44
        centers = cx + rng.uniform(-half * 0.8, half * 0.8, clumps) if clumps else None
        for i in range(n):
            if clumps:
                x = rng.choice(centers) + rng.normal(0, col_w * 0.045)
            else:
                x = cx + np.clip(rng.normal(0, 0.42), -1, 1) * half      # denser centre, sparse card edges
            x = float(np.clip(x, cx - half, cx + half))
            lum = rng.uniform(0.42, 0.78) if rng.rand() > 0.15 else rng.uniform(0.22, 0.35)
            strand(x, rng.uniform(lmin, 1.0) * H, rng.uniform(*wr) * ss / 2, lum, rng.uniform(1, 6) * ss,
                   rng.uniform(2, 6), rng.uniform(0, 6.28), rng.normal(0, col_w * 0.035))
    # lashes / brows: short, dark-grey, curved
    x0, x1 = LASH_COL * col_w + col_w * 0.06, (LASH_COL + 1) * col_w - col_w * 0.06
    for i in range(80):
        x = rng.uniform(x0, x1)
        L = rng.uniform(0.6, 0.95) * H
        bend = rng.uniform(0.04, 0.10) * col_w
        pts = [(x + bend * t * t, t * L) for t in np.linspace(0, 1, 20)]
        for k in range(len(pts) - 1):
            t = k / 19
            ww = max(1, int(round(2.6 * ss * (1 - 0.8 * t))))
            g = int(100 + 45 * rng.rand())
            dr.line([pts[k], pts[k + 1]], fill=(g, g - 2, g), width=ww)
            da.line([pts[k], pts[k + 1]], fill=255, width=ww)
    # opaque scalp column: dense combed strands over dark grey
    x0, x1 = CAP_COL * col_w, (CAP_COL + 1) * col_w
    dr.rectangle([x0, 0, x1, H], fill=(92, 92, 94))
    da.rectangle([x0, 0, x1, H], fill=255)
    for i in range(700):
        x = rng.uniform(x0, x1)
        strand(x, rng.uniform(0.5, 1.0) * H, rng.uniform(1.2, 2.6) * ss / 2, rng.uniform(0.42, 0.74),
               rng.uniform(1, 4) * ss, rng.uniform(2, 5), rng.uniform(0, 6.28), 0, 0.8, 0, alpha_on=False)
    rgb = rgb.resize((size, size), Image.LANCZOS)
    alpha = alpha.resize((size, size), Image.LANCZOS)
    out = rgb.copy()
    out.putalpha(alpha)
    os.makedirs(os.path.dirname(path_color), exist_ok=True)
    out.save(path_color, optimize=True)
    lum = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255.0
    hgt = 0.6 * lum + 0.4 * np.asarray(alpha.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255.0
    gx = np.gradient(hgt, axis=1) * 3.0
    gy = np.gradient(hgt, axis=0) * 0.5
    n = np.stack([-gx, gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(path_normal, optimize=True)
    return path_color, path_normal


def horn_maps(path_color, path_normal, path_orm, size=512, seed=9):
    """Keratin claws / bone spikes. UV: u around, v from base (0) to tip (1)."""
    rng = np.random.RandomState(seed)
    v, u = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    streak = np.zeros_like(u)
    for k in range(6):
        f = 2 ** k
        streak += np.sin(u * 2 * np.pi * f * 3 + rng.uniform(0, 6) + np.sin(v * 7 + k) * 0.5) / (k + 1)
    rings = np.sin(v * 2 * np.pi * 22 + np.sin(u * 2 * np.pi * 2) * 1.5 + streak * 0.3)
    base = np.array([0.085, 0.009, 0.007])
    mid = np.array([0.025, 0.005, 0.004])
    tip = np.array([0.006, 0.005, 0.005])
    t1 = np.clip(v / 0.45, 0, 1)[..., None]
    t2 = np.clip((v - 0.45) / 0.55, 0, 1)[..., None]
    col = base * (1 - t1) + mid * t1
    col = col * (1 - t2) + tip * t2
    col *= (1 + 0.12 * streak[..., None] / 2 + 0.08 * rings[..., None])
    Image.fromarray((_lin2srgb(col) * 255).astype(np.uint8)[::-1]).save(path_color, optimize=True)
    h = 0.5 + 0.25 * rings + 0.1 * streak / 2
    gx, gy = np.gradient(h, axis=1) * 6, np.gradient(h, axis=0) * 6
    n = np.stack([-gx, gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)[::-1]).save(path_normal, optimize=True)
    rough = 0.42 - 0.2 * v + 0.05 * rings
    orm = np.stack([np.ones_like(v), rough, np.zeros_like(v)], -1)
    Image.fromarray((np.clip(orm, 0, 1) * 255).astype(np.uint8)[::-1]).save(path_orm, optimize=True)
    return path_color, path_normal, path_orm


def eye_texture(src, dst):
    """MakeHuman eye recoloured: molten amber iris with a crimson outer ring, faintly bloodshot sclera."""
    im = np.asarray(Image.open(src).convert("RGBA"), dtype=np.float32) / 255.0
    rgb = im[..., :3]
    out = rgb.copy()
    flat = rgb.reshape(-1, 3)
    hs = np.array([colorsys.rgb_to_hsv(*p) for p in flat]).reshape(rgb.shape)
    iris = (hs[..., 1] > 0.45) & (hs[..., 2] > 0.08)
    h = hs[..., 0].copy()
    s = hs[..., 1].copy()
    vv = hs[..., 2].copy()
    # radial position inside each iris: brighter amber toward the pupil
    h[iris] = 0.055 + 0.03 * (vv[iris] / vv[iris].max())
    s[iris] = np.clip(s[iris] * 1.1, 0, 1)
    vv[iris] = np.clip(vv[iris] * 1.35, 0, 1)
    sclera = (~iris) & (hs[..., 2] > 0.55)
    s[sclera] = np.clip(s[sclera] * 1.3 + 0.03, 0, 1)
    rec = np.array([colorsys.hsv_to_rgb(*p) for p in np.stack([h, s, vv], -1).reshape(-1, 3)]).reshape(rgb.shape)
    out = np.concatenate([rec, im[..., 3:]], -1)
    Image.fromarray((out * 255).astype(np.uint8)).save(dst, optimize=True)
    return dst
