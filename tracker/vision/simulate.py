"""Synthetic scene and camera-degradation model.

Kept in the shipped package rather than in tests because its real job is
to be the thing that gets CHECKED against recorded footage. v1's central
process failure was tuning against synthetic renders that quietly omitted
the effects that actually broke it -- notably directional motion blur and
a lighting gradient across the marker. Those are modelled here explicitly,
and the replay harness exists so the same metrics can be computed on real
clips and compared directly against these numbers.
"""
import numpy as np

from .detect import gaussian_blur
from .marker import render_raster


def motion_blur(img, length_px, angle):
    if length_px < 1e-6:
        return img
    n = max(2, int(round(length_px)))
    dx, dy = np.cos(angle), np.sin(angle)
    acc = np.zeros_like(img)
    for t in np.linspace(-0.5, 0.5, n):
        acc += np.roll(np.roll(img, int(round(dy * length_px * t)), axis=0),
                       int(round(dx * length_px * t)), axis=1)
    return acc / n


def degrade(img, rng, defocus_px=0.0, motion_px=0.0, motion_ang=0.0,
            noise=0.0, gain=1.0, gradient=0.0, gamma=1.0):
    out = img.astype(np.float64)
    if defocus_px > 0:
        out = gaussian_blur(out, defocus_px / 2.355)
    if motion_px > 0:
        out = motion_blur(out, motion_px, motion_ang)
    if gradient:
        h, w = out.shape
        gx = np.linspace(-0.5, 0.5, w)[None, :]
        gy = np.linspace(-0.5, 0.5, h)[:, None]
        out = out * (1.0 + gradient * 2 * (np.cos(motion_ang) * gx
                                           + np.sin(motion_ang) * gy))
    out = np.clip(out * gain, 0, 4) ** gamma
    if noise:
        out = out + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 1)


def occlude_wedge(img, cx, cy, start_deg, width_deg, radius_px, level=0.12):
    """Blocks an angular wedge within radius_px of (cx, cy) -- models a
    hand or an overhanging miniature local to ONE token.

    radius_px is required, not optional: angle alone does not bound the
    wedge to the token's own neighbourhood, so an unbounded version paints
    across the ENTIRE frame along that angular ray and can silently dim a
    completely different, far-away token that happens to lie on it. That
    was found as the cause of a handful of spurious high-residual
    detections in the sequence test -- a bug in this simulator, not in the
    detector it was being used to test.
    """
    if width_deg <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - cx, yy - cy
    ang = np.mod(np.degrees(np.arctan2(dy, dx)) - start_deg, 360.0)
    within_radius = (dx * dx + dy * dy) <= radius_px * radius_px
    out = img.copy()
    out[(ang < width_deg) & within_radius] = level
    return out


def paste(scene, patch, cx, cy):
    h, w = patch.shape
    y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1 = min(scene.shape[0], y0 + h)
    sx1 = min(scene.shape[1], x0 + w)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    sub = patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    tgt = scene[sy0:sy1, sx0:sx1]
    scene[sy0:sy1, sx0:sx1] = np.where(np.isnan(sub), tgt, sub)


def disc(radius_px, value, ss=2):
    S = int(radius_px * 2 * ss) + 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    c = (S - 1) / 2
    m = np.hypot(xx - c, yy - c) <= radius_px * ss
    img = np.where(m, float(value), np.nan)
    k = (S // ss) * ss
    img = img[:k, :k].reshape(k // ss, ss, k // ss, ss)
    with np.errstate(invalid='ignore'):
        return np.nanmean(np.nanmean(img, axis=3), axis=1)


def render_frame(tokens, geom, codebook, px_per_mm, size, rng,
                 clutter=(), bg=0.52, noise=0.02, gradient=0.3, gamma=1.2,
                 defocus_px=1.5, exposure_s=1 / 60):
    """tokens: list of dicts with id, x, y, rot_deg, speed_mm_s, occl_deg.

    Motion blur length is derived physically as speed x exposure, rather
    than being a free parameter. This matters: assuming the camera exposes
    for the whole frame interval (the worst case, which a dim room really
    does produce) makes fast drags look undetectable, while at 1/60s the
    same motion is read perfectly. Tying blur to exposure keeps that
    distinction visible instead of baking one arbitrary choice into the
    tests -- which is the kind of hidden assumption that made v1's
    synthetic results over-promise.
    """
    H, W = size
    scene = np.full((H, W), bg) + rng.normal(0, 0.012, (H, W))
    for (cx, cy, rad, val) in clutter:
        paste(scene, disc(rad, val), cx, cy)
    for tk in tokens:
        patch, _, _ = render_raster(codebook[tk["id"]], geom, px_per_mm,
                                    ss=3, rotation_deg=tk.get("rot_deg", 0.0))
        blur_px = tk.get("speed_mm_s", 0.0) * px_per_mm * exposure_s
        if blur_px > 1.0:  # per-token blur along its own travel direction
            filled = np.where(np.isnan(patch), bg, patch)
            blurred = motion_blur(filled, blur_px, tk.get("heading", 0.0))
            patch = np.where(np.isnan(patch), np.nan, blurred)
        paste(scene, patch, tk["x"], tk["y"])
    for tk in tokens:
        if tk.get("occl_deg", 0) > 0:
            scene = occlude_wedge(scene, tk["x"], tk["y"],
                                  tk.get("occl_start", 0.0), tk["occl_deg"],
                                  radius_px=geom.r_out * px_per_mm * 1.25)
    return degrade(scene, rng, defocus_px=defocus_px, noise=noise,
                   gradient=gradient, gamma=gamma,
                   motion_ang=float(rng.uniform(0, np.pi)))
