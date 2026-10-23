"""Synthetic scene and illumination model.

Shipped in the package (not tests/) because its real purpose is to be the
thing CHECKED against recorded footage during the validation-corpus pass.
v1's central process failure was tuning against synthetic renders that
quietly omitted the effects that actually broke it in the field --
directional motion blur, a lighting gradient, and (what this module now
also models explicitly) textured illumination from a ceiling-mounted
projector shining onto the markers.
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
    hand or an overhanging miniature local to ONE token. radius_px is
    required: angle alone does not bound the wedge to the token's own
    neighbourhood, and an unbounded version was found to paint across the
    ENTIRE frame along that angular ray, silently dimming a far-away
    token that happened to lie on it -- a bug in this simulator, caught by
    the end-to-end sequence test, not a bug in the detector it exists to
    exercise."""
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


def _multi_octave_field(shape, rng, scale_mm, n_octaves, strength, px_per_mm):
    h, w = shape
    field = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    for octave in range(n_octaves):
        wl_mm = scale_mm[0] * (scale_mm[1] / scale_mm[0]) ** (octave / max(1, n_octaves - 1))
        wl_px = max(2.0, wl_mm * px_per_mm)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        ang = rng.uniform(0, np.pi)
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        cross = xx * np.sin(ang) - yy * np.cos(ang)
        amp = strength / n_octaves
        field += amp * np.sin(2 * np.pi * proj / wl_px + phase_x) \
            * np.cos(2 * np.pi * cross / (wl_px * 1.7) + phase_y)
    return field


def projector_illumination(shape, rng, scale_mm=(3, 120), n_octaves=5,
                           strength=0.5, grid_spacing_mm=None,
                           grid_strength=0.0, grid_force_through=None,
                           px_per_mm=3.2, monochrome=True,
                           color_fraction=0.5):
    """A multiplicative illumination field standing in for a ceiling
    projector's map image landing on the table.

    Modelled as MULTIPLICATIVE, not additive -- observed = incident(x,y) *
    reflectance(x,y). A printed marker's black/white chips get SCALED by
    whatever light is locally incident, not summed with the projected
    image. This is why the matched filter's zero-mean correlation is
    naturally insensitive to illumination that varies slowly compared to
    one chip: the risk is specifically structure AT OR BELOW chip scale.

    scale_mm therefore spans down to 3mm by default -- a single marker
    chip is ~4.8mm along its own arc at this geometry, so texture coarser
    than ~40mm (the first version of this function used (40, 250)) cannot
    exercise the failure mode this test exists to find at all. That was a
    real bug in this simulator: every projector-robustness trial passed
    100% because nothing in the test could actually disturb a single chip
    independently of its neighbours.

    monochrome=True models forced IR/night mode: pigment-based black ink
    stays dark under both visible and near-IR (near-zero reflectance
    across that whole range, vs. dye-based black jumping to ~80%
    reflectance above 800nm -- a dye-based marker would fail under IR
    regardless of software, which belongs on the validation-corpus
    checklist, not assumed). IR mode does not remove interference
    entirely: it removes the COLOR-only contribution (color_fraction of
    the total field strength here) while the remaining achromatic
    brightness texture -- a lit vs. shadowed map region still differs in
    photon flux reaching the IR-illuminated surface -- passes through
    unchanged. monochrome=False models plain colour mode, where the full
    field strength reaches the sensor.

    grid_force_through, if given as (cx, cy) in pixels, guarantees a map
    grid line passes through that exact point -- used to test the
    specific worst case of a grid line crossing dead center of a marker's
    code band, rather than relying on it coinciding by chance.
    """
    total = _multi_octave_field(shape, rng, scale_mm, n_octaves, strength, px_per_mm)
    if not monochrome:
        total *= 1.0            # full strength reaches the sensor
    else:
        total *= (1.0 - color_fraction)   # color-correlated part washed out
    field = 1.0 + total

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if grid_spacing_mm and grid_strength > 0:
        sp = grid_spacing_mm * px_per_mm
        if grid_force_through is not None:
            fx, fy = grid_force_through
            ox, oy = fx % sp, fy % sp
        else:
            ox, oy = rng.uniform(0, sp), rng.uniform(0, sp)
        grid = (np.abs(np.mod(xx - ox, sp)) < 1.3) | (np.abs(np.mod(yy - oy, sp)) < 1.3)
        field = field * (1.0 - grid_strength * grid)
    return np.clip(field, 0.1, 3.0)


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
                 defocus_px=1.5, exposure_s=1 / 60,
                 projector=None):
    """tokens: list of dicts with id, x, y, rot_deg, speed_mm_s, occl_deg.

    projector: None, or a dict of kwargs forwarded to
    projector_illumination() -- applied as a multiplicative field over the
    WHOLE scene (table, clutter and markers alike) before camera noise,
    matching how a ceiling projector actually illuminates everything in
    its throw, not just the tokens.
    """
    H, W = size
    scene = np.full((H, W), bg) + rng.normal(0, 0.012, (H, W))
    for (cx, cy, rad, val) in clutter:
        paste(scene, disc(rad, val), cx, cy)
    for tk in tokens:
        patch, _, _ = render_raster(codebook[tk["id"]], geom, px_per_mm,
                                    ss=3, rotation_deg=tk.get("rot_deg", 0.0))
        blur_px = tk.get("speed_mm_s", 0.0) * px_per_mm * exposure_s
        if blur_px > 1.0:
            filled = np.where(np.isnan(patch), bg, patch)
            blurred = motion_blur(filled, blur_px, tk.get("heading", 0.0))
            patch = np.where(np.isnan(patch), np.nan, blurred)
        paste(scene, patch, tk["x"], tk["y"])
    for tk in tokens:
        if tk.get("occl_deg", 0) > 0:
            scene = occlude_wedge(scene, tk["x"], tk["y"],
                                  tk.get("occl_start", 0.0), tk["occl_deg"],
                                  radius_px=geom.r_out * px_per_mm * 1.25)
    if projector:
        field = projector_illumination((H, W), rng, px_per_mm=px_per_mm,
                                       **projector)
        scene = scene * field
    return degrade(scene, rng, defocus_px=defocus_px, noise=noise,
                   gradient=gradient, gamma=gamma,
                   motion_ang=float(rng.uniform(0, np.pi)))
