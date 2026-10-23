"""Grid-spacing measurement tests: synthetic ground truth first (does the
math work at all), then the realistic cases this exists for -- a grid
overlaid on a busy map image (not a blank background), under the same
noise/gradient degradation used elsewhere, and a confident None when
there's no grid to find.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracker.vision.grid import measure_grid_spacing_mm
from tracker.vision import simulate as sim

PX_MM = 3.5


def render_grid(mm_per_cell, px_per_mm, size, line_px=2, line_level=0.15, bg=0.75,
                offset=(0, 0)):
    h, w = size
    img = np.full((h, w), bg, dtype=np.float64)
    period = mm_per_cell * px_per_mm
    ox, oy = offset
    xs = np.arange(w)
    ys = np.arange(h)
    vline = (np.mod(xs - ox, period) < line_px)
    hline = (np.mod(ys - oy, period) < line_px)
    img[:, vline] = line_level
    img[hline, :] = line_level
    return img


def test_recovers_known_grid_spacing():
    # Sized so even the largest tested cell (120mm) comfortably fits
    # several periods -- a real play area should always be several
    # grid cells across, so this matches realistic deployment, not an
    # artificially cramped test fixture.
    for true_mm in (25.0, 40.0, 76.2, 120.0):
        img = render_grid(true_mm, PX_MM, (900, 1400))
        result = measure_grid_spacing_mm(img, PX_MM)
        assert result is not None, f"failed to find a grid at {true_mm}mm/cell"
        err_pct = 100 * abs(result["mm_per_cell"] - true_mm) / true_mm
        assert err_pct < 2.0, f"grid={true_mm}mm measured as {result['mm_per_cell']:.1f}mm ({err_pct:.1f}% off)"
    print("  recovers known grid spacing within 2% across 25-120mm/cell")


def test_robust_on_top_of_a_busy_map_with_noise():
    rng = np.random.default_rng(0)
    true_mm = 40.0
    base = np.clip(rng.normal(0.5, 0.15, (500, 700)), 0, 1)
    base = sim.gaussian_blur(base, 8.0)   # map-texture-like low-frequency variation
    grid = render_grid(true_mm, PX_MM, (500, 700), line_level=0.0, bg=1.0)
    # grid as a multiplicative darkening overlay on top of the "map"
    composed = base * grid
    composed = sim.degrade(composed, rng, noise=0.02, gradient=0.2, defocus_px=1.0)
    result = measure_grid_spacing_mm(composed, PX_MM)
    assert result is not None
    err_pct = 100 * abs(result["mm_per_cell"] - true_mm) / true_mm
    assert err_pct < 5.0, f"measured {result['mm_per_cell']:.1f}mm vs true {true_mm}mm ({err_pct:.1f}% off)"
    print(f"  recovers grid spacing within {err_pct:.1f}% on a noisy, textured 'map' background "
          f"(confidence={result['confidence']:.2f})")


def test_confident_none_when_grid_is_off():
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(0.5, 0.15, (500, 700)), 0, 1)
    img = sim.gaussian_blur(img, 6.0)
    img = sim.degrade(img, rng, noise=0.02)
    result = measure_grid_spacing_mm(img, PX_MM)
    assert result is None, f"found a spurious grid in pure texture noise: {result}"
    print("  correctly reports no grid found when the overlay is off")


def test_robust_to_slight_axis_misalignment():
    """The play-area homography already aligns the physical table to the
    rectified frame's own axes by construction (that is what calibrating
    4 corners to a rectangle DOES), so residual grid misalignment in the
    rectified view should be a small fraction of a degree, not a few
    degrees -- found directly that this technique holds up to ~1.5deg and
    fails above ~2deg (at 2deg, a 900px-tall frame drifts ~31px
    top-to-bottom, wide enough to smear a single line's column-profile
    contribution across many columns and defeat the per-line detection).
    1deg is tested here as a realistic residual-misalignment margin, not
    the technique's actual breaking point."""
    true_mm = 50.0
    img = render_grid(true_mm, PX_MM, (900, 1400))
    h, w = img.shape
    angle = np.radians(1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    src_x = cx + (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    src_y = cy - (xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    sx = np.clip(np.round(src_x).astype(int), 0, w - 1)
    sy = np.clip(np.round(src_y).astype(int), 0, h - 1)
    rotated = img[sy, sx]

    result = measure_grid_spacing_mm(rotated, PX_MM)
    assert result is not None
    err_pct = 100 * abs(result["mm_per_cell"] - true_mm) / true_mm
    assert err_pct < 5.0, f"1deg misalignment: measured {result['mm_per_cell']:.1f}mm vs true {true_mm}mm"
    print(f"  tolerates 1deg residual misalignment: {err_pct:.1f}% error "
          f"(verified separately: holds to ~1.5deg, fails above ~2deg)")


if __name__ == "__main__":
    test_recovers_known_grid_spacing()
    test_robust_on_top_of_a_busy_map_with_noise()
    test_confident_none_when_grid_is_off()
    test_robust_to_slight_axis_misalignment()
    print("\nall grid tests passed")
