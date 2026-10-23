"""Printable artwork and a matching raster renderer.

Both the SVG written for printing and the raster used by the tests are
generated from the SAME RingGeometry and the SAME codebook. v1 kept the
artwork generator and the decoder's expectations in separate places and
they silently drifted, which is how a whole print run came out mirrored
relative to what the decoder expected and cost a long debugging detour.
Sharing one source of truth makes that class of bug impossible.

SVG is emitted in real millimetre units, so printing at 100% scale (no
"fit to page") yields a physically correct marker. A printed ruler mark
is included so a mis-scaled print is obvious at a glance rather than
silently producing a marker that decodes but localises wrongly.
"""
import numpy as np


def marker_svg(codeword, geom, marker_id, label=True, bleed_mm=2.0):
    """One marker as a standalone SVG string, in millimetre units."""
    d = geom.base_d_mm + 2 * bleed_mm
    c = d / 2.0
    n = len(codeword)
    step = 360.0 / n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{d}mm" height="{d}mm" '
        f'viewBox="0 0 {d} {d}">',
        f'<rect width="{d}" height="{d}" fill="white"/>',
        # solid detection ring, drawn as a stroked circle on the mid-radius
        f'<circle cx="{c}" cy="{c}" r="{(geom.r_out + geom.det_inner) / 2:.4f}" '
        f'fill="none" stroke="black" stroke-width="{geom.det_ring_mm:.4f}"/>',
    ]
    r_mid = (geom.code_inner + geom.code_outer) / 2.0
    width = geom.code_outer - geom.code_inner
    for i, chip in enumerate(codeword):
        if chip <= 0:          # black chip
            a0 = np.radians(i * step)
            a1 = np.radians((i + 1) * step)
            x0, y0 = c + r_mid * np.cos(a0), c + r_mid * np.sin(a0)
            x1, y1 = c + r_mid * np.cos(a1), c + r_mid * np.sin(a1)
            large = 1 if (step > 180) else 0
            parts.append(
                f'<path d="M {x0:.4f} {y0:.4f} A {r_mid:.4f} {r_mid:.4f} 0 '
                f'{large} 1 {x1:.4f} {y1:.4f}" fill="none" stroke="black" '
                f'stroke-width="{width:.4f}"/>')
    # faint cut guide at the physical base edge
    parts.append(f'<circle cx="{c}" cy="{c}" r="{geom.r_out:.4f}" fill="none" '
                 f'stroke="#bbbbbb" stroke-width="0.08" stroke-dasharray="1,1"/>')
    if label:
        parts.append(
            f'<text x="{c}" y="{d - 0.5:.3f}" font-family="sans-serif" '
            f'font-size="1.6" fill="#888888" text-anchor="middle">'
            f'id{marker_id} &#183; {geom.base_d_mm:.0f}mm &#183; print at 100%</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def sheet_svg(codebook, geom, cols=4, margin_mm=6.0, pitch_extra_mm=4.0):
    """A full printable sheet of all markers, with a 50mm scale bar.

    The scale bar exists so a mis-scaled print is caught immediately: if
    it does not measure 50mm with a ruler, the markers are the wrong size
    and every position the tracker reports will be wrong by that factor.
    """
    pitch = geom.base_d_mm + pitch_extra_mm
    n = len(codebook)
    rows = (n + cols - 1) // cols
    w = margin_mm * 2 + cols * pitch
    h = margin_mm * 2 + rows * pitch + 12
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" '
             f'height="{h}mm" viewBox="0 0 {w} {h}">',
             f'<rect width="{w}" height="{h}" fill="white"/>']
    for i, cw in enumerate(codebook):
        r, c_ = divmod(i, cols)
        ox = margin_mm + c_ * pitch + pitch / 2
        oy = margin_mm + r * pitch + pitch / 2
        parts.append(f'<g transform="translate({ox:.3f},{oy:.3f})">')
        parts.append(_marker_body(cw, geom, i))
        parts.append('</g>')
    by = h - 7
    parts.append(f'<rect x="{margin_mm}" y="{by}" width="50" height="1.2" fill="black"/>')
    parts.append(f'<text x="{margin_mm}" y="{by + 5:.2f}" font-family="sans-serif" '
                 f'font-size="3" fill="black">50mm &#8212; measure this; if it is not '
                 f'50mm the sheet printed at the wrong scale</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def _marker_body(codeword, geom, marker_id):
    """Marker drawn centred on the local origin (for sheet layout)."""
    n = len(codeword)
    step = 360.0 / n
    out = [f'<circle cx="0" cy="0" r="{(geom.r_out + geom.det_inner) / 2:.4f}" '
           f'fill="none" stroke="black" stroke-width="{geom.det_ring_mm:.4f}"/>']
    r_mid = (geom.code_inner + geom.code_outer) / 2.0
    width = geom.code_outer - geom.code_inner
    for i, chip in enumerate(codeword):
        if chip <= 0:
            a0, a1 = np.radians(i * step), np.radians((i + 1) * step)
            x0, y0 = r_mid * np.cos(a0), r_mid * np.sin(a0)
            x1, y1 = r_mid * np.cos(a1), r_mid * np.sin(a1)
            out.append(f'<path d="M {x0:.4f} {y0:.4f} A {r_mid:.4f} {r_mid:.4f} '
                       f'0 0 1 {x1:.4f} {y1:.4f}" fill="none" stroke="black" '
                       f'stroke-width="{width:.4f}"/>')
    out.append(f'<circle cx="0" cy="0" r="{geom.r_out:.4f}" fill="none" '
               f'stroke="#bbbbbb" stroke-width="0.08" stroke-dasharray="1,1"/>')
    out.append(f'<text x="0" y="{geom.r_out + 3.2:.3f}" font-family="sans-serif" '
               f'font-size="2.4" fill="#777777" text-anchor="middle">id{marker_id}</text>')
    return "\n".join(out)


def render_raster(codeword, geom, px_per_mm, ss=4, pad_mm=3.0, rotation_deg=0.0):
    """Antialiased raster of one marker, for tests and simulation.

    Returns (image with NaN outside the disc, centre_px, r_out_px). NaN
    marks "not part of the token" so a caller can composite it onto a
    background instead of pasting a square patch.
    """
    half = geom.r_out + pad_mm
    size = int(round(2 * half * px_per_mm))
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    c = (S - 1) / 2.0
    dx, dy = xx - c, yy - c
    rad = np.hypot(dx, dy) / (px_per_mm * ss)
    ang = np.mod(np.arctan2(dy, dx) - np.radians(rotation_deg), 2 * np.pi)

    img = np.ones((S, S), dtype=np.float64)
    img[(rad >= geom.det_inner) & (rad <= geom.r_out)] = 0.0
    band = (rad >= geom.code_inner) & (rad < geom.code_outer)
    n = len(codeword)
    chip = np.floor(ang / (2 * np.pi / n)).astype(int) % n
    img[band & (np.asarray(codeword)[chip] <= 0)] = 0.0
    img[rad > geom.r_out] = np.nan

    img = img.reshape(size, ss, size, ss)
    with np.errstate(invalid='ignore'):
        img = np.nanmean(np.nanmean(img, axis=3), axis=1)  # NaN column = fully outside disc
    return img, (size - 1) / 2.0, geom.r_out * px_per_mm
