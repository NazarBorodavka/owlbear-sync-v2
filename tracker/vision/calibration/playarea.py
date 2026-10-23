"""Play-area homography: maps the camera's view of the table to a metric,
top-down image at a chosen pixels-per-millimetre scale.

This is where px_per_mm, the single number the whole detection/decode/
track pipeline is built around, actually comes from. It is not measured
indirectly or assumed -- the operator clicks the 4 corners of a real
rectangle on the table and tells the system that rectangle's true width
and height in millimetres (a one-time tape-measure reading), and the
homography is built to map those corners onto an output image whose pixel
dimensions are EXACTLY width_mm * px_per_mm by height_mm * px_per_mm. The
scale is therefore exact by construction, not estimated afterwards.

Choosing px_per_mm is itself informed by RingGeometry.min_px_per_mm() and
the validated sweep in docs/VALIDATION.md -- the calibration UI shows the
operator the resulting accuracy expectation live as they adjust zoom,
rather than asking them to pick a number blind.
"""
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class PlayAreaCalibration:
    def __init__(self, corners_px, width_mm, height_mm, px_per_mm,
                 homography=None):
        """corners_px: 4 (x, y) points in the (lens-undistorted) camera
        image, in this fixed order: top-left, top-right, bottom-right,
        bottom-left of the physical rectangle being calibrated against
        (e.g. the play mat's own printed border, or four marks measured
        out with a tape measure)."""
        self.corners_px = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.px_per_mm = float(px_per_mm)
        self.out_w = int(round(self.width_mm * self.px_per_mm))
        self.out_h = int(round(self.height_mm * self.px_per_mm))
        if homography is not None:
            self.H = np.asarray(homography, dtype=np.float64)
        else:
            self.H = self._compute_homography()

    def _compute_homography(self):
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 is required to compute the play-area homography")
        dst = np.array([[0, 0], [self.out_w, 0],
                        [self.out_w, self.out_h], [0, self.out_h]], dtype=np.float32)
        src = self.corners_px.astype(np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return H

    @classmethod
    def from_dict(cls, d):
        return cls(d["corners_px"], d["width_mm"], d["height_mm"],
                  d["px_per_mm"], homography=d.get("homography"))

    def to_dict(self):
        return {"corners_px": self.corners_px, "width_mm": self.width_mm,
                "height_mm": self.height_mm, "px_per_mm": self.px_per_mm,
                "homography": self.H, "out_w": self.out_w, "out_h": self.out_h}

    def warp(self, img):
        """Rectify a (lens-undistorted) camera frame to the metric
        top-down view the vision pipeline runs on."""
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 is required to warp into the play area")
        return cv2.warpPerspective(img, self.H, (self.out_w, self.out_h),
                                   flags=cv2.INTER_LINEAR)

    def project_point(self, x, y):
        """A point in rectified/output space -> original camera-frame
        space, for drawing an overlay back onto the raw feed."""
        p = np.array([x, y, 1.0])
        Hinv = np.linalg.inv(self.H)
        q = Hinv @ p
        return float(q[0] / q[2]), float(q[1] / q[2])

    def coverage_summary(self, geom):
        """Human-readable accuracy expectation for this px_per_mm, against
        the validated sweep in docs/VALIDATION.md -- shown live in the
        calibration UI, not left for the operator to guess."""
        min_px = geom.min_px_per_mm()
        margin = self.px_per_mm / min_px
        if margin >= 1.5:
            verdict = "comfortable margin; matches the best-validated accuracy (~99%+)"
        elif margin >= 1.2:
            verdict = "workable; validated around 95-99% accuracy under realistic conditions"
        elif margin >= 1.0:
            verdict = "thin margin; expect more refusals under occlusion/motion blur"
        else:
            verdict = "BELOW the validated minimum -- raise zoom or lower the camera"
        return {"px_per_mm": self.px_per_mm, "margin": margin, "verdict": verdict,
                "coverage_cm": (self.width_mm / 10.0, self.height_mm / 10.0)}
