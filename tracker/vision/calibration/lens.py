"""Lens (intrinsic) calibration: camera matrix and distortion coefficients
from checkerboard captures, via cv2.calibrateCamera.

This is the one place pure-math reimplementation genuinely isn't worth
it -- chessboard-based camera calibration is a solved, extremely well
tested OpenCV routine, and the detection/decode/track core (which IS
pure numpy, see tracker/vision/{detect,decode,track}.py) never needs to
re-derive it: it only ever consumes the undistorted output.

Skippable: a cheap wide-angle lens like the Reolink E1 Zoom has real
barrel distortion, but the token tracker tolerates a few percent of
position error just fine (see docs/VALIDATION.md), so running without
lens calibration at all is a legitimate choice for a first setup -- not
every operator owns a printable checkerboard, and this should not block
getting started.
"""
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class LensCalibration:
    def __init__(self, camera_matrix, dist_coeffs, image_size, rms_error):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        # cv2.calibrateCamera returns this as shape (1, 5) or (5, 1)
        # depending on version/path; flattened here once so every caller
        # can index it as a plain 1D array without re-discovering that.
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        self.image_size = tuple(image_size)   # (w, h) at calibration time
        self.rms_error = float(rms_error)
        self._new_matrix = None
        self._map1 = None
        self._map2 = None

    @classmethod
    def identity(cls, image_size):
        """No-distortion fallback, used when lens calibration is skipped.
        Keeps every downstream consumer unconditional -- there is no
        special-cased 'no calibration' branch anywhere else."""
        w, h = image_size
        f = float(max(w, h))      # a plausible focal length guess, in pixels
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
        return cls(K, np.zeros(5), image_size, rms_error=0.0)

    @classmethod
    def from_dict(cls, d):
        return cls(d["camera_matrix"], d["dist_coeffs"], d["image_size"],
                  d["rms_error"])

    def to_dict(self):
        return {"camera_matrix": self.camera_matrix, "dist_coeffs": self.dist_coeffs,
                "image_size": list(self.image_size), "rms_error": self.rms_error}

    def _ensure_maps(self, image_size):
        if self._map1 is not None and image_size == self.image_size:
            return
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 is required to apply lens calibration")
        w, h = image_size
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.0)
        self._new_matrix = new_K
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
        self.image_size = (w, h)

    def undistort_image(self, img):
        h, w = img.shape[:2]
        self._ensure_maps((w, h))
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

    def undistort_points(self, points_xy):
        """points_xy: (N, 2) pixel coordinates in the ORIGINAL (distorted)
        image. Returns their position in the undistorted image this
        calibration produces -- used to carry UI clicks (e.g. play-area
        corners) from the raw feed into rectified space."""
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 is required to apply lens calibration")
        self._ensure_maps(self.image_size)
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs,
                                  P=self._new_matrix)
        return und.reshape(-1, 2)


def find_chessboard(gray, pattern_size):
    """pattern_size: (cols, rows) of INTERNAL corners, e.g. (9, 6) for a
    10x7-square board. Returns the corner points (sub-pixel refined) or
    None if the board was not found in this frame."""
    if not CV2_AVAILABLE:
        raise RuntimeError("cv2 is required for lens calibration")
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def calibrate_from_frames(gray_frames, pattern_size, square_mm):
    """gray_frames: list of grayscale images of the SAME checkerboard at
    different poses (tilt/position) -- more poses and more coverage of
    the frame (especially the corners, where distortion is worst) gives a
    better fit. Returns (LensCalibration, list of per-frame reprojection
    errors) so the caller can show which captures were poor and should be
    retaken, not just a single opaque RMS number.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("cv2 is required for lens calibration")
    cols, rows = pattern_size
    # float32 is a hard requirement of cv2.calibrateCamera (it rejects
    # float64 object points with an opaque "Unsupported format" error) --
    # found directly by testing, not documented clearly anywhere obvious.
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    objpoints, imgpoints, used_sizes = [], [], []
    for gray in gray_frames:
        corners = find_chessboard(gray, pattern_size)
        if corners is None:
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used_sizes.append(gray.shape[::-1])
    if len(objpoints) < 4:
        raise ValueError(
            f"only found the checkerboard in {len(objpoints)} frame(s); "
            f"need at least 4 at different angles/positions for a "
            f"reliable fit")

    image_size = used_sizes[0]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)

    per_frame_err = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = float(np.sqrt(np.mean((proj.reshape(-1, 2) - imgpoints[i].reshape(-1, 2)) ** 2)))
        per_frame_err.append(err)

    return LensCalibration(K, dist, image_size, rms), per_frame_err
