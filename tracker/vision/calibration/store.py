"""Shared persistence for calibration stages.

Each stage (lens, play-area, detector tuning) is its own small JSON file
under a calibration directory, not one combined blob. That is what makes
"each part able to be rerun separately" actually true: re-running lens
calibration cannot accidentally clobber the play-area homography, because
they are different files with no shared state.
"""
import json
import os
import time
import numpy as np


class CalibrationStore:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.directory, f"{name}.json")

    def save(self, name, data):
        """data may contain numpy arrays -- they are converted to plain
        lists so the file is both JSON-valid and human-inspectable,
        which matters for a calibration file someone may want to check
        or hand-edit during setup."""
        payload = {
            "calibrated_at": time.time(),
            "calibrated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data": _to_jsonable(data),
        }
        path = self._path(name)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)   # atomic, so a crash mid-write can't corrupt it

    def load(self, name):
        path = self._path(name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            payload = json.load(f)
        return payload

    def exists(self, name):
        return os.path.exists(self._path(name))

    def clear(self, name):
        path = self._path(name)
        if os.path.exists(path):
            os.remove(path)


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
