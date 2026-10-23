"""Persistent operator-assigned names for detected marker IDs.

This is the web UI's source of truth for token naming (see the project
requirement: naming happens here, with its own persistent storage, and is
SHARED WITH the companion extension via the API below -- the extension
does not own a separate copy of names, it reads this one).
"""
import json
import os
import threading


class NameStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._names = {}   # marker_id (int) -> name (str)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                raw = json.load(f)
            self._names = {int(k): v for k, v in raw.items()}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({str(k): v for k, v in self._names.items()}, f, indent=2)
        os.replace(tmp, self.path)

    def get(self, marker_id):
        with self._lock:
            return self._names.get(int(marker_id))

    def set(self, marker_id, name):
        marker_id = int(marker_id)
        with self._lock:
            if name:
                self._names[marker_id] = name
            else:
                self._names.pop(marker_id, None)
            self._save()

    def all(self):
        with self._lock:
            return dict(self._names)
