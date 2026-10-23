"""Persistent operator settings -- currently just the camera URL.

Separate from NameStore because this holds a CREDENTIAL. RTSP URLs
normally embed the camera login (rtsp://user:pass@host/...), so this
file is the one piece of stored state that must never be echoed back on
a public endpoint. redact() exists for that: the dashboard shows the
redacted form, and the full URL never leaves the host.

The reason this exists at all: the camera URL used to be passed to
POST /api/camera/start and held only in memory, so every container
restart -- a ZimaOS reboot, an app update -- left the tracker sitting
idle until someone opened the dashboard and retyped it. On an appliance
that is indistinguishable from being broken.
"""
import json
import os
import threading
from urllib.parse import urlsplit, urlunsplit


def redact(url):
    """Strips any embedded credentials, for display and logging.

    Falls back to a blanket mask rather than returning the raw string if
    the URL cannot be parsed, because leaking a password is far worse
    than showing an operator something unhelpful.
    """
    if not url:
        return None
    try:
        parts = urlsplit(url)
        if parts.hostname is None:
            return "***"
        netloc = parts.hostname
        if parts.username:
            netloc = f"***@{netloc}"
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:
        return "***"


class SettingsStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # A corrupt settings file must not stop the app booting --
                # it would take the whole tracker down over a remembered
                # convenience value. Start clean instead.
                self._data = {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)   # it holds camera credentials
        except OSError:
            pass

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key, value):
        with self._lock:
            if value is None:
                self._data.pop(key, None)
            else:
                self._data[key] = value
            self._save()
