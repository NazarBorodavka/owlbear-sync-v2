"""Per-frame orchestration: detect -> decode -> track.

The important structural choice is that full-frame search and per-frame
tracking are DIFFERENT, differently-priced operations:

  * Re-localising an already-tracked token is a local edge fit around its
    predicted position. Measured at 0.33ms per token, so six tokens cost
    about 2ms -- it can run every single frame.
  * Searching the whole frame for tokens that are not yet tracked is the
    expensive part (tens of ms at 5MP) and is only needed when something
    new might have appeared. It runs on a slow cadence.

v1 did the equivalent of the expensive thing every frame for every token
and spent ~150ms per frame on it, which is why it ran at 5fps against a
10fps camera. Splitting the two is what buys the headroom.
"""
import time
import numpy as np

from .detect import Detector
from .decode import Matcher
from .track import Tracker


class Pipeline:
    def __init__(self, geom, codebook, px_per_mm,
                 rescan_hz=3.0, expected_tokens=None, **tracker_kw):
        geom.validate()
        self.geom = geom
        self.codebook = np.asarray(codebook)
        self.px_per_mm = px_per_mm
        self.detector = Detector(geom, px_per_mm)
        self.matcher = Matcher(self.codebook, geom)
        self.tracker = Tracker(geom, px_per_mm, **tracker_kw)
        self.rescan_interval = 1.0 / rescan_hz if rescan_hz > 0 else 0.0
        self.expected_tokens = expected_tokens
        self._last_rescan = -1e9
        self.stats = {"frames": 0, "rescans": 0,
                      "t_detect": 0.0, "t_decode": 0.0, "t_track": 0.0}

    def _should_rescan(self, now, n_tracked):
        if self.rescan_interval <= 0:
            return True
        if now - self._last_rescan >= self.rescan_interval:
            return True
        # An expected token is missing, so look for it now rather than
        # waiting out the rescan interval -- this is what makes a token
        # newly placed on the table appear promptly.
        if self.expected_tokens is not None and n_tracked < self.expected_tokens:
            return now - self._last_rescan >= self.rescan_interval * 0.25
        return False

    def process(self, img, now=None):
        now = time.monotonic() if now is None else now
        self.stats["frames"] += 1

        t0 = time.perf_counter()
        dets = []
        # 1) cheap path: re-find each existing track where it is predicted
        for t in self.tracker.tracks:
            tx, ty = t.pos
            d = self.detector.refine_at(img, tx, ty)
            if d is None:
                # The cheap edge fit assumes the prediction is already
                # close, which stops being true the instant a token is
                # picked up and moved. Fall back to a vote search around
                # the prediction -- still far cheaper than a full-frame
                # rescan. This runs on the FIRST failed frame, not after a
                # miss has been recorded: waiting one frame was enough for
                # a flicked token to outrun its own search window and
                # never be recovered.
                #
                # Radius here is deliberately independent of (and more
                # generous than) Track.gate_radius, which is capped tight
                # for geometric safety (see its docstring) -- a wide local
                # SEARCH is fine because the normal association step still
                # has to accept the result, and because this result still
                # goes through full decode+gating before being trusted.
                search_r = min(
                    self.geom.r_out * self.px_per_mm * 2.5
                    + t.last_speed * max(t.coast_s, 1.0 / 15.0),
                    self.geom.r_out * self.px_per_mm * 10.0)
                d = self.detector.search_near(img, tx, ty, radius_px=search_r)
            if d is not None:
                dets.append(d)
        # 2) expensive path: full-frame search, on a slow cadence
        if self._should_rescan(now, len(dets)):
            self._last_rescan = now
            self.stats["rescans"] += 1
            for d in self.detector.detect(img):
                if all(np.hypot(d["x"] - e["x"], d["y"] - e["y"])
                       > self.geom.r_out * self.px_per_mm * 0.8 for e in dets):
                    dets.append(d)
        self.stats["t_detect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        decodes = [self.matcher.decode(img, d["x"], d["y"],
                                       d.get("px_per_mm", self.px_per_mm))
                   for d in dets]
        self.stats["t_decode"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        reported = self.tracker.step(dets, decodes, now=now)
        self.stats["t_track"] += time.perf_counter() - t0
        return reported

    def tokens_mm(self, reported, origin_px=(0.0, 0.0)):
        """Reportable tokens in millimetres relative to the play-area origin."""
        ox, oy = origin_px
        out = []
        for t in reported:
            x, y = t.pos
            out.append({
                "marker_id": t.marker_id,
                "x_mm": (x - ox) / self.px_per_mm,
                "y_mm": (y - oy) / self.px_per_mm,
                "track": t.key,
                "age_s": t.last_seen - t.created,
                "coasting": t.misses > 0,
            })
        return out

    def timing_report(self):
        n = max(1, self.stats["frames"])
        return {k: 1000.0 * v / n for k, v in self.stats.items()
                if k.startswith("t_")} | {
            "frames": self.stats["frames"], "rescans": self.stats["rescans"]}
