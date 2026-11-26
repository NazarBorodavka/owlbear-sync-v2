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
                 rescan_hz=0.0, expected_tokens=None, **tracker_kw):
        # rescan_hz=0 means "full-frame detect EVERY frame", which is now
        # the default. It used to be 3Hz because detect() cost ~124ms on a
        # real 3168x1785 frame; moving its three hot spots onto OpenCV
        # (see detect.py) brought that to ~28ms, which fits inside a 20fps
        # budget. That is not just a speed win -- at 3Hz the tracker had to
        # EXTRAPOLATE a moving token for up to 10 frames between looks, and
        # reporting that extrapolation is exactly what made fast drags
        # overshoot and snap back. Detecting every frame means coasting is
        # a one-frame edge case instead of the normal path during motion.
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
        cand = []
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
                cand.append(d)
        # 2) expensive path: full-frame search, on a slow cadence. Both
        # paths are kept and decoded BEFORE merging now -- see
        # _merge_by_quality for why doing that after a position-only
        # dedup was a real bug, not just extra cost.
        if self._should_rescan(now, len(cand)):
            self._last_rescan = now
            self.stats["rescans"] += 1
            cand += self.detector.detect(img)
        self.stats["t_detect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        decoded = [(d, self.matcher.decode(img, d["x"], d["y"],
                                           d.get("px_per_mm", self.px_per_mm)))
                   for d in cand]
        dets, decodes = self._merge_by_quality(decoded)
        self.stats["t_decode"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        reported = self.tracker.step(dets, decodes, now=now)
        self.stats["t_track"] += time.perf_counter() - t0
        return reported

    def _merge_by_quality(self, decoded):
        """Collapses candidates that are the SAME physical marker --
        typically the cheap path re-finding a track's own position AND
        the full-frame rescan finding it independently -- keeping
        whichever candidate actually reads best, not whichever was
        computed first.

        This replaces a real, measured bug: the old merge decoded AFTER
        discarding near-duplicates by position alone, and always kept the
        cheap-path entry since it was appended first. The cheap path's
        edge fit is seeded at the track's last position and assumes that
        is already close; a few frames into a slow, continuous drag that
        stops being true, and refine_at() can converge to a plausible
        but WRONG nearby edge -- a real circle, legitimate residual, but
        not the token, so it decodes as noise. That wrong candidate then
        sat within the merge radius of the CORRECT full-detect candidate
        (the same physical ring, just found twice) and silently ate it:
        measured on real footage, an established track with 167 confirmed
        hits lost its own detection for 3+ consecutive frames while the
        token was in plain, unoccluded view, because the only candidate
        the tracker ever saw near it decoded at peak 0.59 instead of the
        0.94+ the real ring was reading at a point 32px away.

        Decoded always beats undecoded (real evidence beats none,
        whatever the circle-fit residual looked like); between two
        decoded candidates, higher margin wins; between two undecoded
        ones, lower residual wins, matching Detector._dedup's rule for
        candidates within ONE detection pass."""
        tol = self.geom.r_out * self.px_per_mm * 0.8
        kept = []   # [det, (mid, margin), quality_key]
        for d, (mid, margin) in decoded:
            # (1, margin) sorts above EVERY (0, ...) tuple regardless of
            # margin/residual magnitude -- a decode is always stronger
            # evidence than a bare geometric fit.
            quality = (1, margin) if mid is not None else (0, -d.get("resid", 1e9))
            dup = next((k for k in kept if np.hypot(d["x"] - k[0]["x"],
                                                     d["y"] - k[0]["y"]) <= tol), None)
            if dup is None:
                kept.append([d, (mid, margin), quality])
            elif quality > dup[2]:
                dup[0], dup[1], dup[2] = d, (mid, margin), quality
        return [k[0] for k in kept], [k[1] for k in kept]

    def tokens_mm(self, reported, origin_px=(0.0, 0.0)):
        """Reportable tokens in millimetres relative to the play-area origin."""
        ox, oy = origin_px
        out = []
        for t in reported:
            # report_pos is the measurement while the token is being
            # read, and the filter's short-horizon prediction while it is
            # not -- bounded by Tracker.predict_publish_s, which is what
            # stops this becoming the open-ended extrapolation that was
            # once measured sliding a lost token off the table.
            x, y = t.report_pos
            out.append({
                "marker_id": t.marker_id,
                "x_mm": (x - ox) / self.px_per_mm,
                "y_mm": (y - oy) / self.px_per_mm,
                "track": t.key,
                "age_s": t.last_seen - t.created,
                # True when this position is predicted rather than read,
                # so a consumer can tell the two apart.
                "coasting": t.misses > 0,
            })
        return out

    def timing_report(self):
        n = max(1, self.stats["frames"])
        return {k: 1000.0 * v / n for k, v in self.stats.items()
                if k.startswith("t_")} | {
            "frames": self.stats["frames"], "rescans": self.stats["rescans"]}
