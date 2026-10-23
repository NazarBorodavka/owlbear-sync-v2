"""Multi-object tracking with persistent identity.

This is the piece v1 genuinely did not have, and most of v1's symptoms
trace back to its absence. v1 re-derived identity from scratch every
frame and patched the consequences with ad-hoc vote dictionaries, so a
single bad frame could lose a token's identity, and any unexplained blob
immediately became a visible "phantom" token.

Two structural decisions here remove whole classes of that behaviour:

  * Identity belongs to the TRACK, not to the frame. A decode is evidence
    that accumulates; it is never the sole basis for what a token is. A
    token can go many frames undecoded -- blurred, occluded, face down --
    and keep its identity, because the track persists.

  * A track is only REPORTED once it owns a confident identity. Blobs
    that never resolve to an id stay internal and never reach the UI or
    Owlbear. v1 drew every unmatched circle immediately, which is exactly
    what the "trail of red ghost disks" was.
"""
import time
import numpy as np


class Track:
    _next_id = 1

    def __init__(self, x, y, px_per_mm, now):
        self.key = Track._next_id
        Track._next_id += 1
        # constant-velocity state [x, y, vx, vy] in pixels and px/s
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self._coast_s = 0.0
        self.P = np.diag([4.0, 4.0, 400.0, 400.0])
        self.px_per_mm = px_per_mm
        self.marker_id = None
        self.evidence = {}          # marker_id -> accumulated weight
        self.hits = 1
        self.misses = 0
        self.created = now
        self.last_seen = now
        self.last_decode = 0.0
        # Speed at the moment of the last real detection, undamped. The
        # live velocity gets damped while coasting (see predict), so it is
        # useless for deciding how far away to search for a token that was
        # moving fast when we lost it -- which is exactly when a wide
        # search matters most. v1 hit this too and bolted on a
        # "last_seen_speed"; here it is part of the model from the start.
        self.last_speed = 0.0

    @property
    def pos(self):
        return float(self.state[0]), float(self.state[1])

    @property
    def coast_s(self):
        return self._coast_s

    def note_seen(self, now):
        self._coast_s = 0.0
        self.last_seen = now
        self.last_speed = float(np.hypot(self.state[2], self.state[3]))

    def note_missed(self, dt):
        self._coast_s = getattr(self, "_coast_s", 0.0) + dt

    def predict(self, dt, accel_var=900.0, coast_damping_per_s=0.05,
                coast_grace_s=0.30):
        """Constant velocity, but with velocity DAMPED while coasting.

        A disc pushed across a table decelerates against friction and
        stops; it does not continue ballistically. Undamped extrapolation
        produced the worst failure in the sequence test: a fast-flicked
        token that the detector briefly lost kept flying and drifted
        118mm, ending up closer to a different token's position than its
        own. Damping makes a lost token settle where it was last credibly
        seen, which is both physically right and far less confusing on
        screen than a token sliding off on its own."""
        # Damping only starts after a grace period. A token that was
        # genuinely mid-flick when we lost it really is still travelling,
        # so killing its velocity immediately strands the prediction
        # behind the token and the search then looks in the wrong place.
        # After the grace period the opposite is true -- it has probably
        # stopped, and continuing to extrapolate sends it sliding off.
        if self.misses > 0 and dt > 0 and self.coast_s >= coast_grace_s:
            self.state[2:] *= coast_damping_per_s ** dt
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        self.state = F @ self.state
        G = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt],
                      [dt, 0.0], [0.0, dt]])
        self.P = F @ self.P @ F.T + G @ (accel_var * np.eye(2)) @ G.T

    def update(self, zx, zy, meas_var=1.0):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        R = meas_var * np.eye(2)
        z = np.array([zx, zy], dtype=np.float64)
        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def gate_radius(self, base_px):
        """How far from the prediction a detection may be and still count
        as the same token for GEOMETRIC (identity-blind) association.

        Deliberately capped at a modest multiple of base_px, not allowed
        to grow without bound with coast time. An unbounded gate here was
        tried and is actively dangerous: a track that has been missing for
        a while has a genuinely unreliable position estimate, and a wide
        gate lets it geometrically capture whatever detection happens to
        be nearest -- including a completely different, real token's own
        detection, silently, since distance alone cannot tell them apart.
        That is a worse failure than the one it was meant to fix (a fast
        token outrunning its search window), because it corrupts a
        DIFFERENT, otherwise-healthy track rather than just failing to
        recover the fast one.

        Recovering a token that travelled far while unseen is instead the
        job of the identity-gated rescue in Tracker.step: a decoded id is
        a positive, hard-to-fake match, so it is safe to accept at any
        distance, whereas raw proximity is not."""
        sigma = float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))
        return min(base_px + 3.0 * sigma, 3.0 * base_px)

    def add_evidence(self, marker_id, weight, decay, now):
        for k in list(self.evidence):
            self.evidence[k] *= decay
            if self.evidence[k] < 0.02:
                del self.evidence[k]
        if marker_id is not None and weight > 0:
            self.evidence[marker_id] = self.evidence.get(marker_id, 0.0) + weight
            self.last_decode = now

    def best_identity(self, min_evidence, switch_ratio):
        if not self.evidence:
            return None
        ranked = sorted(self.evidence.items(), key=lambda kv: -kv[1])
        top_id, top_w = ranked[0]
        if top_w < min_evidence:
            return None
        if self.marker_id is not None and top_id != self.marker_id:
            # hysteresis: a challenger must clearly beat the incumbent
            incumbent = self.evidence.get(self.marker_id, 0.0)
            if top_w < incumbent * switch_ratio:
                return self.marker_id
        return top_id


class Tracker:
    def __init__(self, geom, px_per_mm,
                 max_coast_s=2.0, min_evidence=1.0, switch_ratio=1.6,
                 evidence_decay_per_s=0.5, confirm_hits=2,
                 max_report_sigma_mm=12.0):
        self.geom = geom
        self.px_per_mm = px_per_mm
        self.base_px = geom.r_out * px_per_mm
        self.max_coast_s = max_coast_s
        self.min_evidence = min_evidence
        self.switch_ratio = switch_ratio
        self.evidence_decay_per_s = evidence_decay_per_s
        self.confirm_hits = confirm_hits
        self.max_report_sigma_px = max_report_sigma_mm * px_per_mm
        self.tracks = []
        self._last_t = None

    def step(self, detections, decodes, now=None):
        """detections: list of dicts with x, y, px_per_mm.
        decodes: parallel list of (marker_id or None, margin).

        Returns the list of REPORTABLE tracks -- confirmed and identified.
        """
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now

        for t in self.tracks:
            t.predict(dt)

        pairs = []
        for di, d in enumerate(detections):
            for ti, t in enumerate(self.tracks):
                tx, ty = t.pos
                dist = float(np.hypot(d["x"] - tx, d["y"] - ty))
                if dist <= t.gate_radius(self.base_px * 0.6):
                    pairs.append((dist, di, ti))
        pairs.sort(key=lambda p: p[0])

        used_d, used_t = set(), set()
        assignment = {}
        for dist, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            assignment[ti] = di

        # Identity-gated rescue. A detection that positively decodes as a
        # track's established id IS that token, however far it has moved --
        # a geometric gate cannot know that a token was picked up and put
        # down across the table, but the printed code can. This is what
        # lets a fast flick be recovered in one frame instead of waiting
        # for the old track to expire and a new one to earn its identity,
        # and it is safe precisely because it requires a positive id match
        # rather than mere proximity.
        for di, (mid, margin) in enumerate(decodes):
            if di in used_d or mid is None:
                continue
            for ti, t in enumerate(self.tracks):
                if ti in used_t or t.marker_id != mid:
                    continue
                used_d.add(di)
                used_t.add(ti)
                assignment[ti] = di
                break

        decay = self.evidence_decay_per_s ** dt if dt > 0 else 1.0

        for ti, t in enumerate(self.tracks):
            if ti in assignment:
                d = detections[assignment[ti]]
                t.update(d["x"], d["y"])
                t.px_per_mm = d.get("px_per_mm", t.px_per_mm)
                t.hits += 1
                t.misses = 0
                t.note_seen(now)
                mid, margin = decodes[assignment[ti]]
                t.add_evidence(mid, max(0.0, margin), decay, now)
            else:
                t.misses += 1
                t.note_missed(dt)
                t.add_evidence(None, 0.0, decay, now)

        for di, d in enumerate(detections):
            if di in used_d:
                continue
            mid, margin = decodes[di]
            # A new track is only born from a detection that already reads
            # as a valid marker. Without this an arbitrary round object
            # becomes a track and then a visible phantom token, which is
            # precisely what happened in v1.
            if mid is None:
                continue
            t = Track(d["x"], d["y"], d.get("px_per_mm", self.px_per_mm), now)
            t.add_evidence(mid, max(0.0, margin), 1.0, now)
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks
                       if (now - t.last_seen) <= self.max_coast_s]

        for t in self.tracks:
            t.marker_id = t.best_identity(self.min_evidence, self.switch_ratio)
        self._resolve_collisions()

        # Stop REPORTING a track whose position has become too uncertain to
        # be worth drawing, while still keeping it alive internally so it
        # can be reclaimed the moment it is seen again. Reporting a
        # confidently-placed token that is actually a stale guess is worse
        # than briefly reporting nothing.
        return [t for t in self.tracks
                if t.marker_id is not None
                and t.hits >= self.confirm_hits
                and float(np.sqrt(max(t.P[0, 0], t.P[1, 1]))) <= self.max_report_sigma_px]

    def _resolve_collisions(self):
        """No two tracks may hold the same identity at once. The weaker
        claim is dropped rather than both being left ambiguous, but the
        loser keeps being TRACKED -- it just stops being reported until it
        earns an identity again."""
        by_id = {}
        for t in self.tracks:
            if t.marker_id is None:
                continue
            by_id.setdefault(t.marker_id, []).append(t)
        for mid, group in by_id.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda t: (t.evidence.get(mid, 0.0), -t.misses),
                       reverse=True)
            for loser in group[1:]:
                loser.evidence.pop(mid, None)
                loser.marker_id = loser.best_identity(self.min_evidence,
                                                      self.switch_ratio)
