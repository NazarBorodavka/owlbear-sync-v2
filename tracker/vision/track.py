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
        # The id this track has already been PUBLISHED under, latched the
        # first time it is reported. Identity is allowed to settle while a
        # track is still private, but the moment a position goes out under
        # an id, that pairing is a promise: see best_identity() for why
        # breaking it is worse than dropping the track.
        self.published_id = None
        # Where this track was last actually MEASURED, as opposed to
        # where predict() extrapolates it to. Reporting uses this while
        # coasting; see Pipeline.tokens_mm().
        self.last_measured = (float(x), float(y))
        self.evidence = {}          # marker_id -> accumulated weight
        # marker_id -> best single-frame margin ever recorded for it.
        # Separate from accumulated evidence on purpose -- see
        # add_evidence() and _resolve_collisions() for why tenure and
        # peak quality need to be tracked independently.
        self.best_margin = {}
        self.hits = 1
        self.misses = 0
        self.created = now
        self.last_seen = now
        self.last_decode = 0.0
        # How long this track has been fed CONSECUTIVE matched-but-
        # undecoded blobs. Reset by any positive decode of its own id,
        # and NOT advanced by ordinary misses -- see Tracker.step()'s hit
        # handling for why that distinction is the whole point.
        # When a detection last positively decoded as THIS track's id.
        # Publishing is anchored to this, not to 'we matched some
        # blob' -- see Tracker.step().
        self.last_confirm_t = now
        # Seconds since this track last held a CONFIRMED identity (zero
        # at birth, not just after one is lost -- a track that never
        # confirms at all is gated the same way as one that lost its id,
        # by design; see Tracker.step()'s pruning rule).
        # Speed at the moment of the last real detection, undamped. The
        # live velocity gets damped while coasting (see predict), so it is
        # useless for deciding how far away to search for a token that was
        # moving fast when we lost it -- which is exactly when a wide
        # search matters most. v1 hit this too and bolted on a
        # "last_seen_speed"; here it is part of the model from the start.
        self.last_speed = 0.0

    @property
    def pos(self):
        """The filter's CURRENT belief, prediction included. Use this for
        searching (it is what says where to look next), not for
        publishing."""
        return float(self.state[0]), float(self.state[1])

    @property
    def report_pos(self):
        """Position to publish.

        While being read, this is the measurement. During the brief
        unconfirmed window it is the LAST MEASUREMENT, held in place --
        not the filter's extrapolation.

        Extrapolating was tried here and measured strictly worse. Over
        the same clips, published positions that came from a measurement
        were a median 1.0-1.8mm from the real token; positions that came
        from the filter's prediction were 19-36mm out at the median and
        up to 205mm at worst. The velocity estimate at the instant a
        token is lost is exactly the least reliable thing about it --
        usually it was lost BECAUSE something unusual happened to its
        motion -- so running it forward amplifies the error instead of
        tracking the token. Holding the last known position is wrong by
        at most the distance the token travelled during a 2-4 frame gap,
        and it never invents motion that was not observed.

        The filter's prediction is still used, and is the right tool, for
        deciding WHERE TO SEARCH next frame (Track.pos). It is just not
        something to publish."""
        return self.last_measured if self.misses > 0 else self.pos

    @property
    def coast_s(self):
        return self._coast_s

    def note_seen(self, now):
        self._coast_s = 0.0
        self.last_seen = now
        self.last_speed = float(np.hypot(self.state[2], self.state[3]))

    def note_missed(self, dt):
        self._coast_s = getattr(self, "_coast_s", 0.0) + dt

    def predict(self, dt, accel_var=None, coast_damping_per_s=0.05,
                coast_grace_s=0.30, accel_mm_s2=4000.0):
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
        # accel_var was a bare 900 px^2/s^4, which is far too small to be
        # a real process noise for a hand-flicked token: measured on real
        # footage, sigma reached only 2.1px after TEN consecutive missed
        # frames. Nothing downstream could react to "I have lost this" --
        # max_report_sigma_px (42px here) never tripped, so a pure
        # extrapolation kept being reported, and gate_radius stayed ~47px,
        # far too tight to re-acquire a token that had moved 56-700px.
        # Deriving it from a physical acceleration instead keeps it
        # scale-correct: a hand can accelerate a disc at a few m/s^2, and
        # because an arriving measurement shrinks P again, a generous Q
        # costs nothing while a token is actually being seen -- it only
        # takes effect in exactly the case it is needed, a coasting one.
        if accel_var is None:
            accel_var = (accel_mm_s2 * self.px_per_mm) ** 2
        self.P = F @ self.P @ F.T + G @ (accel_var * np.eye(2)) @ G.T

    def update(self, zx, zy, meas_var=1.0):
        self.last_measured = (float(zx), float(zy))
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
        # The cap was 3x (i.e. 1.8x the token radius), which is LESS than
        # one frame of travel for a token moving at the speed this system
        # claims to support: 420mm/s at 15fps is 28mm/frame, ~90px at
        # 3.2px/mm, against a cap of ~109px that 3*sigma could not be
        # allowed to exceed. Measured consequence: a sharp, fast token
        # outran its own track's gate, so the detection at its new
        # position founded a COMPETING track while the original sat
        # coasting and reporting a stale position -- both an identity
        # flip and a mislocated report, from one flick.
        #
        # 5x keeps the anti-stealing property the cap exists for: six
        # tokens on a 905x510mm table sit well over 100mm apart in
        # practice, and this gate is ~57mm at the calibrated scale, so it
        # still cannot reach a different token's detection.
        return min(base_px + 3.0 * sigma, 5.0 * base_px)

    def add_evidence(self, marker_id, weight, decay, now):
        for k in list(self.evidence):
            self.evidence[k] *= decay
            if self.evidence[k] < 0.02:
                del self.evidence[k]
        if marker_id is not None and weight > 0:
            self.evidence[marker_id] = self.evidence.get(marker_id, 0.0) + weight
            # NOT decayed, and deliberately a running max rather than the
            # latest reading: this is "how confidently has this track EVER
            # read this id", which a long run of mediocre-but-consistent
            # reads should not be able to erase. See _resolve_collisions()
            # for why that distinction is load-bearing.
            self.best_margin[marker_id] = max(
                self.best_margin.get(marker_id, 0.0), weight)
            self.last_decode = now

    def best_identity(self, min_evidence, switch_ratio):
        if not self.evidence:
            return None
        ranked = sorted(self.evidence.items(), key=lambda kv: -kv[1])
        top_id, top_w = ranked[0]
        if top_w < min_evidence:
            return None
        if self.published_id is not None:
            # Already published under an id: that is now latched. A track
            # is a claim about one physical token, so "this is token 2,
            # no wait, it is token 0" is never a correct thing to emit --
            # on the table it reads as a mini silently becoming a
            # different character. switch_ratio hysteresis alone was NOT
            # enough: measured on real footage, a track would publish as
            # id2, then during a drag its own id's evidence decayed while
            # a stray early decode of id0 survived, and it relabelled
            # itself mid-life (1-2 such flips per moving clip).
            #
            # So once published, the only transitions allowed are "still
            # this id" or "no longer confident" -- the latter unconfirms
            # the track, which retires it and lets a fresh track
            # re-acquire the token cleanly. That shows up as churn (a
            # brief dropout, then reacquisition) instead of a flip, which
            # is the strictly safer way to be wrong.
            return self.published_id if self.evidence.get(
                self.published_id, 0.0) >= min_evidence else None
        if self.marker_id is not None and top_id != self.marker_id:
            # hysteresis: a challenger must clearly beat the incumbent
            incumbent = self.evidence.get(self.marker_id, 0.0)
            if top_w < incumbent * switch_ratio:
                return self.marker_id
        return top_id


class Tracker:
    def __init__(self, geom, px_per_mm,
                 max_coast_s=5.0, min_evidence=1.0, switch_ratio=1.6,
                 evidence_decay_per_s=0.5, confirm_hits=2,
                 max_report_sigma_mm=12.0, max_unconfirmed_s=1.5,
                 uncorroborated_min_margin=0.25,
                 predict_publish_s=0.25):
        self.geom = geom
        self.px_per_mm = px_per_mm
        self.base_px = geom.r_out * px_per_mm
        self.max_coast_s = max_coast_s
        self.min_evidence = min_evidence
        self.switch_ratio = switch_ratio
        self.evidence_decay_per_s = evidence_decay_per_s
        self.confirm_hits = confirm_hits
        self.max_report_sigma_px = max_report_sigma_mm * px_per_mm
        # Two places below trust a decoded id WITHOUT any nearby existing
        # track to corroborate it -- the identity-rescue match (accepts at
        # ANY distance) and new-track creation (nothing to compare against
        # yet). Both need MORE evidence than an ordinary local match, not
        # the same bare-minimum gate the decoder itself uses for a
        # detection that is already physically close to a track believed
        # to be that token. Measured against real footage: a weak-margin
        # marker (id 0 in this project's own printed set, see HANDOFF) was
        # occasionally misread off background texture (bare branches, a
        # projected map's gradient) elsewhere in the frame. Before this
        # fix, that first teleported an EXISTING track across the whole
        # image via the rescue path; closing that only moved the same
        # failure to the new-track path instead -- a phantom id-0 track
        # kept being born wherever the spurious reading occurred. Fixing
        # only one of the two was tried and measurably insufficient.
        # Set from measurement, not guesswork: real decodes during a
        # drag were measured at margin 0.135-0.212, so the 0.45 this was
        # first set to did not just block background misreads, it blocked
        # the legitimate fast-flick recovery the rescue path exists for.
        # 0.25 clears typical background noise while staying under a real
        # (if blurred) read. Detecting every frame also makes this far
        # less load-bearing than it was: a flicked token is now normally
        # re-detected at its new position directly, so the worst case if
        # this gate does block a rescue is identity churn, not a wrong
        # position.
        # uncorroborated_min_margin is independent of whatever the
        # Matcher's own (possibly loosened-by-calibration) gate is, by
        # design -- a claim made with no corroborating geometry is a much
        # higher-stakes claim than "this detection near where I already
        # expect the token to be is it", and should need correspondingly
        # stronger evidence regardless of how permissively the decoder
        # itself has been tuned.
        self.uncorroborated_min_margin = uncorroborated_min_margin
        # How long after its last CONFIRMED read a track may still be
        # published, from the filter's prediction rather than from a
        # measurement. Measured on real footage: for markers inside the
        # play area, the median publication gap is 100-200ms (2-4
        # frames) -- short blur/occlusion events. A window of 250ms
        # covers those without ever becoming open-ended extrapolation,
        # and beyond it the track stays ALIVE but silent, so identity
        # survives and re-acquisition is instant when it decodes again.
        self.predict_publish_s = predict_publish_s
        # Bounds a real, measured leak: a collision loser keeps being
        # tracked (by design -- it may yet earn its OWN identity if it is
        # a genuine second token, see _resolve_collisions), and a track
        # that stays geometrically glued to the winner keeps refreshing
        # last_seen every frame just by proximity, so max_coast_s's
        # age-since-last-MATCHED check never touches it. Measured against
        # real footage: motion blur occasionally splits one real marker's
        # detection into two, the extra one is born already decoding
        # correctly, loses its very first collision, and then never has
        # an independent existence to ever win one again -- one per bad
        # frame compounds to 1400+ phantom tracks and 11+ seconds/frame
        # within 170 frames. This bounds the population regardless of
        # root cause: any track with no CONFIRMED identity for this long
        # is dropped, win or never-won.
        self.max_unconfirmed_s = max_unconfirmed_s
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
        # for the old track to expire and a new one to earn its identity.
        #
        # Gated on margin, not just "decoded at all": a genuinely relocated
        # marker is still the same well-printed ring, so it decodes with
        # its usual healthy margin -- requiring one here costs that real
        # case nothing. What it stops is a WEAK, barely-gate-passing
        # decode (the marginal end of a marker's own quality range, or a
        # coincidental partial match on background texture) from being
        # trusted to relocate a track across the whole frame, which is a
        # far stronger claim than an ordinary nearby match and was
        # otherwise accepted on the same bare minimum the decoder itself
        # uses for everything else.
        for di, (mid, margin) in enumerate(decodes):
            if di in used_d or mid is None or margin < self.uncorroborated_min_margin:
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
                mid, margin = decodes[assignment[ti]]
                # A matched blob is NOT automatically this token.
                # refine_at() is seeded at the track's own predicted
                # position and will happily fit a circle to whatever
                # local gradient is there -- projected map texture, a
                # shadow, a table edge. Counting that as a hit is how a
                # track whose token had already moved away stayed at
                # misses=0 forever: never coasting, never dropped,
                # keeping its id from decayed evidence, and reporting a
                # FROZEN position. Measured on real footage: tracks
                # frozen for 40+ frames with no positive decode of their
                # own id for up to 1.3s.
                #
                # So a detection only moves a confirmed track if it
                # actually identifies as that track -- or is undecodable
                # but recent enough that a brief occlusion is the likely
                # explanation. A detection decoding as a DIFFERENT id is
                # never this token and must not drag it.
                # Only a positive decode of this track's OWN id may move
                # it. The previous rule also accepted a nearby blob that
                # failed to decode, on the theory that a briefly-blurred
                # token should not be dropped. Measured on real IR
                # footage, that is what produced the false positives:
                # 2.1% of everything published on the occlusion clip sat
                # where its own id does not decode at all -- a hand, a
                # shadow, in one case a spot reading as a DIFFERENT
                # marker. An unidentified blob is not evidence, and the
                # cost of refusing it is now covered by bounded
                # prediction at publish time instead.
                owns_it = (t.marker_id is None or mid == t.marker_id)
                if owns_it:
                    t.update(d["x"], d["y"])
                    t.px_per_mm = d.get("px_per_mm", t.px_per_mm)
                    t.hits += 1
                    t.misses = 0
                    t.note_seen(now)
                    if mid is not None and mid == t.marker_id:
                        t.last_confirm_t = now
                    t.add_evidence(mid, max(0.0, margin), decay, now)
                else:
                    # Seen something, but not this token: coast instead of
                    # being dragged onto it.
                    t.misses += 1
                    t.note_missed(dt)
                    t.add_evidence(None, 0.0, decay, now)
            else:
                t.misses += 1
                t.note_missed(dt)
                t.add_evidence(None, 0.0, decay, now)

        # A detection within gate range of some EXISTING track, but that
        # lost the greedy nearest-match race for it, is contested space --
        # not a new object. Measured directly against real footage: under
        # motion blur, the gradient-vote accumulator can briefly split one
        # real, already-tracked marker into two nearby peaks. The runner-up
        # peak still decodes correctly (it is sampling almost the same
        # code band), so without this check it is eligible to found a
        # brand-new track every single frame the blur persists -- which is
        # exactly what produced 1400+ phantom tracks and an 11+ second
        # frame time from one dragged token. gated_d reuses the SAME
        # gate_radius computation the association step already did, so
        # this costs nothing extra and stays exactly consistent with it.
        gated_d = {di for (_, di, _) in pairs}

        for di, d in enumerate(detections):
            if di in used_d or di in gated_d:
                continue
            mid, margin = decodes[di]
            # A new track is only born from a detection that already reads
            # as a valid marker, and (see Tracker.__init__) reads it with
            # enough margin that nothing is corroborating it except the
            # decode itself. Without the mid check an arbitrary round
            # object becomes a track and then a visible phantom token,
            # which is precisely what happened in v1; without the margin
            # check, a weak/marginal false decode becomes one instead.
            if mid is None or margin < self.uncorroborated_min_margin:
                continue
            t = Track(d["x"], d["y"], d.get("px_per_mm", self.px_per_mm), now)
            t.add_evidence(mid, max(0.0, margin), 1.0, now)
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks
                       if (now - t.last_seen) <= self.max_coast_s]

        for t in self.tracks:
            t.marker_id = t.best_identity(self.min_evidence, self.switch_ratio)
        self._resolve_collisions()

        # Prune on time since this track's OWN decode last confirmed its
        # id -- last_confirm_t -- not merely "does it currently hold an
        # id". Those are not the same thing, and the gap between them is
        # a real bug that was measured hiding a live token: a track that
        # WINS a collision (see _resolve_collisions) keeps t.marker_id
        # set indefinitely even with zero supporting detections, so a
        # rule that reset its clock on "still holds an id" never fired.
        # On real footage: a ghost at a frozen position went 38
        # consecutive misses (1.9s) with its one-time best-ever margin
        # (0.74) still edging out the real, currently-and-correctly-
        # tracking token (reading ~0.73 every single frame) by a hair,
        # every single collision -- the real token never got to publish
        # AT ALL while the ghost sat comfortably inside a grace window
        # that a rule confusing "holds an id" with "earned it recently"
        # kept resetting for it.
        #
        # A track that never establishes an id at all is covered too:
        # last_confirm_t starts at creation and is only ever advanced by
        # a genuine self-confirm, so it ages exactly like one that lost
        # its id would.
        # Uniform rule, no marker_id branch: a track that has NEVER
        # confirmed an id ages on last_confirm_t exactly the same way --
        # it starts at creation time (Track.__init__) and only a genuine
        # self-confirm ever advances it, so "never confirmed" and "used
        # to be confirmed but is not anymore" both show up as the same
        # growing age here.
        self.tracks = [t for t in self.tracks
                       if (now - t.last_confirm_t) <= self.max_unconfirmed_s]

        # Publish ONLY positions measured this frame (misses == 0). A
        # coasting track stays alive internally -- it keeps its identity
        # and its prediction is still what tells the detector where to
        # look next -- but it is not published, because there is no
        # honest answer to "where is it" for a token nobody can currently
        # see. Extrapolating was measured sliding a lost token clean off
        # the table; freezing at the last measurement merely lies more
        # quietly, and at 420mm/s even ONE frame of staleness is 28mm.
        # Measured on real footage, this costs almost nothing: with
        # full-frame detection every frame, only 0.6% of reports were
        # coasting at all.
        #
        # Also stop reporting a track whose position has become too
        # uncertain to be worth drawing. Reporting a
        # confidently-placed token that is actually a stale guess is worse
        # than briefly reporting nothing.
        # Publish a track if its identity was confirmed THIS frame, or
        # recently enough that a short prediction is still honest. The
        # old rule published whenever a blob had been matched, which is
        # how an unidentified blob became a reported token.
        out = [t for t in self.tracks
               if t.marker_id is not None
               and t.hits >= self.confirm_hits
               and (now - t.last_confirm_t) <= self.predict_publish_s
               and float(np.sqrt(max(t.P[0, 0], t.P[1, 1]))) <= self.max_report_sigma_px]
        # Latch identity at the moment of publication, not at
        # confirmation: a track that settled on an id but never actually
        # got reported has promised nothing, so it stays free to change
        # its mind.
        for t in out:
            if t.published_id is None:
                t.published_id = t.marker_id
        return out

    def _resolve_collisions(self):
        """No two tracks may hold the same identity at once. The weaker
        claim is dropped rather than both being left ambiguous, but the
        loser keeps being TRACKED -- it just stops being reported until it
        earns an identity again.

        Ranked by best-ever single-frame margin FIRST, accumulated
        evidence only as a tiebreak -- not the other way round. Measured
        against real footage: a track seeded early (e.g. the very first
        full-frame scan) from a marginal, repeatedly-misread patch of
        background texture accumulates a large total just by surviving
        many frames, each individually weak. A real marker discovered
        later starts from zero and, ranked on raw accumulated evidence
        alone, can mathematically never catch up -- it loses every
        collision from birth, which (combined with Tracker's own
        max_unconfirmed_s pruning) gets it deleted before it can ever
        accumulate enough history to compete on those terms. But its
        single reading is typically far HIGHER quality (a real marker
        decisively beats every other codeword) than anything the
        phantom has ever produced, which is exactly the signal tenure-
        based ranking was throwing away."""
        by_id = {}
        for t in self.tracks:
            if t.marker_id is None:
                continue
            by_id.setdefault(t.marker_id, []).append(t)
        for mid, group in by_id.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda t: (t.best_margin.get(mid, 0.0),
                                      t.evidence.get(mid, 0.0), -t.misses),
                       reverse=True)
            for loser in group[1:]:
                loser.evidence.pop(mid, None)
                loser.marker_id = loser.best_identity(self.min_evidence,
                                                      self.switch_ratio)
