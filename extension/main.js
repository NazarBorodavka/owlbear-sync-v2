/* Owlbear Rodeo extension: syncs physical token positions from the
 * tracker's /api/tokens into OBR scene items, with a mapping between
 * physical marker IDs and OBR item IDs that PERSISTS between sessions.
 *
 * v1's equivalent of that mapping (`tokenMapping`) was a plain in-memory
 * JS variable -- never written to OBR.scene.setMetadata or anywhere else
 * durable, so it was lost on every page reload or extension restart,
 * directly contradicting what this version is required to do. Fixed here
 * by storing the mapping IN the scene's own metadata: it travels with the
 * scene file itself (survives reload, survives re-opening the scene,
 * survives on a different machine if the scene is shared), which is the
 * correct Owlbear Rodeo mechanism for exactly this kind of durable,
 * scene-scoped state.
 */
import OBR from "https://cdn.jsdelivr.net/npm/@owlbear-rodeo/sdk@3.1.0/+esm";

const METADATA_KEY = "com.physicaltokentracker/mapping";
const POLL_INTERVAL_MS = 150;     // ~6.7Hz poll of /api/tokens -- see README for why
                                   // this is plenty: the tracker pipeline itself runs
                                   // at 15fps+, and a local-network JSON poll at this
                                   // rate costs single-digit ms, nowhere near a
                                   // bottleneck worth a WebSocket's added complexity.
// Exponential easing rate, 1/s. Was declared in v2 and never used -- the
// scene was written with the raw tracked position each poll, so a token
// advanced in ~150ms steps. 20/s is a ~50ms time constant: visibly a
// glide rather than a jump, without lagging far enough behind a dragged
// mini to look detached from it.
//
// Steady-state lag behind a mini moving at constant speed is about
// speed/rate, so at 20/s: 5mm at a 100mm/s nudge, 25mm at a 500mm/s
// flick -- under a token base even at the fast end. Lower it for a more
// obvious glide at the cost of trailing further behind the hand; most of
// the visible improvement comes from ticking at PUSH_INTERVAL_MS at all,
// which turns ~7 scene writes a second into 20.
const SMOOTH_RATE = 20;
// A step larger than this is not a hand moving a mini; it is a
// lift-and-place or a re-acquisition. Eased, those would slide the token
// across everything in between. 150mm is well above what continuous
// motion produces between polls -- a brisk 500mm/s drag covers ~75mm in
// a 150ms poll -- so real drags stay smooth and only genuine relocations
// snap.
const SNAP_MM = 150.0;
// 20Hz. The poll runs at 150ms, which is too coarse to ease against, but
// pushing at display rate would broadcast a scene update 60 times a
// second to every connected client.
const PUSH_INTERVAL_MS = 50;
const MIN_MOVE_SCENE_UNITS = 0.5;

let trackerUrl = localStorage.getItem("trackerUrl") || "";
let trackerAuth = localStorage.getItem("trackerAuth") || "";   // "user:pass", base64'd on use
let mapping = {};              // physicalId (string) -> OBR item id
let sceneItemsCache = [];
let virtualTokens = [];
let latestPhysicalTokens = [];
let playAreaMm = null;
// "scene" | "viewport" -- see pushPositions() for what each means.
// Deliberately localStorage and NOT scene metadata: it says which screen
// this particular machine should anchor to, so it is a property of the
// client, not of the shared scene. Storing it in the scene would make the
// host and the projector overwrite each other's choice.
let positionAnchor = localStorage.getItem("positionAnchor") || "scene";
// Travel Mode: while true, positions are not pushed to the scene at all.
// Deliberately NOT persisted -- leaving it on across a reload would look
// exactly like a broken tracker, and it is meant to be held briefly while
// the map is repositioned, not left set.
let travelling = false;
// Deadband, in millimetres: a new reading closer than this to the last
// ACCEPTED one is ignored, so a token at rest stays put instead of
// twitching on measurement noise. Compared against the last accepted
// reading rather than the eased position, which makes it true hysteresis
// -- repeated sub-threshold wobble cannot accumulate into a drift, and
// one reading past the threshold releases it immediately.
//
// Sized against measured stationary noise: mean radial jitter
// 0.35-0.50mm, worst-case excursion 2.06mm over a 30s clip. 2mm sits
// above the mean and around the worst case, and is a twentieth of a
// token base.
let deadbandMm = parseFloat(localStorage.getItem("deadbandMm"));
if (!isFinite(deadbandMm) || deadbandMm < 0) deadbandMm = 2.0;
// The scene-space rectangle the projector is throwing onto the play area:
// {x, y, w, h} in SCENE units. Scene units, not screen pixels, on purpose
// -- Owlbear's "sync view" copies POSITION AND SCALE between clients, not
// window size, so if the projector window is a different pixel size from
// the host's the two show different extents and host pixels would be the
// wrong ruler. A scene rectangle is what the projector actually covers,
// whoever captured it.
let projectorRect = null;
let syncViewActive = false;
let pollTimer = null;
let isReady = false;

function authHeaders() {
  if (!trackerAuth) return {};
  return { Authorization: "Basic " + btoa(trackerAuth) };
}

async function apiFetch(path, options = {}) {
  const url = new URL(path, trackerUrl).toString();
  const res = await fetch(url, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------
// Persistent mapping (scene metadata)
// ---------------------------------------------------------------------
async function loadMapping() {
  try {
    const meta = await OBR.scene.getMetadata();
    mapping = meta[METADATA_KEY] || {};
  } catch (e) {
    console.error("Failed to load token mapping from scene metadata:", e);
    mapping = {};
  }
}

let _saveTimer = null;
function saveMappingDebounced() {
  // Debounced: a flurry of manual re-assignments in the UI should not
  // fire a scene.setMetadata round trip per keystroke/click.
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(async () => {
    try {
      await OBR.scene.setMetadata({ [METADATA_KEY]: mapping });
    } catch (e) {
      console.error("Failed to persist token mapping:", e);
    }
  }, 300);
}

// ---------------------------------------------------------------------
// Scene item tracking
// ---------------------------------------------------------------------
OBR.onReady(async () => {
  isReady = true;
  setStatus("Loading scene...", "ready");

  await loadMapping();
  try {
    const meta = await OBR.scene.getMetadata();
    if (meta[PROJECTOR_RECT_KEY]) projectorRect = meta[PROJECTOR_RECT_KEY];
  } catch (e) {
    console.error("Could not load the stored projector view:", e);
  }
  watchSyncView();

  OBR.scene.items.onChange((items) => {
    sceneItemsCache = items;
    virtualTokens = items.filter(
      (item) => item.layer === "CHARACTER" || item.layer === "MOUNT"
    );
    renderMappingUI();
  });

  try {
    const items = await OBR.scene.items.getItems();
    sceneItemsCache = items;
    virtualTokens = items.filter(
      (item) => item.layer === "CHARACTER" || item.layer === "MOUNT"
    );
  } catch (e) {
    console.error("Initial scene items fetch failed:", e);
  }

  // A mapped item can be deleted from the scene by the DM at any time --
  // drop its mapping entry rather than silently updating nothing forever
  // (which looked, from the UI, like the physical token had simply
  // stopped being tracked).
  OBR.scene.items.onChange((items) => {
    const liveIds = new Set(items.map((i) => i.id));
    let changed = false;
    for (const [physId, itemId] of Object.entries(mapping)) {
      if (!liveIds.has(itemId)) {
        delete mapping[physId];
        changed = true;
      }
    }
    if (changed) saveMappingDebounced();
  });

  wireUi();
  // Restore BOTH saved fields. trackerAuth was being written to
  // localStorage and never read back into its input, so the box looked
  // empty every session even though polling was authenticating fine --
  // which invites retyping it, and a typo there breaks the connection.
  if (trackerUrl) document.getElementById("trackerUrl").value = trackerUrl;
  if (trackerAuth) document.getElementById("trackerAuth").value = trackerAuth;
  if (trackerUrl) startPolling();
  setStatus(trackerUrl ? "Connecting..." : "Enter tracker URL to connect", "ready");
});

// ---------------------------------------------------------------------
// Polling the tracker
// ---------------------------------------------------------------------
function startPolling() {
  stopPolling();
  pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS);
  pollOnce();
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  if (pushTimer) clearInterval(pushTimer);
  pushTimer = null;
  lastPushAt = 0;
}

async function pollOnce() {
  try {
    const data = await apiFetch("/api/tokens");
    latestPhysicalTokens = data.tokens || [];
    playAreaMm = data.play_area_mm;
    setStatus(
      data.connected ? `Tracking (${latestPhysicalTokens.length} tokens, ${data.fps.toFixed(0)}fps)` : "Camera disconnected",
      data.connected ? "ready" : "warn"
    );
    autoMatchByName();
    await pushPositions();
    renderMappingUI();
  } catch (e) {
    setStatus(`Tracker error: ${e.message}`, "error");
  }
}

// Convenience auto-matching: if a physical token's assigned name (set in
// the web dashboard) matches an unmapped scene item's name, map it
// automatically. This never overrides an EXISTING mapping (manual or
// previously auto-matched), only fills gaps -- same non-destructive
// behaviour v1 had, just now actually persisted afterward.
function autoMatchByName() {
  let changed = false;
  for (const pt of latestPhysicalTokens) {
    const key = String(pt.marker_id);
    if (mapping[key] || !pt.name) continue;
    const match = virtualTokens.find(
      (vt) => (vt.text && vt.text.plainText === pt.name) || vt.name === pt.name
    );
    if (match) {
      mapping[key] = match.id;
      changed = true;
    }
  }
  if (changed) saveMappingDebounced();
}

// ---------------------------------------------------------------------
// Position sync
// ---------------------------------------------------------------------
// Smoothing happens in MILLIMETRE space, not scene space, and that is
// what makes a physical drag and a map re-anchor behave differently
// without either being special-cased:
//
//   a mini is moved      -> its millimetres change -> eased, so the
//                           token glides to the new spot
//   the map is re-synced -> the mm->scene CONVERSION changes while the
//                           millimetres do not -> every token jumps at
//                           once, which is what should happen; the map
//                           moved, not the minis
//
// Easing in scene space would smear that jump into a slow slide of every
// token across the map, which is exactly the "tokens walking" being
// avoided elsewhere.
let targetMm = {};        // marker_id -> {x, y, name} most recent READ
let smoothMm = {};        // marker_id -> {x, y} what is on the map now
let anchorCtx = null;     // mm -> scene conversion, refreshed per poll
let pushTimer = null;
let lastPushAt = 0;

// Refreshed once per poll rather than per token: these are properties of
// the scene or the client. The grid used to be queried inside the token
// loop -- six tokens at a 150ms poll meant ~80 SDK round trips a second
// to re-read two numbers that had not changed.
async function refreshAnchorContext() {
  if (!playAreaMm) { anchorCtx = null; return; }
  const [widthMm, heightMm] = playAreaMm;
  try {
    if (positionAnchor === "projector") {
      // While sync view is on the projector IS this view, so read it
      // live; otherwise use the rect frozen when sync stopped.
      const rect = syncViewActive ? await visibleSceneRect() : projectorRect;
      if (!rect) { anchorCtx = null; return; }
      if (syncViewActive) {
        // Keep the stored rect current the whole time sync is live, so
        // that turning sync off leaves a usable calibration behind
        // whether or not the off-transition event is seen. Writing it to
        // scene metadata is left to that transition -- persisting at poll
        // rate would be a scene write several times a second.
        projectorRect = rect;
      }
      anchorCtx = { mode: "projector", widthMm, heightMm, rect };
    } else if (positionAnchor === "viewport") {
      const [w, h] = await Promise.all([
        OBR.viewport.getWidth(),
        OBR.viewport.getHeight(),
      ]);
      if (!(w > 0) || !(h > 0)) { anchorCtx = null; return; }
      anchorCtx = { mode: "viewport", widthMm, heightMm, w, h };
    } else {
      // Only scene anchoring needs the grid, so a scene with no grid
      // configured still syncs in the other two modes.
      const [dpi, scale] = await Promise.all([
        OBR.scene.grid.getDpi(),
        OBR.scene.grid.getScale(),
      ]);
      const mmPerCell = scale.parsed.multiplier * mmPerUnit(scale.parsed.unit);
      if (!(dpi > 0) || !(mmPerCell > 0)) { anchorCtx = null; return; }
      anchorCtx = { mode: "scene", widthMm, heightMm, pxPerMm: dpi / mmPerCell };
    }
  } catch (e) {
    anchorCtx = null;   // not ready yet; try again next poll
  }
}

async function mmToScene(xMm, yMm, ctx) {
  if (ctx.mode === "projector") {
    // The projector maps its whole visible scene rect onto the play
    // area, so a token's millimetres are just a fraction of that rect.
    // No screen pixels in this path at all, which is what makes it
    // independent of the projector window's pixel size.
    return {
      x: ctx.rect.x + (xMm / ctx.widthMm) * ctx.rect.w,
      y: ctx.rect.y + (yMm / ctx.heightMm) * ctx.rect.h,
    };
  }
  if (ctx.mode === "viewport") {
    return OBR.viewport.inverseTransformPoint({
      x: (xMm / ctx.widthMm) * ctx.w,
      y: (yMm / ctx.heightMm) * ctx.h,
    });
  }
  return { x: xMm * ctx.pxPerMm, y: yMm * ctx.pxPerMm };
}

// Called from the poll: records WHERE each token is, and does not touch
// the scene. Separating "what the camera says" from "what gets drawn"
// is what lets the two run at different rates -- the tracker reports at
// its own pace, the easing runs at PUSH_INTERVAL_MS.
function updateTargets() {
  for (const pt of latestPhysicalTokens) {
    // Only act on a position that was actually READ this frame.
    //
    // The tracker also publishes HELD positions -- the last confirmed
    // place of a token whose code it cannot currently read (a hand over
    // it, a mini covering part of the ring). Those are correct as "where
    // it was", but they are not news, and pushing them would let a stale
    // or slightly-off position overwrite a good one. Skipping them
    // leaves whatever is on the map untouched, which is right for a
    // token that is merely unreadable: it has not moved as far as
    // anyone knows.
    if (pt.coasting) continue;
    const key = String(pt.marker_id);
    const prev = targetMm[key];
    if (prev && deadbandMm > 0 &&
        Math.hypot(pt.x_mm - prev.x, pt.y_mm - prev.y) < deadbandMm) {
      // Inside the deadband: hold position, but still let a rename
      // through -- suppressing jitter should not suppress a new name.
      prev.name = pt.name;
      continue;
    }
    targetMm[key] = { x: pt.x_mm, y: pt.y_mm, name: pt.name };
  }
}

// Guards the live rect refresh below. While the map is being panned the
// parent frame is busy, so a round trip can take longer than the tick
// interval; without this, ticks would queue refreshes faster than they
// complete and fall further and further behind.
let _rectRefreshInFlight = false;

async function pushTick() {
  if (travelling) return;          // see Travel Mode
  if (!anchorCtx) return;

  // While sync view is on, the projector IS this view, so the rect has to
  // keep up with the pan rather than with the tracker poll. Refreshing it
  // only once per 150ms poll made following a moving map visibly choppy.
  // This runs at tick rate instead, and is the only SDK read in the hot
  // path.
  if (anchorCtx.mode === "projector" && syncViewActive
      && !_rectRefreshInFlight) {
    _rectRefreshInFlight = true;
    visibleSceneRect()
      .then((r) => { if (r) { anchorCtx.rect = r; projectorRect = r; } })
      .catch(() => {})
      .finally(() => { _rectRefreshInFlight = false; });
  }

  const now = performance.now();
  const dt = lastPushAt ? Math.min((now - lastPushAt) / 1000, 0.5) : 0;
  lastPushAt = now;
  // Frame-rate independent exponential easing: the fraction covered per
  // tick is derived from dt, so a slow tick does not crawl and a fast
  // one does not overshoot.
  const alpha = dt > 0 ? 1 - Math.exp(-SMOOTH_RATE * dt) : 1;

  const updates = [];
  for (const [key, tgt] of Object.entries(targetMm)) {
    const itemId = mapping[key];
    if (!itemId) continue;
    const targetItem = sceneItemsCache.find((i) => i.id === itemId);
    if (!targetItem) continue;

    let cur = smoothMm[key];
    const far = cur && Math.hypot(tgt.x - cur.x, tgt.y - cur.y) > SNAP_MM;
    if (!cur || far) {
      // First sight of this token, or a step too large to be a hand
      // moving it -- a lift-and-place, or a re-acquisition after the
      // code was unreadable for a while. Easing across that would slide
      // the token over everything in between; go straight there.
      cur = { x: tgt.x, y: tgt.y };
    } else {
      cur = { x: cur.x + (tgt.x - cur.x) * alpha,
              y: cur.y + (tgt.y - cur.y) * alpha };
    }
    smoothMm[key] = cur;

    const scenePoint = await mmToScene(cur.x, cur.y, anchorCtx);
    const dist = Math.hypot(
      targetItem.position.x - scenePoint.x,
      targetItem.position.y - scenePoint.y
    );
    const needsNameUpdate =
      tgt.name && targetItem.name !== tgt.name &&
      !(targetItem.text && targetItem.text.plainText === tgt.name);
    // Below the threshold the easing has effectively arrived, so this
    // also stops the ticking once everything is at rest -- no endless
    // sub-pixel writes to a scene nobody is moving.
    if (dist > MIN_MOVE_SCENE_UNITS || needsNameUpdate) {
      updates.push({ id: itemId, position: scenePoint, name: tgt.name });
    }
  }

  if (updates.length === 0) return;
  await OBR.scene.items.updateItems(
    updates.map((u) => u.id),
    (items) => {
      for (const item of items) {
        const u = updates.find((x) => x.id === item.id);
        if (!u) continue;
        item.position = u.position;
        if (u.name) {
          item.name = u.name;
          if (item.text) item.text.plainText = u.name;
        }
      }
    }
  );
}

function startPushLoop() {
  if (pushTimer) return;
  pushTimer = setInterval(() => { pushTick().catch(() => {}); },
                          PUSH_INTERVAL_MS);
}

async function pushPositions() {
  if (travelling) return;
  updateTargets();
  await refreshAnchorContext();
  startPushLoop();
}

function mmPerUnit(unit) {
  // Owlbear Rodeo grid units -- mm equivalents for the auto-zoom math.
  // Confirmed real unit strings via the SDK docs: "ft" and "m" (the
  // default/metric options OBR's own grid settings UI offers); "in" and
  // "cm" included defensively for custom scales. Falls back to feet
  // (the SDK's own default) with a console warning rather than silently
  // guessing wrong with no trace, if an unrecognised unit ever appears.
  const table = { ft: 304.8, m: 1000, in: 25.4, cm: 10, mi: 1609344, km: 1000000 };
  if (!(unit in table)) {
    console.warn(`Unrecognised grid unit "${unit}", assuming feet`);
    return table.ft;
  }
  return table[unit];
}

// ---------------------------------------------------------------------
// Auto-zoom: align OBR's grid to the physical token's real base size
// ---------------------------------------------------------------------
// There used to be a second implementation here that measured the
// projected grid through the camera (POST /api/grid/measure). It is gone:
// the near-IR bandpass filter makes projector light invisible to the
// camera by design, so there was no longer a grid for it to find, and it
// could only ever report "no grid pattern found". The analytic version
// below needs no camera at all.
const TOKEN_BASE_MM = 40.0;   // physical base diameter -- keep in sync with
                               // tracker/vision/geometry.py's RingGeometry.base_d_mm
                               // (the PRINTED marker is 38mm; the physical base it
                               // sits on, which is what should visually match one
                               // grid cell, is 40mm)

// Analytic auto-zoom: no camera measurement, from the play-area size.
//
// The geometry is fully determined without any measurement:
//
//   the projector throws the whole OBR viewport onto the play area, so
//     screen px per real mm  =  viewport px / play-area mm
//   a grid cell is dpi scene px, shown at `scale` screen px per scene px,
//     so its real size is     dpi * scale / (screen px per mm)
//   setting that equal to TOKEN_BASE_MM and solving:
//     scale = TOKEN_BASE_MM * (screen px per mm) / dpi
//
// One assumption, and it is load-bearing: the projected image fills the
// calibrated play area. It is checked below rather than trusted -- the
// two axes give an independent estimate of screen-px-per-mm each, and if
// the projection does NOT line up with the play area they disagree.
async function runAutoZoomFromPlayArea(statusEl) {
  if (!playAreaMm) {
    statusEl.textContent =
      "Need the tracker connected and the play area calibrated first " +
      "(this reads the play-area size from /api/tokens).";
    return;
  }
  const [widthMm, heightMm] = playAreaMm;

  let vw, vh, dpi;
  try {
    [vw, vh, dpi] = await Promise.all([
      OBR.viewport.getWidth(),
      OBR.viewport.getHeight(),
      OBR.scene.grid.getDpi(),
    ]);
  } catch (e) {
    statusEl.textContent = `Auto-zoom failed reading the viewport: ${e.message}`;
    return;
  }
  if (!(dpi > 0) || !(vw > 0) || !(vh > 0)) {
    statusEl.textContent =
      `Auto-zoom failed: viewport ${vw}x${vh}, grid dpi ${dpi} -- ` +
      "is a scene open with its grid configured?";
    return;
  }

  const pxPerMmX = vw / widthMm;
  const pxPerMmY = vh / heightMm;
  // Relative disagreement between the two axes. Zero when the viewport
  // and the play area have the same aspect ratio, which is what "the
  // projection fills the play area" implies.
  const skew = Math.abs(pxPerMmX - pxPerMmY) / Math.min(pxPerMmX, pxPerMmY);
  const pxPerMm = 0.5 * (pxPerMmX + pxPerMmY);

  const newScale = (TOKEN_BASE_MM * pxPerMm) / dpi;
  if (!isFinite(newScale) || newScale <= 0) {
    statusEl.textContent = `Auto-zoom computed a nonsense scale (${newScale}); aborted.`;
    return;
  }

  // Scale ONLY -- deliberately not viewport position. Re-centring would
  // slide the map under minis that have not moved, which is exactly the
  // "tokens walk across the map when you pan" problem; choosing what is
  // under the table is a separate, deliberate action.
  try {
    await OBR.viewport.setScale(newScale);
  } catch (e) {
    statusEl.textContent = `Auto-zoom could not apply the scale: ${e.message}`;
    return;
  }

  // Changing the scale changes how much map the projector covers, so any
  // projector rect captured at the OLD scale is now wrong -- it would
  // place every token against an area the projector no longer shows.
  // While sync view is on the projector is following this client, so the
  // rect can simply be re-read; otherwise it has to be re-pinned, and
  // saying so is better than leaving a silently stale calibration behind.
  let rectNote = "";
  if (syncViewActive) {
    await captureProjectorRect("re-read after auto-zoom");
    rectNote = " Projector view re-read at the new zoom.";
  } else if (projectorRect) {
    projectorRect = null;
    try {
      await OBR.scene.setMetadata({ [PROJECTOR_RECT_KEY]: undefined });
    } catch (e) { /* best effort */ }
    describeProjector();
    rectNote = " The stored projector view was captured at the old zoom "
      + "and has been cleared -- switch sync view on once to recapture "
      + "it at the new zoom.";
  }

  // Reported PER AXIS on purpose. (dpi * newScale) / pxPerMm would just
  // give TOKEN_BASE_MM back -- it is the equation above rearranged, so it
  // would read as a confirmation while actually being an identity that
  // holds no matter how wrong the inputs are. Against each axis's own
  // px/mm it is a real measurement: the two differ by exactly the skew,
  // so a reader can see what the single chosen scale actually achieves in
  // each direction.
  const cellMmX = (dpi * newScale) / pxPerMmX;
  const cellMmY = (dpi * newScale) / pxPerMmY;
  let msg =
    `Done: 1 cell now measures ${cellMmX.toFixed(1)}mm across and ` +
    `${cellMmY.toFixed(1)}mm down on the table (target ${TOKEN_BASE_MM}mm). ` +
    `Play area ${widthMm}x${heightMm}mm shown in a ${vw}x${vh}px viewport, ` +
    `grid dpi ${dpi}, scale ${newScale.toFixed(4)}.`
    + rectNote;
  if (skew > 0.02) {
    msg +=
      ` WARNING: the two axes disagree by ${(100 * skew).toFixed(1)}% ` +
      `(${pxPerMmX.toFixed(3)} vs ${pxPerMmY.toFixed(3)} px/mm), so the ` +
      `projected image probably does not line up with the calibrated ` +
      `play area. Treat this zoom as approximate until that is fixed.`;
  }
  statusEl.textContent = msg;
}

// ---------------------------------------------------------------------
// Projector view
// ---------------------------------------------------------------------
const PROJECTOR_RECT_KEY = "com.physicaltokentracker/projectorRect";

// The scene rectangle currently visible in THIS client's viewport,
// obtained by asking the SDK to convert the two screen corners rather
// than by reconstructing the transform from getPosition()/getScale().
//
// That reconstruction is what put every token in the top-left corner of
// the map. ViewportTransform is just {position, scale} with no stated
// convention, and this assumed position was the scene coordinate at the
// screen origin, so the visible rect was read as position..position+
// size/scale. Owlbear means the opposite -- position is where the scene
// origin lands ON SCREEN -- which makes the real top-left -position/scale.
// With a position near zero that lands everything at scene (0, 0), and
// any small viewport change jitters it.
//
// inverseTransformPoint is the host app's own screen->scene conversion,
// so it is right under either convention and cannot drift from it. The
// lesson is narrower than "use the SDK": do not re-derive a transform
// whose convention is not written down anywhere.
// Viewport pixel size, cached. getWidth()/getHeight() describe the
// Owlbear canvas, which changes only when the window is resized -- not
// when the map is panned or zoomed -- so re-reading them on every rect
// refresh cost a whole extra round trip for two numbers that are almost
// always the same.
let _vpSize = null;
let _vpSizeAt = 0;
const VP_SIZE_TTL_MS = 2000;

async function viewportSize() {
  const now = performance.now();
  if (_vpSize && (now - _vpSizeAt) < VP_SIZE_TTL_MS) return _vpSize;
  const [w, h] = await Promise.all([
    OBR.viewport.getWidth(),
    OBR.viewport.getHeight(),
  ]);
  if (!(w > 0) || !(h > 0)) return null;
  _vpSize = { w, h };
  _vpSizeAt = now;
  return _vpSize;
}

// The scene rectangle currently visible in THIS client's viewport,
// obtained by asking the SDK to convert the two screen corners rather
// than by reconstructing the transform from getPosition()/getScale().
//
// That reconstruction is what put every token in the top-left corner of
// the map. ViewportTransform is just {position, scale} with no stated
// convention, and this assumed position was the scene coordinate at the
// screen origin. Owlbear means the opposite -- position is where the
// scene origin lands ON SCREEN -- which makes the real top-left
// -position/scale. inverseTransformPoint is the host app's own
// screen->scene conversion, so it is right under either convention. The
// lesson is narrower than "use the SDK": do not re-derive a transform
// whose convention is not written down anywhere.
//
// One round trip, not two: the size comes from the cache above, so the
// two corner conversions are the only thing awaited. Every await here is
// a postMessage into the parent frame, and while the map is actively
// being panned that frame is busy drawing -- so an extra serial hop
// showed up directly as tokens lagging behind the map.
async function visibleSceneRect() {
  const size = await viewportSize();
  if (!size) return null;
  const [tl, br] = await Promise.all([
    OBR.viewport.inverseTransformPoint({ x: 0, y: 0 }),
    OBR.viewport.inverseTransformPoint({ x: size.w, y: size.h }),
  ]);
  const rect = { x: tl.x, y: tl.y, w: br.x - tl.x, h: br.y - tl.y };
  if (!(rect.w > 0) || !(rect.h > 0)) return null;
  return rect;
}

async function captureProjectorRect(reason) {
  const rect = await visibleSceneRect();
  if (!rect) return false;
  projectorRect = rect;
  try {
    await OBR.scene.setMetadata({ [PROJECTOR_RECT_KEY]: rect });
  } catch (e) {
    console.error("Could not persist the projector rect:", e);
  }
  describeProjector(reason);
  return true;
}

// Owlbear's sync view makes the players' viewports follow the GM's. So
// while it is ON the projector is showing exactly what this client shows,
// and the projector rect can simply be re-read every poll. The moment it
// goes OFF the projector FREEZES at whatever was on screen at that
// instant -- which is precisely the rect captured on the last tick before
// the transition. That is why this watches the flag rather than asking
// the operator to remember to press a button after every sync.
// Sync state is tracked from the EVENT PAYLOADS, synchronously, and the
// transition is decided before any await. That matters for the one-shot
// "Sync View" button, which fires two changes milliseconds apart (on,
// then immediately off).
//
// The previous version ignored the payloads and re-queried
// getSyncView() inside an async handler. Both invocations then awaited,
// and by the time either resolved the query returned the FINAL value --
// false -- so each saw wasActive false, reading false, and neither
// detected a transition. Nothing was captured, and a one-shot sync moved
// the projector while the stored rect silently kept pointing at the old
// view. The persistent toggle worked only because its two events are
// seconds apart and cannot race.
let _syncMine = false;
let _syncOthers = false;

function onSyncFlagsChanged() {
  const active = Boolean(_syncMine || _syncOthers);
  if (active === syncViewActive) return;
  syncViewActive = active;
  if (active) {
    describeProjector();
  } else {
    // Sync just ended: the projector is parked on what we are showing.
    captureProjectorRect("captured when sync view ended").catch((e) =>
      console.error("Could not capture the projector view:", e));
  }
}

function watchSyncView() {
  // Both halves matter. OBR.party is the OTHER players in the room -- it
  // does not include this client -- so a GM who has switched their own
  // sync view on to broadcast their view does not appear in it at all.
  // Watching only the party left syncViewActive permanently false on the
  // very client driving the projector.
  OBR.player.onChange((pl) => {
    _syncMine = Boolean(pl && pl.syncView);
    onSyncFlagsChanged();
  });
  OBR.party.onChange((players) => {
    _syncOthers = Array.isArray(players) && players.some((pl) => pl.syncView);
    onSyncFlagsChanged();
  });
  Promise.all([
    OBR.player.getSyncView().catch(() => false),
    OBR.party.getPlayers().catch(() => []),
  ]).then(([mine, players]) => {
    _syncMine = Boolean(mine);
    _syncOthers = players.some((pl) => pl.syncView);
    syncViewActive = Boolean(_syncMine || _syncOthers);
    describeProjector();
  });
}

// A one-shot sync performed HERE rather than in Owlbear's own UI, so the
// projector rect is captured in the same breath. Toggling sync on and
// straight back off is exactly what "sync once" does -- followers jump to
// this view, then stop following and stay there -- and it is also exactly
// the manual sequence that was already known to work.
//
// This exists because Owlbear's own one-shot Sync View button does NOT
// touch the syncView flag -- confirmed by testing, after the SDK bundle
// turned out to be a re-export stub that could not settle it either way.
// It moves the projector while changing no state an extension can see:
// there is no viewport event, and another client's viewport cannot be
// read. So that button is unusable here no matter how the listeners are
// written, and this is the supported one-shot path.
const SYNC_SETTLE_MS = 400;

async function syncProjectorToThisView(statusEl) {
  statusEl.textContent = "Syncing the projector to this view...";
  try {
    await OBR.player.setSyncView(true);
    // Give followers time to actually arrive before releasing them;
    // un-syncing too early would park them mid-animation.
    await new Promise((r) => setTimeout(r, SYNC_SETTLE_MS));
    await OBR.player.setSyncView(false);
  } catch (e) {
    statusEl.textContent = `Could not sync the projector: ${e.message}`;
    return;
  }
  // The off-transition above should already have captured it; doing it
  // again is harmless and covers the case where the flag never changed.
  const ok = await captureProjectorRect("synced to this view");
  if (!ok) {
    statusEl.textContent =
      "Synced, but the viewport could not be read -- is a scene open?";
  }
}

function describeProjector(reason) {
  const el = document.getElementById("projectorStatus");
  if (!el) return;
  if (syncViewActive) {
    el.textContent = "Sync view is ON -- the projector mirrors this view, "
      + "so projector anchoring tracks it live.";
    return;
  }
  if (!projectorRect) {
    el.textContent = "No projector view stored yet -- switch Owlbear's sync "
      + "view on once and it is captured automatically.";
    return;
  }
  const r = projectorRect;
  el.textContent = `Projector view: ${r.w.toFixed(0)}x${r.h.toFixed(0)} scene `
    + `units at (${r.x.toFixed(0)}, ${r.y.toFixed(0)})`
    + (reason ? ` -- ${reason}.` : ".");
}

// ---------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------
let lastStatusText = "Loading...";
let lastStatusClass = "ready";

function setStatus(text, cls) {
  // Remembered so Travel Mode can take over the status line and then
  // hand it back, instead of leaving a stale "Travelling" banner up.
  if (!travelling) { lastStatusText = text; lastStatusClass = cls; }
  if (travelling) { text = "Position sync PAUSED"; cls = "warn"; }
  const el = document.getElementById("status");
  if (el) {
    el.textContent = text;
    el.className = `status ${cls}`;
  }
}

function wireUi() {
  document.getElementById("connectBtn").addEventListener("click", () => {
    trackerUrl = document.getElementById("trackerUrl").value.trim();
    trackerAuth = document.getElementById("trackerAuth").value.trim();
    localStorage.setItem("trackerUrl", trackerUrl);
    localStorage.setItem("trackerAuth", trackerAuth);
    if (trackerUrl) startPolling();
  });

  document.getElementById("autoZoomCalcBtn").addEventListener("click", () => {
    runAutoZoomFromPlayArea(document.getElementById("autoZoomStatus"));
  });

  for (const el of document.querySelectorAll('input[name="anchor"]')) {
    el.checked = el.value === positionAnchor;
    el.addEventListener("change", () => {
      if (!el.checked) return;
      positionAnchor = el.value;
      localStorage.setItem("positionAnchor", positionAnchor);
      describeAnchor();
    });
  }
  describeAnchor();

  describeProjector();

  const syncBtn = document.getElementById("syncProjectorBtn");
  if (syncBtn) {
    syncBtn.addEventListener("click", async () => {
      syncBtn.disabled = true;
      try {
        await syncProjectorToThisView(
          document.getElementById("projectorStatus"));
      } finally {
        syncBtn.disabled = false;
      }
    });
  }

  const dbEl = document.getElementById("deadbandMm");
  if (dbEl) {
    dbEl.value = String(deadbandMm);
    dbEl.addEventListener("input", () => {
      deadbandMm = parseFloat(dbEl.value);
      localStorage.setItem("deadbandMm", String(deadbandMm));
      describeDeadband();
    });
  }
  describeDeadband();

  // One switch for all the explanations, so "what does everything here
  // do" does not mean opening eight disclosures by hand -- and the
  // default stays compact, which is the point of collapsing them.
  const helpBtn = document.getElementById("helpToggle");
  if (helpBtn) {
    helpBtn.addEventListener("click", () => {
      const all = [...document.querySelectorAll("details.what")];
      const anyClosed = all.some((d) => !d.open);
      for (const d of all) d.open = anyClosed;
      helpBtn.textContent = anyClosed ? "Hide all help" : "Show all help";
    });
  }

  const travelBtn = document.getElementById("travelBtn");
  if (travelBtn) {
    travelBtn.addEventListener("click", () => {
      travelling = !travelling;
      describeTravel();
    });
  }
  describeTravel();
}

function describeDeadband() {
  const el = document.getElementById("deadbandStatus");
  if (!el) return;
  if (!(deadbandMm > 0)) {
    el.textContent = "Off -- every reading is used, including measurement "
      + "noise (about 0.4mm typical, up to ~2mm).";
    return;
  }
  el.textContent = "Holding still until a token moves more than "
    + `${deadbandMm.toFixed(1)}mm `
    + `(${(100 * deadbandMm / TOKEN_BASE_MM).toFixed(0)}% of a token base).`;
}

function describeTravel() {
  const btn = document.getElementById("travelBtn");
  const st = document.getElementById("travelStatus");
  if (btn) {
    btn.textContent = travelling
      ? "Resume position sync" : "Pause position sync";
  }
  if (st) {
    st.textContent = travelling
      ? "PAUSED -- nothing is being written to the scene. Move the map "
        + "however you like, then resume."
      : "Running -- token positions are being written to the scene.";
  }
  setStatus(
    travelling ? "Position sync PAUSED" : lastStatusText,
    travelling ? "warn" : lastStatusClass);
}

function describeAnchor() {
  const el = document.getElementById("anchorStatus");
  if (!el) return;
  el.textContent = positionAnchor === "viewport"
    ? "Anchored to THIS client's view -- tokens follow the table through "
      + "pan and zoom, and panning moves them across the map."
    : "Anchored to the map -- tokens hold their map position; keep the "
      + "projecting view at its calibrated home for physical alignment.";
}

// Signature of everything renderMappingUI() actually draws. The poll
// loop calls that function every 150ms, and it rebuilds the list from
// innerHTML = "" -- so an open <select> had its DOM node destroyed and
// replaced under the operator's cursor within 150ms of being clicked,
// which closes the dropdown instantly and made token assignment
// impossible. Re-rendering only when the content has really changed
// fixes that; the extra guard below covers the case where the content
// changes WHILE a dropdown is open.
function mappingSignature() {
  return JSON.stringify([
    latestPhysicalTokens.map((t) => [t.marker_id, t.name]),
    virtualTokens.map((v) => [v.id, v.text?.plainText || v.name]),
    mapping,
  ]);
}
let _renderedSignature = null;

function renderMappingUI() {
  const container = document.getElementById("mappingList");
  if (!container) return;

  const sig = mappingSignature();
  if (sig === _renderedSignature) return;

  // Never yank the list out from under an open dropdown, even if
  // something else changed -- the operator is mid-assignment. It will be
  // redrawn on the next poll after the select loses focus.
  const active = document.activeElement;
  if (active && active.tagName === "SELECT" && container.contains(active)) {
    return;
  }
  _renderedSignature = sig;
  container.innerHTML = "";

  if (latestPhysicalTokens.length === 0) {
    container.innerHTML = '<p class="hint">No tokens detected yet.</p>';
    return;
  }

  for (const pt of latestPhysicalTokens) {
    const key = String(pt.marker_id);
    const row = document.createElement("div");
    row.className = "mapping-row";

    const label = document.createElement("span");
    label.className = "mapping-label";
    label.textContent = pt.name || `#${pt.marker_id}`;
    row.appendChild(label);

    const select = document.createElement("select");
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(unmapped)";
    select.appendChild(none);
    for (const vt of virtualTokens) {
      const opt = document.createElement("option");
      opt.value = vt.id;
      opt.textContent = vt.text?.plainText || vt.name || vt.id.slice(0, 8);
      select.appendChild(opt);
    }
    select.value = mapping[key] || "";
    select.addEventListener("change", () => {
      if (select.value) mapping[key] = select.value;
      else delete mapping[key];
      saveMappingDebounced();
    });
    row.appendChild(select);

    container.appendChild(row);
  }
}
