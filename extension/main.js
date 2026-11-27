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
const SMOOTH_RATE = 20;           // 1/s -- higher = snappier, lower = smoother but laggier
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
  if (trackerUrl) {
    document.getElementById("trackerUrl").value = trackerUrl;
    startPolling();
  }
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
async function pushPositions() {
  if (!playAreaMm) return;
  const [widthMm, heightMm] = playAreaMm;

  // Both of these are properties of the SCENE or the CLIENT, not of a
  // token, so they are fetched once per poll rather than once per token.
  // The grid used to be queried inside the loop below: six tokens at a
  // 150ms poll meant ~80 SDK round trips a second (each one a
  // postMessage into the parent frame and back) to re-read two numbers
  // that had not changed.
  let viewportPx = null;
  let pxPerMm = null;
  if (positionAnchor === "viewport") {
    try {
      const [w, h] = await Promise.all([
        OBR.viewport.getWidth(),
        OBR.viewport.getHeight(),
      ]);
      if (!(w > 0) || !(h > 0)) return;
      viewportPx = { w, h };
    } catch (e) {
      return;   // viewport not ready yet; try again next poll
    }
  } else {
    // Only scene anchoring needs the grid, so a scene with no grid
    // configured still syncs in viewport mode.
    try {
      const [dpi, scale] = await Promise.all([
        OBR.scene.grid.getDpi(),
        OBR.scene.grid.getScale(),
      ]);
      const mmPerCell = scale.parsed.multiplier * mmPerUnit(scale.parsed.unit);
      if (!(dpi > 0) || !(mmPerCell > 0)) return;
      pxPerMm = dpi / mmPerCell;
    } catch (e) {
      return;   // grid not ready yet; try again next poll
    }
  }

  const updates = [];
  for (const pt of latestPhysicalTokens) {
    // Only act on a position that was actually READ this frame.
    //
    // The tracker also publishes HELD positions -- the last confirmed
    // place of a token whose code it cannot currently read (a hand over
    // it, a mini covering part of the ring). Those are correct as "where
    // it was", but they are not news, and pushing them would let a
    // stale or slightly-off position overwrite a good one in the scene.
    // Skipping them leaves whatever is already on the map untouched,
    // which is the right behaviour for a token that is merely
    // unreadable: it has not moved as far as anyone knows.
    if (pt.coasting) continue;
    const itemId = mapping[String(pt.marker_id)];
    if (!itemId) continue;
    const targetItem = sceneItemsCache.find((i) => i.id === itemId);
    if (!targetItem) continue;

    // mm position -> OBR scene units (scene-pixel space, independent of
    // viewport pan/zoom). OBR's own grid.getDpi() is scene pixels per
    // grid cell; dividing by how many real-world mm one cell represents
    // (cell count x the grid's configured unit size) gives scene pixels
    // per real mm directly. Deliberately NOT v1's screen-pixel +
    // viewport.inverseTransformPoint() approach -- that depended on
    // whatever the LOCAL player's viewport currently shows, which
    // breaks the moment auto-zoom (below) changes viewport scale
    // mid-session, or if two clients are looking at different areas.
    // Two anchorings, and the difference is which thing stays fixed when
    // the map is panned.
    //
    //   scene    -- play-area origin IS the map origin. Viewport
    //               independent, so every client agrees and nothing
    //               moves on the map when someone pans; but physical
    //               alignment only holds while the projecting view is
    //               parked at its calibrated home position.
    //   viewport -- convert mm to a point on THIS client's screen (the
    //               projected image is assumed to fill the play area,
    //               the same assumption runAutoZoomFromPlayArea() makes
    //               and checks), then back into scene space. The token
    //               stays under its mini through any pan or zoom,
    //               because it is re-derived from the live viewport
    //               every poll -- which is also why panning then drags
    //               tokens across the map.
    let scenePoint;
    if (positionAnchor === "viewport") {
      const screenPoint = {
        x: (pt.x_mm / widthMm) * viewportPx.w,
        y: (pt.y_mm / heightMm) * viewportPx.h,
      };
      try {
        scenePoint = await OBR.viewport.inverseTransformPoint(screenPoint);
      } catch (e) {
        continue;   // viewport not ready; try again next poll
      }
    } else {
      scenePoint = { x: pt.x_mm * pxPerMm, y: pt.y_mm * pxPerMm };
    }

    const dist = Math.hypot(
      targetItem.position.x - scenePoint.x,
      targetItem.position.y - scenePoint.y
    );
    const needsNameUpdate =
      pt.name && targetItem.name !== pt.name &&
      !(targetItem.text && targetItem.text.plainText === pt.name);
    if (dist > MIN_MOVE_SCENE_UNITS || needsNameUpdate) {
      updates.push({ id: itemId, position: scenePoint, name: pt.name });
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
const TOKEN_BASE_MM = 40.0;   // physical base diameter -- keep in sync with
                               // tracker/vision/geometry.py's RingGeometry.base_d_mm
                               // (the PRINTED marker is 38mm; the physical base it
                               // sits on, which is what should visually match one
                               // grid cell, is 40mm)

async function runAutoZoom(statusEl) {
  statusEl.textContent = "Measuring projected grid...";
  let measurement;
  try {
    measurement = await apiFetch("/api/grid/measure", { method: "POST" });
  } catch (e) {
    statusEl.textContent = `Auto-zoom failed: ${e.message}`;
    return;
  }

  const currentScale = await OBR.viewport.getScale();
  // If 1 cell currently measures measurement.mm_per_cell on the table,
  // and we want 1 cell to measure TOKEN_BASE_MM, the viewport scale needs
  // to change by that same ratio -- the DPI/grid-size relationship is
  // linear in viewport scale, so no iteration or re-measurement loop is
  // needed, one shot gets there directly.
  const ratio = measurement.mm_per_cell / TOKEN_BASE_MM;
  const newScale = currentScale * ratio;
  await OBR.viewport.setScale(newScale);

  // Persist as the scene's own "home" zoom, so it is restored automatically
  // next session without re-measuring (the operator can always re-run this
  // if the projector/camera setup changes).
  await OBR.scene.setMetadata({
    "com.physicaltokentracker/homeScale": newScale,
  });

  statusEl.textContent =
    `Done: grid was ${measurement.mm_per_cell.toFixed(0)}mm/cell ` +
    `(confidence ${(measurement.confidence * 100).toFixed(0)}%), ` +
    `now ${TOKEN_BASE_MM}mm/cell to match the token base.`;
}

// Analytic auto-zoom: no camera measurement, from the play-area size.
//
// This is the one that works with the near-IR bandpass filter fitted.
// runAutoZoom() above measures the projected grid IN THE CAMERA IMAGE,
// which was fine when the camera could see the projector -- but the
// filter exists precisely to make projector light invisible to it, so
// that measurement now finds nothing to measure. The geometry is fully
// determined without it:
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
    await OBR.scene.setMetadata({
      "com.physicaltokentracker/homeScale": newScale,
    });
  } catch (e) {
    statusEl.textContent = `Auto-zoom could not apply the scale: ${e.message}`;
    return;
  }

  const cellMm = (dpi * newScale) / pxPerMm;
  let msg =
    `Done: 1 cell is now ${cellMm.toFixed(1)}mm on the table ` +
    `(target ${TOKEN_BASE_MM}mm). Play area ${widthMm}x${heightMm}mm ` +
    `shown in a ${vw}x${vh}px viewport, grid dpi ${dpi}, scale ${newScale.toFixed(4)}.`;
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
// UI
// ---------------------------------------------------------------------
function setStatus(text, cls) {
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

  document.getElementById("autoZoomBtn").addEventListener("click", () => {
    runAutoZoom(document.getElementById("autoZoomStatus"));
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

  // Travel Mode's markup exists but its behaviour does not yet. A button
  // that silently does nothing reads as a bug, so say so plainly instead.
  const travelBtn = document.getElementById("travelBtn");
  if (travelBtn) {
    travelBtn.disabled = true;
    const st = document.getElementById("travelStatus");
    if (st) st.textContent = "Not implemented yet.";
  }
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

function renderMappingUI() {
  const container = document.getElementById("mappingList");
  if (!container) return;
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
