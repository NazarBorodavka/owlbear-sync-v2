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

  const updates = [];
  for (const pt of latestPhysicalTokens) {
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
    let scenePoint;
    try {
      const dpi = await OBR.scene.grid.getDpi();
      const scale = await OBR.scene.grid.getScale();
      const mmPerCell = scale.parsed.multiplier * mmPerUnit(scale.parsed.unit);
      const pxPerMm = dpi / mmPerCell;
      scenePoint = { x: pt.x_mm * pxPerMm, y: pt.y_mm * pxPerMm };
    } catch (e) {
      continue; // grid not ready yet; try again next poll
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
