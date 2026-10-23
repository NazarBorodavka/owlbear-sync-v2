/* Dashboard: thin client over the REST API in tracker/web/app.py. No
 * build step, no framework -- this is an operator setup/monitoring tool
 * used rarely (mostly during initial calibration), not a high-frequency
 * UI, so plain DOM calls are the right amount of complexity. */

// Credentials are prompted once and cached in localStorage, not
// hardcoded -- a hardcoded "admin"/"admin" would silently 401 on every
// API call for anyone who (reasonably) changed TRACKER_USER/
// TRACKER_PASSWORD away from the defaults, with no obvious reason why.
function getAuth() {
  let user = localStorage.getItem("dashboardUser");
  let pass = localStorage.getItem("dashboardPass");
  if (!user || !pass) {
    user = prompt("Tracker username:", "admin") || "admin";
    pass = prompt("Tracker password:", "") || "";
    localStorage.setItem("dashboardUser", user);
    localStorage.setItem("dashboardPass", pass);
  }
  return { user, pass };
}

function authHeader() {
  const { user, pass } = getAuth();
  return "Basic " + btoa(`${user}:${pass}`);
}

function resetAuth() {
  localStorage.removeItem("dashboardUser");
  localStorage.removeItem("dashboardPass");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: authHeader(),
      ...(options.headers || {}),
    },
  });
  if (res.status === 401) {
    resetAuth();
    throw new Error("authentication failed -- re-enter credentials and try again");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

// ---------------------------------------------------------------------
// Camera + preview
// ---------------------------------------------------------------------
const preview = document.getElementById("preview");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
let corners = [];   // clicked points, in PREVIEW pixel space

function resizeOverlay() {
  overlay.width = preview.clientWidth;
  overlay.height = preview.clientHeight;
  drawCorners();
}
window.addEventListener("resize", resizeOverlay);
preview.addEventListener("load", resizeOverlay);

// Reflects the stored/pinned camera config. The URL itself is never
// sent back by the server -- it normally embeds the camera login -- so
// this shows the redacted form and leaves the input empty, and Start
// with an empty input reuses whatever the host already has.
async function refreshCameraConfig() {
  const status = document.getElementById("cameraConfigStatus");
  const input = document.getElementById("cameraUrl");
  try {
    const cfg = await api("/api/camera");
    if (!cfg.configured) {
      status.textContent = "No camera configured yet.";
      input.disabled = false;
      input.placeholder = "rtsp://user:pass@192.168.1.60:554/h264Preview_01_main";
      return;
    }
    if (cfg.editable) {
      status.textContent = `Saved: ${cfg.url_redacted} — ` +
        `${cfg.running ? "running" : "stopped"}. ` +
        `Press Start to reuse it, or enter a new URL to replace it.`;
      input.disabled = false;
      input.placeholder = "leave empty to reuse the saved URL";
    } else {
      // Pinned by TRACKER_CAMERA_URL in compose: editing here would be
      // a lie, since the environment wins and survives a restart.
      status.textContent = `Set by TRACKER_CAMERA_URL: ${cfg.url_redacted} — ` +
        `${cfg.running ? "running" : "stopped"}. ` +
        `Change it in your compose file, not here.`;
      input.disabled = true;
      input.placeholder = "fixed by the deployment";
    }
  } catch (e) {
    status.textContent = "";
  }
}

document.getElementById("startCameraBtn").addEventListener("click", async () => {
  const input = document.getElementById("cameraUrl");
  const url = input.value.trim();
  try {
    // An empty body deliberately means "use the configured URL", so the
    // operator never has to retype a credentialed RTSP URL.
    await api("/api/camera/start", {
      method: "POST",
      body: JSON.stringify(url ? { url } : {}),
    });
    input.value = "";
    preview.src = "/video_feed?ts=" + Date.now();
    refreshCameraConfig();
  } catch (e) {
    alert("Failed to start camera: " + e.message);
  }
});

document.getElementById("stopCameraBtn").addEventListener("click", async () => {
  await api("/api/camera/stop", { method: "POST" });
  preview.removeAttribute("src");
  refreshCameraConfig();
});

document.getElementById("forgetCameraBtn").addEventListener("click", async () => {
  if (!confirm("Forget the saved camera URL? You will need to re-enter it.")) return;
  await api("/api/camera", { method: "DELETE" });
  preview.removeAttribute("src");
  refreshCameraConfig();
});

// If the camera auto-started at boot, show its feed without needing a click.
(async () => {
  await refreshCameraConfig();
  try {
    const cfg = await api("/api/camera");
    if (cfg.running) preview.src = "/video_feed?ts=" + Date.now();
  } catch (e) { /* not authenticated yet; the prompt flow handles it */ }
})();

overlay.addEventListener("click", (e) => {
  const rect = overlay.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  if (corners.length >= 4) corners = [];
  corners.push({ x, y });
  drawCorners();
});

document.getElementById("resetCornersBtn").addEventListener("click", () => {
  corners = [];
  drawCorners();
});

function drawCorners() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (corners.length === 0) return;
  octx.strokeStyle = "#4a6fd4";
  octx.fillStyle = "#4a6fd4";
  octx.lineWidth = 2;
  octx.beginPath();
  corners.forEach((c, i) => {
    octx.beginPath();
    octx.arc(c.x, c.y, 5, 0, 2 * Math.PI);
    octx.fill();
    octx.fillText(String(i + 1), c.x + 8, c.y - 8);
  });
  if (corners.length === 4) {
    octx.beginPath();
    octx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) octx.lineTo(corners[i].x, corners[i].y);
    octx.closePath();
    octx.stroke();
  }
}

// Clicked points are in PREVIEW (displayed) pixel space, which may be
// scaled down from the camera's native resolution -- scale up to native
// coordinates using the <img>'s natural size before sending to the API,
// so calibration is correct regardless of how large the preview element
// happens to be rendered in the browser window.
function cornersToNative() {
  const sx = preview.naturalWidth / preview.clientWidth;
  const sy = preview.naturalHeight / preview.clientHeight;
  return corners.map((c) => [c.x * sx, c.y * sy]);
}

// ---------------------------------------------------------------------
// Play-area calibration
// ---------------------------------------------------------------------
const pxPerMmInput = document.getElementById("pxPerMm");
const scaleHint = document.getElementById("scaleHint");

async function updateScaleHint() {
  const widthMm = parseFloat(document.getElementById("areaWidthMm").value);
  const heightMm = parseFloat(document.getElementById("areaHeightMm").value);
  const pxPerMm = parseFloat(pxPerMmInput.value);
  if (corners.length !== 4) {
    scaleHint.textContent = `${pxPerMm.toFixed(1)} px/mm (click 4 corners to preview accuracy)`;
    return;
  }
  try {
    const result = await api("/api/calibration/playarea/preview", {
      method: "POST",
      body: JSON.stringify({ corners: cornersToNative(), width_mm: widthMm, height_mm: heightMm, px_per_mm: pxPerMm }),
    });
    scaleHint.textContent = `${pxPerMm.toFixed(1)} px/mm, ${result.coverage_cm[0].toFixed(0)}x${result.coverage_cm[1].toFixed(0)}cm -- ${result.verdict}`;
  } catch (e) {
    scaleHint.textContent = `${pxPerMm.toFixed(1)} px/mm (${e.message})`;
  }
}
pxPerMmInput.addEventListener("input", updateScaleHint);
document.getElementById("areaWidthMm").addEventListener("input", updateScaleHint);
document.getElementById("areaHeightMm").addEventListener("input", updateScaleHint);
setInterval(updateScaleHint, 1500);

document.getElementById("calibratePlayareaBtn").addEventListener("click", async () => {
  if (corners.length !== 4) return alert("Click all 4 corners first");
  const widthMm = parseFloat(document.getElementById("areaWidthMm").value);
  const heightMm = parseFloat(document.getElementById("areaHeightMm").value);
  const pxPerMm = parseFloat(pxPerMmInput.value);
  try {
    const result = await api("/api/calibration/playarea", {
      method: "POST",
      body: JSON.stringify({ corners: cornersToNative(), width_mm: widthMm, height_mm: heightMm, px_per_mm: pxPerMm }),
    });
    document.getElementById("playareaResult").textContent =
      `Calibrated: ${result.coverage_cm[0].toFixed(0)}x${result.coverage_cm[1].toFixed(0)}cm at ${result.px_per_mm.toFixed(1)}px/mm -- ${result.verdict}`;
    refreshStatus();
  } catch (e) {
    alert("Calibration failed: " + e.message);
  }
});

document.getElementById("clearPlayareaBtn").addEventListener("click", async () => {
  await api("/api/calibration/playarea", { method: "DELETE" });
  refreshStatus();
});

// ---------------------------------------------------------------------
// Lens calibration
// ---------------------------------------------------------------------
let chessboardFrames = [];

document.getElementById("captureChessboardBtn").addEventListener("click", () => {
  if (!preview.src) return alert("Start the camera first");
  const c = document.createElement("canvas");
  c.width = preview.naturalWidth;
  c.height = preview.naturalHeight;
  c.getContext("2d").drawImage(preview, 0, 0);
  chessboardFrames.push(c.toDataURL("image/jpeg", 0.92));
  document.getElementById("chessboardCount").textContent = chessboardFrames.length;
});

document.getElementById("calibrateLensBtn").addEventListener("click", async () => {
  if (chessboardFrames.length < 4) return alert("Capture at least 4 frames first");
  try {
    const result = await api("/api/calibration/lens", {
      method: "POST",
      body: JSON.stringify({
        frames: chessboardFrames,
        square_mm: parseFloat(document.getElementById("squareMm").value),
      }),
    });
    document.getElementById("lensResult").textContent =
      `Calibrated from ${result.frames_used}/${chessboardFrames.length} frames, RMS error ${result.rms_error.toFixed(2)}px`;
    chessboardFrames = [];
    document.getElementById("chessboardCount").textContent = "0";
    refreshStatus();
  } catch (e) {
    alert("Lens calibration failed: " + e.message);
  }
});

document.getElementById("clearLensBtn").addEventListener("click", async () => {
  await api("/api/calibration/lens", { method: "DELETE" });
  refreshStatus();
});

// ---------------------------------------------------------------------
// Detector calibration
// ---------------------------------------------------------------------
document.getElementById("calibrateDetectorBtn").addEventListener("click", async () => {
  const duration = parseFloat(document.getElementById("detectorDuration").value);
  document.getElementById("detectorResult").textContent = "Capturing...";
  try {
    const report = await api("/api/calibration/detector", {
      method: "POST",
      body: JSON.stringify({ duration_s: duration }),
    });
    document.getElementById("detectorResult").textContent = JSON.stringify(report, null, 2);
    refreshStatus();
  } catch (e) {
    document.getElementById("detectorResult").textContent = "Failed: " + e.message;
  }
});

document.getElementById("resetDetectorBtn").addEventListener("click", async () => {
  await api("/api/calibration/detector", { method: "DELETE" });
  refreshStatus();
});

// ---------------------------------------------------------------------
// Status + live tokens
// ---------------------------------------------------------------------
async function refreshStatus() {
  try {
    const status = await api("/api/calibration/status");
    setBadge("lensBadge", status.lens.calibrated, status.lens.calibrated ? `RMS ${status.lens.rms_error.toFixed(2)}px` : "not calibrated");
    setBadge("playareaBadge", status.playarea.calibrated,
      status.playarea.calibrated ? `${status.playarea.coverage_cm[0].toFixed(0)}x${status.playarea.coverage_cm[1].toFixed(0)}cm` : "not calibrated");
    setBadge("detectorBadge", status.detector.calibrated, status.detector.calibrated ? "tuned" : "using defaults");
  } catch (e) { /* calibration status requires auth; ignore transient failures */ }
}

function setBadge(id, ok, text) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = "badge" + (ok ? " ok" : "");
}

async function refreshTokens() {
  try {
    const data = await fetch("/api/tokens").then((r) => r.json());
    const tbody = document.querySelector("#tokenTable tbody");
    tbody.innerHTML = "";
    for (const t of data.tokens) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${t.marker_id}</td><td>${t.name || ""}</td>` +
        `<td>${t.x_mm.toFixed(0)}, ${t.y_mm.toFixed(0)}</td><td>${t.age_s.toFixed(1)}</td>`;
      tbody.appendChild(tr);
    }
    document.getElementById("globalStatus").textContent =
      data.connected ? `connected, ${data.fps.toFixed(0)}fps, ${data.tokens.length} tokens` : "camera disconnected";
  } catch (e) {
    document.getElementById("globalStatus").textContent = "tracker unreachable";
  }
}

document.getElementById("manifestUrl").textContent =
  window.location.origin + "/extension/manifest.json";

refreshStatus();
refreshTokens();
setInterval(refreshStatus, 5000);
setInterval(refreshTokens, 1000);
