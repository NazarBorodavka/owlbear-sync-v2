// Auto-zoom arithmetic, transcribed from runAutoZoomFromPlayArea.
//
// The projector throws the whole viewport onto the play area, so
//   screen px per mm = viewport px / play-area mm
// and a grid cell, dpi scene px shown at `scale` screen px per scene px,
// measures dpi*scale/(px per mm) on the table. Setting that to the token
// base and solving gives the scale below.
const TOKEN_BASE_MM = 40.0;
const W = 905, H = 510;                 // the calibrated play area

let fail = 0;
const check = (n, c) => { console.log(`  ${c ? "ok  " : "FAIL"}  ${n}`); if (!c) fail++; };

function autoZoom(vw, vh, dpi, widthMm = W, heightMm = H) {
  if (!(dpi > 0) || !(vw > 0) || !(vh > 0)) return null;
  const pxPerMmX = vw / widthMm, pxPerMmY = vh / heightMm;
  const skew = Math.abs(pxPerMmX - pxPerMmY) / Math.min(pxPerMmX, pxPerMmY);
  const pxPerMm = 0.5 * (pxPerMmX + pxPerMmY);
  const scale = (TOKEN_BASE_MM * pxPerMm) / dpi;
  if (!isFinite(scale) || scale <= 0) return null;
  return { scale, skew,
           cellX: (dpi * scale) / pxPerMmX,
           cellY: (dpi * scale) / pxPerMmY };
}

// 1. A 16:9 window over a 905x510 area: both axes land on ~40mm, because
//    905/510 = 1.774 and 16/9 = 1.778.
for (const [vw, vh, dpi] of [[1920,1080,150],[1920,1080,300],[2560,1440,150],[3840,2160,96]]) {
  const r = autoZoom(vw, vh, dpi);
  console.log(`  ${vw}x${vh} dpi${dpi}: scale ${r.scale.toFixed(4)} -> `
            + `${r.cellX.toFixed(2)}mm across, ${r.cellY.toFixed(2)}mm down`);
  check(`${vw}x${vh} dpi${dpi} lands within 0.5mm of a token base`,
        Math.abs(r.cellX - TOKEN_BASE_MM) < 0.5 &&
        Math.abs(r.cellY - TOKEN_BASE_MM) < 0.5);
}

// 2. The scale must be independent of dpi in the sense that the RESULT
//    is: a different grid dpi needs a different scale for the same cell.
const a = autoZoom(1920, 1080, 150), b = autoZoom(1920, 1080, 300);
check("doubling grid dpi halves the scale for the same physical cell",
      Math.abs(b.scale - a.scale / 2) < 1e-9 &&
      Math.abs(b.cellX - a.cellX) < 1e-9);

// 3. The reported cell size must NOT be the input echoed back. Against a
//    deliberately mismatched aspect it has to show the real error.
const odd = autoZoom(1280, 800, 150);
console.log(`  16:10 window: ${odd.cellX.toFixed(2)}mm across, `
          + `${odd.cellY.toFixed(2)}mm down, skew ${(100*odd.skew).toFixed(2)}%`);
check("a mismatched aspect reports a cell size that is NOT 40mm",
      Math.abs(odd.cellX - TOKEN_BASE_MM) > 1.0);
check("a mismatched aspect trips the 2% skew warning", odd.skew > 0.02);
check("the real 16:9 setup stays under the skew warning", a.skew < 0.02);

// 4. Guards: a scene with no grid, or no window, must refuse rather than
//    push a nonsense scale into the viewport.
check("refuses a zero dpi", autoZoom(1920, 1080, 0) === null);
check("refuses a zero-size viewport", autoZoom(0, 0, 150) === null);

// 5. The skew is a property of the aspect ratios only -- not of dpi or
//    of absolute size -- which is why it is a usable alignment check.
check("skew is independent of dpi and window size",
      Math.abs(autoZoom(1920,1080,150).skew - autoZoom(3840,2160,96).skew) < 1e-12);

console.log(fail ? `\n  ${fail} FAILED` : "\n  all auto-zoom checks passed");
process.exit(fail ? 1 : 0);
