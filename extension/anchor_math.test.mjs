// Checks the two anchorings behave as advertised under a pan, using the
// same arithmetic main.js uses. A fake viewport stands in for OBR.
const widthMm = 905, heightMm = 510;
const vw = 1920, vh = 1080;
const dpi = 150, mmPerCell = 40;          // auto-zoomed: 1 cell = 1 base
const pxPerMm = dpi / mmPerCell;

// viewport: scene -> screen is  screen = (scene - pos) * scale
// so inverse is scene = screen/scale + pos
const inverseTransformPoint = (p, vp) =>
  ({ x: p.x / vp.scale + vp.x, y: p.y / vp.scale + vp.y });

const sceneAnchor = (pt) => ({ x: pt.x_mm * pxPerMm, y: pt.y_mm * pxPerMm });
const viewportAnchor = (pt, vp) => inverseTransformPoint(
  { x: (pt.x_mm / widthMm) * vw, y: (pt.y_mm / heightMm) * vh }, vp);

const tok = { x_mm: 452.5, y_mm: 255 };   // dead centre of the play area
// scale such that one 40mm cell spans 40mm of table: screen px per mm is
// vw/widthMm, so scale = TOKEN_BASE_MM*(vw/widthMm)/dpi
const scale = 40 * (vw / widthMm) / dpi;
const home = { x: 0, y: 0, scale };
const panned = { x: 1000, y: -400, scale };

const fmt = (p) => `(${p.x.toFixed(1)}, ${p.y.toFixed(1)})`;
let fail = 0;
const check = (name, cond) => {
  console.log(`  ${cond ? "ok  " : "FAIL"}  ${name}`);
  if (!cond) fail++;
};

const sHome = sceneAnchor(tok), sPan = sceneAnchor(tok);
console.log(`  scene anchor:    home ${fmt(sHome)}  panned ${fmt(sPan)}`);
check("scene anchoring is viewport independent",
      sHome.x === sPan.x && sHome.y === sPan.y);

const vHome = viewportAnchor(tok, home), vPan = viewportAnchor(tok, panned);
console.log(`  viewport anchor: home ${fmt(vHome)}  panned ${fmt(vPan)}`);
check("viewport anchoring shifts with the pan, by exactly the pan",
      Math.abs((vPan.x - vHome.x) - (panned.x - home.x)) < 1e-9 &&
      Math.abs((vPan.y - vHome.y) - (panned.y - home.y)) < 1e-9);

// At the calibrated home position the two must agree closely enough that
// flipping the toggle mid-session does not visibly move anything.
//
// They do NOT agree exactly, and that is correct: auto-zoom picks ONE
// scale from the mean of the two axes, while viewport anchoring maps each
// axis independently, so they differ by the play area's aspect skew
// against the viewport's (0.18% for a 905x510 area in 16:9). Asserted as
// a real-world tolerance -- a quarter of a token base -- rather than as
// exact equality, which would be asserting the skew is zero.
const dMm = { x: (vHome.x - sHome.x) / pxPerMm, y: (vHome.y - sHome.y) / pxPerMm };
console.log(`  difference at home: ${fmt({x: vHome.x - sHome.x, y: vHome.y - sHome.y})} `
            + `scene px = (${dMm.x.toFixed(2)}, ${dMm.y.toFixed(2)})mm`);
check("the two anchorings coincide at home to well under a token base",
      Math.hypot(dMm.x, dMm.y) < 10.0);

// And a token's physical->screen mapping must put the play-area centre
// at the screen centre.
const centreScreen = { x: (tok.x_mm/widthMm)*vw, y: (tok.y_mm/heightMm)*vh };
check("play-area centre maps to viewport centre",
      Math.abs(centreScreen.x - vw/2) < 1e-6 && Math.abs(centreScreen.y - vh/2) < 1e-6);

console.log(fail ? `\n  ${fail} FAILED` : "\n  all anchor-math checks passed");
process.exit(fail ? 1 : 0);
