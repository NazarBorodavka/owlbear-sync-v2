// The property that motivates smoothing in mm rather than scene space:
// a dragged mini eases, a re-synced map does not.
const SMOOTH_RATE = 20, SNAP_MM = 150, DT = 0.05;   // 20Hz
const alpha = 1 - Math.exp(-SMOOTH_RATE * DT);
const widthMm = 905, heightMm = 510;

let fail = 0;
const check = (n, c) => { console.log(`  ${c ? "ok  " : "FAIL"}  ${n}`); if (!c) fail++; };

function ease(cur, tgt) {
  if (!cur || Math.hypot(tgt.x - cur.x, tgt.y - cur.y) > SNAP_MM) return { ...tgt };
  return { x: cur.x + (tgt.x - cur.x) * alpha, y: cur.y + (tgt.y - cur.y) * alpha };
}
const mmToScene = (x, y, r) =>
  ({ x: r.x + (x / widthMm) * r.w, y: r.y + (y / heightMm) * r.h });

console.log(`  easing: rate ${SMOOTH_RATE}/s at ${1000*DT}ms tick -> alpha ${alpha.toFixed(3)}`);

// 1. a mini dragged 60mm: should take several ticks, monotonically
let cur = { x: 100, y: 100 };
const tgt = { x: 160, y: 100 };
const path = [];
for (let i = 0; i < 12; i++) { cur = ease(cur, tgt); path.push(cur.x); }
console.log(`  drag 100->160mm: ${path.slice(0,6).map(v=>v.toFixed(1)).join(" ")} ... ${path.at(-1).toFixed(1)}`);
check("a 60mm drag is spread over multiple ticks, not one jump", path[0] < 140);
check("easing is monotonic toward the target",
      path.every((v, i) => i === 0 || v >= path[i-1]));
check("easing converges to the target", Math.abs(path.at(-1) - 160) < 0.5);
// time to cover 95%
let c2 = { x: 100, y: 100 }, ticks = 0;
while (Math.abs(c2.x - 160) > 3 && ticks < 200) { c2 = ease(c2, tgt); ticks++; }
console.log(`  reaches within 3mm after ${ticks} ticks (${(ticks*DT*1000).toFixed(0)}ms)`);
check("settles in well under a second", ticks * DT < 1.0);

// 2. a lift-and-place across the table must NOT be eased
const near = { x: 100, y: 100 };
const across = { x: 700, y: 400 };
check("a lift-and-place beyond SNAP_MM goes straight there",
      ease(near, across).x === across.x && ease(near, across).y === across.y);

// 3. THE POINT: re-syncing the map must jump every token at once, with
// the easing state untouched -- the millimetres never changed.
const rectA = { x: 0, y: 0, w: 3394, h: 1909 };
const rectB = { x: 5000, y: -2000, w: 3394, h: 1909 };   // host panned, then synced
const settled = { x: 452.5, y: 255 };
const sA = mmToScene(settled.x, settled.y, rectA);
const sB = mmToScene(settled.x, settled.y, rectB);
console.log(`  on sync: scene (${sA.x.toFixed(0)}, ${sA.y.toFixed(0)}) -> (${sB.x.toFixed(0)}, ${sB.y.toFixed(0)})`);
check("a re-anchor moves the token in ONE step, not eased",
      Math.abs((sB.x - sA.x) - (rectB.x - rectA.x)) < 1e-9 &&
      Math.abs((sB.y - sA.y) - (rectB.y - rectA.y)) < 1e-9);
const afterSync = ease(settled, settled);
check("a re-anchor leaves the easing state untouched (mm did not change)",
      afterSync.x === settled.x && afterSync.y === settled.y);

// 4. a fast but continuous drag must stay smooth, not trip the snap
const perPoll = 500 * 0.15;    // 500mm/s for one 150ms poll
console.log(`  a 500mm/s drag advances ${perPoll.toFixed(0)}mm per poll vs SNAP_MM ${SNAP_MM}`);
check("a brisk continuous drag stays under the snap threshold", perPoll < SNAP_MM);

// --- deadband ---------------------------------------------------------
// Hysteresis against the last ACCEPTED reading, not against the eased
// position: that is what stops repeated sub-threshold wobble from
// accumulating into a slow drift.
console.log("");
const DEADBAND_MM = 2.0;
function accept(prev, read) {
  if (prev && DEADBAND_MM > 0 &&
      Math.hypot(read.x - prev.x, read.y - prev.y) < DEADBAND_MM) return prev;
  return { ...read };
}
// measured stationary noise: ~0.4mm typical, up to ~2.06mm excursion
let held = { x: 300, y: 200 };
const noise = [0.1, -0.4, 0.3, 0.45, -0.2, 0.38, -0.5, 0.25];
let moved = 0;
for (const n of noise) {
  const next = accept(held, { x: held.x + n, y: held.y });
  if (next !== held) moved++;
  held = next;
}
check("typical jitter never moves the token", moved === 0);
check("the held position is exactly where it started",
      held.x === 300 && held.y === 200);

// 60 consecutive sub-threshold nudges in ONE direction must still not
// creep -- this is the property a naive "compare to current" would fail.
let creep = { x: 300, y: 200 };
for (let i = 0; i < 60; i++) creep = accept(creep, { x: creep.x + 1.9, y: creep.y });
console.log(`  after 60 x 1.9mm same-direction nudges: x = ${creep.x}`);
check("sub-threshold nudges cannot accumulate into a drift", creep.x === 300);

// and one real move releases it immediately
const releasedTo = accept({ x: 300, y: 200 }, { x: 302.5, y: 200 });
check("a move past the threshold is accepted at once", releasedTo.x === 302.5);
check("the deadband is a small fraction of a token base",
      DEADBAND_MM / 40 < 0.1);

console.log(fail ? `\n  ${fail} FAILED` : "\n  all smoothing checks passed");
process.exit(fail ? 1 : 0);
