// Sync-view transition detection, with the one-shot case that broke it.
//
// Owlbear's persistent sync toggle produces two change events seconds
// apart. Its one-shot Sync View produces two MILLISECONDS apart (on,
// then immediately off). Only the second ordering exposes the bug, so
// both are replayed here.
let fail = 0;
const check = (n, c) => { console.log(`  ${c ? "ok  " : "FAIL"}  ${n}`); if (!c) fail++; };

// --- the old, broken handler: ignores the payload and re-queries ------
function makeBroken() {
  const st = { active: false, captures: 0, flag: false };
  st.onEvent = async (syncView) => {
    st.flag = syncView;                     // what the host now reports
    const was = st.active;
    await Promise.resolve();                // the await that loses the race
    st.active = st.flag;                    // re-query: sees the FINAL value
    if (was && !st.active) st.captures++;
  };
  return st;
}

// --- the fixed handler: payload read synchronously --------------------
function makeFixed() {
  const st = { active: false, captures: 0 };
  st.onEvent = (syncView) => {
    const active = Boolean(syncView);
    if (active === st.active) return;
    st.active = active;
    if (!active) st.captures++;             // decided before any await
  };
  return st;
}

async function oneShot(st) {                // on and off, same tick
  const a = st.onEvent(true), b = st.onEvent(false);
  await a; await b;
}
async function toggle(st) {                 // well separated
  await st.onEvent(true);
  await new Promise((r) => setTimeout(r, 5));
  await st.onEvent(false);
}

const brokenOneShot = makeBroken();  await oneShot(brokenOneShot);
const brokenToggle  = makeBroken();  await toggle(brokenToggle);
const fixedOneShot  = makeFixed();   await oneShot(fixedOneShot);
const fixedToggle    = makeFixed();  await toggle(fixedToggle);

console.log(`  broken handler: one-shot captured ${brokenOneShot.captures}, `
          + `toggle captured ${brokenToggle.captures}`);
console.log(`  fixed  handler: one-shot captured ${fixedOneShot.captures}, `
          + `toggle captured ${fixedToggle.captures}`);

// This is the reported bug, reproduced: the old code handled the toggle
// and dropped the one-shot. A test that only exercised the toggle would
// have passed against the broken version.
check("the OLD handler drops a one-shot sync (the reported bug)",
      brokenOneShot.captures === 0);
check("the OLD handler does catch the slow toggle (why it seemed to work)",
      brokenToggle.captures === 1);
check("the fixed handler captures a one-shot sync", fixedOneShot.captures === 1);
check("the fixed handler still captures the toggle", fixedToggle.captures === 1);

// Idempotence: repeated events at the same value must not re-capture,
// or every party change would clobber a good calibration.
const st = makeFixed();
st.onEvent(true); st.onEvent(true); st.onEvent(false); st.onEvent(false);
check("repeated events at the same value capture exactly once",
      st.captures === 1);

// Either source can hold sync on: self (GM broadcasting) or a party
// member. The off-transition only fires when BOTH are clear.
function makeTwoSource() {
  const s = { mine: false, others: false, active: false, captures: 0 };
  s.set = (mine, others) => {
    s.mine = mine; s.others = others;
    const active = Boolean(mine || others);
    if (active === s.active) return;
    s.active = active;
    if (!active) s.captures++;
  };
  return s;
}
const two = makeTwoSource();
two.set(true, false);     // GM syncs
two.set(true, true);      // a player syncs too
two.set(false, true);     // GM stops, player still synced
check("sync held by a party member alone does not trigger a capture",
      two.captures === 0);
two.set(false, false);    // now everyone is clear
check("the capture fires when the last sync source clears",
      two.captures === 1);

// --- live rect refresh, and the in-flight guard -----------------------
// While sync view is on, the rect must keep up with the PAN, not with the
// tracker poll. Refreshing it once per 150ms poll is what made following
// a moving map choppy. Refreshing on every 50ms tick fixes that, but a
// round trip can take longer than a tick while the parent frame is busy
// drawing the pan -- so without a guard, ticks queue refreshes faster
// than they complete.
console.log("");
async function runTicks({ ticks, tickMs, roundTripMs, guard }) {
  let inFlight = false, started = 0, completed = 0, queued = 0;
  const pending = [];
  for (let i = 0; i < ticks; i++) {
    if (guard && inFlight) {
      queued++;
    } else {
      inFlight = true; started++;
      pending.push(new Promise((r) => setTimeout(() => {
        completed++; inFlight = false; r();
      }, roundTripMs)));
    }
    // Awaited on the skip path too: the real loop is a setInterval, so a
    // tick that skips still consumes its interval. Skipping the wait made
    // every remaining tick run in one synchronous burst with no chance
    // for the in-flight timer to clear -- an artefact of the harness, not
    // of the guard.
    await new Promise((r) => setTimeout(r, tickMs));
  }
  await Promise.all(pending);
  return { started, completed, skipped: queued };
}

// A slow round trip (120ms) against a 50ms tick: without the guard, more
// than twice as many refreshes are launched as can complete in time.
const unguarded = await runTicks({ ticks: 10, tickMs: 50, roundTripMs: 120, guard: false });
const guarded   = await runTicks({ ticks: 10, tickMs: 50, roundTripMs: 120, guard: true });
console.log(`  10 ticks, 50ms apart, 120ms round trip:`);
console.log(`    unguarded: ${unguarded.started} launched`);
console.log(`    guarded:   ${guarded.started} launched, ${guarded.skipped} skipped`);
check("without a guard, a slow round trip over-launches refreshes",
      unguarded.started === 10);
check("the guard launches only what can complete", guarded.started < 10);
check("the guard never leaves a refresh uncompleted",
      guarded.started === guarded.completed);
check("the guard still refreshes repeatedly (it does not latch off)",
      guarded.started >= 3);

// Rate comparison: the point of moving the refresh off the poll.
const POLL_MS = 150, TICK_MS = 50;
console.log(`  refresh rate: ${(1000/POLL_MS).toFixed(1)}Hz on the poll `
          + `vs ${(1000/TICK_MS).toFixed(1)}Hz on the tick`);
check("the tick refreshes the rect at least 3x as often as the poll",
      (1000 / TICK_MS) >= 3 * (1000 / POLL_MS));

console.log(fail ? `\n  ${fail} FAILED` : "\n  all sync-view checks passed");
process.exit(fail ? 1 : 0);
