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

console.log(fail ? `\n  ${fail} FAILED` : "\n  all sync-view checks passed");
process.exit(fail ? 1 : 0);
