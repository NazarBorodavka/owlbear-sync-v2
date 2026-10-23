# Deploying on ZimaOS

Two paths. **Custom Install** works immediately and is the one to use
first. The **app store** path gives you "Update available" prompts, but
only after CI has published at least one image.

---

## 0. Prerequisites (one-time, on GitHub)

### Create the repo and push

The repo is committed locally with a remote already configured, but
nothing has been pushed — this machine has no GitHub credentials
(no `gh`, no token, no SSH key), so I could not create it or push.

Create an **empty** repo named `owlbear-sync-v2` under your account
(no README, no .gitignore — the local repo has both), then:

```bash
cd "/home/nazar/My Files/Projects/Code/owlbear-sync/owlbear-sync-v2"
git push -u origin master
```

The remote is already set to
`https://github.com/NazarBorodavka/owlbear-sync-v2.git`. If you'd rather
use SSH, `git remote set-url origin git@github.com:NazarBorodavka/owlbear-sync-v2.git`.

Also push the store repo, which now has the v2 app entry:

```bash
cd ../owlbear-sync-store
git push
```

> That repo also has an **uncommitted** edit of yours in
> `Apps/OwlbearTracker/docker-compose.yml` (VAAPI/OpenCL comments for
> v1). I left it alone — commit or discard it as you prefer.

### Make the GHCR package pullable

The first push to `master` builds and publishes
`ghcr.io/nazarborodavka/owlbear-tracker-v2`. **New GHCR packages are not
necessarily public**, and ZimaOS pulls anonymously, so check:

> github.com → your profile → **Packages** → `owlbear-tracker-v2` →
> *Package settings* → **Change visibility** → Public

Same step you'll have done for v1's `owlbear-tracker`. If you'd rather
keep it private, you must instead `docker login ghcr.io` over SSH on the
ZimaOS box with a PAT that has `read:packages`.

### Store dispatch token

The v2 workflow notifies the store repo using the **existing**
`STORE_DISPATCH_TOKEN` secret — but secrets are per-repo, so add the same
fine-grained PAT (scoped to `owlbear-sync-store`, contents write) to the
new `owlbear-sync-v2` repo:

> Settings → Secrets and variables → Actions → New repository secret →
> `STORE_DISPATCH_TOKEN`

Without it the image still publishes; only the store bump is skipped
(the step is `continue-on-error`).

### Confirm CI went green

Actions → the run for your push. The `test` job must pass before `build`
runs — that gate is deliberate. Then confirm the image exists and the
tag it published (`master-1`, `master-2`, …).

---

## 1. Install on ZimaOS — Custom Install (do this first)

ZimaOS → **App Store** → the **⋮** menu (top right) → **Install a
customized app**, then paste:

```yaml
name: owlbear-tracker-v2

services:
  tracker:
    image: ghcr.io/nazarborodavka/owlbear-tracker-v2:latest
    container_name: owlbear-tracker-v2
    ports:
      - target: 5000
        published: "5001"
        protocol: tcp
    volumes:
      - type: bind
        source: /DATA/AppData/owlbear-tracker-v2
        target: /data
    environment:
      - PYTHONUNBUFFERED=1
      - TRACKER_USER=admin
      - TRACKER_PASSWORD=pick-something-real
    restart: unless-stopped

x-casaos:
  id: com.nazarborodavka.owlbearsync.v2
  main: tracker
  index: /
  port_map: "5001"
  scheme: http
  title:
    en_US: Owlbear Token Tracker v2
```

Three things in there matter:

- **Port 5001**, not 5000. v1 uses 5000, and these are separate apps that
  can run at the same time while you migrate.
- **Target `/data`**, matching `TRACKER_DATA_DIR` in the image. Do *not*
  point it anywhere under `/app`: shipped files live there, including
  `tracker/vision/codebook_data.json`, which the decoder reads and which
  your printed markers are generated from. A bind mount there shadows it
  and the app won't start. (This was a real bug — bind mounts don't seed
  from the image the way named volumes do.)
- **Change the password.** It's plain HTTP basic auth and the default is
  `admin`/`admin`.

Then open `http://<zima-ip>:5001`.

### Verify it came up

```bash
curl http://<zima-ip>:5001/api/version
```

You should get the build version, commit, and the codebook fingerprint
(`n_chips: 20`, `n_ids: 24`). That last part is the useful bit: it
confirms the running image carries the same codebook your markers were
printed from.

---

## 2. Install from your app store (for update prompts)

Once CI has published and the store repo has rebuilt (check its Actions
tab — `Update app v2 from new image build` should have run), the entry
appears in your existing store at
`https://nazarborodavka.github.io/owlbear-sync-store/`.

If that store source is already added to ZimaOS, **Owlbear Token Tracker
v2** shows up as a *new app* alongside v1 — it has a distinct app id on
purpose, so installing it does not replace your v1 install.

This path is what gives you "Update available" when you push to master.
If you installed via Custom Install first, uninstall that before
installing from the store (same ports and AppData path), or they'll
conflict.

> **Keep the pinned tag pinned.** The store entry's image tag
> (`...-v2:master-N`) is rewritten by CI on every master build. Don't
> hand-edit it to `latest` — ZimaOS detects updates by comparing that
> exact string against what's installed, and a floating tag never
> changes, so it would silently stop prompting.

---

## 3. First-run setup, in order

The calibration steps depend on each other in this order. Each is
independently rerunnable later without disturbing the others.

1. **Change the password** if you didn't in compose, then reload.

2. **Camera.** Enter the RTSP URL once:
   `rtsp://user:pass@<camera-ip>:554/h264Preview_01_main` (Reolink's main
   stream). Press **Start**.

   It's remembered in the data volume (`0600`, since it carries the
   camera login) and auto-starts on every later boot — you won't retype
   it after a reboot or an update. The dashboard only ever shows a
   redacted form. Alternatively set `TRACKER_CAMERA_URL` in compose,
   which pins it and wins over the UI.

   If the camera is unreachable the dashboard still loads fine; the
   connection attempt runs in the background. Press Start again once
   it's up.

3. **Camera settings** (on the camera's own web UI — easier than its CGI
   API for one-time setup), in order of how much they matter:
   - **Exposure capped at 1/60s or faster.** This is the single biggest
     factor in whether fast token moves survive, because blur is
     `speed × exposure`. Measured: 1/60s tracks to 250mm/s; a dim-room
     1/15s already fails at 130mm/s.
   - **Force night/IR mode on.** The projector's changing brightness is
     the main nuisance light, and IR sits outside its output.
   - **Lock autofocus.** A refocus mid-session changes px/mm and
     quietly invalidates your play-area calibration.

4. **Lens calibration** *(optional but do it once)*. Print a chessboard,
   hold it at several angles. Corrects barrel distortion. Only needs
   redoing if you change camera or zoom.

5. **Play area.** Click the four corners of your play surface on the live
   preview (top-left, top-right, bottom-right, bottom-left), and enter
   its real width and height in mm.

   Watch the reported **px/mm**: 2.5 is the warned-at minimum, 3.2 is
   where all the validated numbers come from. If it's low, zoom in or
   lower the camera — those trade against area covered. **Redo this
   whenever the camera moves, zooms or refocuses.**

6. **Print markers.** On the ZimaOS host:

   ```bash
   docker exec owlbear-tracker-v2 \
     python tools/generate_markers.py --outdir /data/markers --n-ids 8
   ```

   The SVG lands in `/DATA/AppData/owlbear-tracker-v2/markers/`. Then:
   print at **100% / actual size** (no "fit to page"), **measure the
   50mm bar** with a ruler, use a **laser printer or pigment ink** on
   **matte** paper, and mount on 40mm round bases.

   Pigment is not optional — dye-based black is nearly transparent above
   ~800nm, so under IR a dye-printed marker shows almost no contrast and
   won't decode at all. Gloss paper retroreflects the IR LEDs and blows
   out the code ring.

   **v1's markers do not work.** Different codebook entirely.

7. **Detector tuning** *(optional)*. With markers on the table under
   real lighting, this measures the peak/margin distribution they
   actually produce and tightens the accept thresholds. Improves
   rejection of non-markers (dice, mugs).

8. **Grid.** With a map projected, measures the projected grid spacing so
   the extension can auto-zoom until one cell matches a 40mm base.

9. **Name your tokens** in the dashboard. Names persist and drive the
   extension's automatic matching.

---

## 4. The Owlbear Rodeo extension

Add this as a custom extension in Owlbear Rodeo:

```
http://<zima-ip>:5001/extension/manifest.json
```

> **Expect this to need HTTPS.** Owlbear Rodeo is served over `https://`,
> and browsers block a secure page from loading `http://` subresources as
> mixed content. A plain LAN IP over HTTP will most likely be refused —
> `http://localhost` is exempt as a secure context, but the ZimaOS box
> isn't localhost from your laptop.
>
> Try it first; if the panel stays blank, check the browser console for a
> mixed-content error. The fix is to put TLS in front of it — a
> Cloudflare Tunnel, Tailscale Serve, or any reverse proxy with a cert —
> and register the `https://` manifest URL instead. That also handles
> playing from outside your LAN.
>
> This applies to v1 too, so whatever you already do for v1 works here.

In the extension panel, map each scene token to a tracked marker id.
Mappings live in **scene metadata**, so they survive reloads and come
back with the scene next session. Tokens whose names match a tracked
token's name are matched automatically.

Positions cross the API in **millimetres** relative to the play-area
origin, and the extension converts to the scene's own grid using its DPI
and scale. That's what makes it work on a cast receiver as well as the
host — nothing depends on anyone's window size.

---

## 5. Updating

Push to `master`. CI runs the tests, publishes the image, smoke-tests it,
then tells the store. ZimaOS shows "Update available" for store installs.

Calibration, names and the camera URL all live in
`/DATA/AppData/owlbear-tracker-v2` and survive updates.

> If a build ever changes `tracker/vision/codebook_data.json`, **every
> printed marker is invalidated** and must be reprinted. CI warns when
> that file changes in a commit, and `/api/version` reports the running
> codebook so you can check what's deployed.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Container won't start, `FileNotFoundError: codebook_data.json` | The `/data` bind is pointed somewhere under `/app` and is shadowing shipped files. Target must be `/data`. |
| `denied` / `unauthorized` pulling the image | GHCR package is private. Make it public, or `docker login ghcr.io` on the host. |
| Dashboard loads, preview stays black | Camera not started, or the RTSP URL is wrong. Check `GET /api/camera` (needs auth) for what it thinks is configured. |
| Tokens detected but never named / no ids | Markers printed at the wrong scale (measure the 50mm bar), or dye-based black ink under IR. |
| Ids flicker or refuse to decode | px/mm too low — recheck the play-area step's verdict. Or exposure too long; cap at 1/60s. |
| Extension panel blank in Owlbear | Mixed content — the manifest is `http://` on an `https://` page. Put TLS in front (see §4). |
| ZimaOS never offers an update | The store entry's image tag was edited to a floating tag, or the store repo's workflow didn't run — check its Actions tab. |
| Everything works, positions are wrong scale | Play area width/height entered in the wrong units, or the camera moved since calibrating. |

To score what's actually happening, record footage and replay it:

```bash
docker exec owlbear-tracker-v2 python tools/replay.py \
    /data/recording.mp4 --calibration /data/calibration \
    --expect-tokens 6 --annotate /data/annotated.mp4
```

It reports identity flips, phantoms, id churn and frame timing with no
hand-labelling. See [VALIDATION.md](VALIDATION.md).
