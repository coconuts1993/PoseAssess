# Wii Balance Board integration in PoseAssess: design and work split

Status: the scaffold is in place (branch `claude/wizardly-sagan-tv7pap`, not committed). This
document is the contract for the 5 implementation agents: CORE-WII, CORE-BOARD, GUI-WII,
GUI-CAPTURE and GUI-VIEW. The scaffold files are also part of the contract: their module and
function docstrings give details this document only summarizes. Section 13 assigns every file
to exactly one owner.

Repository: `/home/user/poseassess`. Do not commit or push; the orchestrator does that.
Python environment: `/home/user/venv-pa/bin/python`.

---

## 0. Goal and hard requirements

The user wants to keep every feature of the original PoseAssess (Pose2Sim 0.10 offline
pipeline, PySide6 GUI, 3D View) and add the Wii Balance Board.

- **R1: nothing existing is removed or changes behaviour.**
  - Existing pages keep their content, titles and numbers. New pages and new sections are
    allowed.
  - The app must start and work when hidapi is missing, when no board is connected, and when
    a project has no Wii data.
  - Scaffold check: the content area of every existing page is pixel-identical to the
    original commit (screenshots, section 12).
- **R2: plug-and-play.**
  - A Bluetooth-paired board is found and connected automatically (hot-plug polling), and
    reconnected after a drop.
  - Tare is remembered per device.
  - Auto-connect is on by default at app start.
- **R3: data collection inside PoseAssess.** Two workflows:
  - (a) Capture page: record the multi-camera trial videos and the Wii data together. The
    videos go straight into `videos/camNN.mp4`.
  - (b) Record Wii-only (Capture page with cameras off, or the CLI), import the external
    videos with the existing Videos page, then align.
- **R4: board location, the same way as clicking the checkerboard corners.**
  - Click TL, TR, BR, BL, C in camera frames. "Front" is the long edge **opposite** the power
    button (the TL/TR sensors).
  - The pose is computed in the Calib.toml world.
  - With 2 or more calibrated cameras: triangulate. With 1 camera: PnP.
  - Fit the board flat on the floor, reject mirrored click orders, and offer a live corner
    check to fix a 180° front/back swap.
- **R5: alignment.**
  - Every Wii sample has `t` (perf_counter), `t_rel` and `t_unix`. Video frames have
    timestamps. Events can be marked.
  - Imported videos are aligned by a sync event (jump or stomp detected in the force vs the
    vertical marker motion in the .trc) or by a manual offset. The alignment is stored per
    trial.
- **R6: outputs.**
  - 3D View: board, COP point and trail, ground-reaction-force arrow, and whole-body COM with
    plumb line, all synchronized with the .trc playback.
  - Results: COP sway metrics, COM-COP relation, force plots, and an export of the fused
    per-frame table.

Language:
- UI text is **English**, like the rest of the app.
- Code comments and docstrings are English.
- Every module has a docstring and `from __future__ import annotations`.

Reference implementation:
- PoseBoard commit **3ea6d8a** only. Read-only extraction:
  `/tmp/claude-0/-home-user-PoseBoard/80080f02-f6a3-5192-9efd-bb7471921107/scratchpad/ref3ea/`,
  or `git -C /home/user/PoseBoard show 3ea6d8a:<path>`.
- Do **not** read or copy the PoseBoard working tree; another workflow is editing it.
- PoseAssess must not import `poseboard` at runtime.

---

## 1. Architecture

```
                       ┌──────────────────────── GUI (PySide6) ─────────────────────────┐
 MainWindow ── nav ──► 1 Project │ 2 Calibration │ 2b Wii Board │ 3 Videos │ 3b Capture │
                       4 Run │ 5 3D View (+overlays) │ 6 Results (+Balance tab) │ 7 Benchmark
      │ status bar: WiiStatusWidget (permanent)
      ▼
 AppState ── project_changed(Project|None)            (existing)
          ── balance_changed(str)                     (NEW: "board"|"trial"|"recording"|"alignment")
          ── wii: WiiController (NEW, lazy)  ── owns ONE ForceSource (WiiAutoConnect | HID | Simulated)
                                                   signals on the GUI thread; tare memory; tare lock
 ─────────────────────────────── core (no Qt) ─────────────────────────────────────────────
 poseassess/wii/            protocol · device (ForceSource…) · recorder (WiiRecorder) · io (formats) · record_cli
 poseassess/core/capture/   camera (CameraStream) · settings · session (CaptureSession) · export (→ videos/)
 poseassess/core/balance/   paths · trial · calib · geometry · board · trc · com · analysis · alignment · fusion · overlay
 existing (unchanged)       project · config_gen · calibration/* · calib_read · calib_check · trc_io · pipeline …
```

Data flow of a trial:

1. Calibration (existing) writes `calibration/Calib.toml`.
2. Wii Board page: clicks go to `wii/board/camNN/points.json`, then `compute_board` writes
   `wii/board/board.json`.
3. Capture page: a take goes to `wii/recordings/<id>/` (wii.csv, events.csv,
   camNN.mkv + timestamps). The export writes `videos/camNN.mp4` + `frames.csv`, sets
   `project.frame_rate`, and updates `wii/trial.json` (method "recorded").

   Alternatively: import the videos (Videos page) and a Wii recording, then align via sync
   event or manual offset. The alignment goes to `wii/trial.json`.
4. Run (existing Pose2Sim) writes `pose-3d/*.trc`.
5. `fusion.fuse_trial(project)` builds a `FusedTrial` (one row per .trc frame).
   - 3D View: `BalanceSkeleton3DViewer.set_balance(fused, board)`.
   - Results: `BalancePanel`, which shows metrics and plots and exports
     `wii/exports/<trc>_fused.csv`, `_balance_summary.json` and `_grf.mot`.

---

## 2. Navigation and pages

`gui/pages/__init__.py` is already done and frozen:

| Row | nav_title | Class | Owner | needs_project |
|---|---|---|---|---|
| 0 | `1. Project` | ProjectPage (unchanged) | – | False |
| 1 | `2. Calibration` | CalibrationPage (unchanged) | – | True |
| 2 | **`2b. Wii Board`** | `WiiBoardPage` (new) | GUI-WII | **False**: the device tab works without a project; the board tab disables itself |
| 3 | `3. Videos` | VideosPage (unchanged) | – | True |
| 4 | **`3b. Capture`** | `CapturePage` (new) | GUI-CAPTURE | True |
| 5 | `4. Run` | RunPage (unchanged) | – | True |
| 6 | `5. 3D View` | Viz3DPage (**+overlays**) | GUI-VIEW | True |
| 7 | `6. Results` | ResultsPage (**+"Balance (Wii)" tab**) | GUI-VIEW | True |
| 8 | `7. Benchmark` | BenchmarkPage (unchanged) | – | True |

Existing titles are not renumbered. The new pages use sub-numbers.

Shared shell changes, done in the scaffold:

- **`BasePage`** has two new overridable hooks: `can_close() -> bool` (default True) and
  `shutdown()` (default no-op).
- **`MainWindow`**:
  - adds `WiiStatusWidget(state.wii)` with `statusBar().addPermanentWidget`, so the project
    message never overwrites it;
  - calls `state.wii.start_default()` through `QTimer.singleShot(0, …)`;
  - `closeEvent` asks every page `can_close()` (any False cancels), then calls every page's
    `shutdown()`, then `state.shutdown()`. Every call is wrapped in try/except with logging,
    so a Wii failure never blocks startup or exit.
- **`AppState`**:
  - `balance_changed = Signal(str)`;
  - `notify_balance_changed(what)`;
  - `wii`: a lazily created `WiiController` that never imports hid;
  - `shutdown()`.

Who emits `balance_changed` (always after the file on disk is written):

| Emitter | Value | When |
|---|---|---|
| GUI-WII | `"board"` | compute board, swap front/back, corner check, geometry change |
| GUI-CAPTURE | `"recording"` | a take was saved |
| GUI-CAPTURE | `"trial"` | a take was exported to `videos/` (trial.json changed) |
| GUI-VIEW | `"trial"` | recording selected or imported |
| GUI-VIEW | `"alignment"` | alignment saved |

Listeners:
- Viz3DPage and BalancePanel reload on any value.
- WiiBoardPage reloads on `"board"`.
- CapturePage refreshes its takes list on `"recording"` and `"trial"`.

UI conventions (copy the existing style):
- Left control column of fixed width 230–260 with `QGroupBox` sections.
- Main view on the right with stretch 1.
- Help text in `<small>` `QLabel`s; tooltips are full sentences.
- Primary buttons carry a glyph ("▶ …", "✓ …", "📸 …"). Buttons that open a dialog end with "…".
- Status colours: good `#3ad35a`, warn `#e0a94c`, bad `#e05c5c`, accent `#3d8bfd`.
- Errors are shown with `QMessageBox.warning/critical`. Broad catches are written
  `except Exception as e:  # noqa: BLE001`.
- Heavy imports (cv2, core modules) go inside methods.
- Every handler starts with `proj = self.state.project; if not proj: return`.
- Tab titles contain no bare `&`; write `&&`.
- Size: `MainWindow` asks for 1000×680, but the existing pages already push the effective
  minimum width to about 1380–1420 px (measured with the original code; it depends on the project). New pages must **not** increase it: keep their
  minimum width at or below that, and use tabs or splitters rather than wide rows.

---

## 3. Files inside a project (all Wii data under `<root>/wii/`)

The rules come from the Pose2Sim and PoseAssess code (`core/balance/paths.py`). Do not break
them:
1. No top-level folder with "calib" in its name.
2. Never write `*.toml` or `*.yml` into `calibration/`.
3. Only `camNN.<video>` files in `videos/`, where every video file counts as a camera. No
   temp files, previews or timestamps there, ever. Never leave both `camNN.mkv` and
   `camNN.mp4`.
4. Nothing new in `pose/`, `pose-sync/`, `pose-associated/`, `pose-3d/` or `kinematics/`.
5. No `Config.toml` in sub-folders.
6. No Wii keys in `project.toml`. `ProjectConfig` drops unknown keys and cannot store None.

```
<root>/wii/
  board/camNN/<any>.jpg        reference frames for clicking (FrameSelector NNNNNN.jpg, snapshot_*.jpg, board.jpg …)
  board/camNN/points.json      clicks of camera NN                          (board.CameraClicks)
  board/board.json             board registration                           (board.BoardRegistration)
  recordings/<id>/             one take; id = YYYYmmdd_HHMMSS[_subject][_2…] (poseassess.wii.io)
      session.json  wii.csv  events.csv  camNN.mkv  camNN_timestamps.csv  frames.csv  export_tmp/ (during export only)
  trial.json                   the trial's recording + alignment            (trial.TrialInfo)
  capture.json                 Capture page settings                        (capture.settings.CaptureSettings)
  exports/                     <trc stem>_fused.csv, _balance_summary.json, _grf.mot
```

Camera identity is **positional**:
- camera i (1-based) = `camNN` = the i-th table of Calib.toml = the i-th sorted video =
  `calibration/extrinsics/camNN`.
- Calib table names differ between producers ("1", "01", "cam01"). Never match by name.
- `calib.project_cameras(project)` returns them keyed `camNN`.

### 3.1 CSV and JSON formats

All CSVs:
- comma-separated, header row, `%.6f`, empty field = NaN;
- `events.csv` is UTF-8 **with BOM**;
- readers are in `poseassess/wii/io.py` (already implemented; use them, and do not
  hand-parse).

**wii.csv** (`io.WII_HEADER`), about 100 Hz:

```
t,t_rel,t_unix,TR_kg,BR_kg,TL_kg,BL_kg,total_kg,cop_x_board,cop_y_board,cop_x_world,cop_y_world,cop_z_world
```

- `t`: `time.perf_counter()` of the recording PC, in s.
- `t_rel = t - t0`: seconds since the recording start. This is the time base of every
  alignment.
- `t_unix = t + clock_offset_unix`: the offset is measured once at start by
  `capture_clock_offset()`, to about 1 ms.
- Sensor order is **TR, BR, TL, BL**, tared, in kg.
- COP is in metres in the board frame, NaN when `total_kg < min_total_kg` (default 5 kg).
- The `cop_*_world` columns are filled only when a board pose was given at record start.
  **Readers must ignore them** and recompute from the current `board.json`.

**events.csv** (`io.EVENTS_HEADER`): `t,t_rel,t_unix,label`.
- Labels are whitespace-normalized. An empty label becomes "event".
- Automatic labels: `wii_connected`, `wii_disconnected`.
- User labels are free (default `mark_<n>`, or `sync`).

**camNN_timestamps.csv** (`io.TIMESTAMPS_HEADER`): `frame,t,t_rel,t_unix`.
- One row per frame written to `camNN.mkv`, by `CameraStream`.
- Frames of another size are dropped and get no row.

**frames.csv**: written by the capture export, one row per frame of the exported
`videos/camNN.mp4`.

```
frame,t,t_rel,t_unix,cam01_src,cam01_dt_ms,cam02_src,cam02_dt_ms,...
```

- `frame` counts 0..M-1 = Pose2Sim Frame#.
- `t`/`t_rel`/`t_unix` are the grid time of that exported frame (uniform `1/fps`).
- `camNN_src` is the source frame index in `camNN.mkv`.
- `camNN_dt_ms` is the source time minus the grid time.

**session.json** (`io.SESSION_SCHEMA = "poseassess.wii.session/1"`), written by
`WiiRecorder`:

- At start: `schema`, `app`, `created`, `subject`, `notes`, `clock` (text), `t0`, `t0_unix`,
  `clock_offset_unix`, `start_time_iso`, `has_wii`, `has_video`, `camera_names`,
  `streams: [{name, source, fps, video, timestamps}]`, `world_frame`, `board_geometry`,
  `board_pose`, `force_source` (`ForceSource.info()`), `force_source_history`
  (`[{t_rel, t_unix, reason, type, device, tare_kg, min_total_kg}]`).
- `extra_meta` from the Capture page: `project: {name, root}` and
  `capture: {settings: <capture.json dict>, calib_sizes: {camNN: [w, h]}}`.
- At stop: `t_stop`, `t_stop_unix`, `stop_time_iso`, `duration_s`, `samples: {wii, events}`,
  and `has_wii` (updated).
- Export adds:
  ```
  export: {fps, n_frames, t_start_rel, videos: ["videos/cam01.mp4", …], frames_csv: "frames.csv",
           max_dt_ms: {camNN: f}, duplicated: {camNN: n}, skipped: {camNN: n}, keep_raw, time}
  ```
- Import adds `imported_from: "<abs path>"`.

**points.json** (per camera; readable by the existing `extrinsic.load_clicked_points` because
of the `points` key):

```json
{"points": [[x, y], ...], "labels": ["TL","TR","BR","BL","C"], "image": "000123.jpg",
 "image_size": [1920, 1080], "source": "videos/cam01.mp4#frame=123"}
```

- Up to 5 points, in pixels of `image`.
- `image` is a file in the same folder, or a project-relative path.

**board.json** (`"poseassess.wii.board/1"`, see the `board.py` docstring): `schema`,
`created`, `calib {file, sha1, mtime_ns}`, `world_frame`, `world_up_sign`,
`geometry {length_mm 511, width_mm 316, sensor_dx_mm 433, sensor_dy_mm 238, height_mm 53}`,
`pose` (`BoardPose.to_dict()`: `board_to_world {R, t}`, `method` "triangulation"|"pnp",
`reproj_error_px {camNN: px}`, `world_camera`, `floor_constrained`, `tilt_deg`, `warnings[]`,
`notes[]`), `cameras_used`, `clicks {camNN: points.json dict}` (snapshot), `corners_world`
(TL, TR, BR, BL), `center_world`, `sensors_world` (TR, BR, TL, BL), `up_world`, and
`corner_check` (null or `{result: ok|swapped|redo, pressed, time}`).
- `BoardRegistration.stale` is runtime-only. `load_board` sets it when the sha1 of the
  current `find_calib_toml()` differs from the stored one (recalibration, Flip Z), or when
  the calibration is missing.

**trial.json** (`"poseassess.wii.trial/1"`, see the `trial.py` docstring):

```json
{"schema": "poseassess.wii.trial/1", "recording": "20260924_153000", "source": "capture|external",
 "alignment": {"method": "recorded|sync_event|xcorr|manual|none", "offset_s": 0.0,
               "frames_csv": "frames.csv", "trc": "pose-3d/x.trc", "details": {}, "updated": "iso"},
 "videos_fingerprint": {"cam01.mp4": [size, mtime_ns]}, "body_mass_kg": null, "notes": ""}
```

**capture.json**:

```json
{"cameras": {"cam01": {"source": 0, "width": 1920, "height": 1080, "fps": 30.0, "enabled": true}},
 "output_fps": 30, "keep_raw": true, "subject": ""}
```

`source` is a device index, or a string (a URL, or a video file that loops, for tests).

---

## 4. Coordinate frames

| Frame | Definition |
|---|---|
| **Board B** | Metres. Origin = centre of the top surface. **+x = subject's right** (TR/BR side), **+y = front** (the TL/TR long edge opposite the power button), **+z = up** (the board normal). Right-handed. `BoardGeometry.landmarks()` gives TL(−hx,+hy,0), TR(+hx,+hy,0), BR(+hx,−hy,0), BL(−hx,−hy,0), C(0,0,0). `sensors()` gives TR, BR, TL, BL. |
| **World W** | The Calib.toml world = the extrinsic checkerboard frame, in metres. `x_cam = R·x_W + t` with `R = Rodrigues(rotation)`. Z = X×Y of the clicked checkerboard axes, so **+Z can point down** (`invert_z` or the click origin). `calib.world_up_sign(cams)`: +1 if the median camera-centre z ≥ 0, else −1. The checkerboard plane is Z = 0, normally the floor. The board's top surface is about 53 mm above it along the physical up. |
| **TRC T** | Pose2Sim .trc, Y-up. `p_T = p_W[[1,2,0]]` (X′=Y, Y′=Z, Z′=X). Inverse: `p_W = p_T[[2,0,1]]`. Use `trc.zup_to_yup` and `trc.yup_to_zup`, which apply to points and directions. |
| **Viewer V** | `Skeleton3DViewer._tf(p_T) = p_T @ M.T − center`, where `M = _orient_matrix(trc)`, a proper rotation that points the subject's up to display +Z. |

Transforms:
- Board to world: `p_W = R_bw · p_B + t_bw`, i.e. `BoardPose.board_to_world.apply`.
  `up_world = R_bw[:, 2]`, which **always points physically up**. `register_board` with
  `up_sign = −1` works in the world conjugated by `F = diag(1,−1,−1)` and maps back, so the
  board frame stays right-handed with z up.
- World point to viewer: `viewer._tf(zup_to_yup(P_W))`. This is exactly what `set_cameras`
  does.
- World direction to viewer: `zup_to_yup(d_W) @ viewer._M.T`, with no centre subtraction.
- COM is computed with `com.center_of_mass` on markers. It is linear, so any frame works. The
  fusion computes it in W.

Derived quantities:
- COP in world: `cop_W = board_to_world.apply([cop_x, cop_y, 0])`, on the top surface.
- Ground reaction force: the Wii measures vertical load only (no shear), so
  `F_W = up_world · total_kg · g` with `g = 9.80665`, applied at `cop_W`.
  - 3D arrow length: `total_kg · 0.005 m` (`fusion.COP_ARROW_M_PER_KG`), i.e. 0.35 m for
    70 kg.
- COM plumb line: `plumb = com_W − ((com_W − t_bw)·up_world) · up_world`, the projection on
  the board's top plane (`fusion.plumb_point`).
- COM in the board frame: `com_B = board_to_world.inverse().apply(com_W)`.
- COM − COP in the board frame: x = medio-lateral (ML), y = antero-posterior (AP).
- Sway metrics are in mm, with x = ML and y = AP.

---

## 5. Time model and alignment

- **Clock.** Everything recorded in-app (Wii samples, camera frames, events) uses one
  `time.perf_counter()` clock per recording, with `t0` = record start.
- **Pose2Sim .trc.**
  - `Frame#` = 0-based index of the `videos/camNN.*` frame (json index). It can start above
    0 after `frame_range` trimming.
  - `Time = Frame# / Config.frame_rate`.
  - `read_trc_full` keeps Frame#. The old `trc_io.read_trc` drops it and mis-parses NaN
    rows (see section 11).
- **Alignment methods** (in `wii/trial.json`; function `alignment.trc_rows_to_wii_time`):
  - `recorded`: the videos came from this take's export. `t_rel(row) = frames.t_rel[Frame#]`.
    This holds even when the project fps differs from the measured camera fps. Frame stamps
    are taken when `cap.read()` returns; the export subtracts the camera latency of
    `wii/capture.json` (`latency_ms`, global or per camera; measured with
    `alignment.check_recorded_alignment`, "Check timing with a sync event").
    `offset_s` = frames.t_rel[0] is the fallback. Pose2Sim's synchronization stage must stay
    **off** for recorded takes (it is off by default).
  - `sync_event`, `xcorr`, `manual`: the videos were recorded elsewhere.
    `t_rel = Time + offset_s`.
  - `none`: not aligned. Fusion raises, and the GUI asks the user to align.
- **`auto_align`** (CORE-BOARD), for imported videos:
  1. Detect force events with `analysis.detect_force_events(recording)`: `step_on`,
     `step_off`, `takeoff`, `landing`, `stomp`, in `t_rel`.
  2. Detect TRC events with `analysis.detect_trc_events(Time, names, world coords, up)`:
     `takeoff`, `landing`, `stomp`, in TRC `Time`. `up` comes from `BoardRegistration.up_world`,
     else `world_up_sign·Z`.
  3. `analysis.match_events` gives a type-aware offset (at least 1 matched jump or stomp; 2 or
     more recommended).
  4. `analysis.xcorr_offset` correlates the measured total force with `m·(g + z̈_COM)` from
     the .trc. It refines ±1 s around the event offset, or searches globally when no events
     matched.
  5. The first method that succeeds wins, and `details` stores all candidates.
  6. Ask the subject to do **2 small jumps or 2 stomps** at the start of an externally
     recorded trial. The UI help text says this.
- **Videos changed.** When `trial.source == "capture"` and `videos_changed()` is true
  (the videos were replaced on the Videos page), the recorded alignment is invalid. The
  status turns into a warning and the user must re-align or re-export.

---

## 6. Module APIs (scaffold state: **impl** = working now, **STUB** = raises NotImplementedError)

The full docstrings in the files are part of the contract. Owners may add optional kwargs,
private helpers or new functions. They must **not** rename or remove documented names, or
change their semantics. If a change is unavoidable, say so in the final report.

### 6.1 `poseassess/wii/` (CORE-WII)

- **`protocol.py`**: impl, verbatim port. `VENDOR_ID`, `PRODUCT_IDS`, report builders,
  `parse_read_data`, `is_balance_board`, `Calibration.from_bytes/to_kg`, `parse_sensor_raw`,
  `center_of_pressure(kg, dx, dy, min_total_kg) -> (x, y)`, `SENSOR_ORDER`,
  `pressed_sensor(baseline_kg, pressed_kg, min_rise_kg=3.0) -> "TR"|"BR"|"TL"|"BL"|None`,
  `KG_TO_N`.
- **`device.py`**: impl, verbatim port plus `hidapi_status() -> (bool, str)`.
  - `ForceSample(t, kg(4), total_kg, cop_board(2), raw)`.
  - `ForceSource`: `start`, `stop`, `running`, `status`, `connected`, `add/remove_listener`,
    `add/remove_status_listener`, `latest`, `recent(s)`, `device_key`, `do_tare(s)`,
    `adopt_tares(other)`, `info`.
  - `BalanceBoardHID(path=None, silence_timeout_s=5)` with the static methods
    `list_devices()` and `open_device(path)`, which tests monkeypatch.
  - `WiiAutoConnect(path=None, poll_interval_s=2, retry_failed_s=10)`: attributes
    `connections`, `disconnects`, `last_error`, `fatal_error`.
  - `SimulatedBoard(mass_kg=70, rate_hz=100)`.
  - Exceptions: `BoardDisconnected`, `HidapiUnavailable`, `NotABalanceBoard`.
  - `sort_boards`, `DEFAULT_COP_MIN_KG = 5`.
  - `import hid` happens only in `_import_hid()`.
- **`io.py`**: impl. Constants `SESSION_JSON`, `WII_CSV`, `EVENTS_CSV`, `FRAMES_CSV`,
  `TIMESTAMPS_SUFFIX`, `SESSION_SCHEMA`, `WII_HEADER`, `SENSOR_COLS`, `EVENTS_HEADER`,
  `TIMESTAMPS_HEADER`, `FRAMES_HEADER_BASE`. Functions `timestamps_csv_name(cam)`,
  `is_recording_folder`, `read_csv_columns`, `read_wii_csv(file|folder)` (all WII_HEADER
  columns, sorted by t_rel), `read_events_csv`, `read_timestamps_csv`, `read_frames_csv`,
  `read_session_json`, `write_session_json`.
- **`recorder.py`**: impl, a port of `SessionRecorder` without pose or post-processing.
  - `VideoSink` protocol: `name`, `source`, `fps`,
    `start_recording(path, clock_offset_unix, t0) -> Path`, `stop_recording()`.
  - `capture_clock_offset()`, `iso_time()`, `on_console_close(cb)`,
    `remove_console_close(h)`, `new_recording_folder(root, subject="", t_unix=None)`.
  - `WiiRecorder`: `start(*, folder=None, root=None, force=None, cams=None, board=None,
    geometry=None, subject="", notes="", extra_meta=None) -> Path`, `attach_force(force)`,
    `log_force_source(force, reason)`, `add_event(label, t=None) -> dict|None`,
    `update_meta(**items)`, `stop() -> Path|None`, `to_unix(t)`. Attributes `folder`, `t0`,
    `t0_unix`, `clock_offset_unix`, `meta`, `counts`, `recording`.
- **`record_cli.py`**: STUB. `build_parser()`, `main(argv) -> int`. Exit codes 0/2/3/4/130.
  Options `--project DIR` (records into `DIR/wii/recordings/`) or `--out DIR`, `--subject`,
  `--notes`, `--seconds`, `--simulate`, `--wait`, `--tare`, `--min-kg`, `--device PATH|N`,
  `--list`. `__main__.py` calls it: `python -m poseassess.wii`.

### 6.2 `poseassess/core/balance/` (CORE-BOARD)

- **`paths.py`**: impl.
  - `cam_name(i)`, `cam_index(name)`.
  - `WiiPaths(project|root)` with: `wii_dir`, `board_dir`, `board_cam_dir(i)`,
    `board_points_file(i)`, `board_json`, `recordings_dir`, `recording_dir(id)`,
    `trial_json`, `capture_json`, `exports_dir`, `relative(p)`, `resolve(rel)`, `ensure()`,
    `list_recordings()`.
- **`trial.py`**: impl except `import_recording`.
  - `Alignment`, `TrialInfo` (to/from dict), `now_iso`.
  - `load_trial`, `save_trial` (atomic).
  - `videos_fingerprint`, `videos_changed`.
  - `set_recording(project, rec_id, source="external", alignment=None) -> TrialInfo`.
  - STUB `import_recording(project, src, name=None, make_active=True) -> rec_id`.
- **`calib.py`**: impl.
  - `CameraCalibration` (port).
  - `load_calib_toml(path) -> list`, in file order, with dist padded to 5 and all-zero
    extrinsics treated as none.
  - `find_calib_toml(project)`: the first of `sorted(Calib*.toml)`, like the GUI.
  - `project_cameras(project, calib_path=None) -> {camNN: cam}`.
  - `calib_fingerprint(path)`, `world_up_sign(cams)`.
- **`geometry.py`**: data classes impl; algorithms STUB.
  - impl: `LANDMARK_NAMES`, `SENSOR_NAMES`, `FLOOR_MAX_TILT_DEG = 10`, `TILT_NOTE_DEG = 1`,
    `MIRRORED_CLICKS_MSG`, `BoardGeometry`, `RigidTransform`, `BoardPose`,
    `rotate_board_frame`.
  - STUB (port from 3ea6d8a): `kabsch`, `procrustes_2d`, `triangulate_point`,
    `backproject_to_plane`, `project_to_image`, `reprojection_error`.
  - STUB (port and adapt): `register_board(geometry, cams, clicks, floor=True, up_sign=None)`,
    `solve_board_pnp(cam, model, img_px, return_ambiguity=False, up_world=None)`. In UI
    strings, "tab 2" becomes "2. Calibration".
- **`board.py`**:
  - impl: `CameraClicks`, `BoardRegistration` (with `corners_world`, `center_world`,
    `sensors_world`, `up_world`, `cameras_used`), `load_clicks`, `save_clicks`,
    `load_all_clicks`, `swap_front_back_clicks`, `load_board` (with the stale check),
    `save_board`, `set_corner_check`, `board_paths`.
  - STUB: `compute_board(project, geometry=None, floor=True, save=True,
    keep_corner_check=False)`. It uses every camera with 5 clicks and skips clicks whose
    `image_size` differs from the Calib size (with a note). It keeps the geometry of the
    existing board.json. It raises `ValueError` with user-readable text.
- **`trc.py`**:
  - impl: `ZUP_TO_YUP`, `YUP_TO_ZUP`, `zup_to_yup`, `yup_to_zup`, `trc_sort_key`,
    `find_trc_files(project)` (3D View order), and `TrcTrajectories` (`path`, `frames`,
    `times`, `names`, `coords` (N,K,3), `data_rate`, `header`, `world()`, `marker()`).
  - STUB: `read_trc_full(path)`.
- **`com.py`**: impl, verbatim port. `center_of_mass(keypoints (K,3), names,
  return_segments=False)`. It resolves Pose2Sim HALPE names; the head falls back to "Head".
- **`analysis.py`**: STUB. `interp`, `sway_metrics`, `body_mass_from_force`,
  `detect_force_events`, `write_events_csv`, `estimate_time_offset`, `detect_trc_events`,
  `match_events`, `xcorr_offset`. Constants `EVENT_TYPES`, `EVENT_COLS`, `G`.
- **`alignment.py`**: STUB. `AlignmentResult(ok, alignment, message, force_events,
  trc_events, signals)`, `trc_rows_to_wii_time`, `recorded_alignment`, `auto_align`,
  `apply_alignment`, `set_manual_offset`, `alignment_status -> (level, text)`.
- **`fusion.py`**:
  - impl: `FusedTrial` (fields listed in the file; `index_for_time`, `table`, `force_n`,
    `com_minus_cop`), `FUSED_COLUMNS`, `COP_ARROW_M_PER_KG`.
  - STUB: `cop_to_world`, `plumb_point`, `fuse_trial(project, trc_path=None)`,
    `balance_summary(fused, t_from=None, t_to=None)`, `export_fused_csv`, `export_grf_mot`,
    `export_all`.
- **`overlay.py`**: STUB. `project_world`, `draw_clicks`, `draw_board`, `draw_cop`,
  `board_check_image`.

### 6.3 `poseassess/core/capture/` (GUI-CAPTURE)

- **`camera.py`**: impl, verbatim port. `VIDEO_EXT = ".mkv"`, `Frame(index, t, image)`,
  `configure_capture`, `open_capture`, `open_video_writer`,
  `CameraStream(name, source, width, height, fps)` with `start`, `stop`, `running`,
  `latest`, `frame_age`, `measured_fps`, `error`, `dropped_frames`,
  `start_recording(path, clock_offset_unix, t0)`, `stop_recording`, `video_path`. It
  satisfies `VideoSink`.
- **`settings.py`**: impl. `CameraSetting`, `CaptureSettings`, `default_settings(n)`,
  `load_capture_settings(project)`, `save_capture_settings(project, s)`.
- **`session.py`**: STUB.
  - `probe_cameras(max_index=8, timeout_s=3)`.
  - `CaptureSession(project, settings)` with `open() -> {cam: error}`, `close()`,
    `latest(cam)`, `camera_status()`, `save_board_snapshots()`, `recording`, `folder`,
    `elapsed_s`, `start_recording(force=None, board=None, geometry=None, subject="",
    notes="")`, `add_event`, `attach_force`, `stop_recording`.
- **`export.py`**: STUB.
  - `ExportPlan`, `ExportResult`.
  - `plan_export(recording_dir, fps=None)`.
  - `export_recording_to_project(project, rec_id, fps=None, keep_raw=True,
    update_project=True, progress=None, cancel=None)`.
  - It builds the `recorded` `Alignment` **itself**:
    `Alignment("recorded", t_rel[0], "frames.csv", None, {...}, now_iso())` with
    `trial.set_recording(..., source="capture")` + `videos_fingerprint`. It does not depend
    on the CORE-BOARD stub.

### 6.4 GUI

- **`gui/wii_controller.py`** (GUI-WII): `WiiController(QObject)`, the only owner of a
  ForceSource.
  - Signals: `state_changed(str)`, `status_changed(str)`, `source_changed(object)`,
    `connection_event(str)` ("wii_connected"/"wii_disconnected"), `tare_changed(object)`.
  - Methods: `source`, `connected`, `auto_connect`, `hidapi_status()`, `start_default()`,
    `start_auto(path=None)`, `connect_device(path)`, `start_simulator(mass_kg=70)`,
    `disconnect()`, `list_devices()`, `tare(seconds=1)`, `set_min_load(kg)`,
    `lock_tare(reason|None)`, `tare_locked`, `latest(max_age_s=1.0)` (None when stale),
    `recent(s)`, `status_text()`, `shutdown()`.
  - Also `wii_start_mode()` reads `POSEASSESS_WII`: `0`/`off` means none, `sim` means the
    simulator, anything else means auto.
  - Scaffold: minimal and inert.
- **`gui/widgets/wii_status.py`** (GUI-WII): `WiiStatusWidget(controller)`.
- **`gui/widgets/cop_view.py`** (GUI-WII, impl port): `CopView.set_data(cop_trail (N,2) m,
  com_trail (N,2) m, total_kg, text="")`.
- **`gui/widgets/board_picker.py`** (GUI-WII): `BoardPointPicker(CornerPicker)` with
  `LABELS`, `HINTS`, `next_label()`, STUB `undo_last()`, STUB `swap_front_back()`.
- **`gui/pages/wii_board_page.py`** (GUI-WII): `WiiBoardPage`, a placeholder.
- **`gui/pages/capture_page.py`** + **`gui/widgets/camera_grid.py`** (GUI-CAPTURE):
  placeholders. `CameraGrid.set_cameras(names)`, `set_frame(name, bgr, text)`.
- **`gui/widgets/balance_3d.py`** (GUI-VIEW): `BalanceSkeleton3DViewer(Skeleton3DViewer)`
  with `set_balance(fused, board)` and `clear_balance()`.
- **`gui/widgets/balance_panel.py`**, **`wii_alignment.py`**, **`balance_plots.py`**
  (GUI-VIEW): `BalancePanel(state)` with `refresh()`; `AlignmentPanel(state)` with the
  signal `alignment_saved` and `refresh()`; `BalancePlotCanvas.plot_fused(fused, t_from,
  t_to)`.

---

## 7. Capture: how recorded videos reach the pipeline (GUI-CAPTURE)

1. `CaptureSession.open()` starts one `CameraStream` per enabled `camNN` (settings from
   `wii/capture.json`).
   - Preview: poll `latest(cam)` with a QTimer at 15–20 Hz while the page is visible.
2. **Record** needs one of two modes:
   - *all* project cameras (`num_cameras`) running, for a video take;
   - "Record cameras" unchecked, for a Wii-only take (the external-video workflow).

   Recording steps:
   - `WiiRecorder.start(root=WiiPaths.recordings_dir, force=state.wii.source,
     cams=running streams, board=load_board().pose if any, geometry=…, extra_meta=…)`.
   - While recording:
     - call `state.wii.lock_tare("recording")`;
     - connect `state.wii.source_changed` to `session.attach_force`;
     - connect `state.wii.connection_event` to `session.add_event`.
   - Mark event: F9 (and M when no text field has focus) plus a button, with an optional
     label edit. The label defaults to `mark_<n>`.
3. **Stop**:
   - Unlock tare.
   - Emit `balance_changed("recording")`.
   - Offer to export: automatically when `videos/` is empty, else ask "Replace the trial
     videos in videos/?".
4. **Export** runs in a QThread worker with progress and cancel.
   - `plan_export`: the common grid runs from the latest first frame to the earliest last
     frame, at `fps = settings.output_fps or project.config.frame_rate`. Warn when a camera's
     measured fps is below 90 % of it.
   - Nearest source frame per grid time.
   - Write `<rec>/export_tmp/camNN.mp4` with OpenCV `mp4v` at the recorded size, then move
     the files into `videos/` only when every camera succeeded.
   - Delete the other `videos/camNN.<video ext>` files of those cameras.
   - Write `frames.csv`.
   - Write session.json `export`.
   - If the fps changed: `project.config.frame_rate = fps`, `project.save()`,
     `config_gen.generate_config(project)`, then `state.set_project(project)` so the other
     pages reload.
   - Update trial.json: `set_recording(project, rec_id, "capture", Alignment("recorded",
     …))` and `videos_fingerprint`.
   - Delete the raw `camNN.mkv` unless `keep_raw` (default True).
   - Emit `balance_changed("trial")`.
   - Warn when a recorded size ≠ the Calib.toml `size` of that camera (intrinsics are
     resolution-specific).
5. The Videos page then shows every camera "ready" (it checks `camNN.mp4`). Run and the
   pipeline work unchanged.
6. **Takes list.** Show the recordings in `wii/recordings` (date, duration, cameras, Wii
   yes/no, exported yes/no). Actions: "Export to videos/" (needs the raw MKVs) and "Use as
   the trial's Wii recording" (`set_recording(..., "external")` for Wii-only takes; alignment
   then happens on the Results page).
7. **"📸 Board snapshots"** writes `wii/board/camNN/snapshot_<stamp>.jpg` for the Wii Board
   page.

---

## 8. The Wii is optional (all agents)

- Importing any module must never import `hid`. Only `wii.device._import_hid()` does,
  lazily. `tests/test_scaffold_smoke.py` asserts that `"hid" not in sys.modules` after the
  MainWindow is built.
- hidapi missing:
  - `WiiAutoConnect` sets `fatal_error` and stops.
  - `WiiController.status_text()` gives "Wii: off (hidapi not installed — pip install
    hidapi)".
  - The device tab shows the hint and keeps "Use simulator" enabled.
- No board: status "searching for a paired board…" with a pairing hint. Windows: Settings >
  Bluetooth > Add device > press the red SYNC button in the battery compartment; leave the
  PIN empty.
- Projects without `wii/` data:
  - The 3D View looks exactly as before: overlay toggles hidden or disabled, and no extra
    status text except a subtle "Wii: no data".
  - The Results "Joint angles" tab is unchanged.
  - The Balance tab explains what to do.
- A stub that raises `NotImplementedError` (during parallel work) is caught at the UI
  boundary and shown as "not available yet". It must never crash the app.
- Portable deploy (CORE-WII):
  - Add `run_wii.bat`, which runs `venv\Scripts\python.exe -m poseassess.wii %*` and pauses
    at the end.
  - Add a "Wii Balance Board" section to `README_DEPLOY.txt`: install hidapi into the bundle
    on the build PC with `venv\Scripts\python -m pip install hidapi`, which includes the DLL;
    pairing; CLI usage.
  - Do not edit `pip_freeze.txt` (it documents the current bundle).

---

## 9. Threading and shutdown

| Thread | Rule |
|---|---|
| ForceSource reader (daemon) | Calls listeners (the recorder writes CSV under its own lock) and status listeners. `WiiController` forwards state through a private `Signal(object, str)` with a queued connection (PoseBoard `app.py` l.180-182 / 746-747 pattern); only the controller touches Qt. |
| CameraStream (one per camera) | Owns its VideoCapture and never releases it during a blocking read. The GUI polls `latest()`. |
| GUI timers | Live COP (WiiBoard and Capture pages) at about 20 Hz and camera previews at 15–20 Hz, **only while the page is visible** (`showEvent`/`hideEvent`). The 3D View uses its existing timer. |
| QThread workers | Capture export, `auto_align`, `probe_cameras`, and optionally `compute_board`. Use the existing page pattern (`self._thread`, `self._worker`, `moveToThread`, `done`→`quit`, `deleteLater`). Never start a second worker while one runs. |

Shutdown:
- `MainWindow.closeEvent` calls `can_close()` on every page. `CapturePage` asks whether to
  stop a running recording; answering No cancels the close.
- Then every page's `shutdown()`: the capture page stops recording and cameras; pages stop
  their timers and workers.
- Then `state.shutdown()`, which calls `WiiController.shutdown()` to stop the source (LED
  off).
- Tare is blocked while recording: `WiiController.tare` raises, and the Tare button is
  disabled with a tooltip.

---

## 10. Per-agent tasks and acceptance criteria

### CORE-WII (`poseassess/wii/*`, CLI, deploy files, Wii tests)

1. Review the ported `protocol.py` / `device.py` (keep behaviour), and `recorder.py`
   (already implemented in the scaffold; fix anything you find, and keep the API).
2. Implement `record_cli.py`, a port of PoseBoard `wii/record.py` on `WiiRecorder`:
   - `--project` records into `<project>/wii/recordings/`;
   - events come from stdin lines;
   - `wii_connected` / `wii_disconnected` events;
   - console-close handler;
   - the same exit codes.
3. Add `run_wii.bat` and the `README_DEPLOY.txt` section.
4. Tests:
   - `tests/test_wii_protocol_device.py`: port `test_wii_auto.py` (all of it) plus the Wii
     tests of `test_core.py`.
   - `tests/test_wii_recorder.py`: Wii-only, video-only with a fake VideoSink, both,
     attach_force, events BOM, same-second folders, stop twice, a camera start failure rolls
     back, an abrupt `os._exit` child still leaves readable CSVs.
   - `tests/test_wii_cli.py`: port the CLI tests of `test_wii_auto.py` and `test_runtime.py`.
   - Use `tests/wii_fakes.py` (FakeHid, FakeBus, sensor_report, CAL, BOARD, PATH,
     wait_until).
5. Done when: all tests pass; `python -m poseassess.wii --simulate --seconds 2 --project <p>`
   writes a valid recording; nothing imports hid at import time.

### CORE-BOARD (`poseassess/core/balance/*`, shared test infra, balance tests)

1. `geometry.py`: port the stubs from 3ea6d8a. `register_board`/`solve_board_pnp` gain
   `up_sign`/`up_world` (conjugation with `F = diag(1,-1,-1)` for a Z-down world). Messages
   mention "2. Calibration".
2. `board.compute_board` and `trial.import_recording`.
3. `trc.read_trc_full`: port `pose/external.read_trc`, plus Frame#, plus Units.
4. `analysis.py`:
   - port `interp`, `sway_metrics`, `detect_force_events` (input: folder / wii.csv / dict /
     `(t_rel, total)` pair; output in `t_rel`), `write_events_csv`, `estimate_time_offset`;
   - new: `body_mass_from_force`, `detect_trc_events`, `match_events`, `xcorr_offset`.
5. `alignment.py`: all functions (section 5).
6. `fusion.py`: `cop_to_world`, `plumb_point`, `fuse_trial`, `balance_summary`,
   `export_fused_csv`, `export_grf_mot`, `export_all`.
   - COP world is always recomputed from the current board.
   - Full-rate COP metrics inside the window.
   - `warnings` for: stale board, no board, videos changed, alignment none, trc not covered.
7. `overlay.py`: port the drawing functions, minus pose.
8. Tests:
   - `test_balance_geometry.py`: port `test_geometry_io.py` and the geometry parts of
     `test_core.py`, plus **Z-down** cases.
   - `test_balance_board.py`: compute from `tests/synth` clicks with up ±1 and both Calib
     styles; noisy clicks give < 1 cm error; mirrored clicks are rejected; swapping
     front/back and recomputing equals a 180° rotation; stale after a Calib change;
     `import_recording`.
   - `test_balance_trc_com.py`.
   - `test_balance_analysis.py`: PoseBoard KNOTS/TRUTH; synth jump and stomp in
     `detect_trc_events`.
   - `test_balance_alignment_fusion.py`: `auto_align` recovers `demo_trial["offset_s"]`
     within 1 frame for both `sync_event` and `xcorr`; `fuse_trial` matches
     `synth.fused_from_demo` (COP on the board top, force along up, COM near COP); exports;
     Z-down.
   - `test_balance_overlay.py`.
   - Keep `tests/test_scaffold_contracts.py` passing.
9. Owns `tests/conftest.py`, `tests/synth.py`, `tests/deps.py` and `tests/__init__.py`:
   additive changes only, because other agents import them.

### GUI-WII (`2b. Wii Board` page, WiiController, status widget, picker, shell files)

1. **`WiiController`**:
   - Plug-and-play by default: `start_default()` follows `wii_start_mode()`; `auto` calls
     `start_auto()`.
   - Only one source at a time; replacing it calls `adopt_tares(old)`, `old.stop()`,
     `source_changed`.
   - Thread-safe state forwarding.
   - `connection_event` on connected ↔ lost.
   - Status text: battery, connections/disconnects, last error, hidapi hint.
   - `latest()` never returns a sample older than `STALE_FORCE_S`.
   - Tare lock.
   - `shutdown()`.
2. **`WiiStatusWidget`**: a coloured dot + text. A click may jump to the Wii Board page.
   GUI-WII may add a `MainWindow` hook for that; it is the only allowed shell edit.
3. **`BoardPointPicker`**: the behaviour in the module docstring. `CornerPicker` stays
   unmodified.
4. **`WiiBoardPage`**: two tabs.
   - **"Device && live COP"**:
     - Auto-connect checkbox (checked).
     - Device combo + Scan, Connect, "Use simulator", Disconnect.
     - "Tare (board empty)".
     - COP min. load spin (kg).
     - Monospace status.
     - `CopView` with the 3 s COP trail and total kg (plus body-weight % when known).
   - **"Board location"**:
     - Camera combo.
     - Frame list from `wii/board/camNN/*.jpg`, with "Grab frame from video…" (FrameSelector
       on `videos/camNN.*` or a chosen file, output dir `wii/board/camNN`), "Use calibration
       frame" (copy the first `calibration/extrinsics/camNN/*.jpg`), and "Load image…".
     - `BoardPointPicker` with a next-landmark hint, Undo last, Clear, "Swap front/back
       (180°)", "Save clicks for this camera" (`board.save_clicks` with image name and size).
     - "✓ Compute board position" (`board.compute_board`): result text (method, cameras,
       reprojection px per camera, tilt, notes, warnings) and a QC grid of
       `overlay.board_check_image` per camera (like the Verify tab, 3 columns).
     - Stale banner + "Recompute".
     - Geometry: height of the top surface above the checkerboard (default 53 mm) and board
       dimensions (advanced).
     - **Corner check**, enabled when a board is registered and connected:
       1. The prompt says: "Press firmly on the FRONT-LEFT corner (TL)…".
       2. Baseline = mean of the last 0.5 s.
       3. Poll every 100 ms with the mean of the last 0.3 s and
          `protocol.pressed_sensor(baseline, now, 3.0)`.
       4. Timeout after 15 s.
       5. Result TL → "OK". BR → `swap_front_back_clicks` + `compute_board` + message
          ("front and back were swapped; fixed"). TR/BL → "Redo the clicks".
       6. Record the result with `set_corner_check`.
   - Emit `balance_changed("board")` after compute, swap or check.
   - The board tab is disabled without a project. The device tab always works.
5. Tests:
   - `tests/test_gui_wii_controller.py`: FakeBus board appears and drops, reconnect, tare
     kept per board, signals arrive on the GUI thread, hidapi missing, simulator,
     `POSEASSESS_WII` modes, shutdown.
   - `tests/test_gui_wii_board_page.py`: on `demo_trial` images and clicks; corner check
     with a manual ForceSource (PoseBoard `test_gui_fixes.py` `ManualBoard` pattern); the
     page works while `compute_board` is still a stub (monkeypatch it, or use
     `tests.deps.call_or_skip`).
   - Keep `tests/test_scaffold_smoke.py` passing.
   - Screenshots of both tabs, with the simulator and `demo_trial`.

### GUI-CAPTURE (`3b. Capture` page + `core/capture/*`)

1. `session.py`, `export.py` (section 7), and `camera_grid.py`.
2. **`CapturePage`**:
   - Camera table (camNN, source: device index spin / file, resolution, fps, enabled,
     measured fps / status), "Scan cameras" (worker), Open / Close cameras.
   - Preview grid with a "● REC" overlay.
   - Wii box: status + small `CopView` from `state.wii`, Tare, and a hint when no board is
     connected. Recording without a board is allowed after a confirmation.
   - Subject, notes, "Record cameras" checkbox, "● Record" / "■ Stop", elapsed time, "Mark
     event" + label, recent-events list.
   - Takes list with Export / Use / Open folder.
   - Output fps, "Keep raw camera files".
   - "📸 Board snapshots".
   - `can_close()` / `shutdown()`.
   - Help text explaining both workflows: record here; or record Wii only and import the
     videos on "3. Videos" + do 2 jumps for syncing.
3. Tests:
   - `tests/test_capture_camera.py`: port the camera tests of `test_runtime.py` and
     `test_session.py`.
   - `tests/test_capture_session_export.py`: file sources (`synth.write_test_video`) +
     `SimulatedBoard`; a synthetic take with jittered, offset and dropped timestamps gives
     equal frame counts and correct nearest-frame choice; `frames.csv` is consistent;
     other-extension files are removed; Config fps is updated; trial.json is `recorded`;
     cancel leaves `videos/` untouched.
   - `tests/test_gui_capture_page.py`: record 1–2 s with file cameras and the controller
     simulator (call `state.wii.start_simulator()`; if the GUI-WII controller is still a
     stub, pass a `SimulatedBoard` via monkeypatching `state.wii.source`).
   - Screenshots.

### GUI-VIEW (3D View overlays, Results Balance tab, alignment UI)

1. **`BalanceSkeleton3DViewer`**:
   - Board: closed outline through corners TL-TR-BR-BL, with the front edge TL-TR thicker or
     coloured; sensor dots; optionally a translucent surface.
   - COP: point + trail (last about 2 s of rows).
   - Force: arrow (line + cone or thick line + head), length `total_kg ·
     COP_ARROW_M_PER_KG`, direction `up_world`.
   - COM: sphere/point + plumb line to `fusion.plumb_point`.
   - Toggles "Board", "COP", "Force", "COM" next to "Cameras".
   - Per frame: `row = fused.index_for_time(self._trc.times[i])`; −1 or NaN hides the item.
   - Re-transform on `load_trc`. `clear()` hides everything.
   - Colours must be visible on the LIGHT GL background (the existing quirk): e.g. board
     dark grey `#333`, COP red `#e0303a`, force green `#1a9a3a`, COM magenta/violet
     `#8a2be2`.
   - With no data, the viewer is identical to `Skeleton3DViewer` (toggles hidden).
   - `skeleton_3d.py` may only get **additive** hooks (e.g. `self.ctl = ctl`), with no
     behaviour change.
2. **`Viz3DPage`**:
   - Use `BalanceSkeleton3DViewer`.
   - After `set_cameras` in `_load_selected`, call `fusion.fuse_trial(project,
     trc_path=path)` + `board.load_board(project)` + `viewer.set_balance(...)`. Catch
     `ValueError` / `NotImplementedError`, draw the board only, and put the reason in the
     status text.
   - A compact status line: "Wii: <recording> · <alignment_status> · <warnings>".
   - Reload on `balance_changed`.
   - `_browse` (a .trc from elsewhere): overlays only when the file is inside this project's
     `pose-3d/`.
3. **`ResultsPage`**:
   - Wrap the existing `split` in a `QTabWidget`. Tab "Joint angles" holds the unchanged
     widgets; tab "Balance (Wii)" holds a `BalancePanel`.
   - The top bar and note stay as they are.
4. **`BalancePanel`**:
   - `AlignmentPanel`:
     - recording combo;
     - "Import Wii recording…" (folder or wii.csv);
     - status (`alignment_status`);
     - "Detect sync event" (worker, `auto_align`), with the result and plot (force vs
       foot/COM height with matched events);
     - manual offset spin (s, ±600, 0.001 step) + "Save alignment";
     - when the method is `recorded`: read-only info, with "Re-align manually…" as an
       explicit override.
   - Trc combo (default `find_trc_files()[0]`).
   - Time window (from/to spin boxes, default the whole trial; "Use events" optional).
   - Metrics table from `balance_summary`:
     - COP: mean/range/RMS ML and AP, path length, mean velocity, 95 % ellipse area;
     - COM: the same;
     - COM–COP: RMS distance, RMS ML/AP, correlation;
     - load: mean kg, body mass.
   - `BalancePlotCanvas`: panels for total force (kg, with a body-weight line), COP x/y +
     COM x/y (board frame, mm), events as vertical lines, and a shaded window.
   - "Export fused table…" (`export_all` to a folder, default `wii/exports/`).
   - `refresh()` on project, trial or alignment change.
5. Tests:
   - `tests/test_gui_balance_view.py`: on `demo_trial` with `synth.fused_from_demo`
     (monkeypatch `fusion.fuse_trial` so the tests do not depend on CORE-BOARD progress),
     check the overlay item positions (`viewer._tf(zup_to_yup(board.corners_world))`), the
     per-frame update, toggles, and clear. Rendering screenshot tests are
     `@pytest.mark.gl`.
   - `tests/test_gui_results_balance.py`: the joint-angles tab is unchanged; the balance tab
     fills with fused_from_demo; the manual offset is saved to trial.json; the export writes
     files (via `call_or_skip` when core is a stub).
   - Screenshots under xvfb (3D View with overlays, Results Balance tab).

---

## 11. Pre-existing issues: report them, do not fix them silently (R1)

- `core/trc_io.read_trc` drops empty fields. A .trc with NaN markers (Pose2Sim writes NaN as
  empty) shifts columns or fails, and Frame# is dropped. The balance code uses
  `balance.trc.read_trc_full`. Mention the issue in reports; do not change `trc_io`.
- Videos page status and the plugin backends know only `.mp4` (plus avi/mov for the
  plugins). The capture therefore writes `.mp4`.
- `project_page._save` resets Flip Z (invert_z). After any recalibration, the board shows as
  stale. That is intended.
- `synchronization_gui` in Config.toml is true, so the sync stage may pop up a UI. Keep sync
  off for captured takes.
- The 3D View grid is drawn at hip height, not at the floor. The board appears near the
  feet, below the grid.
- The `Calib_easymocap.toml` produced by the Pose2Sim "calibration" stage has wrong sizes. It
  sorts after `Calib.toml`, and `find_calib_toml` takes the first one.

---

## 12. Test plan and commands

- Unit and GUI tests (headless, from the repo root):
  `cd /home/user/poseassess && /home/user/venv-pa/bin/python -m pytest -q tests`
- `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`, `POSEASSESS_WII=0` (no auto-connect
  thread in tests unless a test asks for one) and `MPLBACKEND=Agg`, and provides the
  fixtures `qapp`, `project` (an empty 3-camera, 30 fps project) and `demo_trial`
  (`synth.make_demo_trial`).
- **`tests/synth.py`**, ground-truth generators:
  - `look_at_camera`, `ring_cameras(up_sign=±1)`, `write_calib_toml(style="poseassess"|"opencap")`;
  - `project_points`, `board_landmarks`, `board_pose`, `board_clicks(noise_px)`,
    `render_board_image`;
  - `kg_from_cop`, `standing_trial` (sway + jump + stomp, TRC and Wii on one time base);
  - `write_trc` (Pose2Sim format, NaN → empty);
  - `write_wii_recording`, `write_test_video`;
  - `make_demo_trial(root, n_cams=3, up_sign=1, fps=30, …, offset_s=2.5)`;
  - `fused_from_demo(demo)`.
- Other helpers:
  - `tests/qtutil.py`: `pump`, `pump_until`.
  - `tests/deps.py`: `call_or_skip(fn, …)` skips while a dependency is a stub. The final
    merged run must have **no** such skips.
  - `tests/wii_fakes.py`: the fake hidapi bus.
- OpenGL tests: mark them `@pytest.mark.gl` and run with
  `POSEASSESS_TEST_GL=1 QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1600x1000x24" /home/user/venv-pa/bin/python -m pytest -q -m gl tests`
- Lint (new or changed files only; the existing code is not ruff-clean and must not be
  reformatted): `/home/user/venv-pa/bin/ruff check --select F,E9,I <your files>`
- Screenshots: `QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1600x1000x24" /home/user/venv-pa/bin/python /tmp/claude-0/-home-user-PoseBoard/80080f02-f6a3-5192-9efd-bb7471921107/scratchpad/pa_screens_wii.py <OUT_DIR> [--sim] [--size=1000x680]`.
  It builds `OUT_DIR/pa_demo` with `make_demo_trial` and saves `page_<n>_<title>.png` for
  every page.
  - Each agent copies it to `scratchpad/<AGENT>/` to add its own states (tabs, clicks,
    recording) and writes output there.
  - Check at 1400×860 **and** with `--size=1000x680` (the window clamps to the effective
    minimum, about 1380–1420 px wide today; your page must not make it wider).
- R1 regression check (the orchestrator runs it after merging):
  1. Screenshot the original commit with the original `pa_screens.py`:
     `git archive HEAD | tar -x -C <dir>`, pointing `sys.path` there.
  2. Screenshot the merged tree.
  3. Compare the content area (x > 195, without the status bar) of pages 1, 2, 3, 4, 5, 6, 7
     (old numbering) against the new rows 0, 1, 3, 5, 6, 7, 8. Only project-path text may
     differ.
  4. On the 3D View and Results pages the comparison applies to a project **without** `wii/`
     data. The Results page gains a tab bar, which is accepted.

  The scaffold passed this check: identical except the path text.

---

## 13. File ownership (every new or changed file has exactly ONE owner)

Rules:
- Edit only your own files.
- Files marked *frozen* are complete. Change them only to fix a bug, and report it.
- Anything not listed here is **read-only for everyone**, including all existing PoseAssess
  files not listed and `docs/WII_INTEGRATION.md`.
- If you need a change in someone else's file, work around it locally (wrap or monkeypatch in
  tests) and describe the needed change in your final report.

| File | Owner | State after scaffold |
|---|---|---|
| `poseassess/wii/__init__.py`, `__main__.py` | CORE-WII | impl |
| `poseassess/wii/protocol.py` | CORE-WII | impl (verbatim port) |
| `poseassess/wii/device.py` | CORE-WII | impl (port + `hidapi_status`) |
| `poseassess/wii/io.py` | CORE-WII | impl, *frozen format* (additive only) |
| `poseassess/wii/recorder.py` | CORE-WII | impl (port); test and harden |
| `poseassess/wii/record_cli.py` | CORE-WII | STUB |
| `run_wii.bat` (new), `README_DEPLOY.txt` (append section) | CORE-WII | to do |
| `tests/wii_fakes.py`, `tests/test_wii_protocol_device.py`, `tests/test_wii_recorder.py`, `tests/test_wii_cli.py` | CORE-WII | fakes impl; tests empty |
| `poseassess/core/balance/__init__.py`, `paths.py` | CORE-BOARD | impl, *frozen* (additive only) |
| `poseassess/core/balance/trial.py` | CORE-BOARD | impl + STUB `import_recording` |
| `poseassess/core/balance/calib.py` | CORE-BOARD | impl |
| `poseassess/core/balance/geometry.py` | CORE-BOARD | data impl, algorithms STUB |
| `poseassess/core/balance/board.py` | CORE-BOARD | I/O impl, STUB `compute_board` |
| `poseassess/core/balance/trc.py` | CORE-BOARD | helpers impl, STUB `read_trc_full` |
| `poseassess/core/balance/com.py` | CORE-BOARD | impl (verbatim port) |
| `poseassess/core/balance/analysis.py`, `alignment.py`, `overlay.py` | CORE-BOARD | STUB |
| `poseassess/core/balance/fusion.py` | CORE-BOARD | `FusedTrial` impl, functions STUB |
| `tests/__init__.py`, `tests/conftest.py`, `tests/synth.py`, `tests/deps.py` | CORE-BOARD | impl, additive changes only |
| `tests/test_scaffold_contracts.py`, `tests/test_balance_geometry.py`, `tests/test_balance_board.py`, `tests/test_balance_trc_com.py`, `tests/test_balance_analysis.py`, `tests/test_balance_alignment_fusion.py`, `tests/test_balance_overlay.py` | CORE-BOARD | contracts impl; others empty |
| `poseassess/gui/state.py`, `poseassess/gui/main_window.py`, `poseassess/gui/pages/base_page.py`, `poseassess/gui/pages/__init__.py` | GUI-WII | impl, *frozen* (shared shell) |
| `poseassess/gui/wii_controller.py` | GUI-WII | minimal inert stub |
| `poseassess/gui/widgets/wii_status.py`, `cop_view.py`, `board_picker.py` | GUI-WII | minimal / port / STUB |
| `poseassess/gui/pages/wii_board_page.py` | GUI-WII | placeholder |
| `tests/qtutil.py`, `tests/test_scaffold_smoke.py`, `tests/test_gui_wii_controller.py`, `tests/test_gui_wii_board_page.py` | GUI-WII | helpers and smoke impl; others empty |
| `poseassess/core/capture/__init__.py`, `camera.py`, `settings.py` | GUI-CAPTURE | impl (camera = verbatim port) |
| `poseassess/core/capture/session.py`, `export.py` | GUI-CAPTURE | STUB |
| `poseassess/gui/pages/capture_page.py`, `poseassess/gui/widgets/camera_grid.py` | GUI-CAPTURE | placeholder / STUB |
| `tests/test_capture_camera.py`, `tests/test_capture_session_export.py`, `tests/test_gui_capture_page.py` | GUI-CAPTURE | empty |
| `poseassess/gui/widgets/balance_3d.py`, `balance_panel.py`, `wii_alignment.py`, `balance_plots.py` | GUI-VIEW | STUB / placeholder |
| `poseassess/gui/widgets/skeleton_3d.py` (existing) | GUI-VIEW | unchanged; additive hooks only |
| `poseassess/gui/pages/viz3d_page.py`, `poseassess/gui/pages/results_page.py` (existing) | GUI-VIEW | unchanged; to extend |
| `tests/test_gui_balance_view.py`, `tests/test_gui_results_balance.py` | GUI-VIEW | empty |
| `docs/WII_INTEGRATION.md` | orchestrator (read-only for agents) | this document |

Dependencies during parallel work, and what is ready now:

| Agent | Depends on | Ready now | If still a stub |
|---|---|---|---|
| CORE-WII | – | everything it needs | – |
| CORE-BOARD | `wii.io`, `com`, `calib`, `paths`, `trial` | all ready | – |
| GUI-WII | `device` (ready), `board` I/O (ready), `compute_board` / `overlay` (CORE-BOARD) | – | monkeypatch in tests; catch `NotImplementedError` in the UI |
| GUI-CAPTURE | `camera`, `settings`, `WiiRecorder`, `trial.set_recording` (all ready), `WiiController` (GUI-WII) | – | use `state.wii.source` or a `SimulatedBoard` directly in tests; `load_board` is ready |
| GUI-VIEW | `fusion` / `alignment` / `analysis` functions (CORE-BOARD) | – | develop and test with `synth.fused_from_demo` and monkeypatching; catch `NotImplementedError` in the UI |

---

## 14. Final report of each agent (returned to the orchestrator)

Include:
- the files changed;
- API additions or deviations (with reasons);
- test results (pytest summary line + GL tests if any);
- screenshot paths;
- known limitations;
- any change you need in a file you do not own;
- pre-existing bugs you noticed (section 11 style).

Keep the UI English and the existing pages' behaviour intact.
