# Pixel-matched MuJoCo scene for the bimanual YAM rig

A MuJoCo scene whose top camera reproduces the real D405's intrinsics and pose,
so a render and a real frame are directly comparable. Built for SPD-style
sim pre-training ([spd.bot](https://spd.bot/), arXiv 2608.15917), where the sim
table has to look like the real table and the background is handled by
augmentation.

## Measured, not assumed

| Quantity | Value | Source |
|---|---|---|
| Top cam intrinsics | fx 391.533, fy 390.515, ppx 325.038, ppy 235.265 | D405 calibration EEPROM |
| Top cam FOV @640x480 | HFOV 78.518°, VFOV 63.148° | derived from the above |
| Distortion | inverse_brown_conrady, k1 −0.0542, k2 0.0625, k3 −0.0212 | D405 EEPROM |
| Tabletop | 1.60 × 1.60 m | measured |
| Top cam height | **0.9256 m** above the tabletop | fitted (0.90 m measured) |
| Top cam position | x +0.016, y −0.561 (0.239 m in from the near edge) | fitted |
| Top cam rotation | pitch 34.318°, yaw 1.124°, roll −3.277° (55.68° below horizontal) | fitted |
| Tabletop colour, as shot | RGB (112, 179, 180) — cyan, via the camera's AWB | measured off `greenscreen/top.png` |

The 640x480 capture mode is **not** the D405's full field of view: its 16:9
modes give HFOV 89.06°, the 4:3 mode crops horizontally to 78.52° and gains
vertically. Anything quoting the datasheet's 87°x58° for this stream is wrong.

Fit quality: **0.409 px RMS** over 888 edge points on three independent
features (far edge, left edge, board seam). Freeing the camera height put it at
0.9256 m against a 0.90 m tape measurement — a 2.5 cm agreement, about what
measuring to a lens's optical centre is worth. The right edge was held out of
the fit and lands at +5.5 px, limited by its own segmentation (it runs through
shadow behind the shelving), not by the pose.

## Layout

    sim/build_scene.py            generates spd_scene.xml
    sim/spd_scene.xml             the scene (generated -- edit the builder)
    sim/camera_intrinsics.json    frozen factory intrinsics, all three cameras
    sim/top_camera_pose.json      fitted top-camera pose
    sim/top_distortion_map.npz    cached ideal->distorted remap

    tools/sim_dump_intrinsics.py  read intrinsics off the cameras (rig powered)
    tools/sim_verify_intrinsics.py prove MuJoCo reproduces them (0.09 px)
    tools/sim_fit_top_camera.py   fit the pose to greenscreen/top.png
    tools/sim_calibrate_lighting.py match render brightness to the real frame
    tools/sim_compare_top.py      render vs real: overlay, blend, diff, IoU
    tools/sim_viser.py            browser viewer, works over SSH

## Rendering needs a GL context

EGL and OSMesa are both unavailable here, so use an X display:

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_compare_top.py

## Regenerating

    python sim/build_scene.py --table-color asshot --arm-style dark
    python sim/build_scene.py --table-color neutral --arm-style translucent

`--table-color asshot` uses the measured cyan (pixel match against real
frames); `neutral` uses the grey-world-corrected albedo, for when the colour
cast is handled downstream. `--arm-style translucent` follows the paper.

## Validated

* **Camera intrinsics**, all three cameras — read from each D405's calibration
  EEPROM and reproduced in MuJoCo to **0.09 px mean / 0.18 px max**
  (`tools/sim_verify_intrinsics.py`).
* **Top camera pose** — **0.409 px RMS** over 888 edge points on three
  independent features. Freeing the height put it at 0.9256 m against a 0.90 m
  tape measurement.
* **Table** — tabletop mask IoU **0.941**, area within **+0.71%**, median
  outline agreement **1.00 px**.
* **Joint mapping** — sim `qpos` *is* real `joint_pos`; i2rt drives the real
  robot's kinematics from the same `yam.xml`. Confirmed visually: rendering
  `calib/pose_04`'s recorded joint angles reproduces the real arm posture
  (`sim/compare_arms/pose04_side_by_side.png`).
* **Arm base yaw = +90 deg.** Was -90 deg, i.e. ~180 deg wrong: at the recorded
  joint angles that put the grippers at y ~ -1.1, reaching away from the table
  and behind the camera, so zero arm pixels rendered where reality has ~10k.

## Known gaps

* **Arm base position is not fitted.** Yaw is settled but the bases are still at
  their estimated 0.50 m separation on the near edge. Fitting it from imagery
  stalled on segmentation: a YAM arm is white links plus black housings, on a
  white table, and neither the bright nor the dark term separates it cleanly
  from bright tabletop, shadows and cables. **A tape measure settles this
  faster** — base separation, and distance from the table's near edge.
* **Wrist camera extrinsics are unfitted**, and blocked on the above: solving a
  mount transform needs a correct gripper world pose. Their *intrinsics* are
  exact. Attempts and why they failed are documented in
  `tools/sim_fit_wrist_cameras.py` — worth reading before retrying, especially
  that the table silhouette is contaminated by the gripper jaws and clutter,
  and that the board seam loses to any thin dark cable.
* **The "asshot" cyan table colour is not reproducible.** Same tabletop, same
  camera: the greenscreen still has B-R = +64, the later `calib/` recording
  B-R = -7. That cyan was one capture's auto-white-balance, not a property of
  the table, so prefer `--table-color neutral` unless matching those specific
  frames. `LIGHT_SCALE` was calibrated against the cyan frame.
* **Table height off the floor is assumed 0.75 m** — affects only the strip of
  floor past the far edge.
* Background, floor and the ceiling-light specular blooms differ by design —
  SPD swaps background and table textures as augmentation anyway.

## Teleop and recording in sim

`examples/record_sim_cubes.py` drives this scene from the Quest and records
LeRobot episodes, with the same controls and the same dataset schema as
`examples/record_bi_yam.py`.

**This is not `record_bi_yam.py --sim`.** That flag uses i2rt's `SimRobot`,
which loads a single-arm XML *per arm* -- two arms in two separate worlds, with
no table and nothing to pick up. Stacking needs both arms and the objects in one
model.

Three terminals:

    VR_TELEOP_SIM_FRAMES=1 vr-teleop-relay          # 1
    DISPLAY=:0 MUJOCO_GL=glfw .venv/bin/python examples/record_sim_cubes.py \
        --repo-id Dimios45/sim_cube_stacking_spd --task "stack the cubes" \
        --root data/sim_cube_stacking_spd --num-episodes 25            # 2
    adb reverse tcp:8443 tcp:8443                   # 3, USB only
    # Quest browser -> http://localhost:8443/ , enable the camera stream

Add `--resume` to append to a dataset that already exists; `--num-episodes`
then counts *new* episodes rather than the total. Plain `LeRobotDataset(...)`
opens read-only and `add_frame` raises, so this goes through
`LeRobotDataset.resume()`, which attaches a writer. `--overwrite` still
*deletes* and starts over, and the two flags are mutually exclusive.

Rendered frames reach the headset through `relay/sim_frames.py`, a
shared-memory bus. `SimFrameReader` deliberately mirrors `CameraReader`'s
interface (`.spec` + `.latest()`), which is all `relay.server.CameraTrack`
uses, so sim frames travel the same WebRTC path as real ones. The relay change
is opt-in: without `VR_TELEOP_SIM_FRAMES` it opens real cameras exactly as
before. A 921 KB frame is far too big to push through a queue every tick, and
physics cannot live in the relay's event loop, hence shared memory.

**Runs at 20 fps, not 30.** Each offscreen `render()` costs ~11 ms of GL
readback on this machine, and that is per-call overhead rather than pixels --
640x480 costs 32.8 ms for three cameras and 160x120 still costs 28.3 ms, while
`update_scene` is 0.01 ms. Three cameras therefore do not fit a 33 ms tick but
sit comfortably in 50 ms. Physics still runs at the scene's 480 Hz (24 substeps
per tick, ~1 ms).

Each episode start re-spawns the three cubes uniformly in an 8 cm square
(`--spawn-region`, centred 44 cm out so the near edge sits at 40 cm) and
jitters their tints, in the spirit of the paper's reset randomization. Colour
is assigned *after* the draw, by sorted x, so black is always leftmost and the
two reds take middle and right whatever the spawn. `--seed` is drawn from the
OS per session and logged, so a session is reproducible but two recordings
never share a layout sequence.

## Contact physics

`tools/sim_check_grasp.py` verifies the scene can actually do the task, and both
checks pass:

* **grasp + lift** -- jaws close on a cube, it stays within 12 mm of the grasp
  site while the arm lifts it 188 mm.
* **stack** -- one cube resting on another drifts 0.02 mm and sinks 0.07 mm over
  3 s.

Two settings were needed, and both matter:

* **`impratio=10`.** At MuJoCo's default of 1, frictional contacts are soft
  enough that a stacked cube creeps ~13 mm sideways in 3 seconds. This is the
  single change that made stacking stable.
* **`gravcomp="1"` on every arm body.** i2rt's `yam.xml` caps each joint
  actuator at +-10 Nm, which cannot hold the arm against gravity: a
  position-controlled arm sags ~0.6 rad at joint 2 and cannot lift at all. The
  real robot does not fight gravity either -- `get_yam_robot` runs
  `zero_gravity_mode=True` and feeds gravity torques forward -- so cancelling it
  in sim matches the hardware rather than hiding the force limit.

Gripper convention, measured rather than assumed: **joint7/joint8 at 0 is
closed, 0.0475 is open**, with a 92-104 mm gap across the grasping region. Do
not infer this from the distance between the two tip *bodies* -- they travel
along their own axes and their origin separation runs the opposite way (102 mm
at "open" 0, 48 mm at "closed"), which reads as though the gripper closes when
it opens. The teleop's `gripper.pos` is the other way round again (0 = open,
1 = closed).

## Bugs worth not re-introducing

* `<compiler angle="radian">` is mandatory (i2rt joint ranges are radians) and
  applies to `euler` too — degrees there get silently wrapped.
* MuJoCo's `principal` x and y take **opposite** signs; one wrong flips the
  render by 2x the offset (~10 px here).
* i2rt's `yam.xml` link geoms are **unnamed**. Prefixing only touches geoms that
  already have a name, so segmentation masks, arm hiding and per-arm styling all
  matched nothing and returned an empty arm mask rather than an error.
  `build_scene.py` now names every arm geom.
* LeRobot writes AV1 video; this OpenCV build has no software AV1 path and
  returns no frames, silently. Decode with PyAV/libdav1d.
* `scipy.optimize.least_squares` needs a **fixed-length** residual. Dropping
  entries for points behind the camera changes the length and raises "could not
  broadcast input array" for every seed.
