# vr-teleop-kit

A kit for teleoperating robot manipulators with a WebXR headset
(Meta Quest). A web page served from your workstation runs in the Quest
browser, streams 6-DoF controller poses over a WebSocket, and a Python
teleoperator turns them into joint commands via differential inverse
kinematics (forward kinematics and Jacobians read from a MuJoCo model
of the arm). Plugs into [LeRobot](https://github.com/huggingface/lerobot)
as a drop-in `Teleoperator`.

> **Full write-up:** [**VR Teleoperation Stack for Robot Manipulation**](https://aurelarnold.xyz/blog/vr-teleoperation-stack/) walks through the whole stack: the inverse kinematics, the safety features, and the camera and haptic feedback that close the loop.

The robot currently supported end-to-end is a bimanual pair of
[I2RT YAM](https://i2rt.com) arms (with the linear_4310 gripper),
driven through [i2rt](https://github.com/i2rt-robotics/i2rt). The pose
mapping and the relay are robot-agnostic; supporting another arm means
porting the IK layer (see "Adapting to a different arm" below).

Highlights:

- **Clutch-relative mapping**: hold the grip button and the robot follows
  your hand's relative motion; release, reposition, grab again.
- **Controller buttons**: grip = clutch; a precision modifier (hold to
  lower the gains for fine work); and a button that sends the arm back to
  its home pose.
- **Reach limit**: the target pose can never run more than a fixed
  distance or angle ahead of the robot. Pressing past a joint limit or the
  workspace boundary feels like a wall (with haptic feedback), and
  reversing bites immediately, like a mouse cursor at the edge of the
  screen.
- **Decoupled IK**: joints 1-3 track position, the
  wrist tracks orientation, both as damped-least-squares steps with
  manipulability-adaptive damping (graceful near singularities and
  gimbal lock).
- **Wrist-pivot calibration**: a 5-second in-VR ritual aligns the
  controller read-out point with your anatomical wrist pivot, so pure
  wrist twists don't drag the arm around.
- **Haptics**: gripper grasp force and IK trouble (joint limits, reach,
  gimbal proximity) are mapped to controller vibration.
- **Camera streaming**: optional WebRTC streams of robot cameras,
  rendered as world-locked panels inside VR.
- **Two ways to connect**: USB cable (`adb reverse`, ~1 ms latency,
  recommended) or Wi-Fi (LAN HTTPS with a self-signed certificate). A
  public tunnel works for pose streaming too, but the WebRTC camera media
  is not configured off-LAN (it needs a TURN server), so full remote
  operation isn't set up.
- Live-tunable settings (gains, smoothing, velocity caps, haptic
  thresholds) from the web page, applied on the next solver tick.

## Architecture

Three layers, separated by how robot-specific they are:

```
src/vr_teleop_kit/
├── core/      robot-agnostic: clutch-relative pose mapping + reach limit
├── relay/     robot-agnostic: FastAPI WebSocket relay, WebRTC cameras,
│              and the WebXR page served to the Quest
├── ik/        YAM-tuned: decoupled IK (MuJoCo model built from the i2rt
│              yam.xml + gripper MJCF, wrist-anchor geometry, mount frames)
└── lerobot/   thin adapter: LeRobot Teleoperator classes wiring core+ik
               into the LeRobot action interface
```

```mermaid
flowchart LR
    OP(["Operator"])

    subgraph HS["VR headset"]
        WEB["Web app"]
    end

    subgraph WS["Workstation"]
        RELAY["Relay"]
        IK["Inverse kinematics"]
        RELAY -->|controller pose| IK
    end

    subgraph RB["Robot"]
        ARMS["Robot arms"]
    end

    OP -->|controller input| WEB
    WEB -->|"controller pose (cable / Wi-Fi)"| RELAY
    IK -->|joint angles| ARMS

    ARMS -.->|haptics| RELAY
    RELAY -.-> WEB
    WEB -.-> OP

    linkStyle 4,5,6 stroke:#9a9a9a,stroke-dasharray:4 3
```

For pose and state messages the relay is a pure broadcast hub: any number
of clients (teleop processes, MuJoCo viewers) can subscribe to the same
stream. (It also publishes the optional WebRTC camera tracks.)

## Install

```bash
git clone https://github.com/Dream-Machines-Robotics/vr-teleop-kit
cd vr-teleop-kit
pip install -e ".[relay,lerobot]"           # or: uv pip install -e ".[relay,lerobot]"

# YAM model files (and, for real hardware, the driver):
git clone https://github.com/i2rt-robotics/i2rt
pip install -e ./i2rt        # only needed to drive real hardware

# The IK finds the model files in the ./i2rt clone automatically. If the
# clone lives elsewhere, point at the arm MJCF instead (or pass
# model_path in the teleop config):
export YAM_XML=/path/to/i2rt/i2rt/robot_models/arm/yam/yam.xml
```

The `relay` extra covers the server (FastAPI, aiortc, OpenCV); the
`lerobot` extra covers the Teleoperator adapter. The bare package (pose
mapping + IK) only needs numpy, mujoco and websockets.

## Run the relay and connect the headset

Both transports serve the same page on port 8443; they differ only in
how the Quest reaches it. WebXR requires a secure context, which is why
the two paths exist.

**USB (recommended)**: plain HTTP on localhost (a secure context per the
WebXR spec), forwarded over the cable:

```bash
vr-teleop-relay                               # binds 127.0.0.1:8443
adb reverse tcp:8443 tcp:8443                 # forward Quest's localhost over USB
# Quest browser → http://localhost:8443/
```

One-time Quest setup: enable Developer Mode (Meta Quest mobile app →
Devices → your headset → Developer Mode), plug in the cable, accept the
"Allow USB debugging" prompt in the headset. `adb reverse` must be re-run
after re-plugging the cable.

**LAN**: HTTPS with a self-signed certificate; the Quest shows a
"not secure" warning the first time, which you accept:

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -nodes -days 825 \
    -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=$(hostname)" \
    -addext "subjectAltName=DNS:$(hostname),DNS:localhost,IP:127.0.0.1,IP:<your-lan-ip>"
vr-teleop-relay --host 0.0.0.0 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
# Quest browser → https://<your-lan-ip>:8443/
```

`certs/` is gitignored; never commit key material. USB has ~1 ms RTT and
no jitter; LAN works but Wi-Fi adds occasional >100 ms spikes. A public
tunnel (e.g. `cloudflared tunnel --url http://localhost:8443`) also works
for pose streaming, but WebRTC camera media is peer-to-peer and needs a
TURN server off-LAN (not configured here).

On the page: **Calibrate wrist** once per operator (squeeze both grips
in VR, then for 5 s twist your hands while keeping each wrist roughly in
place: the hand rotates, the wrist pivot stays still), then
**Start Teleop**. Settings (gains, smoothing, velocity caps, haptics)
are on the same page and apply live.

## Cameras

The relay auto-discovers RealSense color streams at startup — no
`source cams.env` needed. Discovery globs `/dev/v4l/by-id` for the color
node (index 4 on the D405) and matches each camera's serial to a role
(`top` / `left` / `right`) via the map in
`src/vr_teleop_kit/relay/cameras.py`. Update that map when you swap a
camera, or point `CAM_MAP` at a JSON file (`{"<serial>": "<role>"}`) to
override without editing code.

```bash
python tools/gen_cams_env.py          # print what's detected (no writes)
python tools/gen_cams_env.py --write  # (re)generate ./cams.env to source elsewhere
```

`cams.env` is now a generated, gitignored artifact — handy for a shell
that needs the `CAM_*` vars (e.g. the recorder). An explicit `CAM_TOP` /
`CAM_LEFT` / `CAM_RIGHT` still overrides discovery for that role;
`CAM_*_ROTATE` (0/90/180/270) and `CAM_COLOR_INDEX` tune capture.

## Try it without a robot

```bash
vr-teleop-relay                       # terminal 1
python tools/viewer_client.py         # terminal 2: MuJoCo viewer (YAM model)
python examples/pure_sim.py           # terminal 3: IK loop, no hardware
# Quest browser → Start Teleop → squeeze a grip
```

`tools/smoke_test.py` drives the full pipeline with a fake Quest client
and asserts on the resulting actions (no headset needed).

(`pure_sim.py` and `smoke_test.py` use the LeRobot Teleoperator adapter,
so they need the `[lerobot]` extra — installed by the Install command
above, not just the bare package.)

## Use as a LeRobot Teleoperator

The adapter emits a bimanual joint action dict
(`{left,right}_joint_{1..6}.pos`, `{left,right}_gripper.pos`; gripper
0 = open, 1 = closed), so it pairs with any follower using that schema.
Nothing is copied into LeRobot's tree; you instantiate and hand it to
your loop:

```python
import time

from vr_teleop_kit.lerobot import BiQuestTeleoperator, BiQuestTeleoperatorConfig

teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
    id="vr-teleop",
    ws_url="ws://127.0.0.1:8443/ws",
    model_path="/path/to/i2rt/.../arm/yam/yam.xml",  # or set YAM_XML / rely on ./i2rt
))

teleop.connect(); follower.connect()
while True:
    follower.send_action(teleop.get_action())
    time.sleep(1 / 200)
```

`examples/teleop_bi_yam.py` is the complete hardware version of this
loop: it drives two i2rt YAMs (one CAN channel per arm) directly via
`get_yam_robot`, with the rest ramp, gripper-convention conversion
(i2rt's normalized gripper is 0 = closed, 1 = open — the inverse of the
teleop's), haptic feedback, and timing. For LeRobot CLIs, import
`vr_teleop_kit.lerobot` so the `@register_subclass` decorators run, then
use `--teleop.type=bi_quest_teleop` (bimanual) or
`single_arm_quest_teleop` (one arm, unprefixed action keys).

For human-in-the-loop data collection: the teleop exposes intervention
hooks (`is_engaged`, `is_handoff_pressed`, `is_pause_pressed`,
`is_reverse_pressed`, `seed_qpos_from_obs`, `publish_state`) that an
orchestrator can poll to hand control between a policy and the operator.
We use these for an HG-DAgger workflow built on top of this stack in our
LeRobot fork; that workflow is not part of this repository.

## Enabling grasp-force haptics

The grasp-force haptic (the controller buzzing as the gripper closes on an
object) reads the gripper torque through an optional follower method,
`get_joint_torques()`, returning
`{"{left_,right_}gripper.torque": Nm, "{left_,right_}gripper.pos": 0..1}`.
This step is optional: the teleop feature-detects the method via `getattr`
and degrades gracefully without it, disabling only the grasp-force
vibration (the IK-trouble haptics still fire).

On YAM this works out of the box: i2rt reports the gripper motor effort
in `get_observations()` (`gripper_eff`), and `examples/teleop_bi_yam.py`
forwards it (plus the gripper position for velocity masking) to
`teleop.send_feedback({"torques": ...})` every tick. If you write your
own follower, implement `get_joint_torques()` with the contract above —
any driver that does gets grasp-force haptics with no other changes.

## Configuration that is YAM-specific

- **Model path**: the IK builds its MuJoCo model from
  `robot_models/arm/yam/yam.xml` + the linear_4310 gripper MJCF inside an
  [i2rt](https://github.com/i2rt-robotics/i2rt) clone (not vendored
  here). Resolution order: `model_path` in the config, the `YAM_XML` env
  var, then an `./i2rt` clone in the working directory / repo root.
- **`r_calib`** (config): the fixed rotation from the Quest's
  `local-floor` world frame into the arm base frame. The default assumes
  the operator faces the robot's front; if your mounting differs,
  re-derive it by mapping the operator's forward/left/up directions onto
  arm-base axes (the per-engage yaw correction handles the operator
  turning in the room, so only the axis convention matters).
- **Rest poses** (`rest_qpos_left/right`): where the arms park and what
  the IK's posture bias pulls toward.

## Adapting to a different arm

The `core/` mapping, the `relay/`, and the web client carry over to any
arm unchanged. The IK does not: `ik/` is written against the YAM's
geometry (the wrist-anchor site placement, the gripper-mount frame, a
6-DoF arm with a roughly spherical wrist whose joints split 3+3 into
position/orientation). Porting means rebuilding `ik/model.py`'s model
assembly and site construction for your robot's model files and checking
the decoupling assumption, not just retuning gains. (The camera panel ids — `top`, `left_wrist`,
`right_wrist` — are also fixed in the relay and web client; rename or
extend them there if your arm has a different camera set.)

## License

Apache-2.0.
