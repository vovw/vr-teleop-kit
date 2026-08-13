"""Mujoco viewer showing BOTH YAM arms in one scene.

`viewer_client.py` renders one arm per window (`--arm left|right`), which
means two overlapping windows to watch a bimanual session. This variant
attaches two copies of the combined arm+gripper spec into a single scene
-- prefixed `left_` / `right_` -- and drives each from the matching key in
the broadcast `ik_state`.

The arms are placed side by side, separated along Y by `--separation`,
defaulting to 0.45 m -- the measured base-to-base spacing of this rig.
The IK works in each arm's own base frame (r_calib maps the headset world
into it), so the model files carry no inter-arm transform: this offset is
render-only and changing it cannot affect the solver or the commanded
joint angles. Re-measure and update the default if the arms are remounted.

Lighting: the default MuJoCo scene is lit only by a dim headlight, which
reads as near-black on the arms' dark meshes. This adds a key/fill/rim
spotlight rig aimed at the arms plus a brighter headlight; tune with
--brightness.

    python tools/viewer_both.py
    python tools/viewer_both.py --separation 0.8 --brightness 1.4

Quit by closing the viewer window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import mujoco
import mujoco.viewer
import websockets

from vr_teleop_kit.ik.model import (
    _combined_arm_gripper_xml,
    _GRIPPER_XML_REL,
    resolve_model_path,
)

DEFAULT_URL = "ws://127.0.0.1:8443/ws"
ARMS = ("left", "right")

# Redraw rate. Independent of the teleop's broadcast rate -- see consume().
RENDER_HZ = 60.0
# Cap on messages coalesced per render pass.
MAX_DRAIN = 400


def _arm_spec(model_path: str | None) -> mujoco.MjSpec:
    """One combined arm+gripper spec, same assembly the IK uses."""
    arm_xml = resolve_model_path(model_path)
    gripper_xml = arm_xml.resolve().parents[2] / _GRIPPER_XML_REL
    if not gripper_xml.exists():
        raise FileNotFoundError(
            f"gripper MJCF not found at {gripper_xml} -- expected the arm MJCF "
            "to live inside an i2rt clone at .../i2rt/robot_models/arm/yam/yam.xml"
        )
    return mujoco.MjSpec.from_string(_combined_arm_gripper_xml(arm_xml, gripper_xml))


def build_bimanual(model_path: str | None, separation: float, yaw_deg: float,
                   brightness: float):
    """Attach two arm specs into one scene and light it."""
    scene = mujoco.MjSpec()
    scene.modelname = "yam bimanual"   # shows in the viewer window title
    scene.compiler.degree = True

    # Ground plane: gives the eye a reference for depth and catches the
    # spotlights, so the arms read as solid rather than floating.
    scene.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=512, height=512,
        rgb1=[0.24, 0.26, 0.30], rgb2=[0.31, 0.33, 0.38],
    )
    mat = scene.add_material(name="gridmat", texrepeat=[6, 6], reflectance=0.08)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    scene.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.05],
        pos=[0, 0, -0.35], material="gridmat",
    )

    half = separation / 2.0
    # +Y is the operator's left when facing the robot's front (the frame
    # DEFAULT_R_CALIB maps into), so left goes to +Y.
    placement = {"left": (+half, +yaw_deg), "right": (-half, -yaw_deg)}
    for arm in ARMS:
        y, yaw = placement[arm]
        frame = scene.worldbody.add_frame(pos=[0.0, y, 0.0], euler=[0.0, 0.0, yaw])
        scene.attach(_arm_spec(model_path), prefix=f"{arm}_", frame=frame)

    # Key / fill / rim rig. Spotlights (cutoff < 90) aimed at the arms so
    # the highlights track the geometry instead of flat-washing it.
    b = brightness
    spot = mujoco.mjtLightType.mjLIGHT_SPOT
    directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    scene.worldbody.add_light(
        pos=[0.9, 0.7, 1.5], dir=[-0.5, -0.35, -1.0], type=spot,
        cutoff=55, exponent=12,
        diffuse=[0.85 * b, 0.85 * b, 0.82 * b], specular=[0.35 * b] * 3,
    )
    scene.worldbody.add_light(
        pos=[0.6, -1.1, 1.0], dir=[-0.35, 0.7, -0.9], type=spot,
        cutoff=65, exponent=8,
        diffuse=[0.45 * b, 0.47 * b, 0.55 * b], specular=[0.12 * b] * 3,
    )
    scene.worldbody.add_light(
        pos=[-1.3, 0.0, 0.9], dir=[1.0, 0.0, -0.45], type=directional,
        diffuse=[0.30 * b, 0.30 * b, 0.34 * b], specular=[0.10 * b] * 3,
    )

    model = scene.compile()

    # Headlight follows the camera, so it fills whatever you rotate toward
    # and stops any pose from going black. Ambient lifts the shadow floor.
    model.vis.headlight.ambient = [0.34 * b, 0.34 * b, 0.36 * b]
    model.vis.headlight.diffuse = [0.55 * b, 0.55 * b, 0.55 * b]
    model.vis.headlight.specular = [0.18 * b, 0.18 * b, 0.18 * b]
    return model, mujoco.MjData(model)


def qpos_addresses(model) -> dict[str, list[int]]:
    """qpos address of every hinge/slide joint, grouped by arm prefix and
    kept in model order -- which is the order the teleop packs its qpos."""
    out: dict[str, list[int]] = {a: [] for a in ARMS}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for arm in ARMS:
            if name.startswith(f"{arm}_"):
                out[arm].append(model.jnt_qposadr[j])
    return out


def apply(model, data, addrs, msg) -> bool:
    """Write both arms' qpos from one ik_state. Returns True if anything set."""
    touched = False
    for arm in ARMS:
        qpos = msg.get(f"{arm}_qpos")
        # Legacy compat: bare `qpos` is the right arm, same as viewer_client.
        if not qpos and arm == "right":
            qpos = msg.get("qpos")
        if not qpos or len(qpos) < 6:
            continue
        slots = addrs[arm]
        for value, adr in zip(qpos, slots):
            data.qpos[adr] = value
        touched = True
    return touched


async def consume(ws, model, data, viewer, addrs, from_id: str | None) -> None:
    """Drain ik_state as fast as it arrives, but render on a fixed clock.

    The teleop broadcasts at its loop rate (200 Hz). Rendering every message
    means mj_forward + viewer.sync() 200x/s on the event loop thread, which
    starves the websocket keepalive: the server closes with 1011 and the
    exception tears down the GL context mid-flight (SIGABRT). Only the newest
    ik_state is worth drawing anyway, so coalesce -- keep the last one seen
    and paint it at RENDER_HZ.
    """
    period = 1.0 / RENDER_HZ
    next_render = time.perf_counter()
    latest = None

    while viewer.is_running():
        # Coalesce whatever is already queued. Bounded, so a fast producer
        # cannot starve rendering entirely.
        for _ in range(MAX_DRAIN):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.001)
            except asyncio.TimeoutError:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") != "ik_state":
                continue
            if from_id is not None and msg.get("teleop_id") != from_id:
                continue
            latest = msg

        now = time.perf_counter()
        if now < next_render:
            # Yield so pings/pongs and window events get serviced.
            await asyncio.sleep(min(period, next_render - now))
            continue
        next_render = now + period

        if latest is not None and apply(model, data, addrs, latest):
            mujoco.mj_forward(model, data)
        viewer.sync()              # also pumps window events when idle


def _ssl_context_for(url: str):
    """Self-signed LAN cert -- same permissive handling as viewer_client."""
    if not url.startswith("wss://"):
        return None
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def runner(url, model, data, viewer, addrs, from_id) -> None:
    """Keep the viewer attached across relay restarts and dropped sockets.

    A closed connection is not fatal here: the window stays up showing the
    last pose and reconnects, so bouncing the relay mid-session doesn't cost
    you the viewer (and never takes the GL context down with it).
    """
    print(f"connecting to {url} (rendering both arms)", flush=True)
    announced = False
    while viewer.is_running():
        try:
            async with websockets.connect(
                url,
                ssl=_ssl_context_for(url),
                # Rendering briefly occupies this loop; keep the keepalive
                # far looser than any single frame could take.
                ping_interval=20,
                ping_timeout=60,
                max_queue=64,
            ) as ws:
                if not announced:
                    print("connected, listening for ik_state ...", flush=True)
                    announced = True
                await consume(ws, model, data, viewer, addrs, from_id)
            return                      # consume() returned: window closed
        except (OSError, websockets.exceptions.WebSocketException) as e:
            if not viewer.is_running():
                return
            print(f"relay connection lost ({type(e).__name__}); retrying in 1s",
                  flush=True)
            announced = False
            await asyncio.sleep(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help=f"WebSocket URL (default: {DEFAULT_URL})")
    ap.add_argument("--from-id", default=None, help="only render ik_state from this teleop id")
    ap.add_argument("--model", default=None, help="path to the YAM arm MJCF inside an i2rt clone")
    ap.add_argument("--separation", type=float, default=0.45,
                    help="metres between the two arm bases along Y "
                         "(default: 0.45, the measured spacing on this rig)")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="degrees each arm is toed in toward the centre (default: 0)")
    ap.add_argument("--brightness", type=float, default=1.0,
                    help="scales the whole light rig (default: 1.0; try 1.4 if still dark)")
    args = ap.parse_args()

    model, data = build_bimanual(args.model, args.separation, args.yaw, args.brightness)
    addrs = qpos_addresses(model)
    print(f"joints per arm: " + ", ".join(f"{a}={len(addrs[a])}" for a in ARMS), flush=True)

    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        try:
            asyncio.run(runner(args.url, model, data, viewer, addrs, args.from_id))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            # Never let an exception unwind through launch_passive's exit --
            # tearing the GL context down mid-flight aborts the process
            # ("free(): invalid pointer") instead of reporting the cause.
            print(f"viewer stopping: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
