"""Serve the pixel-matched scene in a browser, for inspection over SSH.

MuJoCo's own viewer needs a local GL window, which is awkward on a headless
box and useless inside tmux. Viser renders in the browser instead, so the
scene is inspectable from anywhere the port is reachable.

Alongside the 3D scene the page carries the two things you actually want to
judge -- the real top-camera frame and the MuJoCo render of the same camera --
so the pixel match can be eyeballed without leaving the page. The render is
produced through the real lens distortion, matching what the camera would
output rather than an idealised pinhole.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_viser.py
    # then open the printed URL, or forward it:
    #   ssh -L 8080:localhost:8080 <this-host>

Joint sliders drive the arms live; the render panel refreshes with them.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tools"))


def mat2quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z), the ordering viser expects."""
    t = R.trace()
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--live", action="store_true",
                    help="mirror a running simulator instead of showing a "
                         "standalone copy of the scene: attaches to the shared-"
                         "memory bus that examples/record_sim_cubes.py writes "
                         "and follows its qpos. Costs the host no GL work, "
                         "because viser's 3D view renders in the browser.")
    ap.add_argument("--camera-panel", action="store_true",
                    help="with --live, also show the top-camera render. Off by "
                         "default: it is an 11 ms GL readback per refresh, and "
                         "the simulator needs that budget for its own three "
                         "cameras.")
    args = ap.parse_args()

    import cv2
    import mujoco
    import viser

    from sim_compare_top import distortion_maps

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    intr_all = json.loads((_REPO_ROOT / "sim" / "camera_intrinsics.json").read_text())
    want_panel = (not args.live) or args.camera_panel
    maps = (distortion_maps(intr_all["top"],
                            cache=_REPO_ROOT / "sim" / "top_distortion_map.npz")
            if want_panel else None)

    bus = None
    if args.live:
        from vr_teleop_kit.relay.sim_frames import SimFrameBus
        try:
            bus = SimFrameBus(create=False)
        except FileNotFoundError:
            raise SystemExit(
                "--live: no shared-memory bus found. Start the simulator first "
                "(examples/record_sim_cubes.py); it creates the bus.")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    # ── static geometry ─────────────────────────────────────────────────
    # Every geom is pushed once; only its transform is updated later, which
    # keeps the slider callback cheap.
    handles: list[tuple[int, object]] = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
        gtype = model.geom_type[gid]
        rgba = model.geom_rgba[gid]
        colour = tuple(int(255 * c) for c in rgba[:3])
        pos = data.geom_xpos[gid].copy()
        quat = mat2quat(data.geom_xmat[gid].reshape(3, 3))
        size = model.geom_size[gid]
        path = f"/scene/{name}"

        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            h = server.scene.add_box(path, color=colour,
                                     dimensions=(8.0, 8.0, 0.002),
                                     position=pos, wxyz=quat)
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            h = server.scene.add_box(path, color=colour,
                                     dimensions=tuple(2 * size[:3]),
                                     position=pos, wxyz=quat)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            h = server.scene.add_icosphere(path, radius=float(size[0]),
                                           color=colour, position=pos, wxyz=quat)
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            verts = model.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float32)
            faces = model.mesh_face[fa:fa + fn].reshape(-1, 3).astype(np.uint32)
            h = server.scene.add_mesh_simple(path, vertices=verts, faces=faces,
                                             color=colour, position=pos, wxyz=quat)
        else:
            continue
        handles.append((gid, h))

    # ── camera frustums, drawn with the real measured FOV ───────────────
    cam_handles: list[tuple[int, object]] = []
    for cname, intr in intr_all.items():
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cname)
        if cid < 0:
            continue
        # viser's frustum takes vertical FOV in radians; MuJoCo's camera looks
        # down -Z while viser's looks down +Z, so spin 180 deg about X.
        R = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        h = server.scene.add_camera_frustum(
            f"/cameras/{cname}",
            fov=np.radians(intr["vfov_deg"]),
            aspect=intr["width"] / intr["height"],
            scale=0.12, color=(255, 90, 90),
            position=data.cam_xpos[cid].copy(), wxyz=mat2quat(R))
        cam_handles.append((cid, h))
        server.scene.add_label(f"/cameras/{cname}/label", cname,
                               position=data.cam_xpos[cid].copy())

    # ── GUI ─────────────────────────────────────────────────────────────
    cam_top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    real_top = renderer = sim_img = blend_img = None
    if want_panel:
        real_top = cv2.imread(str(_REPO_ROOT / "greenscreen" / "top.png"))
        renderer = mujoco.Renderer(model, intr_all["top"]["height"],
                                   intr_all["top"]["width"])
        with server.gui.add_folder("Top camera: real vs sim"):
            server.gui.add_markdown(
                f"fx={intr_all['top']['fx']:.2f} fy={intr_all['top']['fy']:.2f}  "
                f"HFOV={intr_all['top']['hfov_deg']:.2f}° "
                f"VFOV={intr_all['top']['vfov_deg']:.2f}°")
            server.gui.add_image(cv2.cvtColor(real_top, cv2.COLOR_BGR2RGB),
                                 label="real (greenscreen/top.png)")
            sim_img = server.gui.add_image(np.zeros_like(real_top),
                                           label="sim render")
            blend_img = server.gui.add_image(np.zeros_like(real_top),
                                             label="50/50 blend")
    else:
        server.gui.add_markdown(
            "**live mirror** — following the running simulator.\n\n"
            "Camera views are in the relay page (http://localhost:8443/); "
            "pass `--camera-panel` to render one here too.")

    show_arms = server.gui.add_checkbox("Render arms", False)

    # MuJoCo's GL context belongs to the thread that created it (this one), and
    # viser dispatches GUI callbacks on its websocket thread. Rendering from
    # there makes glXMakeCurrent fail with BadAccess and kills the process the
    # moment a browser connects. So callbacks only mark state dirty; the main
    # loop below owns every render.
    dirty = threading.Event()
    dirty.set()

    def refresh_render() -> None:
        if renderer is None:
            return
        saved = model.geom_rgba[:, 3].copy()
        if not show_arms.value:
            for gid in range(model.ngeom):
                n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if n.startswith(("left_", "right_")):
                    model.geom_rgba[gid, 3] = 0.0
        renderer.update_scene(data, camera=cam_top)
        rgb = renderer.render()
        model.geom_rgba[:, 3] = saved
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dist = cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR)
        sim_img.image = cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)
        blend_img.image = cv2.cvtColor(
            cv2.addWeighted(real_top, 0.5, dist, 0.5, 0), cv2.COLOR_BGR2RGB)

    def sync_scene() -> None:
        """Push MuJoCo state to viser and re-render. Main thread only."""
        mujoco.mj_forward(model, data)
        for gid, h in handles:
            h.position = data.geom_xpos[gid].copy()
            h.wxyz = mat2quat(data.geom_xmat[gid].reshape(3, 3))
        for cid, h in cam_handles:
            h.position = data.cam_xpos[cid].copy()
            h.wxyz = mat2quat(data.cam_xmat[cid].reshape(3, 3)
                              @ np.diag([1.0, -1.0, -1.0]))
        refresh_render()

    show_arms.on_update(lambda _: dirty.set())

    for side in (() if args.live else ("left", "right")):
        with server.gui.add_folder(f"{side} arm"):
            for j in range(1, 8):
                jname = f"{side}_joint{j}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                qadr = model.jnt_qposadr[jid]
                lo, hi = model.jnt_range[jid]
                if lo == hi:
                    lo, hi = -3.14, 3.14
                s_h = server.gui.add_slider(
                    jname, min=float(lo), max=float(hi), step=0.005,
                    initial_value=float(data.qpos[qadr]))
                s_h.on_update(
                    lambda _, adr=qadr, h=s_h: (data.qpos.__setitem__(adr, h.value),
                                                dirty.set()))

    reset = server.gui.add_button("Reset pose", disabled=bool(args.live))

    @reset.on_click
    def _(_) -> None:
        if args.live:
            return
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(model, data, kid)
        else:
            mujoco.mj_resetData(model, data)
        dirty.set()

    pose = json.loads((_REPO_ROOT / "sim" / "top_camera_pose.json").read_text())
    print(f"\nviser serving on http://0.0.0.0:{args.port}")
    print(f"  over SSH:  ssh -L {args.port}:localhost:{args.port} <this-host>")
    print(f"  top camera: pos={np.round(pose['pos'], 4).tolist()} "
          f"euler_deg={np.round(pose['euler_deg'], 3).tolist()}")
    nfit = sum(pose["features"][k] for k in pose.get("fit_features", pose["features"]))
    print(f"  fit RMS {pose['fit_rms_px']:.3f} px on {nfit} edge points "
          f"({', '.join(pose.get('fit_features', []))})")
    print("Ctrl-C to stop.")
    try:
        last_seq = -1
        while True:
            if args.live:
                st = bus.read_state()
                if st is not None and st[0] != last_seq:
                    last_seq = st[0]
                    n = min(len(st[1]), model.nq)
                    data.qpos[:n] = st[1][:n]
                    dirty.set()
                elif st is None:
                    time.sleep(0.2)
            if dirty.is_set() or (not args.live and dirty.wait(timeout=0.5)):
                dirty.clear()
                try:
                    sync_scene()
                except Exception:
                    # A failure here must not take the server down.
                    traceback.print_exc()
            if args.live:
                time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if bus is not None:
            bus.close()


if __name__ == "__main__":
    main()
