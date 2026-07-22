"""FastAPI relay + WebRTC publisher for VR teleop.

Two responsibilities, both per WebSocket client:

  1. Broadcast relay for pose/state messages (xr_frame, ik_state,
     config_update, … — the full set is RELAY_TYPES below) between the
     Quest browser and the pose-streaming client
     (e.g. `examples/teleop_bi_yam.py`).

  2. WebRTC publisher for camera tracks. Cameras are auto-discovered by
     serial (see relay/cameras.py); CAM_TOP / CAM_LEFT / CAM_RIGHT env vars
     override a role's device. On webrtc_request the server opens any cameras that exist,
     creates an RTCPeerConnection with one VideoStreamTrack per camera,
     and exchanges SDP/ICE over the same WebSocket. Per-camera enable
     toggles (camera_toggle messages) mute the track by repeating the
     last frame — H.264 inter-frame compression drops the bandwidth to
     near zero without renegotiating SDP.

Topology:
    Quest browser  ── xr_frame ─────►  server  ── xr_frame ──►  teleop process
                                                                       │
                   ◄── ik_state ──   server  ◄── ik_state ──   (teleop publishes)
                   ◄═ WebRTC video ═ server                    (cv2 → aiortc tracks)

Run (console script from `pip install vr-teleop-kit[relay]`):
    vr-teleop-relay                            # bind 127.0.0.1 (USB / tunnel)
    vr-teleop-relay --host 0.0.0.0 \
        --ssl-keyfile  certs/key.pem \
        --ssl-certfile certs/cert.pem       # LAN HTTPS for direct Quest access
"""

import argparse
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vr_teleop_kit.relay.capture import CameraReader, CameraSpec, build_camera_specs
from vr_teleop_kit.log import get_logger, setup_logging

setup_logging(level=logging.INFO)
logger = get_logger("vr_teleop", "relay")
cam_log = get_logger("vr_teleop.camera", "camera")
ws_log = get_logger("vr_teleop.ws", "ws")
rtc_log = get_logger("vr_teleop.webrtc", "webrtc")
WEB_DIR = Path(__file__).parent / "web"

# WebSocket message types broadcast verbatim to every other connected client.
# Anything not in this set is handled inline (signaling, toggles, ping).
RELAY_TYPES = {
    "xr_frame", "ik_state", "config_update", "request_settings",
    # Web UI ↔ teleop: gripper-haptic threshold calibration.
    "haptic_calibrate", "haptic_calibrate_result",
}


# ── Camera capture ───────────────────────────────────────────────────────
# Frame grabbing lives in relay/capture.py (CameraReader / CameraSpec /
# build_camera_specs), shared with the dataset recorder. CameraTrack below is
# the relay-only WebRTC adapter: it pulls from a CameraReader and hands frames
# to aiortc, so it stays here with the av/aiortc deps.

class CameraTrack(VideoStreamTrack):
    """aiortc track that pulls from a CameraReader. `enabled=False` returns
    the previously-sent frame, so H.264's P-frames compress to nothing and
    the receiver freezes on the last image — no SDP renegotiation needed."""

    kind = "video"

    def __init__(self, reader: CameraReader) -> None:
        super().__init__()
        self.reader = reader
        self.enabled = True
        self._last_sent: np.ndarray | None = None
        # Pre-build a black fallback so the encoder has *something* to chew
        # on if the camera hasn't produced a frame yet by the first recv().
        # 90/270 rotations swap the frame dimensions.
        h, w = reader.spec.height, reader.spec.width
        if reader.spec.rotate in (90, 270):
            h, w = w, h
        self._black = np.zeros((h, w, 3), dtype=np.uint8)

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        if self.enabled:
            frame = self.reader.latest()
            if frame is None:
                frame = self._last_sent if self._last_sent is not None else self._black
            else:
                self._last_sent = frame
        else:
            frame = self._last_sent if self._last_sent is not None else self._black
        # cv2 returns BGR; PyAV's bgr24 format matches.
        vf = av.VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf


# ── Camera registry ──────────────────────────────────────────────────────
# Only cameras whose device path exists are registered (build_camera_specs in
# relay/capture.py). The Quest UI is populated from this list (camera_list
# message on WS open), so missing cameras silently disappear instead of
# erroring at peer-connect time.

CAMERA_SPECS: list[CameraSpec] = build_camera_specs()
CAMERA_READERS: dict[str, CameraReader] = {}


def _ensure_readers() -> None:
    """Lazy-open the cameras on first WebRTC request. Avoids holding v4l2
    locks during dev iterations where the operator only wants the relay."""
    for spec in CAMERA_SPECS:
        if spec.id in CAMERA_READERS:
            continue
        try:
            CAMERA_READERS[spec.id] = CameraReader(spec)
        except Exception:
            cam_log.exception("failed to open camera %s", spec.id)


# ── Codec preference ─────────────────────────────────────────────────────
# Force H.264 because Quest's hardware video decoder is H.264-strongest.
# VP8 is software-decoded → higher CPU, worse latency under load.

def _prefer_h264(pc: RTCPeerConnection) -> None:
    caps = RTCRtpSender.getCapabilities("video")
    h264 = [c for c in caps.codecs if c.mimeType == "video/H264"]
    if not h264:
        return
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(h264)


# ── Per-WebSocket client state ───────────────────────────────────────────
# Each WS holds its own RTCPeerConnection + the live CameraTracks it added.
# We key tracks by camera id so camera_toggle messages can flip the right one.

@dataclass
class ClientState:
    ws: WebSocket
    pc: RTCPeerConnection | None = None
    tracks: dict[str, CameraTrack] | None = None  # id -> track


_clients: dict[WebSocket, ClientState] = {}


async def _broadcast(text: str, exclude: WebSocket | None = None) -> None:
    if not _clients:
        return
    targets = [ws for ws in list(_clients) if ws is not exclude]
    if not targets:
        return
    await asyncio.gather(
        *[ws.send_text(text) for ws in targets],
        return_exceptions=True,
    )


# ── FastAPI app ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for reader in CAMERA_READERS.values():
        reader.stop()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Force-disable browser cache for the page and JS so operators don't
    have to clear the Quest's cache after every web-asset change."""
    response = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


# ── WebRTC signaling ─────────────────────────────────────────────────────
# Server-as-publisher pattern:
#   1. Client sends `webrtc_request` (optionally with enabled_cameras list).
#   2. Server creates PC, adds CameraTracks, generates *offer*, sends back
#      as `webrtc_offer` carrying a `cameras` list ({id, label}). The client
#      matches incoming tracks to cameras via MediaStream.id, which we force
#      to the camera id (sender._stream_id below).
#   3. Client sets remote desc, generates *answer*, sends `webrtc_answer`.
#   4. The client trickles `ice_candidate` messages; the server drops them
#      (see ws_handler — the offer SDP already carries our candidates).

async def _handle_webrtc_request(state: ClientState, msg: dict) -> None:
    if state.pc is not None:
        # Already negotiated for this WS — tear down before re-offering.
        await _close_pc(state)

    _ensure_readers()
    enabled_set: set[str] = set(msg.get("enabled_cameras") or [s.id for s in CAMERA_SPECS])

    pc = RTCPeerConnection()
    state.pc = pc
    state.tracks = {}

    for spec in CAMERA_SPECS:
        reader = CAMERA_READERS.get(spec.id)
        if reader is None:
            continue
        track = CameraTrack(reader)
        track.enabled = spec.id in enabled_set
        sender = pc.addTrack(track)
        # Force the MediaStream id to the camera id. aiortc otherwise
        # assigns a random UUID, leaving the client unable to associate
        # an incoming track with a UI slot without scraping SDP msid.
        # Private attr, but stable across aiortc 1.9+.
        sender._stream_id = spec.id
        state.tracks[spec.id] = track

    _prefer_h264(pc)

    @pc.on("iceconnectionstatechange")
    async def _on_ice_state() -> None:
        rtc_log.info("ice state: %s", pc.iceConnectionState)
        if pc.iceConnectionState in ("failed", "closed"):
            await _close_pc(state)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    await state.ws.send_text(json.dumps({
        "type": "webrtc_offer",
        "sdp": pc.localDescription.sdp,
        "sdp_type": pc.localDescription.type,
        "cameras": [
            {"id": spec.id, "label": spec.label}
            for spec in CAMERA_SPECS if spec.id in state.tracks
        ],
    }))


async def _handle_webrtc_answer(state: ClientState, msg: dict) -> None:
    if state.pc is None:
        rtc_log.warning("webrtc_answer with no pending pc")
        return
    answer = RTCSessionDescription(sdp=msg["sdp"], type=msg["sdp_type"])
    await state.pc.setRemoteDescription(answer)


async def _handle_camera_toggle(state: ClientState, msg: dict) -> None:
    cam_id = msg.get("camera_id")
    enabled = bool(msg.get("enabled", True))
    if state.tracks and cam_id in state.tracks:
        state.tracks[cam_id].enabled = enabled
        cam_log.info("camera %s -> %s", cam_id, "on" if enabled else "off")


async def _close_pc(state: ClientState) -> None:
    if state.pc is None:
        return
    try:
        await state.pc.close()
    except Exception:
        pass
    state.pc = None
    state.tracks = None


# ── WebSocket endpoint ───────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_handler(websocket: WebSocket) -> None:
    await websocket.accept()
    state = ClientState(ws=websocket)
    _clients[websocket] = state
    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    ws_log.info("ws connect %s  (now %d clients)", peer, len(_clients))

    # Tell the client which cameras exist before any signaling starts. The
    # UI uses this to render the toggle row even if the operator hasn't
    # asked for video yet.
    await websocket.send_text(json.dumps({
        "type": "camera_list",
        "cameras": [{"id": s.id, "label": s.label} for s in CAMERA_SPECS],
    }))

    types_seen: dict[str, int] = {}

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t = msg.get("type", "?")
            if t not in types_seen:
                ws_log.info("first %r msg from %s | keys=%s", t, peer, sorted(msg.keys()))
            types_seen[t] = types_seen.get(t, 0) + 1

            if t in RELAY_TYPES:
                await _broadcast(raw, exclude=websocket)
            elif t == "webrtc_request":
                await _handle_webrtc_request(state, msg)
            elif t == "webrtc_answer":
                await _handle_webrtc_answer(state, msg)
            elif t == "ice_candidate":
                # Trickle ICE from the browser — intentionally dropped.
                # aiortc gathers all local candidates before the offer is
                # sent (non-trickle), so on LAN the connection establishes
                # from the SDP candidates alone.
                pass
            elif t == "camera_toggle":
                await _handle_camera_toggle(state, msg)
            elif t == "latency_report":
                # Client-side RTT stats from latency mode (?latency=1). Logged
                # here so transport comparisons (USB vs LAN) can be read off the
                # workstation terminal. `host` self-labels the transport
                # (localhost:8443 = USB tether, <lan-ip>:8443 = LAN).
                logger.info(
                    "latency_report host=%s n=%s mean=%.1f p50=%.1f p95=%.1f min=%.1f max=%.1f ms (one-way~%.1f)",
                    msg.get("host"), msg.get("n"),
                    msg.get("mean", 0.0), msg.get("p50", 0.0), msg.get("p95", 0.0),
                    msg.get("min", 0.0), msg.get("max", 0.0), msg.get("p50", 0.0) / 2.0,
                )
            else:
                await websocket.send_json({"echo": msg, "server_time": time.time()})

    except WebSocketDisconnect as e:
        ws_log.info("ws disconnect %s code=%s totals=%s", peer, e.code, types_seen)
    finally:
        await _close_pc(state)
        _clients.pop(websocket, None)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Use 0.0.0.0 for LAN access.")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--ssl-keyfile",  default=None, help="Path to TLS private key (PEM).")
    ap.add_argument("--ssl-certfile", default=None, help="Path to TLS cert chain (PEM).")
    args = ap.parse_args()

    import uvicorn
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="info",
        ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
