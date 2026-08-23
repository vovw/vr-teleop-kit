"""Shared-memory frame bus, so the Quest can watch the simulator.

The relay owns the WebRTC connection to the headset and pulls frames from a
``CameraReader`` per camera. Physics has to live somewhere else -- stepping
MuJoCo at 480 Hz inside the relay's event loop would stall the WebRTC stack --
so the simulator runs in its own process and hands frames over through shared
memory. A frame is 921 KB; copying it through a socket or a queue every tick
would cost more than the render.

``SimFrameReader`` deliberately mirrors ``CameraReader``'s shape (a ``spec``
attribute and a ``latest()`` returning BGR or ``None``), which is the entire
interface ``relay.server.CameraTrack`` uses. That means the relay streams sim
frames through exactly the same path as real ones, with no changes to the
WebRTC or encoding code.

Reads are guarded by a sequence counter rather than a lock: the writer bumps a
per-camera counter after each frame, and the reader re-checks it after copying.
A lock shared between the physics loop and the event loop would let a slow
reader stall the simulator, which matters more here than the occasional
retried copy.

Enable in the relay with ``VR_TELEOP_SIM_FRAMES=1``; without it the relay
opens real cameras exactly as before.
"""

from __future__ import annotations

import mmap
import os
import time

import numpy as np

from vr_teleop_kit.relay.capture import CameraSpec

SHM_NAME = "vrteleop_sim_frames"
CAM_IDS = ("top", "left_wrist", "right_wrist")
CAM_LABELS = {"top": "Top (sim)", "left_wrist": "Left wrist (sim)",
              "right_wrist": "Right wrist (sim)"}
WIDTH, HEIGHT = 640, 480
_FRAME_BYTES = WIDTH * HEIGHT * 3
_SEQ_BYTES = 8
_SLOT = _SEQ_BYTES + _FRAME_BYTES

# A small state channel after the frame slots, carrying the simulator's live
# qpos. This is what lets a viewer show the *running* session rather than its
# own separate copy of the scene: a viser process reads qpos, applies it, and
# pushes transforms to the browser, which is where its 3D view is actually
# rendered -- so watching live costs the host no GL work at all.
MAX_NQ = 256
_STATE_OFF = _SLOT * len(CAM_IDS)
_STATE_BYTES = _SEQ_BYTES + 8 + MAX_NQ * 8      # seq, nq, qpos[MAX_NQ]
_TOTAL = _STATE_OFF + _STATE_BYTES


def sim_frames_enabled() -> bool:
    return os.environ.get("VR_TELEOP_SIM_FRAMES", "").strip().lower() not in (
        "", "0", "false", "no")


def build_sim_camera_specs(fps: int = 30) -> list[CameraSpec]:
    """CameraSpecs for the three simulated cameras.

    Ids match the real ones so the Quest UI, the WS ``camera_list`` message and
    any recorded dataset keys are identical between sim and hardware.
    """
    return [CameraSpec(cid, CAM_LABELS[cid], f"sim://{cid}",
                       WIDTH, HEIGHT, fps, 0) for cid in CAM_IDS]


class SimFrameBus:
    """The shared memory block itself. One writer, any number of readers.

    Mapped with plain ``os.open`` + ``mmap`` rather than
    ``multiprocessing.shared_memory``. That module registers every block with a
    ``resource_tracker`` process which unlinks it when *any* attached process
    exits -- not just the creator (CPython bpo-38119). A reader coming and going
    therefore destroyed the simulator's bus, after which every other reader held
    a stale "(deleted)" mapping full of zeros and served black frames. There is
    no per-process refcount to opt out of, and unregistering clobbers the
    creator's own entry, so the tracker is best avoided altogether here.
    """

    _PATH = f"/dev/shm/{SHM_NAME}"

    def __init__(self, create: bool) -> None:
        self.create = create
        if create:
            # A block left behind by a killed writer would otherwise be reused
            # at whatever size and contents it had.
            try:
                os.unlink(self._PATH)
            except FileNotFoundError:
                pass
            fd = os.open(self._PATH, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, _TOTAL)
        else:
            fd = os.open(self._PATH, os.O_RDWR)
            if os.fstat(fd).st_size < _TOTAL:
                os.close(fd)
                raise FileNotFoundError(
                    f"{self._PATH} is smaller than expected — writer mid-setup?")
        self._fd = fd
        self._mm = mmap.mmap(fd, _TOTAL)
        if create:
            self._mm[:_TOTAL] = b"\x00" * _TOTAL
        # Identity of the block we mapped, so a reader can notice the writer
        # replaced it. A recreated block keeps the same *name* but gets a new
        # inode; our old mapping survives as "(deleted)" and silently freezes.
        self.inode = self._current_inode()

        self._seq: dict[str, np.ndarray] = {}
        self._img: dict[str, np.ndarray] = {}
        for i, cid in enumerate(CAM_IDS):
            base = i * _SLOT
            # np.ndarray(buffer=...) rather than np.frombuffer: the latter can
            # hand back a read-only view, which the writer needs to mutate.
            self._seq[cid] = np.ndarray(1, np.uint64, self._mm, base)
            self._img[cid] = np.ndarray((HEIGHT, WIDTH, 3), np.uint8, self._mm,
                                        base + _SEQ_BYTES)
        self._state_seq = np.ndarray(1, np.uint64, self._mm, _STATE_OFF)
        self._state_nq = np.ndarray(1, np.uint64, self._mm, _STATE_OFF + 8)
        self._state_qpos = np.ndarray(MAX_NQ, np.float64, self._mm,
                                      _STATE_OFF + 16)

    @staticmethod
    def _current_inode() -> int | None:
        try:
            return os.stat(SimFrameBus._PATH).st_ino
        except OSError:
            return None

    def is_stale(self) -> bool:
        """True if the block we mapped has been unlinked or replaced."""
        cur = self._current_inode()
        return cur is None or cur != self.inode

    def write(self, cam_id: str, frame: np.ndarray) -> None:
        if cam_id not in self._img:
            return
        if frame.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(
                f"{cam_id}: expected {(HEIGHT, WIDTH, 3)}, got {frame.shape}")
        self._img[cam_id][:] = frame
        # Bump the counter only after the pixels are in place, so a reader that
        # sees a new sequence number is looking at a complete frame.
        self._seq[cam_id][0] += 1

    def read(self, cam_id: str) -> np.ndarray | None:
        slot = self._img.get(cam_id)
        if slot is None:
            return None
        for _ in range(4):
            before = int(self._seq[cam_id][0])
            if before == 0:
                return None
            out = slot.copy()
            if int(self._seq[cam_id][0]) == before:
                return out
        return out

    def write_state(self, qpos: np.ndarray) -> None:
        """Publish the simulator's qpos for live viewers."""
        n = int(min(len(qpos), MAX_NQ))
        self._state_qpos[:n] = qpos[:n]
        self._state_nq[0] = n
        self._state_seq[0] += 1

    def read_state(self) -> tuple[int, np.ndarray] | None:
        """(sequence, qpos) or None if the simulator has not published yet."""
        for _ in range(4):
            before = int(self._state_seq[0])
            if before == 0:
                return None
            n = int(self._state_nq[0])
            out = self._state_qpos[:n].copy()
            if int(self._state_seq[0]) == before:
                return before, out
        return before, out

    def close(self) -> None:
        # Drop the numpy views first: they hold references into the mapping and
        # mmap.close() raises while any export is outstanding.
        self._seq.clear()
        self._img.clear()
        self._state_seq = self._state_nq = self._state_qpos = None
        try:
            self._mm.close()
        finally:
            try:
                os.close(self._fd)
            finally:
                if self.create:
                    try:
                        os.unlink(self._PATH)
                    except FileNotFoundError:
                        pass


class SimBusHolder:
    """A reader-side handle that survives the simulator restarting.

    The simulator clears any stale block when it starts, so a long-lived reader
    (the relay) that cached a bus ends up mapping an orphaned block and serving
    black frames for the rest of its life. This re-attaches when the block is
    replaced, rate-limited so the check costs one stat per interval rather than
    one per frame.
    """

    def __init__(self, check_period: float = 0.5) -> None:
        self._bus: SimFrameBus | None = None
        self._checked = 0.0
        self._period = check_period

    def bus(self) -> SimFrameBus | None:
        now = time.monotonic()
        if self._bus is not None and (now - self._checked) < self._period:
            return self._bus
        self._checked = now
        if self._bus is not None and not self._bus.is_stale():
            return self._bus
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        try:
            self._bus = SimFrameBus(create=False)
        except FileNotFoundError:
            self._bus = None
        return self._bus

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            finally:
                self._bus = None


class SimFrameReader:
    """CameraReader-shaped view of one camera's slot in the bus.

    ``relay.server.CameraTrack`` only ever touches ``.spec`` and ``.latest()``,
    so presenting the same two members is enough for simulated frames to travel
    the relay's existing WebRTC path unchanged.

    Takes a ``SimBusHolder`` (preferred -- keeps working when the simulator is
    restarted) or a bare ``SimFrameBus``.
    """

    def __init__(self, source: "SimFrameBus | SimBusHolder",
                 spec: CameraSpec) -> None:
        self.source = source
        self.spec = spec

    def latest(self) -> np.ndarray | None:
        bus = self.source.bus() if isinstance(self.source, SimBusHolder) else self.source
        if bus is None:
            return None
        return bus.read(self.spec.id)

    def stop(self) -> None:
        """No-op: the bus outlives individual readers."""
