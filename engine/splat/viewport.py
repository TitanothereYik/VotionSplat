"""Real-time viewport transport: shared-memory frame ring + stdin camera protocol.

Stdlib only (``multiprocessing.shared_memory``, ``struct``, ``threading``) so the Qt
process can import the reader without torch.

Layout of the block ``votion_splat_<pid>``::

    header (64 bytes, little endian)
      0  magic   4s  b"VSPL"
      4  version u32 = 1
      8  seq     u32  monotonically increasing frame counter (0 = nothing yet)
     12  slot    u32  index of the slot holding frame ``seq``
     16  width   u32  of the frame in ``slot``
     20  height  u32
     24  max_w   u32  slot capacity
     28  max_h   u32
     32  slots   u32
     36  pad     to 64
    slot k at 64 + k * max_w * max_h * 3 : RGB8 rows, tightly packed (stride = width*3)

Writer fills the slot that is *not* current, then publishes ``seq``/``slot``/``w``/``h``
in one header write. Reader checks ``seq``; a torn frame is impossible because the
reader never reads the slot the writer is filling (2-slot ring, one writer).

Stdin lines from Qt (tab separated, one per line)::

    VIEW\t{"yaw":..,"pitch":..,"zoom":..,"pan_x":..,"pan_y":..,"w":..,"h":..,"cam":-1,"gt":false}
    PAUSE / RESUME / STOP

Stdout lines to Qt (worker -> Qt), see ``engine.splat.trainer`` for the full list::

    SPLAT\tshm\t<name>\t<bytes>\t<max_w>\t<max_h>
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
from multiprocessing import shared_memory

from engine.splat.camera import ViewState

MAGIC = b"VSPL"
VERSION = 1
HEADER_SIZE = 64
_HDR = struct.Struct("<4sIIIIIIII")  # magic, version, seq, slot, w, h, max_w, max_h, slots
DEFAULT_MAX_W = 1920
DEFAULT_MAX_H = 1080


def shm_name(pid: int | None = None) -> str:
    return f"votion_splat_{pid if pid is not None else os.getpid()}"


class SharedFrame:
    """Writer side (worker process)."""

    def __init__(self, name: str | None = None, max_w: int = DEFAULT_MAX_W, max_h: int = DEFAULT_MAX_H, slots: int = 2) -> None:
        self.name = name or shm_name()
        self.max_w = int(max_w)
        self.max_h = int(max_h)
        self.slots = max(2, int(slots))
        self.slot_bytes = self.max_w * self.max_h * 3
        self.size = HEADER_SIZE + self.slot_bytes * self.slots
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
        except FileExistsError:
            old = shared_memory.SharedMemory(name=self.name, create=False)
            old.close()
            try:
                old.unlink()
            except Exception:
                pass
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
        self.seq = 0
        self.slot = 0
        self._write_header(0, 0, 0, 0)

    def _write_header(self, seq: int, slot: int, w: int, h: int) -> None:
        _HDR.pack_into(self.shm.buf, 0, MAGIC, VERSION, seq, slot, w, h, self.max_w, self.max_h, self.slots)

    def write(self, rgb) -> int:
        """``rgb``: numpy uint8 array [H, W, 3] (C-contiguous). Returns the new seq."""
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        if w > self.max_w or h > self.max_h:
            raise ValueError(f"frame {w}x{h} exceeds slot {self.max_w}x{self.max_h}")
        import numpy as np

        nxt = (self.slot + 1) % self.slots
        off = HEADER_SIZE + nxt * self.slot_bytes
        n = w * h * 3
        dst = np.frombuffer(self.shm.buf, dtype=np.uint8, offset=off, count=n)
        dst[:] = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(-1)[:n]
        self.seq += 1
        self.slot = nxt
        self._write_header(self.seq, self.slot, w, h)
        return self.seq

    def announce_line(self) -> str:
        return f"SPLAT\tshm\t{self.name}\t{self.size}\t{self.max_w}\t{self.max_h}"

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass
        try:
            self.shm.unlink()
        except Exception:
            pass


class SharedFrameReader:
    """Reader side (Qt process). ``poll()`` returns None when nothing new."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.shm = shared_memory.SharedMemory(name=name, create=False)
        magic, version, _s, _slot, _w, _h, self.max_w, self.max_h, self.slots = _HDR.unpack_from(self.shm.buf, 0)
        if magic != MAGIC or version != VERSION:
            self.close()
            raise ValueError(f"shared frame {name}: bad header {magic!r} v{version}")
        self.slot_bytes = self.max_w * self.max_h * 3
        self.last_seq = 0

    def poll(self):
        """-> (seq, w, h, bytes) for a new frame, else None. Copies the frame (<= 6 MB)."""
        try:
            _m, _v, seq, slot, w, h, _mw, _mh, _slots = _HDR.unpack_from(self.shm.buf, 0)
        except (ValueError, struct.error):
            return None
        if seq == 0 or seq == self.last_seq or w == 0 or h == 0:
            return None
        off = HEADER_SIZE + int(slot) * self.slot_bytes
        n = int(w) * int(h) * 3
        data = bytes(self.shm.buf[off : off + n])
        # If the writer lapped us during the copy, the header seq moved on; the slot we
        # read is then the one being refilled. Re-check and drop the torn frame.
        _m, _v, seq2, slot2, _w2, _h2, _mw, _mh, _slots = _HDR.unpack_from(self.shm.buf, 0)
        if seq2 != seq and slot2 != slot:
            return None
        self.last_seq = seq
        return seq, int(w), int(h), data

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass


class ControlReader:
    """Background thread over stdin. Keeps the latest VIEW, queues control words."""

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._lock = threading.Lock()
        self._view: ViewState | None = None
        self._view_seq = 0
        self._commands: list[str] = []
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="splat-stdin", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for raw in self._stream:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("VIEW\t"):
                    try:
                        data = json.loads(line.split("\t", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    with self._lock:
                        self._view = ViewState.from_dict(data)
                        self._view_seq += 1
                else:
                    word = line.split("\t", 1)[0].upper()
                    with self._lock:
                        self._commands.append(word)
        except (ValueError, OSError):
            pass
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def latest_view(self) -> tuple[int, ViewState | None]:
        with self._lock:
            return self._view_seq, self._view

    def pop_commands(self) -> list[str]:
        with self._lock:
            out = self._commands
            self._commands = []
        return out
