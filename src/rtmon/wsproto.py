"""Minimal RFC 6455 WebSocket, enough for one local page.

Deliberately stdlib-only, in the same spirit as ``beat_app.server``: the project
already ships a dependency-light HTTP server, and adding an async web framework to
pull one socket upgrade would be the tail wagging the dog. What is implemented is
exactly what a browser talking to ``localhost`` uses -- the handshake, binary and
text data frames, continuation, ping/pong and close. No extensions (so no
permessage-deflate: the payload here is already-decimated float32, which does not
compress usefully and would cost CPU on every frame).

Server-to-client frames are never masked; client-to-server frames always are.
:meth:`WebSocket.send` is serialised by a lock, so the broadcast thread and the
per-connection reader can share one socket.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct
import threading

_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

# A frame larger than this is treated as hostile rather than allocated. Nothing the
# page sends is more than a few KB of JSON.
MAX_FRAME_BYTES = 1 << 20


def accept_key(client_key: str) -> str:
    """The ``Sec-WebSocket-Accept`` value for a client's ``Sec-WebSocket-Key``."""
    digest = hashlib.sha1(client_key.strip().encode("ascii") + _GUID).digest()
    return base64.b64encode(digest).decode("ascii")


class WebSocketClosed(Exception):
    pass


class WebSocket:
    """A connected socket after a successful upgrade."""

    def __init__(self, rfile, wfile):
        self._rfile = rfile
        self._wfile = wfile
        self._wlock = threading.Lock()
        self.closed = False

    # ----------------------------------------------------------------- write
    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        if n < 126:
            header = struct.pack("!BB", 0x80 | opcode, n)
        elif n < (1 << 16):
            header = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        with self._wlock:
            if self.closed:
                raise WebSocketClosed()
            try:
                # One write, not header-then-body: two writes on a TCP_NODELAY socket
                # put a frame header in its own packet at the frame rates used here.
                self._wfile.write(header + payload)
                self._wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError, OSError) as exc:
                self.closed = True
                raise WebSocketClosed() from exc

    def send_binary(self, payload: bytes) -> None:
        self._send_frame(OP_BINARY, payload)

    def send_text(self, text: str) -> None:
        self._send_frame(OP_TEXT, text.encode("utf-8"))

    def send_pong(self, payload: bytes = b"") -> None:
        self._send_frame(OP_PONG, payload)

    def close(self, code: int = 1000) -> None:
        if self.closed:
            return
        try:
            self._send_frame(OP_CLOSE, struct.pack("!H", code))
        except WebSocketClosed:
            pass
        finally:
            self.closed = True

    # ------------------------------------------------------------------ read
    def _read_exact(self, n: int) -> bytes:
        buf = self._rfile.read(n)
        if buf is None or len(buf) != n:
            raise WebSocketClosed()
        return buf

    def recv(self):
        """Next complete message as ``(opcode, payload)``.

        Blocks. Ping is answered here and never surfaces to the caller; close raises
        :class:`WebSocketClosed`. Fragmented messages are reassembled.
        """
        frag_op = None
        frag: list[bytes] = []
        while True:
            b0, b1 = self._read_exact(2)
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                (length,) = struct.unpack("!H", self._read_exact(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._read_exact(8))
            if length > MAX_FRAME_BYTES:
                self.close(1009)
                raise WebSocketClosed()

            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(length) if length else b""
            if mask:
                # bytes(a ^ b) via translate is not available for a 4-cycle key; the
                # numpy-free int-list form is fast enough for control-sized payloads.
                payload = bytes(byte ^ mask[i & 3] for i, byte in enumerate(payload))

            if opcode == OP_CLOSE:
                self.closed = True
                raise WebSocketClosed()
            if opcode == OP_PING:
                self.send_pong(payload)
                continue
            if opcode == OP_PONG:
                continue

            if opcode == OP_CONT:
                if frag_op is None:
                    raise WebSocketClosed()
                frag.append(payload)
            else:
                frag_op, frag = opcode, [payload]

            if fin:
                return frag_op, b"".join(frag)


def handshake_headers(client_key: str) -> list[tuple[str, str]]:
    """Response headers completing the upgrade (status 101 is the caller's job)."""
    return [
        ("Upgrade", "websocket"),
        ("Connection", "Upgrade"),
        ("Sec-WebSocket-Accept", accept_key(client_key)),
    ]


def is_upgrade(headers) -> bool:
    return (headers.get("Upgrade", "").lower() == "websocket"
            and "upgrade" in headers.get("Connection", "").lower()
            and bool(headers.get("Sec-WebSocket-Key")))


def origin_is_local(headers) -> bool:
    """Reject cross-origin upgrades.

    A browser will happily let any page on the internet open a WebSocket to
    ``localhost`` -- the same-origin policy does not apply to WebSockets and there is
    no preflight. Since this server exposes device control and recording, the Origin
    header is checked here rather than trusting that "it's only bound to loopback".
    """
    origin = headers.get("Origin")
    if not origin:
        return True  # non-browser client (curl, a test); no ambient credentials to abuse
    if os.environ.get("RTMON_ALLOW_ANY_ORIGIN"):
        return True
    return any(origin.startswith(p) for p in ("http://127.0.0.1", "http://localhost", "http://[::1]"))
