"""Beat-marking web app — backend.

A small dependency-light (stdlib + numpy/scipy/pyyaml, all already in the project)
HTTP server that:

  * serves the single-page frontend under ``static/``,
  * loads an uploaded WAV into :class:`analyze.data.Audio`,
  * lists and runs the ``analyze.hr`` beat detectors (in a background thread, so
    the request returns immediately and the UI polls for the result),
  * exports edited beat timestamps as YAML or ``.npy``,
  * loads timestamps back from a ``.npy`` file.

Run it with::

    python src/beat_app/server.py            # then open http://127.0.0.1:8000
    python src/beat_app/server.py --port 9000

The browser decodes the audio itself for display/playback; the backend keeps its
own copy only so the Python detectors can run. Both sides agree on the time axis
in *seconds*, so beat times line up regardless of any browser resampling.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Make ``analyze`` / ``constants`` importable exactly as the rest of the project does.
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from beat_app import audio_io, detectors  # noqa: E402
from constants import FETAL_BPM_RANGE  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 MB guard rail

# In-memory state. Single-user local tool, so a plain dict guarded by a lock is fine.
_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}   # session_id -> {"audio": Audio, "name": str}
_JOBS: dict[str, dict] = {}       # job_id -> {"status", "times"|"error"}


# ---------------------------------------------------------------------------
# Detection jobs (non-blocking)
# ---------------------------------------------------------------------------

def _start_detection(session_id: str, detector_id: str, bpm_range) -> str:
    with _LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError("unknown session; upload a file first")

    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {"status": "running"}

    def worker():
        try:
            times = detectors.run_detector(detector_id, session["audio"], bpm_range)
            with _LOCK:
                _JOBS[job_id] = {"status": "done", "times": [float(t) for t in times]}
        except Exception as exc:
            traceback.print_exc()
            with _LOCK:
                _JOBS[job_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    threading.Thread(target=worker, name=f"detect-{job_id[:8]}", daemon=True).start()
    return job_id


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "BeatApp/1.0"

    # -- helpers ------------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status=200, filename: str | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload too large ({length} bytes)")
        return self.rfile.read(length) if length else b""

    def _read_json(self) -> dict:
        raw = self._read_body()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/", "/index.html"):
                return self._serve_static("index.html")
            if route == "/api/detectors":
                return self._send_json({"detectors": detectors.list_detectors()})
            if route == "/api/job":
                qs = parse_qs(parsed.query)
                job_id = (qs.get("id") or [""])[0]
                with _LOCK:
                    job = _JOBS.get(job_id)
                if job is None:
                    return self._send_json({"error": "unknown job"}, status=404)
                return self._send_json(job)
            if route.startswith("/static/"):
                return self._serve_static(route[len("/static/"):])
            # allow bare asset names (style.css, app.js, hr.html)
            if route.lstrip("/") in ("style.css", "app.js", "hr.html"):
                return self._serve_static(route.lstrip("/"))
            return self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            traceback.print_exc()
            return self._send_json({"error": str(exc)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if route == "/api/upload":
                return self._handle_upload(qs)
            if route == "/api/detect":
                return self._handle_detect()
            if route == "/api/export":
                return self._handle_export()
            if route == "/api/load_npy":
                return self._handle_load_npy(qs)
            return self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            traceback.print_exc()
            return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    # -- endpoints ----------------------------------------------------------
    def _serve_static(self, rel: str):
        # Prevent path traversal outside STATIC_DIR.
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return self._send_json({"error": "not found"}, status=404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), ctype)

    def _handle_upload(self, qs):
        name = (qs.get("name") or ["audio.wav"])[0]
        raw = self._read_body()
        if not raw:
            return self._send_json({"error": "empty upload"}, status=400)
        audio = audio_io.load_audio_from_bytes(raw)
        session_id = uuid.uuid4().hex
        with _LOCK:
            _SESSIONS[session_id] = {"audio": audio, "name": name}
        return self._send_json({
            "id": session_id,
            "name": name,
            "sample_rate": int(audio.hz),
            "num_samples": int(audio.data.shape[0]),
            "duration": float(audio.data.shape[0] / audio.hz),
            "default_bpm_range": list(FETAL_BPM_RANGE),
        })

    def _handle_detect(self):
        body = self._read_json()
        session_id = body.get("id")
        detector_id = body.get("detector")
        bpm_range = body.get("bpm_range") or list(FETAL_BPM_RANGE)
        bpm_range = (float(bpm_range[0]), float(bpm_range[1]))
        if not session_id or not detector_id:
            return self._send_json({"error": "id and detector are required"}, status=400)
        job_id = _start_detection(session_id, detector_id, bpm_range)
        return self._send_json({"job_id": job_id})

    def _handle_export(self):
        body = self._read_json()
        times = np.asarray(body.get("times", []), dtype=float)
        fmt = (body.get("format") or "yaml").lower()
        name = body.get("name") or "beats"
        stem = Path(name).stem or "beats"
        if fmt in ("yaml", "yml"):
            text = audio_io.times_to_yaml(times, name=name)
            return self._send_bytes(text.encode("utf-8"), "application/x-yaml",
                                    filename=f"{stem}_beats.yaml")
        if fmt == "npy":
            data = audio_io.times_to_npy_bytes(times)
            return self._send_bytes(data, "application/octet-stream",
                                    filename=f"{stem}_beats.npy")
        return self._send_json({"error": f"unknown format {fmt!r}"}, status=400)

    def _handle_load_npy(self, qs):
        raw = self._read_body()
        if not raw:
            return self._send_json({"error": "empty upload"}, status=400)
        times = audio_io.npy_bytes_to_times(raw)
        return self._send_json({"times": [float(t) for t in times]})


def main():
    parser = argparse.ArgumentParser(description="Beat-marking web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = parser.parse_args()

    # Warm the detector registry so import errors surface at startup, not mid-click.
    try:
        found = detectors.list_detectors()
        print(f"[beat_app] discovered {len(found)} detector(s): "
              + ", ".join(d["id"] for d in found))
    except Exception:
        traceback.print_exc()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[beat_app] serving on {url}  (Ctrl-C to stop)")
    if not args.no_browser and not os.environ.get("BEAT_APP_NO_BROWSER"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[beat_app] shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
