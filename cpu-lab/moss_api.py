"""Private, bounded CPU MOSS API. Run under the supplied systemd user unit.

POST /v1/audio/transcriptions accepts mono 16 kHz PCM16 WAV bytes. Requests
are authenticated and inference is serialized; busy callers receive 503.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

MAX_BYTES = 4_000_000
MAX_SECONDS = 120
POLICY = "moss-cpp-cpu-marker-retries-v2"
MODEL_SHA256 = "ed6c35d0d527c5d03171c3eb448e2150a42c76a51e3e73aa821e351c3da8307c"
COMMIT = "190a569c13b4b247450f2fb3b2a431244e84833e"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


class Runtime:
    def __init__(self):
        self.token = os.environ["MOSS_API_TOKEN"]
        if len(self.token) < 32:
            raise ValueError("API token must contain at least 32 characters")
        self.binary = Path(os.environ["MOSS_CPP_BINARY"])
        self.model = Path(os.environ["MOSS_CPP_MODEL"])
        if digest(self.model) != MODEL_SHA256:
            raise ValueError("MOSS weights do not match pinned Q8 digest")
        self.slot = threading.Lock()
        self.identity = {"backend": "moss-transcribe.cpp", "device": "cpu", "quantization": "q8_0",
                         "sourceCommit": COMMIT, "modelRepository": "mudler/moss-transcribe.cpp-gguf",
                         "modelRevision": "54e4bbd17da3f84adf1c1bcf7791b9b9266f741e",
                         "modelSha256": MODEL_SHA256, "binarySha256": digest(self.binary),
                         "threads": 16, "policy": POLICY, "maxAudioSeconds": MAX_SECONDS}

    def infer(self, audio, max_tokens):
        with wave.open(io.BytesIO(audio)) as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
                raise ValueError("Expected mono 16 kHz PCM16 WAV")
            frames = wav.getnframes()
            duration = frames / 16000
            if not 0 < duration <= MAX_SECONDS:
                raise ValueError("Expected audio of at most 120 seconds")
            if len(wav.readframes(frames)) != frames * 2:
                raise ValueError("Truncated WAV payload")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="moss-api-") as temporary:
            path = Path(temporary) / "input.wav"
            path.write_bytes(audio)
            attempts = []
            for penalty in (1.0, 1.1, 1.5):
                remaining = 240 - (time.perf_counter() - started)
                if remaining <= 0:
                    raise RuntimeError("MOSS inference exceeded 240 seconds")
                attempt_started = time.perf_counter()
                env = {**os.environ, "MTD_DEVICE": "cpu", "MTD_THREADS": "16", "OMP_NUM_THREADS": "16",
                       "MTD_LOOP_GUARD": "1", "MTD_REPETITION_PENALTY": str(penalty)}
                proc = subprocess.Popen([str(self.binary), "transcribe", str(self.model), str(path),
                                         "--max-new", str(max_tokens)], env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, start_new_session=True)
                try:
                    raw, log = proc.communicate(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate()
                    raise RuntimeError("MOSS inference exceeded 240 seconds") from None
                generated = re.search(r"BENCH_GENERATION tokens=(\d+) stop=(\w+)", log)
                timing = re.search(r"BENCH_TIMING load=([\d.]+) inference=([\d.]+)", log)
                attempts.append({"repetitionPenalty": penalty,
                                 "seconds": time.perf_counter() - attempt_started,
                                 "modelSetupSeconds": float(timing[1]) if timing else 0,
                                 "inferenceSeconds": float(timing[2]) if timing else time.perf_counter() - attempt_started,
                                 "stopReason": generated[2] if generated else "failed",
                                 "generationTokens": int(generated[1]) if generated else 0,
                                 "rawText": raw.strip()})
                # Upstream CLI returns 1 for an empty decoded string even when
                # the model completed normally with EOS. This is a valid empty
                # result; missing EOS and all other failures stay rejected.
                empty_complete = (proc.returncode == 1 and not raw.strip()
                                  and "transcription failed" in log)
                if (proc.returncode == 0 or empty_complete) and generated and generated[2] == "eos":
                    break
                if not generated or generated[2] not in ("token_limit", "empty_marker_loop"):
                    raise RuntimeError("MOSS inference failed; partial output was rejected")
            else:
                raise RuntimeError("MOSS exhausted bounded retries; partial output was rejected")
        # Import the same complete-text parser used by the laptop; no SDK regex.
        import moss_output
        segments = moss_output.parse(raw.strip(), duration)
        elapsed = time.perf_counter() - started
        return {"schema": 1, "identity": self.identity, "inputSha256": hashlib.sha256(audio).hexdigest(),
                "audioSeconds": duration, "text": raw.strip(), "segments": segments,
                "generationTokens": int(generated[1]), "stopReason": "eos",
                "performance": {"backend": "remote-cpu", "policy": POLICY, "totalSeconds": elapsed,
                                "attempts": attempts,
                                "inferenceSeconds": sum(a["inferenceSeconds"] for a in attempts),
                                "modelSetupSeconds": sum(a["modelSetupSeconds"] for a in attempts)}}


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if status == 503:
            self.send_header("Retry-After", "5")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        value = self.headers.get("Authorization", "")
        if not hmac.compare_digest(value, "Bearer " + self.server.runtime.token):
            self.send_json(401, {"error": "Unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if self.path in ("/health", "/v1/model"):
            self.send_json(200, {"status": "ok", "busy": self.server.runtime.slot.locked(),
                                 "identity": self.server.runtime.identity})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorized():
            return
        parts = urlsplit(self.path)
        if parts.path != "/v1/audio/transcriptions":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            max_tokens = int(parse_qs(parts.query).get("max_tokens", ["4096"])[0])
            if not 0 < length <= MAX_BYTES or not 512 <= max_tokens <= 8192:
                raise ValueError("Invalid payload size or token limit")
            if self.headers.get("Content-Type", "").split(";")[0] != "audio/wav":
                raise ValueError("Expected Content-Type: audio/wav")
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        if not self.server.runtime.slot.acquire(blocking=False):
            self.send_json(503, {"error": "MOSS worker is busy; retry later"})
            return
        try:
            audio = self.rfile.read(length)
            if len(audio) != length:
                raise ValueError("Incomplete upload")
            result = self.server.runtime.infer(audio, max_tokens)
            self.send_json(200, result)
        except (ValueError, wave.Error, EOFError) as exc:
            self.send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.send_json(422, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            self.server.runtime.slot.release()

    def log_message(self, fmt, *args):
        # Avoid query parameters and never log audio/transcript/token contents.
        print(json.dumps({"event": "http", "method": self.command,
                          "path": urlsplit(self.path).path, "client": self.client_address[0]}), flush=True)


def main():
    runtime = Runtime()
    server = ThreadingHTTPServer((os.environ["MOSS_BIND_ADDRESS"], int(os.getenv("MOSS_API_PORT", "8787"))), Handler)
    server.daemon_threads = True
    server.runtime = runtime
    print(json.dumps({"event": "ready", "identity": runtime.identity}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
