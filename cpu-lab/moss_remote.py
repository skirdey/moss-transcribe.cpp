"""Pinned hp-fury MOSS client: bounded uploads, complete outputs, no MLX fallback."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

MODEL_SHA256 = "ed6c35d0d527c5d03171c3eb448e2150a42c76a51e3e73aa821e351c3da8307c"
SOURCE_COMMIT = "190a569c13b4b247450f2fb3b2a431244e84833e"
POLICY = "moss-cpp-cpu-marker-retries-v2"


def legacy_signature():
    return f"remote-cpu:{SOURCE_COMMIT}:{MODEL_SHA256}:moss-cpp-cpu-eos-v1"


def configured():
    return bool(os.getenv("MOSS_API_URL") and os.getenv("MOSS_API_TOKEN"))


def signature():
    return f"remote-cpu:{SOURCE_COMMIT}:{MODEL_SHA256}:{POLICY}"


def _request(route, body=None, timeout=270):
    if not configured():
        raise RuntimeError("MOSS remote API is not configured; set MOSS_API_URL and MOSS_API_TOKEN")
    headers = {"Authorization": "Bearer " + os.environ["MOSS_API_TOKEN"]}
    if body is not None:
        headers["Content-Type"] = "audio/wav"
    request = urllib.request.Request(os.environ["MOSS_API_URL"].rstrip("/") + route, data=body, headers=headers)
    # Tailnet traffic goes directly to hp-fury, independent of HTTP proxy envs.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.loads(response.read(4 << 20))
    except urllib.error.HTTPError as exc:
        # Server errors are bounded and contain no request secrets.
        detail = exc.read(4096).decode(errors="replace")
        raise RuntimeError(f"MOSS API returned HTTP {exc.code}: {detail}") from None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"MOSS API is unavailable: {type(exc).__name__}") from None
    return payload


def _validate_identity(identity):
    if not isinstance(identity, dict) or any(identity.get(k) != value for k, value in {
        "modelSha256": MODEL_SHA256, "sourceCommit": SOURCE_COMMIT,
        "policy": POLICY, "device": "cpu", "quantization": "q8_0",
    }.items()):
        raise RuntimeError("MOSS remote model/runtime does not match the pinned identity")


@lru_cache(maxsize=1)
def identity():
    value = _request("/v1/model", timeout=10)["identity"]
    _validate_identity(value)
    return value


def transcribe(path: Path, max_tokens: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    duration = len(samples) / rate
    if not 0 < duration <= 120 or not np.isfinite(samples).all():
        raise ValueError("Remote MOSS expects finite audio of at most 120 seconds")
    mono = samples.mean(axis=1)
    if rate != 16000:
        factor = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // factor, rate // factor)
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(mono, -1, 1), 16000, format="WAV", subtype="PCM_16")
    audio = buffer.getvalue()
    max_tokens = min(8192, max(512, max_tokens))
    payload = _request(f"/v1/audio/transcriptions?max_tokens={max_tokens}", audio)
    _validate_identity(payload.get("identity"))
    if (payload.get("schema") != 1 or payload.get("stopReason") != "eos"
        or payload.get("inputSha256") != hashlib.sha256(audio).hexdigest()
        or abs(payload.get("audioSeconds", -1) - duration) > .001
        or not isinstance(payload.get("text"), str) or not isinstance(payload.get("segments"), list)
        or not isinstance(payload.get("generationTokens"), int)
        or not 0 < payload["generationTokens"] < max_tokens):
        raise RuntimeError("MOSS API returned incomplete or mismatched inference output")
    return payload


class Model:
    """Adapter for the separate podcast processor's generate interface."""
    def generate(self, path, max_tokens=4096, temperature=0.0):
        if temperature != 0.0:
            raise ValueError("Remote MOSS supports greedy decoding only")
        result = transcribe(Path(path), max_tokens)
        return SimpleNamespace(text=result["text"], segments=result["segments"],
                               generation_tokens=result["generationTokens"], prompt_tokens=0,
                               total_time=result["performance"]["totalSeconds"])
