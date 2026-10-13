import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request
import wave

import pytest
import moss_output

spec = importlib.util.spec_from_file_location("moss_api", Path(__file__).resolve().parent / "moss_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def wav_bytes(rate=16000):
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x01\x00" * rate)
    return target.getvalue()


def test_api_authentication_busy_worker_and_bounded_input():
    runtime = SimpleNamespace(token="x" * 48, slot=threading.Lock(), identity={})
    server = api.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    server.runtime = runtime
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(url + "/health")
        assert denied.value.code == 401
        headers = {"Authorization": "Bearer " + runtime.token, "Content-Type": "audio/wav"}
        with urllib.request.urlopen(urllib.request.Request(url + "/health", headers=headers)) as response:
            assert json.load(response)["status"] == "ok"
        runtime.slot.acquire()
        with pytest.raises(urllib.error.HTTPError) as busy:
            urllib.request.urlopen(urllib.request.Request(url + "/v1/audio/transcriptions", data=wav_bytes(), headers=headers))
        assert busy.value.code == 503
        runtime.slot.release()
        with pytest.raises(urllib.error.HTTPError) as invalid:
            urllib.request.urlopen(urllib.request.Request(url + "/v1/audio/transcriptions?max_tokens=99999", data=wav_bytes(), headers=headers))
        assert invalid.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("stop", ["eos", "token_limit"])
def test_runtime_rejects_partial_generation_and_keeps_complete_text(monkeypatch, stop):
    runtime = object.__new__(api.Runtime)
    runtime.binary = Path("unused")
    runtime.model = Path("unused.gguf")
    runtime.identity = {}
    class Process:
        returncode = 0
        def communicate(self, timeout):
            return "[0][S01]First[.5][S02]Second[1]", f"BENCH_GENERATION tokens=20 stop={stop}\nBENCH_TIMING load=0.1 inference=1.2"
    monkeypatch.setattr(api.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setitem(sys.modules, "moss_output", moss_output)
    if stop == "token_limit":
        with pytest.raises(RuntimeError, match="partial output"):
            runtime.infer(wav_bytes(), 512)
    else:
        result = runtime.infer(wav_bytes(), 512)
        assert [row["text"] for row in result["segments"]] == ["First", "Second"]
        assert result["inputSha256"] == hashlib.sha256(wav_bytes()).hexdigest()
    with pytest.raises(ValueError, match="16 kHz"):
        runtime.infer(wav_bytes(48000), 512)


def test_loop_retry_keeps_original_audio_and_discards_aborted_output(monkeypatch):
    runtime = object.__new__(api.Runtime)
    runtime.binary = Path("unused")
    runtime.model = Path("unused.gguf")
    runtime.identity = {}
    calls = []
    class Process:
        returncode = 0
        def __init__(self, attempt): self.attempt = attempt
        def communicate(self, timeout):
            if self.attempt == 1:
                return "[0][S01][0]", "BENCH_GENERATION tokens=128 stop=empty_marker_loop\nBENCH_TIMING load=0.1 inference=1.0"
            return "[0][S01]Recovered[1]", "BENCH_GENERATION tokens=12 stop=eos\nBENCH_TIMING load=0.1 inference=2.0"
    def popen(args, **kwargs):
        calls.append((Path(args[3]).read_bytes(), kwargs["env"]["MTD_REPETITION_PENALTY"]))
        return Process(len(calls))
    monkeypatch.setattr(api.subprocess, "Popen", popen)
    monkeypatch.setitem(sys.modules, "moss_output", moss_output)
    result = runtime.infer(wav_bytes(), 512)
    assert [p for _, p in calls] == ["1.0", "1.1"]
    assert all(audio == wav_bytes() for audio, _ in calls)
    assert result["text"] == "[0][S01]Recovered[1]"
    assert result["performance"]["inferenceSeconds"] == 3.0
    assert result["performance"]["attempts"][0]["stopReason"] == "empty_marker_loop"


def test_eos_only_is_complete_empty_output_despite_upstream_cli_exit_one(monkeypatch):
    runtime = object.__new__(api.Runtime)
    runtime.binary = Path("unused")
    runtime.model = Path("unused.gguf")
    runtime.identity = {}
    class Process:
        returncode = 1
        def communicate(self, timeout):
            return "", "BENCH_GENERATION tokens=1 stop=eos\ntranscription failed"
    monkeypatch.setattr(api.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setitem(sys.modules, "moss_output", moss_output)
    result = runtime.infer(wav_bytes(), 512)
    assert result["text"] == "" and result["segments"] == []
    assert result["stopReason"] == "eos" and result["generationTokens"] == 1


def test_inference_environment_forces_validated_profile():
    base={"MTD_CPU_OPT":"786480", "MTD_THREADS_WHISPER":"1", "MTD_THREADS_DECODE":"2",
          "OMP_DYNAMIC":"TRUE", "OMP_NUM_THREADS":"32", "OMP_WAIT_POLICY":"PASSIVE",
          "MTD_TRACE_TOKENS":"1", "LD_LIBRARY_PATH":"example-libraries"}
    result=api.inference_environment(base,1.1)
    assert result["MTD_CPU_OPT"]=="48" and result["OMP_NUM_THREADS"]=="16"
    assert result["OMP_DYNAMIC"]=="FALSE" and result["MTD_TRACE_TOKENS"]=="0"
    assert result["MTD_REPETITION_PENALTY"]=="1.1"
    assert "MTD_THREADS_WHISPER" not in result and "MTD_THREADS_DECODE" not in result
    assert "OMP_WAIT_POLICY" not in result and result["LD_LIBRARY_PATH"]==base["LD_LIBRARY_PATH"]
    assert base["MTD_CPU_OPT"]=="786480"


def test_startup_rejects_wrong_deployment_binary(monkeypatch,tmp_path):
    binary=tmp_path/'program';binary.write_bytes(b'example program')
    model=tmp_path/'model.gguf';model.write_bytes(b'example model')
    monkeypatch.setenv('MOSS_API_TOKEN','x'*48)
    monkeypatch.setenv('MOSS_CPP_BINARY',str(binary));monkeypatch.setenv('MOSS_CPP_MODEL',str(model))
    monkeypatch.setattr(api,'MODEL_SHA256',api.digest(model))
    monkeypatch.setenv('MOSS_CPP_EXPECTED_SHA256','0'*64)
    with pytest.raises(ValueError,match='deployment digest'):api.Runtime()
    monkeypatch.setenv('MOSS_CPP_EXPECTED_SHA256',api.digest(binary))
    runtime=api.Runtime()
    assert runtime.identity['cpuOpt']==48 and runtime.identity['binarySha256']==api.digest(binary)
