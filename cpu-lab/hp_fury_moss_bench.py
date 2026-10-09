"""Reproducible CPU-only moss-transcribe.cpp benchmark (Python stdlib only).

Run on hp-fury with --root /home/stan/hw-audio-bench. Each run starts a new
process: wall time includes model load, and subsequent runs have warm OS caches.
The optional instrumentation adds timing/count logs without changing inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import wave
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instrument(root):
    source = root / "moss-transcribe.cpp/src/generate.cpp"
    text = source.read_text()
    if "BENCH_GENERATION" not in text:
        # The EOS check is immediately before the token-cap check upstream.
        old = "if (t == eos) break;"
        assert text.count(old) == 1
        text = text.replace(old, 'if (t == eos) { MT_LOGI("BENCH_GENERATION tokens=%zu stop=eos", ids.size()); break; }')
        old = "if ((int)ids.size() >= max_new) break;"
        assert text.count(old) == 1
        text = text.replace(old, 'if ((int)ids.size() >= max_new) { MT_LOGI("BENCH_GENERATION tokens=%zu stop=token_limit", ids.size()); break; }')
        source.write_text(text)
    source = root / "moss-transcribe.cpp/src/cli.cpp"
    text = source.read_text()
    if "BENCH_TIMING" not in text:
        text = '#include <chrono>\n' + text
        old = "mt::ModelLoader m;\n    if (!m.load(gguf))"
        assert text.count(old) == 1
        text = text.replace(old, "auto bench_start = std::chrono::steady_clock::now();\n    " + old)
        old = "std::string text = mt::transcribe_wav(m, wav, max_new);"
        assert text.count(old) == 1
        text = text.replace(old, 'auto bench_loaded = std::chrono::steady_clock::now();\n    ' + old + '\n    auto bench_end = std::chrono::steady_clock::now();\n    std::fprintf(stderr, "BENCH_TIMING load=%.6f inference=%.6f\\n",\n        std::chrono::duration<double>(bench_loaded-bench_start).count(),\n        std::chrono::duration<double>(bench_end-bench_loaded).count());')
        source.write_text(text)
    patch = subprocess.check_output(["git", "diff"], cwd=root / "moss-transcribe.cpp", text=True)
    (root / "results/instrumentation.patch").write_text(patch)


def add_recovery(root):
    """Port the project's empty-marker guard and 100-token retry penalty."""
    source = root / "moss-transcribe.cpp/src/generate.cpp"
    text = source.read_text()
    if "MTD_REPETITION_PENALTY" in text:
        return
    helper = r'''
// Dense empty timestamp/speaker turns are decoder loops. Lexical repetitions
// and event labels break the run and are never removed by this guard.
static bool empty_marker_loop(const std::string& text) {
    static const std::regex marker(R"(\[(\d+(?:\.\d+)?)\]\s*\[S\d+\]\s*\[(\d+(?:\.\d+)?)\])");
    std::deque<std::pair<double,double>> run;
    size_t previous_end = 0;
    for (std::sregex_iterator it(text.begin(), text.end(), marker), end; it != end; ++it) {
        const auto& match = *it;
        auto gap = text.substr(previous_end, static_cast<size_t>(match.position()) - previous_end);
        if (gap.find_first_not_of(" \t\r\n") != std::string::npos) run.clear();
        try { run.emplace_back(std::stod(match[1].str()), std::stod(match[2].str())); }
        catch (...) { run.clear(); }
        if (run.size() > 12) run.pop_front();
        previous_end = static_cast<size_t>(match.position() + match.length());
        if (run.size() == 12) {
            double lo = run.front().first, hi = lo;
            for (const auto& pair : run) { lo = std::min({lo,pair.first,pair.second}); hi = std::max({hi,pair.first,pair.second}); }
            if (hi - lo <= 2.0) return true;
        }
    }
    return false;
}

'''
    text = '#include <algorithm>\n#include <cstdlib>\n#include <deque>\n#include <regex>\n#include <unordered_set>\n#include "tokenizer.hpp"\n' + text
    text = text.replace("std::vector<int32_t> greedy_generate(", helper + "std::vector<int32_t> greedy_generate(", 1)
    old = "ids.reserve((size_t)max_new);"
    assert text.count(old) == 1
    text = text.replace(old, '''const char* penalty_env = std::getenv("MTD_REPETITION_PENALTY");
    float penalty = penalty_env ? static_cast<float>(std::atof(penalty_env)) : 1.0f;
    penalty = std::max(1.0f, std::min(1.5f, penalty));
    const bool guarded = std::getenv("MTD_LOOP_GUARD") != nullptr;
    Tokenizer guard_tokenizer;
    if (guarded && !guard_tokenizer.load(m)) return ids;
    ''' + old)
    old = "int t = argmax_first(logits);"
    assert text.count(old) == 1
    text = text.replace(old, '''if (penalty > 1.0f) {
            std::unordered_set<int32_t> recent;
            for (int i = std::max(0, static_cast<int>(ids.size()) - 100); i < static_cast<int>(ids.size()); ++i) recent.insert(ids[i]);
            for (int32_t id : recent) if (id >= 0 && id < static_cast<int32_t>(logits.size())) {
                logits[id] = logits[id] > 0 ? logits[id] / penalty : logits[id] * penalty;
            }
        }
        ''' + old)
    old = 'if ((int)ids.size() >= max_new) { MT_LOGI("BENCH_GENERATION tokens=%zu stop=token_limit", ids.size()); break; }'
    assert text.count(old) == 1
    text = text.replace(old, old + '''
        if (guarded && ids.size() >= 128 && ids.size() % 32 == 0) {
            auto begin = ids.begin() + std::max(0, static_cast<int>(ids.size()) - 512);
            std::vector<int32_t> tail(begin, ids.end());
            if (empty_marker_loop(guard_tokenizer.decode(tail))) {
                MT_LOGI("BENCH_GENERATION tokens=%zu stop=empty_marker_loop", ids.size());
                break;
            }
        }''')
    source.write_text(text)
    patch = subprocess.check_output(["git", "diff"], cwd=root / "moss-transcribe.cpp", text=True)
    (root / "results/instrumentation.patch").write_text(patch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--suite", choices=["pilot", "matrix", "confirm"], default="pilot")
    parser.add_argument("--quant", default="q8_0")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.instrument:
        instrument(root)
        if args.recovery:
            add_recovery(root)
        return
    results = root / "results"
    report_path = results / "benchmark.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        repo = root / "moss-transcribe.cpp"
        report = {
            "createdAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": subprocess.check_output(["hostname"], text=True).strip(),
            "cpu": subprocess.check_output(["lscpu"], text=True),
            "source": "https://github.com/localai-org/moss-transcribe.cpp",
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "ggmlCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo / "third_party/ggml", text=True).strip(),
            "modelRepository": "mudler/moss-transcribe.cpp-gguf",
            "modelRevision": "54e4bbd17da3f84adf1c1bcf7791b9b9266f741e",
            "protocol": "Sequential CPU runs; fresh process per run; wall includes load; no production cache; 300-second timeout; max-new=4096; warm OS cache after first access.",
            "limitations": ["Short real-audio samples without a scored human reference; no DER/WER claim.", "Anonymous speaker IDs are scoped to each inference window.", "This is a quantized C++ port, not the project's MLX weights/runtime.", "No reference PyTorch run: no measured speedup against PyTorch.", "Shared machine; other workloads may affect results."],
            "runs": [],
        }
    if args.suite == "pilot":
        cases = [("q8_0", 8, "meeting-60s", "pilot")]
    elif args.suite == "matrix":
        cases = [(q, t, "meeting-60s", "matrix") for q in ("q8_0", "q5_0") for t in (4, 8, 16)]
    else:
        cases = [(args.quant, args.threads, "meeting-60s", f"repeat{i}") for i in range(2)]
        cases += [(args.quant, args.threads, clip, "confirm") for clip in ("meeting-120s", "dinner-60s")]
    for quant, threads, clip, label in cases:
        key = f"{quant}-{threads}t-{clip}-{label}"
        if any(r["id"] == key for r in report["runs"]):
            continue
        model = root / f"models/moss-transcribe-{quant}.gguf"
        audio = root / f"audio/{clip}.wav"
        with wave.open(str(audio)) as wav:
            duration = wav.getnframes() / wav.getframerate()
        command = [str(root / "moss-transcribe.cpp/build/moss-transcribe"), "transcribe", str(model), str(audio), "--max-new", "4096"]
        timing = results / f"{key}.time.json"
        raw = results / f"{key}.txt"
        logs = results / f"{key}.log"
        env = {**os.environ, "MTD_DEVICE": "cpu", "MTD_THREADS": str(threads), "OMP_NUM_THREADS": str(threads)}
        wrapped = ["/usr/bin/time", "-f", '{"userSeconds":%U,"systemSeconds":%S,"maxRssKiB":%M,"cpuPercent":"%P"}', "-o", str(timing), *command]
        started = time.perf_counter()
        print(json.dumps({"started": key}), flush=True)
        with raw.open("w") as stdout, logs.open("w") as stderr:
            try:
                proc = subprocess.run(["timeout", "--signal=TERM", "--kill-after=10", "300", *wrapped], env=env, stdout=stdout, stderr=stderr)
                return_code = proc.returncode
            except OSError as exc:
                return_code = -1
                stderr.write(str(exc))
        wall = time.perf_counter() - started
        text = raw.read_text()
        log = logs.read_text()
        generation = re.search(r"BENCH_GENERATION tokens=(\d+) stop=(\w+)", log)
        phases = re.search(r"BENCH_TIMING load=([\d.]+) inference=([\d.]+)", log)
        segments = [{"start": float(m[1]), "speaker": m[2], "text": m[3].strip(), "end": float(m[4])}
                    for m in re.finditer(r"\[([\d.]+)\]\[(S\d+)\](.*?)\[([\d.]+)\]", text, re.S)]
        nonempty = [s for s in segments if s["text"]]
        row = {"id": key, "quant": quant, "threads": threads, "audio": clip, "audioSeconds": duration,
               "inputSha256": sha256(audio), "modelSha256": sha256(model), "modelBytes": model.stat().st_size,
               "wallSeconds": wall, "wallRtf": wall / duration, "returnCode": return_code,
               "generationTokens": int(generation[1]) if generation else None,
               "stopReason": generation[2] if generation else "unknown",
               "completed": return_code == 0 and generation is not None and generation[2] == "eos",
               "segments": segments, "nonemptySegments": len(nonempty),
               "speakers": sorted(set(s["speaker"] for s in nonempty)),
               "invalidTimingSegments": sum(s["end"] <= s["start"] or s["start"] < 0 or s["end"] > duration + 0.5 for s in segments),
               "rawText": text, "logFile": logs.name, "command": command,
               "loadAverageAfter": os.getloadavg()}
        if phases:
            row.update(loadSeconds=float(phases[1]), inferenceSeconds=float(phases[2]), inferenceRtf=float(phases[2]) / duration)
        try:
            row.update(json.loads(timing.read_text().splitlines()[-1]))
        except (OSError, ValueError, IndexError):
            pass
        report["runs"].append(row)
        temporary = report_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(report_path)
        print(json.dumps({k: row.get(k) for k in ("id", "wallSeconds", "inferenceSeconds", "wallRtf", "maxRssKiB", "completed", "generationTokens", "speakers", "invalidTimingSegments")}), flush=True)


if __name__ == "__main__":
    main()
