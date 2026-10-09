"""Paired CPU speed and exact-output regression gate for MOSS (stdlib only).

Run on hp-fury. Raw outputs remain in the experiment directory. Report contains
hashes, timings and completion status. Exit 1 rejects changed output, incomplete
generation, or >5% median slowdown on any case. This proves sample equivalence,
not corpus-level accuracy. Public reference scoring is a separate check.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import threading
import time
import wave


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(runs, baseline="baseline", tolerance=0.05):
    failures, comparisons = [], []
    if not runs:
        failures.append("No benchmark measurements")
    for case in sorted({r["case"] for r in runs}):
        reference = [r for r in runs if r["case"] == case and r["variant"] == baseline]
        if not reference:
            failures.append(f"{case}: missing baseline")
            continue
        def fingerprint(r):
            return (r["inputSha256"],r["duration"],r["outputSha256"],r["tokens"],r["stop"],r["returncode"])
        fingerprints = {fingerprint(r) for r in reference}
        if len(fingerprints) != 1 or not all(r["complete"] and not r.get("concurrentMoss") for r in reference):
            failures.append(f"{case}: unstable or incomplete baseline")
        base_time = statistics.median(r["wallSeconds"] for r in reference)
        variants = {r["variant"] for r in runs if r["case"] == case} - {baseline}
        if not variants:
            failures.append(f"{case}: missing candidate")
        for variant in sorted(variants):
            selected = [r for r in runs if r["case"] == case and r["variant"] == variant]
            parity = all(r["complete"] and not r.get("concurrentMoss") and fingerprint(r) in fingerprints for r in selected)
            median = statistics.median(r["wallSeconds"] for r in selected)
            ratio = median / base_time
            comparisons.append({"case": case, "variant": variant, "repeats": len(selected), "baselineSeconds": base_time,
                                "candidateSeconds": median, "speedup": base_time / median, "exactOutputParity": parity,
                                "maxRssKiB": max(r["maxRssKiB"] for r in selected), "speedPass": ratio <= 1+tolerance})
            if not parity:
                failures.append(f"{case}/{variant}: output or completion regression")
            if ratio > 1 + tolerance:
                failures.append(f"{case}/{variant}: median slowdown {100*(ratio-1):.2f}%")
    return {"passed": not failures, "failures": failures, "comparisons": comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=["meeting-60s", "dinner-60s", "meeting-120s", "silence-60s"])
    p.add_argument("--variants", nargs="+", default=["baseline", "0", "1", "2", "4", "7"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--name", default="regression")
    p.add_argument("--skip-warmup", action="store_true", help="For a follow-on stress check after the same binaries/model were warmed in a preceding suite")
    p.add_argument("--pin-physical", action="store_true", help="Bind one OpenMP worker to each physical core, avoiding sibling/core collisions")
    p.add_argument("--timeout",type=float,default=300.)
    p.add_argument("--reference-root",type=Path,default=Path("/home/stan/hw-moss-cache-layout"),help="Fixed validated transposed-cache build; variant cache-reference uses bit 16")
    p.add_argument("--model",type=Path,default=Path("/home/stan/hw-audio-bench/models/moss-transcribe-q8_0.gguf"))
    p.add_argument("--audio-dir",type=Path,default=Path("/home/stan/hw-audio-bench/audio"))
    p.add_argument("--baseline",type=Path,default=Path("/home/stan/hw-moss-api/bin/moss-transcribe"))
    p.add_argument("--baseline-lib-dir",default="/home/stan/hw-moss-api/lib")
    p.add_argument("--threads",type=int,default=16)
    args = p.parse_args()
    if args.repeats < 1 or args.threads < 1 or "baseline" not in args.variants or len(args.variants) < 2:
        p.error("Need at least one repeat, baseline, and a candidate")
    root = args.root.resolve()
    results = root / args.name
    results.mkdir(parents=True, exist_ok=True)
    model, baseline = args.model.resolve(), args.baseline.resolve()
    reference = args.reference_root.resolve() / "build/moss-transcribe"
    def configuration(variant):
        if variant == "baseline":
            return baseline, "0", args.baseline_lib_dir
        if variant == "cache-reference":
            return reference, "16", ""
        if not variant.isdecimal():
            raise ValueError(f"Unknown variant: {variant}")
        return candidate, variant, ""
    candidate = root / "build/moss-transcribe"
    report_path = results / "report.json"
    if report_path.exists():
        raise ValueError("Use a new --name; never mix measurements from different sessions")
    report = {"protocol": f"Sequential fresh processes, {args.threads} threads, pinned model, alternating variant order each round, warm filesystem cache. Production auto processing paused; API stays available. Wall includes load. Shared host; no hardware counter support.",
              "modelSha256": digest(model), "binarySha256": {"baseline": digest(baseline), "candidate": digest(candidate)},
              "libraries": {str(path): digest(path) for path in sorted((root / "build").rglob("*.so"))},
              "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / "source", text=True).strip(),
              "cpu": subprocess.check_output(["lscpu"], text=True), "runs": []}
    report["harnessSha256"] = digest(Path(__file__))
    report["threads"] = args.threads
    report["sourceSha256"] = {str(path.relative_to(root / "source")):digest(path)
                              for path in sorted((root / "source/src").glob("*")) if path.is_file()}
    report["sourceSha256"]["CMakeLists.txt"] = digest(root / "source/CMakeLists.txt")
    if "cache-reference" in args.variants:
        report["binarySha256"]["cache-reference"] = digest(reference)
        report["referenceRoot"] = str(args.reference_root.resolve())
        report["referenceLibraries"] = {str(path):digest(path) for path in sorted((args.reference_root / "build").rglob("*.so"))}
    report["productionLibraries"] = {str(path):digest(path) for path in sorted(Path(args.baseline_lib_dir).glob("*.so"))}
    report["warmupDiscarded"] = not args.skip_warmup
    if args.skip_warmup:
        report["protocol"] = report["protocol"].replace("discarded warmups", "warmups skipped for this follow-on suite after priming in preceding pilot")
    report["ggmlCommit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root / "source/third_party/ggml",text=True).strip()
    affinity = {}
    if args.pin_physical:
        cores = {}
        for line in subprocess.check_output(["lscpu","-p=CPU,CORE,SOCKET"],text=True).splitlines():
            if line.startswith("#"):
                continue
            cpu,core,socket = map(int,line.split(","))
            if cpu in os.sched_getaffinity(0):
                cores.setdefault((socket,core),cpu)
        selected_cores = list(cores.values())[:args.threads]
        if len(selected_cores) < args.threads:
            raise ValueError("Not enough allowed physical cores for the requested thread count")
        affinity = {"OMP_PROC_BIND":"spread","OMP_PLACES":",".join("{"+str(cpu)+"}" for cpu in selected_cores)}
        report["physicalCoreAffinity"] = selected_cores
    report["timeoutSeconds"] = args.timeout
    # One discarded warmup for each binary, with full inference.
    audio0 = args.audio_dir / (args.cases[0] + ".wav")
    for binary, opt, lib in ([] if args.skip_warmup else [configuration(v) for v in args.variants]):
        subprocess.run([str(binary), "transcribe", str(model), str(audio0), "--max-new", "4096"],
                       env={**os.environ,**affinity,"MTD_DEVICE":"cpu","MTD_THREADS":str(args.threads),"OMP_NUM_THREADS":str(args.threads),"MTD_CPU_OPT":opt,"MTD_LOOP_GUARD":"1","MTD_REPETITION_PENALTY":"1.0","LD_LIBRARY_PATH":lib},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=args.timeout)
    for repeat in range(args.repeats):
        for case in args.cases:
            audio = args.audio_dir / (case + ".wav")
            with wave.open(str(audio)) as w:
                duration = w.getnframes() / w.getframerate()
            variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            for variant in variants:
                key = f"{case}-{variant}-{repeat}"
                binary, opt, lib = configuration(variant)
                env = {**os.environ,**affinity,"MTD_DEVICE":"cpu","MTD_THREADS":str(args.threads),"OMP_NUM_THREADS":str(args.threads),"MTD_CPU_OPT":opt,"MTD_LOOP_GUARD":"1","MTD_REPETITION_PENALTY":"1.0","LD_LIBRARY_PATH":lib}
                raw, log, timer = [results / (key + ext) for ext in (".txt", ".log", ".time")]
                cmd = ["/usr/bin/time", "-f", "%M", "-o", str(timer), str(binary), "transcribe", str(model), str(audio), "--max-new", "4096"]
                start = time.perf_counter()
                load_before = os.getloadavg()
                stopped, concurrent = threading.Event(), threading.Event()
                def monitor():
                    while not stopped.wait(0.5):
                        processes = subprocess.run(["pgrep", "-x", "moss-transcribe"],capture_output=True,text=True)
                        if len(processes.stdout.splitlines()) > 1:
                            concurrent.set()
                monitor_thread = threading.Thread(target=monitor,daemon=True)
                monitor_thread.start()
                with raw.open("w") as out, log.open("w") as err:
                    proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
                    try:
                        rc = proc.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        rc = 124
                stopped.set()
                monitor_thread.join()
                wall = time.perf_counter()-start
                stderr = log.read_text()
                generation = re.search(r"BENCH_GENERATION tokens=(\d+) stop=(\w+)", stderr)
                timing = re.search(r"BENCH_TIMING load=([\d.]+) inference=([\d.]+)", stderr)
                tokens, stop = (int(generation[1]), generation[2]) if generation else (None, None)
                complete = stop == "eos" and tokens < 4096 and (rc == 0 or (rc == 1 and not raw.read_text().strip()))
                rss = re.findall(r"^\d+$", timer.read_text(), re.M)
                row = {"case":case,"variant":variant,"repeat":repeat,"inputSha256":digest(audio),"duration":duration,"outputSha256":digest(raw),"returncode":rc,"tokens":tokens,"stop":stop,"complete":complete,"concurrentMoss":concurrent.is_set(),"wallSeconds":wall,"maxRssKiB":int(rss[-1]) if rss else 0,
                       "hostLoadBefore":load_before,"hostLoadAfter":os.getloadavg(),
                       "timing": {"load":float(timing[1]),"inference":float(timing[2])} if timing else None,
                       "profile": {k:float(v) for k,v in re.findall(r"(prefill|embedding|decoder|logits)=([\d.]+)",stderr)}}
                report["runs"].append(row)
                report_path.write_text(json.dumps(report,indent=2))
                print(json.dumps(row),flush=True)
    artifact_hashes = {str(model):report["modelSha256"], str(baseline):report["binarySha256"]["baseline"],
                       str(candidate):report["binarySha256"]["candidate"], **report["libraries"], **report["productionLibraries"]}
    if "cache-reference" in args.variants:
        artifact_hashes.update({str(reference):report["binarySha256"]["cache-reference"], **report["referenceLibraries"]})
    artifact_hashes.update({str(root / "source" / name):value for name,value in report["sourceSha256"].items()})
    changed = [path for path,value in artifact_hashes.items() if digest(path) != value]
    report["artifactGate"] = {"passed":not changed,"changed":changed}
    report["gate"] = evaluate(report["runs"])
    if "cache-reference" in args.variants:
        compared = [r for r in report["runs"] if r["variant"] != "baseline"]
        report["cacheReferenceGate"] = evaluate(compared, baseline="cache-reference")
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report["gate"],indent=2))
    return 0 if report["artifactGate"]["passed"] and report["gate"]["passed"] and report.get("cacheReferenceGate",{"passed":True})["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
