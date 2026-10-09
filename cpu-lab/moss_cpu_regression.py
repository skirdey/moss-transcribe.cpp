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
    args = p.parse_args()
    if args.repeats < 1 or "baseline" not in args.variants or len(args.variants) < 2:
        p.error("Need at least one repeat, baseline, and a candidate")
    root = args.root.resolve()
    results = root / args.name
    results.mkdir(parents=True, exist_ok=True)
    model = Path("/home/stan/hw-audio-bench/models/moss-transcribe-q8_0.gguf")
    baseline = Path("/home/stan/hw-moss-api/bin/moss-transcribe")
    candidate = root / "build/moss-transcribe"
    report_path = results / "report.json"
    if report_path.exists():
        raise ValueError("Use a new --name; never mix measurements from different sessions")
    report = {"protocol": "Sequential fresh processes, 16 threads, Q8, alternating variant order each round, warm filesystem cache. Production auto processing paused; API stays available. Wall includes load. Shared host; no hardware counter support.",
              "modelSha256": digest(model), "binarySha256": {"baseline": digest(baseline), "candidate": digest(candidate)},
              "libraries": {str(path): digest(path) for path in sorted((root / "build").rglob("*.so"))},
              "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / "source", text=True).strip(),
              "cpu": subprocess.check_output(["lscpu"], text=True), "runs": []}
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
        selected_cores = list(cores.values())[:16]
        if len(selected_cores) < 16:
            raise ValueError("This 16-thread benchmark requires 16 allowed physical cores")
        affinity = {"OMP_PROC_BIND":"spread","OMP_PLACES":",".join("{"+str(cpu)+"}" for cpu in selected_cores)}
        report["physicalCoreAffinity"] = selected_cores
    report["timeoutSeconds"] = args.timeout
    # One discarded warmup for each binary, with full inference.
    audio0 = Path("/home/stan/hw-audio-bench/audio") / (args.cases[0] + ".wav")
    for binary, opt, lib in ([] if args.skip_warmup else [(baseline,"0","/home/stan/hw-moss-api/lib"), (candidate,"0","")]):
        subprocess.run([str(binary), "transcribe", str(model), str(audio0), "--max-new", "4096"],
                       env={**os.environ,**affinity,"MTD_DEVICE":"cpu","MTD_THREADS":"16","OMP_NUM_THREADS":"16","MTD_CPU_OPT":opt,"MTD_LOOP_GUARD":"1","MTD_REPETITION_PENALTY":"1.0","LD_LIBRARY_PATH":lib},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=args.timeout)
    for repeat in range(args.repeats):
        for case in args.cases:
            audio = Path("/home/stan/hw-audio-bench/audio") / (case + ".wav")
            with wave.open(str(audio)) as w:
                duration = w.getnframes() / w.getframerate()
            variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            for variant in variants:
                key = f"{case}-{variant}-{repeat}"
                binary = baseline if variant == "baseline" else candidate
                env = {**os.environ,**affinity,"MTD_DEVICE":"cpu","MTD_THREADS":"16","OMP_NUM_THREADS":"16","MTD_CPU_OPT":"0" if variant == "baseline" else variant,"MTD_LOOP_GUARD":"1","MTD_REPETITION_PENALTY":"1.0","LD_LIBRARY_PATH":"/home/stan/hw-moss-api/lib" if variant == "baseline" else ""}
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
    report["gate"] = evaluate(report["runs"])
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report["gate"],indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
