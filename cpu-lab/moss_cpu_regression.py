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
from moss_ngram_audit import read_trace


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def variant_settings(variant, default_threads):
    """Numeric OPT, OPT@THREADS, or OPT@THREADS-passive; old labels still work."""
    match = re.fullmatch(r"([0-9]+)(?:@([0-9]+))?(-passive)?", variant)
    if not match:
        raise ValueError(f"Unknown variant: {variant}")
    threads = int(match[2]) if match[2] else default_threads
    if threads < 1:
        raise ValueError("Variant thread count must be positive")
    return match[1], threads, bool(match[3])


def run_environment(base, affinity, opt, threads, lib, profile=False, passive=False, phase_threads=None, trace_tokens=False):
    env = {**base, **affinity}
    # A named default is reproducible even if the invoking shell tuned libgomp.
    for key in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_DYNAMIC", "OMP_THREAD_LIMIT"):
        env.pop(key, None)
    for phase in ("WHISPER", "ADAPTOR", "PREFILL", "DECODE", "LOGITS"):
        env.pop("MTD_THREADS_"+phase, None)
    env.update({"MTD_PROFILE": "1" if profile else "0", "MTD_TRACE_TOKENS": "1" if trace_tokens else "0", "MTD_DEVICE": "cpu",
                "MTD_THREADS": str(threads), "OMP_NUM_THREADS": str(threads),
                "OMP_DYNAMIC": "FALSE", "MTD_CPU_OPT": opt,
                "MTD_LOOP_GUARD": "1", "MTD_REPETITION_PENALTY": "1.0",
                "LD_LIBRARY_PATH": lib})
    if passive:
        env.update({"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"})
    for phase, count in (phase_threads or {}).items():
        env["MTD_THREADS_"+phase.upper()] = str(count)
    return env


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


def parse_storage(stderr):
    records = re.findall(r"BENCH_MODEL_STORAGE mode=(mapped|copied)(?: fileBytes=(\d+))? tensorBytes=(\d+) copiedBytes=(\d+)", stderr)
    if len(records) != 1:
        return None  # Old binaries have no record; duplicate records are ambiguous.
    mode, file_bytes, tensors, copied = records[0]
    return {"mode":mode, "fileBytes":int(file_bytes) if file_bytes else None,
            "tensorBytes":int(tensors), "copiedBytes":int(copied)}


def evaluate_storage(runs, settings):
    failures = []
    for r in runs:
        if not (settings.get(r["variant"],{}).get("opt",0) & 4096):
            continue
        storage = r.get("modelStorage") or {}
        if (storage.get("mode") != "mapped" or storage.get("copiedBytes") != 0 or
                not storage.get("tensorBytes") or not storage.get("fileBytes") or
                storage["fileBytes"] < storage["tensorBytes"]):
            failures.append(f'{r["case"]}/{r["variant"]}/{r["repeat"]}: requested CPU mapping was not verified')
    return {"passed":not failures,"failures":failures}


def evaluate_candidate_reference(runs, reference):
    """Compare numeric variants in the same build; old/frozen binaries are separate."""
    return evaluate([r for r in runs if r["variant"] not in ("baseline","cache-reference")],baseline=reference)


def token_fingerprint(path):
    tokens,eos,_ = read_trace(path)
    if not tokens or tokens[-1] != eos or eos in tokens[:-1]:
        raise ValueError("Token trace lacks unique terminal EOS")
    # Hash canonical IDs and EOS; never include reconstructable IDs in reports.
    encoded=json.dumps({"eos":eos,"tokens":tokens},separators=(",",":")).encode()
    return {"sha256":hashlib.sha256(encoded).hexdigest(),"count":len(tokens)}


def evaluate_token_traces(runs, reference):
    selected=[r for r in runs if r["variant"] not in ("baseline","cache-reference")]
    failures=[]
    if not selected: failures.append("No same-build token traces")
    for case in sorted({r["case"] for r in selected}):
        controls=[r for r in selected if r["case"]==case and r["variant"]==reference]
        hashes={r.get("tokenTrace",{}).get("sha256") for r in controls}
        if not controls or None in hashes or len(hashes)!=1:
            failures.append(case+": missing or unstable same-build token control")
        for row in [r for r in selected if r["case"]==case]:
            trace=row.get("tokenTrace") or {}
            if (not row["complete"] or row.get("concurrentMoss") or not trace.get("sha256")
                or trace.get("count") != row["tokens"] or trace.get("sha256") not in hashes):
                failures.append(case+"/"+row["variant"]+": missing, incomplete or changed full token trace")
    return {"passed":not failures,"failures":failures,
            "scope":"Complete same-build greedy token IDs including unique EOS; old production/frozen binaries retain full-output/count/EOS checks."}


def evaluate_shared_activation(runs, settings):
    failures=[]
    for r in runs:
        opt=settings.get(r["variant"],{}).get("opt",0)
        if not opt & (8192|16384): continue
        phases=r.get("phaseProfile") or {}
        decoder=sum(phases.get(p,{}).get("sharedQ8Nodes",0) for p in ("prefill","decode"))
        encoder=phases.get("whisper",{}).get("sharedQ8CastNodes",0)
        if (opt & 8192 and decoder<=0) or (opt & 16384 and encoder<=0):
            failures.append(f'{r["case"]}/{r["variant"]}/{r["repeat"]}: requested shared conversion was not observed')
    return {"passed":not failures,"failures":failures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=["meeting-60s", "dinner-60s", "meeting-120s", "silence-60s"])
    p.add_argument("--variants", nargs="+", default=["baseline", "0", "1", "2", "4", "7"],
                   help="baseline, cache-reference, OPT, OPT@THREADS, or OPT@THREADS-passive")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--name", default="regression")
    p.add_argument("--skip-warmup", action="store_true", help="For a follow-on stress check after the same binaries/model were warmed in a preceding suite")
    p.add_argument("--pin-physical", action="store_true", help="Bind one OpenMP worker to each physical core, avoiding sibling/core collisions")
    p.add_argument("--timeout",type=float,default=300.)
    p.add_argument("--reference-root",type=Path,default=Path("/home/stan/hw-moss-cache-layout"),help="Fixed validated reference build; variant cache-reference uses --reference-opt")
    p.add_argument("--reference-opt",type=int,default=16,help="Opt bitmask of the fixed reference (48 for validated parallel softmax)")
    p.add_argument("--candidate-reference",help="Numeric variant label in this same compiled binary; additionally reject regression against this control")
    p.add_argument("--model",type=Path,default=Path("/home/stan/hw-audio-bench/models/moss-transcribe-q8_0.gguf"))
    p.add_argument("--audio-dir",type=Path,default=Path("/home/stan/hw-audio-bench/audio"))
    p.add_argument("--baseline",type=Path,default=Path("/home/stan/hw-moss-api/bin/moss-transcribe"))
    p.add_argument("--baseline-lib-dir",default="/home/stan/hw-moss-api/lib")
    p.add_argument("--threads",type=int,default=16)
    p.add_argument("--decode-threads",type=int,default=0,help="Candidate-only decode budget; zero keeps startup threads")
    p.add_argument("--logits-threads",type=int,default=0,help="Candidate-only logits budget; zero keeps startup threads")
    p.add_argument("--profile",action="store_true",help="Collect numeric CPU phase/graph timings from instrumented candidates")
    p.add_argument("--trace-tokens",action="store_true",help="Require identical complete token IDs against --candidate-reference; private logs only, numeric hashes in report")
    args = p.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.decode_threads < 0 or args.logits_threads < 0 or "baseline" not in args.variants or len(args.variants) < 2:
        p.error("Need at least one repeat, baseline, and a candidate")
    candidates = [v for v in args.variants if v not in ("baseline","cache-reference")]
    if args.candidate_reference and (args.candidate_reference not in candidates or len(set(candidates)) < 2):
        p.error("Candidate reference must be a numeric variant with another same-build candidate")
    if args.trace_tokens and not args.candidate_reference:
        p.error("Token parity needs an explicit same-build candidate reference")
    root = args.root.resolve()
    results = root / args.name
    results.mkdir(parents=True, exist_ok=True)
    model, baseline = args.model.resolve(), args.baseline.resolve()
    reference = args.reference_root.resolve() / "build/moss-transcribe"
    phase_threads = {phase:count for phase,count in (("decode",args.decode_threads),("logits",args.logits_threads)) if count}
    def configuration(variant):
        if variant == "baseline":
            return baseline, "0", args.baseline_lib_dir, args.threads, False, {}
        if variant == "cache-reference":
            return reference, str(args.reference_opt), "", args.threads, False, {}
        opt, threads, passive = variant_settings(variant, args.threads)
        return candidate, opt, "", threads, passive, phase_threads
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
    report["traceReaderSha256"] = digest(Path(__file__).with_name("moss_ngram_audit.py"))
    report["sourceTree"] = subprocess.check_output(["git","rev-parse","HEAD^{tree}"],cwd=root / "source",text=True).strip()
    report["sourceClean"] = not subprocess.check_output(["git","status","--porcelain"],cwd=root / "source",text=True).strip()
    if not report["sourceClean"]:
        raise ValueError("Freeze and commit source before benchmarking")
    report["threads"] = args.threads
    report["variantSettings"] = {v: {"opt": int(configuration(v)[1]), "threads": configuration(v)[3],
                                       "waitPolicy": "passive" if configuration(v)[4] else "default",
                                       "phaseThreads": configuration(v)[5]}
                                 for v in args.variants}
    report["protocol"] += " Candidate OPT@THREADS labels override per-run threads; -passive sets OMP_WAIT_POLICY=PASSIVE/GOMP_SPINCOUNT=0. Default variants clear inherited wait/dynamic/thread-limit settings."
    report["phaseProfiling"] = args.profile
    report["tokenTracingSameBuild"] = args.trace_tokens
    report["sourceSha256"] = {str(path.relative_to(root / "source")):digest(path)
                              for path in sorted((root / "source/src").glob("*")) if path.is_file()}
    report["sourceSha256"]["CMakeLists.txt"] = digest(root / "source/CMakeLists.txt")
    for path in sorted((root / "source/third_party/pocketfft").glob("*")):
        if path.is_file():
            report["sourceSha256"][str(path.relative_to(root / "source"))] = digest(path)
    if "cache-reference" in args.variants:
        report["binarySha256"]["cache-reference"] = digest(reference)
        report["referenceRoot"] = str(args.reference_root.resolve())
        report["referenceOpt"] = args.reference_opt
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
        max_threads = max(configuration(v)[3] for v in args.variants)
        selected_cores = list(cores.values())[:max_threads]
        if len(selected_cores) < max_threads:
            raise ValueError("Not enough allowed physical cores for the requested thread count")
        affinity = {"OMP_PROC_BIND":"spread","OMP_PLACES":",".join("{"+str(cpu)+"}" for cpu in selected_cores)}
        report["physicalCoreAffinity"] = selected_cores
    report["timeoutSeconds"] = args.timeout
    # One discarded warmup for each binary, with full inference.
    audio0 = args.audio_dir / (args.cases[0] + ".wav")
    for binary, opt, lib, threads, passive, budgets in ([] if args.skip_warmup else [configuration(v) for v in args.variants]):
        subprocess.run([str(binary), "transcribe", str(model), str(audio0), "--max-new", "4096"],
                       env=run_environment(os.environ, affinity, opt, threads, lib, args.profile, passive, budgets),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=args.timeout)
    for repeat in range(args.repeats):
        for case in args.cases:
            audio = args.audio_dir / (case + ".wav")
            with wave.open(str(audio)) as w:
                duration = w.getnframes() / w.getframerate()
            variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            for variant in variants:
                key = f"{case}-{variant}-{repeat}"
                binary, opt, lib, threads, passive, budgets = configuration(variant)
                env = run_environment(os.environ, affinity, opt, threads, lib, args.profile, passive, budgets,
                                      args.trace_tokens and variant not in ("baseline","cache-reference"))
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
                row = {"case":case,"variant":variant,"repeat":repeat,"threads":threads,"phaseThreads":budgets,"waitPolicy":"passive" if passive else "default","inputSha256":digest(audio),"duration":duration,"outputSha256":digest(raw),"returncode":rc,"tokens":tokens,"stop":stop,"complete":complete,"concurrentMoss":concurrent.is_set(),"wallSeconds":wall,"maxRssKiB":int(rss[-1]) if rss else 0,
                       "hostLoadBefore":load_before,"hostLoadAfter":os.getloadavg(),
                       "timing": {"load":float(timing[1]),"inference":float(timing[2])} if timing else None,
                       "profile": {k:float(v) for k,v in re.findall(r"(prefill|embedding|decoder|logits)=([\d.]+)",stderr)},
                       "phaseProfile": json.loads(phase_match[1]) if (phase_match := re.search(r"^CPU_PHASE_PROFILE (\{.*\})$", stderr, re.M)) else None,
                       "modelStorage":parse_storage(stderr)}
                if args.trace_tokens and variant not in ("baseline","cache-reference"):
                    try: row["tokenTrace"]=token_fingerprint(log)
                    except ValueError as error: row["tokenTrace"]={"error":str(error)}
                report["runs"].append(row)
                report_path.write_text(json.dumps(report,indent=2))
                print(json.dumps(row),flush=True)
    artifact_hashes = {str(model):report["modelSha256"], str(baseline):report["binarySha256"]["baseline"],
                       str(candidate):report["binarySha256"]["candidate"], **report["libraries"], **report["productionLibraries"]}
    if "cache-reference" in args.variants:
        artifact_hashes.update({str(reference):report["binarySha256"]["cache-reference"], **report["referenceLibraries"]})
    artifact_hashes.update({str(root / "source" / name):value for name,value in report["sourceSha256"].items()})
    artifact_hashes.update({str(Path(__file__)):report["harnessSha256"],str(Path(__file__).with_name("moss_ngram_audit.py")):report["traceReaderSha256"]})
    changed = [path for path,value in artifact_hashes.items() if digest(path) != value]
    report["artifactGate"] = {"passed":not changed,"changed":changed}
    report["storageGate"] = evaluate_storage(report["runs"],report["variantSettings"])
    report["sharedActivationGate"] = evaluate_shared_activation(report["runs"],report["variantSettings"])
    report["gate"] = evaluate(report["runs"])
    if "cache-reference" in args.variants:
        compared = [r for r in report["runs"] if r["variant"] != "baseline"]
        report["cacheReferenceGate"] = evaluate(compared, baseline="cache-reference")
    if args.candidate_reference:
        report["candidateReference"] = args.candidate_reference
        report["candidateReferenceGate"] = evaluate_candidate_reference(report["runs"],args.candidate_reference)
    if args.trace_tokens:
        report["tokenTraceGate"] = evaluate_token_traces(report["runs"],args.candidate_reference)
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report["gate"],indent=2))
    if args.candidate_reference:
        print(json.dumps({"candidateReferenceGate":report["candidateReferenceGate"]},indent=2))
    return 0 if all(report.get(g,{"passed":True})["passed"] for g in
        ("artifactGate","storageGate","sharedActivationGate","gate","cacheReferenceGate","candidateReferenceGate","tokenTraceGate")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
