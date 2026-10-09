"""Check a final native build against a frozen paired report, without a speed claim.

Outputs stay local; public reports contain hashes and numeric diagnostics only.
Use the paired regression harness separately for any latency acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import threading


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference-report', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/home/stan/hw-audio-bench/models/moss-transcribe-q8_0.gguf'))
    p.add_argument('--audio-dir', type=Path, default=Path('/home/stan/hw-audio-bench/audio'))
    p.add_argument('--cases', nargs='+', default=['meeting-60s','dinner-60s','vox-vmaiq-60s'])
    p.add_argument('--opt', type=int, default=2096)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--name', default='final-smoke')
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--trace-tokens', action='store_true', help='Write actual greedy token IDs to private stderr logs for replay audits')
    args = p.parse_args()
    if args.threads < 1 or args.timeout <= 0 or not re.fullmatch(r'[A-Za-z0-9_-]+', args.name) or not all(re.fullmatch(r'[A-Za-z0-9_-]+', c) for c in args.cases):
        p.error('Invalid thread count, timeout or case name')
    root = args.root.resolve()
    ref = json.loads(args.reference_report.read_text())
    if not ref.get('artifactGate',{}).get('passed'):
        raise ValueError('Reference artifacts did not pass their integrity gate')
    baseline = [r for r in ref['runs'] if r['variant'] == 'cache-reference']
    if not baseline:
        raise ValueError('Need the frozen validated cache-reference runs')
    result_dir = root / args.name
    result_dir.mkdir(exist_ok=False)
    binary = root / 'build/moss-transcribe'
    artifacts = {str(path):digest(path) for path in [args.model, binary, args.reference_report, Path(__file__)]}
    for path in [*sorted((root/'build').rglob('*.so')), *sorted((root/'source/src').glob('*')), root/'source/CMakeLists.txt', *sorted((root/'source/third_party/pocketfft').glob('*'))]:
        if path.is_file(): artifacts[str(path)] = digest(path)
    if artifacts[str(args.model)] != ref['modelSha256']:
        raise ValueError('Different model from validated reference')
    cores = {}
    for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines():
        if line.startswith('#'): continue
        cpu, core, socket = map(int,line.split(','))
        if cpu in os.sched_getaffinity(0): cores.setdefault((socket,core),cpu)
    selected = list(cores.values())[:args.threads]
    if len(selected) != args.threads: raise ValueError('Insufficient allowed physical cores')
    env = {**os.environ,'MTD_DEVICE':'cpu','MTD_THREADS':str(args.threads),'OMP_NUM_THREADS':str(args.threads),
           'OMP_PROC_BIND':'spread','OMP_PLACES':','.join('{'+str(c)+'}' for c in selected),
           'MTD_CPU_OPT':str(args.opt),'MTD_PROFILE':'1','MTD_TRACE_TOKENS':'1' if args.trace_tokens else '0','MTD_LOOP_GUARD':'1','MTD_REPETITION_PENALTY':'1.0','LD_LIBRARY_PATH':''}
    report = {'protocol':'Final-build output/EOS smoke only. Fresh processes, primed model filesystem cache, pinned physical cores, shared host. Diagnostic wall time; no matched latency gate or new corpus accuracy claim.',
              'referenceReportSha256':digest(args.reference_report),'modelSha256':digest(args.model),'binarySha256':digest(binary),
              'sourceCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'source',text=True).strip(),
              'opt':args.opt,'threads':args.threads,'tokenTracing':args.trace_tokens,'physicalCoreAffinity':selected,'artifacts':artifacts,'runs':[]}
    for case in args.cases:
        expected = [r for r in baseline if r['case'] == case]
        if not expected or any(not r['complete'] or r.get('concurrentMoss') for r in expected): raise ValueError('Incomplete or competing reference')
        fingerprints = {(r['inputSha256'],r['outputSha256'],r['tokens'],r['stop']) for r in expected}
        if len(fingerprints) != 1: raise ValueError('Unstable reference')
        input_sha, output_sha, tokens, stop = fingerprints.pop()
        audio = args.audio_dir / (case+'.wav')
        if digest(audio) != input_sha: raise ValueError('Different audio from reference')
        artifacts[str(audio)] = input_sha
        if subprocess.run(['pgrep','-x','moss-transcribe'],capture_output=True).returncode != 1: raise ValueError('MOSS window is busy')
        raw, log = result_dir/(case+'.txt'), result_dir/(case+'.log')
        start = time.perf_counter()
        stopped, competing = threading.Event(), threading.Event()
        def monitor():
            while not stopped.wait(0.5):
                snapshot = subprocess.run(['pgrep','-x','moss-transcribe'],capture_output=True,text=True)
                if snapshot.returncode not in (0,1) or len(snapshot.stdout.splitlines()) > 1: competing.set()
        watcher = threading.Thread(target=monitor,daemon=True); watcher.start()
        with raw.open('w') as out, log.open('w') as err:
            proc = subprocess.Popen([str(binary),'transcribe',str(args.model),str(audio),'--max-new','4096'],env=env,stdout=out,stderr=err,start_new_session=True)
            try: rc = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
        stopped.set(); watcher.join()
        generation = re.search(r'BENCH_GENERATION tokens=(\d+) stop=(\w+)',log.read_text())
        got_tokens, got_stop = (int(generation[1]),generation[2]) if generation else (None,None)
        passed = not competing.is_set() and (rc == 0 or (rc == 1 and not raw.read_text().strip())) and got_tokens is not None and got_tokens < 4096 and got_stop == 'eos' and got_stop == stop and got_tokens == tokens and digest(raw) == output_sha
        row = {'case':case,'inputSha256':input_sha,'outputSha256':digest(raw),'returncode':rc,'tokens':got_tokens,'stop':got_stop,'wallSeconds':time.perf_counter()-start,'concurrentMoss':competing.is_set(),'passed':passed}
        report['runs'].append(row); print(json.dumps(row),flush=True)
    changed = [path for path,value in artifacts.items() if digest(Path(path)) != value]
    report['artifactGate'] = {'passed':not changed,'changed':changed}
    report['gate'] = {'passed':not changed and all(r['passed'] for r in report['runs'])}
    (result_dir/'report.json').write_text(json.dumps(report,indent=2))
    return 0 if report['gate']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
