"""Measure the explicit synthetic activation graph probe on a Linux CPU host.

Conversion/graph dispatch are timed; model loading, audio and generation are
absent. This is not a full-input performance gate. Raw output contains numbers
only. An immutable source/build and no concurrent MOSS are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',default='activation-graph-v1')
    p.add_argument('--timeout',type=float,default=1200)
    args=p.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name) or args.timeout<=0:
        p.error('Invalid result name or timeout')
    root=args.root.resolve();source=root/'source';binary=root/'build/tests/test_cpu_activation'
    results=root/args.name;results.mkdir(exist_ok=False)
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
        raise ValueError('Source is not clean')
    cores={}
    for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines():
        if line.startswith('#'):continue
        cpu,core,socket=map(int,line.split(','))
        if cpu in os.sched_getaffinity(0):cores.setdefault((socket,core),cpu)
    selected=list(cores.values())[:16]
    if len(selected)!=16:raise ValueError('Probe requires 16 physical cores')
    files=[binary,Path(__file__),source/'CMakeLists.txt',source/'tests/CMakeLists.txt',source/'tests/test_cpu_activation.cpp',
           *sorted((source/'src').glob('*')),*sorted((root/'build').rglob('*.so'))]
    artifacts={str(f):digest(f) for f in files if f.is_file()}
    if subprocess.run(['pgrep','-x','moss-transcribe'],capture_output=True).returncode!=1:
        raise ValueError('MOSS is busy')
    stopped,competing=threading.Event(),threading.Event()
    def monitor():
        while not stopped.wait(.5):
            if subprocess.run(['pgrep','-x','moss-transcribe'],capture_output=True).returncode!=1:competing.set()
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    env={**os.environ,'OMP_NUM_THREADS':'16','OMP_PROC_BIND':'spread','OMP_DYNAMIC':'FALSE',
         'OMP_PLACES':','.join('{'+str(c)+'}' for c in selected)}
    env.pop('OMP_WAIT_POLICY',None)
    load_before=list(os.getloadavg());start=time.monotonic()
    try:
        with (results/'probe.jsonl').open('wb') as out,(results/'probe.log').open('wb') as err:
            proc=subprocess.Popen(['taskset','-c',','.join(map(str,selected)),str(binary),'--benchmark'],env=env,stdout=out,stderr=err,start_new_session=True)
            try:rc=proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
    finally:stopped.set();watcher.join()
    rows=[json.loads(line) for line in (results/'probe.jsonl').read_text().splitlines() if line.startswith('{')]
    changed=[str(f) for f,h in artifacts.items() if digest(Path(f))!=h]
    passed=rc==0 and len(rows)==72 and not changed and not competing.is_set() and all(
        r['activationByteDifferences']==r['castFloatBitDifferences']==r['sharedFloatBitDifferences']==0 for r in rows)
    report={'protocol':'Synthetic same-process warm projection groups. Ordinary weights/reference mul_mat; conversion and graph dispatch included. Five cyclic-order median samples after one discarded warmup. No model load/audio/generation or full-input latency claim.',
        'sourceCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        'sourceTree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=source,text=True).strip(),
        'ggmlCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source/'third_party/ggml',text=True).strip(),
        'artifacts':artifacts,'physicalCoreAffinity':selected,'environment':{k:env[k] for k in ('OMP_NUM_THREADS','OMP_PROC_BIND','OMP_DYNAMIC','OMP_PLACES')},
        'hostLoadBefore':load_before,'hostLoadAfter':list(os.getloadavg()),'elapsedSeconds':time.monotonic()-start,
        'returncode':rc,'concurrentMoss':competing.is_set(),'rows':rows,'artifactGate':{'passed':not changed,'changed':changed},
        'gate':{'passed':passed},'probeOutputSha256':digest(results/'probe.jsonl'),'probeLogSha256':digest(results/'probe.log')}
    (results/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('returncode','concurrentMoss','elapsedSeconds','gate')},indent=2),flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
