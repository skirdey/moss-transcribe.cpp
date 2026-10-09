"""Prepare/score a pinned VoxConverse development smoke subset.

Dependencies: pyarrow, soundfile, scipy, pyannote.metrics. This is a small
regression subset, not an official full-corpus VoxConverse/OpenBench result.
DER uses optimal speaker assignment, includes overlap, and reports collars
0 and 0.25 seconds. Reference boundaries are clipped to each 60-second cut.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

REVISION = "3acfa1b45ca4b7419aee999d67d94c617f9c9d47"
PARQUET_SHA = "f36c54412f0ac9cfe7ec2682e27f70f3e3df63d8e2b8f288d4f62341a595917d"


def prepare(data):
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    parquet = data / "vox-dev.parquet"
    assert hashlib.file_digest(parquet.open("rb"), "sha256").hexdigest() == PARQUET_SHA
    samples = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)[:rate * 60]
        if rate != 16000:
            factor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // factor, rate // factor)
        duration = len(audio) / 16000
        case = "vox-" + Path(row["audio"]["path"]).stem + "-60s"
        output = data / (case + ".wav")
        sf.write(output, audio, 16000, subtype="PCM_16")
        reference = [{"start": max(0., start), "end": min(duration, end), "speaker_id": speaker}
                     for start,end,speaker in zip(row["timestamps_start"],row["timestamps_end"],row["speakers"])
                     if end > 0 and start < duration]
        samples.append({"case":case,"duration":duration,"inputSha256":hashlib.file_digest(output.open("rb"),"sha256").hexdigest(),"reference":reference})
        if len(samples) == 3:
            break
    manifest = {"source":"https://huggingface.co/datasets/diarizers-community/voxconverse",
                "revision":REVISION,"parquetSha256":PARQUET_SHA,"split":"dev","selection":"First three rows of dev-00000-of-00005, first 60 seconds, chosen before candidate evaluation.",
                "license":"CC BY 4.0; VoxConverse, Chung et al. 2020; HF packaging by diarizers-community.","samples":samples}
    (data / "vox-reference.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"cases":[s["case"] for s in samples]}))


def score(data, experiment, destination):
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from moss_output import parse
    manifest = json.loads((data / "vox-reference.json").read_text())
    samples = {s["case"]:s for s in manifest["samples"]}
    report = json.loads((experiment / "report.json").read_text())
    output = {"benchmark":"VoxConverse development 3 x 60s smoke subset", "reference":{k:v for k,v in manifest.items() if k != "samples"},
              "protocol":"pyannote.metrics optimal speaker permutation, overlap included, UEM=cut interval, macro averages over three cuts; each variant scored once after deterministic repeat parity checks.","scores":[]}
    for variant in sorted({r["variant"] for r in report["runs"]}):
        rows = []
        for case,sample in samples.items():
            run = next(r for r in report["runs"] if r["case"] == case and r["variant"] == variant)
            assert run["inputSha256"] == sample["inputSha256"] and run["complete"], "Cannot score mismatched/incomplete output"
            raw = (experiment / f"{case}-{variant}-{run['repeat']}.txt").read_text()
            reference, hypothesis = Annotation(uri=case), Annotation(uri=case)
            for i,s in enumerate(sample["reference"]):
                reference[Segment(s["start"],s["end"]),i] = s["speaker_id"]
            segments = parse(raw,sample["duration"])
            for i,s in enumerate(segments):
                hypothesis[Segment(s["start"],s["end"]),i] = s["speaker_id"]
            uem = Timeline([Segment(0,sample["duration"])],uri=case)
            scores = {str(c):DiarizationErrorRate(collar=c,skip_overlap=False)(reference,hypothesis,uem=uem,detailed=True) for c in (0.,.25)}
            rows.append({"case":case,"scores":scores,"segments":len(segments)})
        output["scores"].append({"variant":variant,"cases":rows,"macroDER":{str(c):sum(r["scores"][str(c)]["diarization error rate"] for r in rows)/len(rows) for c in (0.,.25)}})
    baseline = next(row for row in output["scores"] if row["variant"] == "baseline")
    reference_scores = {row["case"]:row["scores"] for row in baseline["cases"]}
    failures = []
    for variant in output["scores"]:
        if variant["variant"] == "baseline":
            continue
        for row in variant["cases"]:
            for collar,metrics in row["scores"].items():
                delta = metrics["diarization error rate"] - reference_scores[row["case"]][collar]["diarization error rate"]
                if delta > 1e-12:
                    failures.append({"variant":variant["variant"],"case":row["case"],"collar":float(collar),"DERIncrease":delta})
    output["qualityGate"] = {"passed":not failures,"allowedDERIncrease":0.,"numericalTolerance":1e-12,"failures":failures}
    destination.write_text(json.dumps(output,indent=2))
    print(json.dumps([{k:v for k,v in row.items() if k != "cases"} for row in output["scores"]],indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--experiment",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.data)
    else:
        raise SystemExit(score(args.data,args.experiment,args.output))
