"""Read-only GGUF v3 audit of lossless compression opportunities (stdlib only).

Deterministic beginning/middle/end tensor samples; no weights are written.
Q8 symbol entropies describe an independent-symbol coding model, not a proof
that more complex compression cannot do better. Codec timings are warm sample
decompression, not inference throughput. Unknown tensor types are inventoried.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import lzma
import math
from pathlib import Path
import statistics
import struct
import time
import zlib


def entropy(counts):
    n = sum(counts.values())
    return -sum((v/n)*math.log2(v/n) for v in counts.values()) if n else 0.


def audit(path, sample_bytes):
    file_bytes = path.stat().st_size
    with path.open('rb') as f:
        def unpack(fmt):
            size = struct.calcsize('<'+fmt)
            data = f.read(size)
            if len(data) != size:
                raise ValueError('Truncated GGUF header')
            return struct.unpack('<'+fmt, data)[0]
        def string():
            return f.read(unpack('Q')).decode('utf-8')
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        def skip_value(kind):
            if kind == 8:
                f.seek(unpack('Q'), 1)
            elif kind == 9:
                child, n = unpack('I'), unpack('Q')
                if child in sizes:
                    f.seek(n*sizes[child], 1)
                else:
                    for _ in range(n): skip_value(child)
            elif kind in sizes:
                f.seek(sizes[kind], 1)
            else:
                raise ValueError('Unknown GGUF metadata type')
        if f.read(4) != b'GGUF' or unpack('I') != 3:
            raise ValueError('Expected little-endian GGUF version 3')
        nt, nk = unpack('Q'), unpack('Q')
        alignment = 32
        for _ in range(nk):
            name, kind = string(), unpack('I')
            if name == 'general.alignment' and kind == 4:
                alignment = unpack('I')
            else:
                skip_value(kind)
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError('GGUF alignment must be a positive power of two')
        tensors = []
        for _ in range(nt):
            name, nd = string(), unpack('I')
            dims = [unpack('Q') for _ in range(nd)]
            kind, off = unpack('I'), unpack('Q')
            tensors.append((name,dims,kind,off))
        data_start = (f.tell()+alignment-1)//alignment*alignment
        rows = []
        totals = defaultdict(lambda: {'tensors':0,'elements':0,'bytes':0})
        for name,dims,kind,off in tensors:
            elements = math.prod(dims)
            fmt = {0:'F32',1:'F16',8:'Q8_0',30:'BF16'}.get(kind, str(kind))
            size = elements//32*34 if kind == 8 else elements*{0:4,1:2,30:2}.get(kind,0)
            if kind == 8 and (not dims or dims[0] % 32):
                raise ValueError('Q8_0 rows must contain complete 32-weight blocks')
            if size and data_start + off + size > file_bytes:
                raise ValueError('Truncated GGUF tensor: ' + name)
            totals[fmt]['tensors'] += 1
            totals[fmt]['elements'] += elements
            totals[fmt]['bytes'] += size
            row = {'name':name,'dims':dims,'type':fmt,'bytes':size}
            if not size:
                rows.append(row); continue
            unit = 34 if kind == 8 else {0:4,1:2,30:2}[kind]
            part = min(size//unit, max(1,sample_bytes//(3*unit))) * unit
            offsets = sorted({0, ((size-part)//(2*unit))*unit, size-part})
            pieces = []
            for pos in offsets:
                f.seek(data_start+off+pos)
                piece = f.read(part)
                if len(piece) != part:
                    raise ValueError('Truncated tensor sample: ' + name)
                pieces.append(piece)
            sample = b''.join(pieces)
            row['sampleBytes'] = len(sample)
            row['byteEntropy'] = entropy(Counter(sample))
            if kind == 8:
                counts = Counter(sample)
                scales = b''.join(sample[i:i+2] for i in range(0,len(sample),34))
                counts.subtract(Counter(scales))
                counts = +counts
                scale_counts = Counter(struct.unpack('<'+'H'*(len(scales)//2),scales))
                row['q8CodeEntropyBits'] = entropy(counts)
                row['q8ScaleEntropyBits'] = entropy(scale_counts)
                row['independentSymbolBitsPerWeight'] = entropy(counts)+entropy(scale_counts)/32
            for codec, encode, decode in [('zlib1',lambda s:zlib.compress(s,1),zlib.decompress),
                                          ('xz1',lambda s:lzma.compress(s,preset=1),lzma.decompress)]:
                compressed = encode(sample)
                assert decode(compressed) == sample
                times = []
                for _ in range(3):
                    start = time.perf_counter(); restored = decode(compressed)
                    times.append(time.perf_counter()-start)
                assert hashlib.sha256(restored).digest() == hashlib.sha256(sample).digest()
                row[codec] = {'ratio':len(compressed)/len(sample),
                              'decodeMBps':len(sample)/statistics.median(times)/1e6,
                              'roundTripExact':True}
            rows.append(row)
    with path.open('rb') as model:
        model_hash = hashlib.file_digest(model,'sha256').hexdigest()
    return {'modelSha256':model_hash,
            'modelBytes':path.stat().st_size,'tensorTypes':dict(totals),
            'protocol':'Deterministic start/middle/end samples per tensor, capped by --sample-bytes. Per-tensor codec ratios and warm decompression; not full-model compression or inference speed.',
            'sampleLimitBytes':sample_bytes,'tensors':rows}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-bytes',type=int,default=65536)
    args=p.parse_args()
    if args.sample_bytes < 102: p.error('Need at least 102 sample bytes')
    result=audit(args.model,args.sample_bytes)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'},indent=2))
