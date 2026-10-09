"""Independent synthetic GGUF fixtures for inventory and block sampling."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from moss_lossless_audit import audit


def u32(n): return struct.pack('<I', n)
def u64(n): return struct.pack('<Q', n)
def string(s):
    b = s.encode()
    return u64(len(b)) + b


def fixture():
    # Non-default alignment and string-array metadata exercise header skipping.
    header = b'GGUF' + u32(3) + u64(2) + u64(2)
    header += string('general.alignment') + u32(4) + u32(64)
    header += string('tokenizer.ggml.tokens') + u32(9) + u32(8) + u64(2) + string('a') + string('b')
    header += string('projection.weight') + u32(2) + u64(32) + u64(6) + u32(8) + u64(0)
    header += string('norm.weight') + u32(1) + u64(3) + u32(0) + u64(256)
    header += bytes((-len(header)) % 64)
    # Six blocks: scale 1.0 (0x3c00), all code 7. Known entropy is zero.
    q8 = (struct.pack('<H', 0x3c00) + bytes([7]) * 32) * 6
    data = q8 + bytes(256 - len(q8)) + struct.pack('<fff', 1, 2, 3)
    return header + data


class AuditTests(unittest.TestCase):
    def test_inventory_entropy_and_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fixture.gguf'
            payload = fixture(); path.write_bytes(payload)
            report = audit(path, 102)
            self.assertEqual(report['modelSha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(report['tensorTypes']['Q8_0'], {'tensors': 1, 'elements': 192, 'bytes': 204})
            self.assertEqual(report['tensorTypes']['F32']['bytes'], 12)
            tensor = report['tensors'][0]
            self.assertEqual(tensor['sampleBytes'], 102)
            self.assertEqual(tensor['q8CodeEntropyBits'], 0)
            self.assertEqual(tensor['q8ScaleEntropyBits'], 0)
            for tensor in report['tensors']:
                for codec in ('zlib1', 'xz1'):
                    self.assertTrue(tensor[codec]['roundTripExact'])

    def test_truncated_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fixture.gguf'; path.write_bytes(fixture()[:-1])
            with self.assertRaisesRegex(ValueError, 'Truncated GGUF tensor'):
                audit(path, 102)

    def test_truncated_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fixture.gguf'; path.write_bytes(fixture()[:20])
            with self.assertRaisesRegex(ValueError, 'Truncated GGUF header'):
                audit(path, 102)


if __name__ == '__main__': unittest.main()
