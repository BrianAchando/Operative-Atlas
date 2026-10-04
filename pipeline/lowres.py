"""A half-resolution preview of the reference CT and its labels (about an eighth of the size), so the atlas opens
quickly on slow connections; the app swaps in the full-resolution volume when it has downloaded."""
import gzip
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / 'public' / 'data'
A = json.loads((OUT / 'atlas.json').read_text())
c = A['ct']; d0, d1, d2 = c['dims']


def half(name, out):
    a = np.frombuffer(gzip.decompress((OUT / name).read_bytes()), np.uint8).reshape(d2, d1, d0)[::2, ::2, ::2]   # x fastest, as in the app
    (OUT / out).write_bytes(gzip.compress(np.ascontiguousarray(a).tobytes(), 9))
    return [a.shape[2], a.shape[1], a.shape[0]]


dims = half(c['file'], 'ct.lo.hu8.gz'); half(A['labels']['file'], 'labels.lo.u8.gz')
aff = [[row[0] * 2, row[1] * 2, row[2] * 2, row[3]] for row in c['affine']]
A['ct']['lo'] = {'file': 'ct.lo.hu8.gz', 'labels': 'labels.lo.u8.gz', 'dims': dims, 'affine': aff, 'spacing': c['spacing'] * 2}
(OUT / 'atlas.json').write_text(json.dumps(A))
print('preview', dims, (OUT / 'ct.lo.hu8.gz').stat().st_size // 1024, 'KB')
