"""Cardiac step: the heart's chambers, valves' surroundings and coronaries, for the cardiac module.

    totalseg_set_license -l <your licence key>        (once; free academic licence from totalsegmentator.com)
    python pipeline/segment_heart.py [--work work/] [--device cpu|gpu]

Needs work/ct.nii.gz and work/total.nii.gz from segment.py. Crops the CT to the heart (with a margin) and runs three
licensed TotalSegmentator tasks on the crop:
  heartchambers_highres  -> work/heart.nii.gz      myocardium, LA, LV, RA, RV, aorta, pulmonary artery (sub-millimetre)
  aortic_sinuses         -> work/sinuses.nii.gz    LV outflow tract, the three aortic sinuses (cusps)
  coronary_arteries      -> work/coronary.nii.gz   coronary arteries
Each output is reused if it exists, so a stopped run can be restarted. A task that is not licensed or fails is
skipped with a message; build.py uses whatever is there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
TS = {n: int(k) for k, n in json.loads((HERE / 'config' / 'totalsegmentator_total_v2.json').read_text()).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, default=HERE.parent / 'work')
    ap.add_argument('--device', default='cpu', help="'gpu' or 'cpu'")
    a = ap.parse_args(); w = a.work
    ct = nib.load(str(w / 'ct.nii.gz')); L = np.asanyarray(nib.load(str(w / 'total.nii.gz')).dataobj)
    crop = w / 'heart_input.nii.gz'
    if not crop.exists():
        heart = np.argwhere(L == TS['heart'])
        # the heart and 3 cm around it, up to the top of the aortic arch
        mm = np.abs(np.diag(ct.affine)[:3])
        lo = np.maximum(heart.min(0) - (30 / mm).astype(int), 0); hi = np.minimum(heart.max(0) + (30 / mm).astype(int) + 1, L.shape)
        arch = np.argwhere(L == TS['aorta']); hi[2] = min(L.shape[2], max(hi[2], arch[:, 2].max() + 5))
        d = np.asanyarray(ct.dataobj)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        aff = ct.affine.copy(); aff[:3, 3] = ct.affine[:3, :3] @ lo + ct.affine[:3, 3]
        nib.save(nib.Nifti1Image(d.astype(np.int16), aff), crop)
        print('  wrote', crop.name, d.shape)
    from totalsegmentator.python_api import totalsegmentator
    for task, name in (('heartchambers_highres', 'heart'), ('aortic_sinuses', 'sinuses'), ('coronary_arteries', 'coronary')):
        out = w / f'{name}.nii.gz'
        if out.exists():
            print('  reuse', out.name); continue
        print(f'  TotalSegmentator {task} ({a.device}) ...')
        try:
            totalsegmentator(str(crop), str(out), ml=True, task=task, device=a.device, nr_thr_resamp=1, nr_thr_saving=1)
            print('  wrote', out.name)
        except Exception as e:                                             # noqa: BLE001  (no licence, out of memory ...)
            print(f'  skipped {task}: {e}')


if __name__ == '__main__':
    main()
