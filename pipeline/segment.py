"""Step 1: segment a chest CT with TotalSegmentator into the work folder that build.py reads.

    python pipeline/segment.py --input <DICOM folder | ct.nii.gz> [--work work/] [--device gpu|cpu] [--lowmem]

Writes work/ct.nii.gz (cropped to the body, RAS), work/total.nii.gz (task 'total': lobes, heart, great vessels,
bones) and work/vessels.nii.gz (task 'lung_vessels' on the left hemithorax: airways, arteries, veins).
Each output is reused if it already exists, so a run that stops can be restarted.

Memory: 'lung_vessels' works at 0.7 mm and needs about 8 GB for one hemithorax. With --lowmem the hemithorax is
segmented in three overlapping blocks of ~100 slices and merged, which fits in about 5 GB. A GPU (--device gpu)
turns ~40 min of CPU time into a few minutes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from scipy import ndimage

HERE = Path(__file__).resolve().parent
TS = {n: int(k) for k, n in json.loads((HERE / 'config' / 'totalsegmentator_total_v2.json').read_text()).items()}


def run_ts(inp: Path, out: Path, task: str, device: str, lowmem: bool) -> None:
    if out.exists():
        print(f'  reuse {out.name}'); return
    from totalsegmentator.python_api import totalsegmentator
    print(f'  TotalSegmentator {task} on {inp.name} ({device}) ...')
    totalsegmentator(str(inp), str(out), ml=True, task=task, device=device, force_split=lowmem and task == 'total', nr_thr_resamp=1, nr_thr_saving=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True, help='DICOM folder of one CT series, or a .nii/.nii.gz')
    ap.add_argument('--work', type=Path, default=HERE.parent / 'work')
    ap.add_argument('--device', default='gpu', help="'gpu' or 'cpu'")
    ap.add_argument('--lowmem', action='store_true', help='split the heavy steps to fit in ~5 GB of RAM')
    a = ap.parse_args()
    w = a.work; w.mkdir(parents=True, exist_ok=True)

    # ---- 1. the CT as RAS NIfTI, cropped to the body
    ct_path = w / 'ct.nii.gz'
    if not ct_path.exists():
        src = a.input
        if src.is_dir():
            from totalsegmentator.dicom_io import dcm_to_nifti
            tmp = w / 'ct_raw.nii.gz'
            print('  DICOM -> NIfTI ...'); dcm_to_nifti(src, tmp, tmp_dir=w, verbose=False)
            src = tmp
        img = nib.as_closest_canonical(nib.load(str(src)))
        d = np.asanyarray(img.dataobj).astype(np.int16)
        body = ndimage.binary_opening(d > -500, iterations=2)
        lab, n = ndimage.label(body)
        body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        ii = np.argwhere(body); lo = np.maximum(ii.min(0) - 8, 0); hi = np.minimum(ii.max(0) + 9, d.shape)
        aff = img.affine.copy(); aff[:3, 3] = img.affine[:3, :3] @ lo + img.affine[:3, 3]
        nib.save(nib.Nifti1Image(d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], aff), ct_path)
        print('  wrote', ct_path.name, tuple(hi - lo))

    # ---- 2. whole-body structures
    run_ts(ct_path, w / 'total.nii.gz', 'total', a.device, a.lowmem)

    # ---- 3. hilar vessels and airways on the left hemithorax at the model's own spacing
    out = w / 'vessels.nii.gz'
    if out.exists():
        print('  reuse', out.name); return
    crop = w / 'vessels_input.nii.gz'
    if not crop.exists():
        ct = nib.load(str(ct_path)); L = np.asanyarray(nib.load(str(w / 'total.nii.gz')).dataobj)
        ii = np.argwhere(np.isin(L, [TS['lung_upper_lobe_left'], TS['lung_lower_lobe_left'], TS['trachea'], TS['pulmonary_vein']]))
        lo = np.maximum(ii.min(0) - 6, 0); hi = np.minimum(ii.max(0) + 7, L.shape)
        # keep 15 mm to the right of the trachea so the carina and the left PA origin are inside
        tx = np.median(np.argwhere(L == TS['trachea'])[:, 0]); hi[0] = min(hi[0], int(tx + 15 / abs(ct.affine[0, 0])))
        d = np.asanyarray(ct.dataobj)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        aff = ct.affine.copy(); aff[:3, 3] = ct.affine[:3, :3] @ lo + ct.affine[:3, 3]
        r = resample_to_output(nib.Nifti1Image(d.astype(np.int16), aff), (0.703125, 0.703125, 1.0), order=1, cval=-1024)
        nib.save(nib.Nifti1Image(np.asanyarray(r.dataobj).astype(np.int16), r.affine), crop)
    img = nib.load(str(crop)); d = np.asanyarray(img.dataobj); A = img.affine
    if not a.lowmem:
        run_ts(crop, out, 'lung_vessels', a.device, False); return
    nz = d.shape[2]; size, overlap = 100, 20
    starts = list(range(0, max(1, nz - overlap), size - overlap))
    merged = np.zeros(d.shape, np.uint8)
    for n, k0 in enumerate(starts):
        k1 = min(nz, k0 + size)
        blk = w / f'vessels_block{n}.nii.gz'; blk_out = w / f'vessels_block{n}_seg.nii.gz'
        if not blk_out.exists():
            aff = A.copy(); aff[:3, 3] = A[:3, :3] @ [0, 0, k0] + A[:3, 3]
            nib.save(nib.Nifti1Image(d[:, :, k0:k1], aff), blk)
        run_ts(blk, blk_out, 'lung_vessels', a.device, False)
        s = np.asanyarray(nib.load(str(blk_out)).dataobj)
        # each block owns the middle of its overlap with the next
        a_ = k0 + (overlap // 2 if n else 0); b_ = k1 - (overlap // 2 if k1 < nz else 0)
        merged[:, :, a_:b_] = s[:, :, a_ - k0:b_ - k0]
    nib.save(nib.Nifti1Image(merged, A), out)
    print('  wrote', out.name)


if __name__ == '__main__':
    main()
