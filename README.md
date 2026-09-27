# Hilum — operative anatomy atlas

A 3D thoracic operative atlas linked to a scrollable CT. Built for VATS left upper lobectomy (anterior and posterior
approaches); the same data model takes other operations.

- **CT and 3D in one frame** (RAS mm, carina at the origin). Scroll or click the CT and the 3D follows; click in 3D and
  the CT jumps there. Labelled structures are outlined on the CT.
- **Procedure mode**: each step moves the camera to the surgeon's view, highlights the structure being dissected
  (teal) and what to protect (red), asks the next move before revealing it, and staples at a computed staple line.
- **Your CT**: drop an uncompressed DICOM series (or .nii / .nii.gz). It is read in the browser and never uploaded.
  Click the carina once to place it against the 3D reference (translation only).

## Build it

Needs Python 3.10–3.12, Node 20+, and `uv` (or plain pip). 16 GB RAM, or `--lowmem`; an NVIDIA GPU makes step 2 minutes instead of ~45 min.

```
npm install                                             # app + gltf-transform (mesh compression)
cd pipeline && uv sync --extra segment && cd ..         # Python deps incl. TotalSegmentator
uv run --project pipeline python pipeline/segment.py --input /path/to/dicom-folder --device gpu    # add --lowmem on 8 GB
uv run --project pipeline python pipeline/build.py      # names hilar branches, meshes, writes public/data/
uv run --project pipeline python pipeline/procedures.py # the two approaches, cameras from the computed anatomy
npm run dev                                             # http://localhost:5174
npm run build                                           # static site in dist/, host anywhere
```

`pipeline/hilum.py` explains how branches are named (skeleton trees, lobe territory, order along the trunk). Nerves,
the ligamentum arteriosum and node stations are schematic, placed on landmarks, and labelled as such.

**Not for clinical use.** One reference CT (3D Slicer's CTA-cardio sample, no stated licence: teaching use only).
Branch names come from this scan's geometry and need a surgeon's review.
