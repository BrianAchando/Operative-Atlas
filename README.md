# COVA: Cardiothoracic Operative and Vascular Atlas

An interactive 3D operative teaching atlas for cardiothoracic and vascular surgery, built for residents and
registrars. Repo: Operative-Atlas (the local project folder is still called `hilum`, the platform's first name).

Each operation is laid out the same way: **pathophysiology → anatomy → case practice → surgical steps → consent →
post-operative care**, with randomised multiple-choice questions and cited evidence.

## What is in it

118 operations and approaches across 13 menu groups:

| Group | Operations |
| --- | --- |
| Cardiac (21) | Aortic valve (standard, hemisternotomy, right anterior mini-thoracotomy, endocarditis with root abscess, double valve); mitral valve (standard, minimally invasive, septal approach, repair, endocarditis); tricuspid (replacement, ring, minimally invasive); CABG (1 and 2 vessel, on-pump, off-pump); aortic root (Bentall, David, Ross); pericardiectomy for TB constrictive pericarditis |
| Congenital cardiac (8) | ASD, VSD, PDA (3rd and 4th space), coarctation, tetralogy, modified BT shunt, pulmonary artery banding |
| Vascular (16) | Abdominal aortic aneurysm (infrarenal, juxtarenal, suprarenal, EVAR); thoracic aortic aneurysm (ascending, open, TEVAR); aorto-iliac occlusive disease (aortobifemoral, axillobifemoral, endovascular); acute limb ischemia (embolectomy and fasciotomy); infrainguinal bypass and amputation levels; dialysis access fistulae |
| Lobectomy (26), segmentectomy (3), pneumonectomy (4) | All five lobes, open, uniportal, biportal, anterior and posterior VATS approaches; post-TB bronchiectasis; CLE and CPAM |
| Esophagus (4), mediastinum (3), pleura (4), airway (1) | Esophagectomy (Ivor Lewis, McKeown, transhiatal), thoracic duct ligation; thymectomy (3 approaches); empyema (open, VATS), post-pneumonectomy empyema and BPF; cervical tracheal resection |
| Trauma (6) | Resuscitative thoracotomy, clamshell, cardiorrhaphy, hilar control (clamp, twist), pulmonary tractotomy |
| Access and positioning (16) | Positioning, thoracotomy and VATS port layouts |
| CTICU protocol (6) | Consent, core ICU care, doses and cardiac, thoracic and vascular post-operative management. A Kenyan CTICU protocol did not exist; this is being built from scratch and **has not yet had expert review** |

## How it works

- **CT and 3D in one frame** (RAS mm, carina at the origin). Scroll or click the CT and the 3D follows; click in 3D
  and the CT jumps there. Labelled structures are outlined on the CT.
- **Procedure mode**: each step moves the camera to the surgeon's view, highlights the structure being dissected
  (teal) and what to protect (red), asks the next move before revealing it, and staples at a computed staple line.
- **Your CT**: drop an uncompressed DICOM series (or .nii / .nii.gz). It is read in the browser and never uploaded.
  Click the carina once to place it against the 3D reference (translation only).
- **Review flags**: anyone can flag a step as wrong, outdated, unclear or missing. Listing and resolving flags needs
  the review key (`functions/api/flags.js`, Cloudflare D1; set up with `review/schema.sql`).

## Editing the text

Every operation has one file in `content/procedures/<group>/<operation>.md`. See [content/README.md](content/README.md)
for the format and the rules the build checks. The standing rule: **every clinical statement needs a source a reader
can check** (a guideline, trial or series), cited in an `> **Evidence:**` line and listed under `## Sources`. Unit
practice is marked **KNH practice**. Atlas text uses American spelling.

```
npm install
npm run content   # checks the text and merges it with the 3D skeleton; fails on any error
npm run dev       # http://localhost:5174
npm run build     # static site in dist/
```

## Building the 3D data (only when anatomy or steps change)

Needs Python 3.10–3.12, Node 20+, and `uv` (or plain pip). 16 GB RAM, or `--lowmem`; an NVIDIA GPU makes step 2 minutes
instead of ~45 min.

```
npm install                                             # app + gltf-transform (mesh compression)
cd pipeline && uv sync --extra segment && cd ..         # Python deps incl. TotalSegmentator
uv run --project pipeline python pipeline/segment.py --input /path/to/dicom-folder --device gpu    # add --lowmem on 8 GB
uv run --project pipeline python pipeline/build.py      # names hilar branches, meshes, writes public/data/
uv run --project pipeline python pipeline/procedures.py # operations, approaches and cameras from the computed anatomy
node scripts/content-export.mjs                         # adds text files for any new operation or step, without touching existing text
npm run build
```

`pipeline/hilum.py` explains how branches are named (skeleton trees, lobe territory, order along the trunk). Nerves,
the ligamentum arteriosum and node stations are schematic, placed on landmarks, and labelled as such.

## Deploying

```
git push
npm run build
npx wrangler pages deploy dist
```

## Limits

**Not for clinical use. A teaching aid, not clinical guidance.** One reference CT (3D Slicer's CTA-cardio sample, no
stated licence: teaching use only), so anatomy is a normal adult reference; other regions (for example the pelvis and
groin vessels) are schematic until a de-identified CTA replaces them. Branch names come from this scan's geometry and
need a surgeon's review. Do not upload identifiable patient imaging anywhere.
