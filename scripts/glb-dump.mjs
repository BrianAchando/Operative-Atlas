// Decode meshopt-compressed GLBs back to vertex positions (atlas frame), for pipeline steps that fit new shapes to
// existing ones (pipeline/anatomy_heart.py reads the valve annuli this way).
// usage: node scripts/glb-dump.mjs out.json id1 id2 ...   (reads public/data/meshes/<id>.glb)
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
import fs from 'node:fs';
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const [out, ...ids] = process.argv.slice(2); const res = {};
for (const id of ids) {
  const file = `public/data/meshes/${id}.glb`; if (!fs.existsSync(file)) continue;
  const doc = await io.read(file); const P = [];
  for (const node of doc.getRoot().listNodes()) {
    const mesh = node.getMesh(); if (!mesh) continue; const M = node.getWorldMatrix();
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute('POSITION'); const el = [0, 0, 0];
      for (let i = 0; i < pos.getCount(); i++) { pos.getElement(i, el); const [x, y, z] = el;
        P.push(M[0] * x + M[4] * y + M[8] * z + M[12], M[1] * x + M[5] * y + M[9] * z + M[13], M[2] * x + M[6] * y + M[10] * z + M[14]); }
    }
  }
  res[id] = { P };
}
fs.writeFileSync(out, JSON.stringify(res));
