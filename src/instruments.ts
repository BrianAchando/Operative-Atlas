import * as THREE from 'three';

/**
 * Operating instruments drawn to scale in world millimetres: an endoscopic stapler (articulating, with a coloured
 * reload), a peanut (Kittner) dissector on a long holder, a knot pusher, and silk ligatures. Each is built with its
 * working tip at the local origin, so it can be dropped onto a target and aimed back at the port it comes through.
 */

const metal = () => new THREE.MeshStandardMaterial({ color: 0xbcc4cc, metalness: 0.85, roughness: 0.28 });
const dark = () => new THREE.MeshStandardMaterial({ color: 0x2e333b, metalness: 0.35, roughness: 0.5 });
/** shafts are drawn see-through so they never hide the structure they work on */
const ghost = (c = 0x3a4048) => new THREE.MeshStandardMaterial({ color: c, metalness: 0.5, roughness: 0.4, transparent: true, opacity: 0.32, depthWrite: false });

/** reload colours: vascular (thin) and thick tissue (bronchus, parenchyma) */
export const RELOAD = { vascular: 0xd8c08e, tissue: 0x7a5bc6 } as const;

/** a cylinder between two points */
export function rod(a: THREE.Vector3, b: THREE.Vector3, r: number, mat: THREE.Material): THREE.Mesh {
  const d = b.clone().sub(a); const len = d.length();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 20, 1), mat);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  return m;
}

export interface Stapler {
  group: THREE.Group;
  /** 0 = open, 1 = clamped */
  setClamp(f: number): void;
  setFire(f: number): void;
  dispose(): void;
}

/**
 * Stapler clamped across a structure at `at`: jaws run along `jaw` (perpendicular to the structure), open and close
 * along `sep`; the shaft leaves the jaw base toward `port` (the bend at the base is the articulation).
 */
export function stapler(at: THREE.Vector3, jaw: THREE.Vector3, sep: THREE.Vector3, port: THREE.Vector3, radius: number, reload: number, jawLen = 45): Stapler {
  const g = new THREE.Group();
  const Z = jaw.clone().normalize(); const X = sep.clone().sub(Z.clone().multiplyScalar(sep.dot(Z))).normalize(); const Y = new THREE.Vector3().crossVectors(Z, X);
  const frame = new THREE.Group(); frame.matrixAutoUpdate = false;
  frame.matrix.makeBasis(X, Y, Z).setPosition(at); g.add(frame);
  const back = radius + 7;                               // jaw base sits this far behind the structure
  const T = 3.4, W = 9.5;
  const mkJaw = (mat: THREE.Material) => { const piv = new THREE.Group(); piv.position.set(0, 0, -back); const b = new THREE.Mesh(new THREE.BoxGeometry(T, W, jawLen), mat); b.position.z = jawLen / 2; piv.add(b); frame.add(piv); return { piv, b }; };
  const cartMat = new THREE.MeshStandardMaterial({ color: reload, metalness: 0.2, roughness: 0.45, emissive: 0x000000 });
  const anvil = mkJaw(metal()); const cart = mkJaw(cartMat);
  // knuckle at the articulation
  const knuckle = new THREE.Mesh(new THREE.SphereGeometry(5.6, 20, 14), ghost()); knuckle.position.set(0, 0, -back - 3); frame.add(knuckle);
  // shaft from the knuckle to the port, 12 mm diameter
  const base = new THREE.Vector3(0, 0, -back - 3).applyMatrix4(frame.matrix);
  const toPort = port.clone().sub(base); const L = Math.max(60, toPort.length() + 30);
  const shaft = rod(base, base.clone().addScaledVector(toPort.normalize(), L), 6, ghost()); g.add(shaft);
  const setClamp = (f: number) => {
    const open = THREE.MathUtils.degToRad(13) * (1 - f);
    const gap = T / 2 + 1.2 + (radius * 0.9) * (1 - f);
    anvil.piv.position.x = gap; anvil.piv.rotation.y = open;
    cart.piv.position.x = -gap; cart.piv.rotation.y = -open;
  };
  setClamp(0);
  return {
    group: g, setClamp,
    setFire: (f: number) => { cartMat.emissive.setRGB(0.5 * f, 0.45 * f, 0.9 * f); },
    dispose: () => g.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); } }),
  };
}

/** peanut (Kittner) dissector: gauze ellipsoid on a long holder; tip at the origin, aimed at the port */
export function peanut(): { group: THREE.Group; aim(tip: THREE.Vector3, port: THREE.Vector3): void } {
  const g = new THREE.Group();
  const gauze = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshStandardMaterial({ color: 0xf3eee2, roughness: 1, metalness: 0 }));
  gauze.scale.set(4.2, 4.2, 7); gauze.position.z = 5; g.add(gauze);
  const jaws = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 1.6, 10, 16), metal()); jaws.rotation.x = Math.PI / 2; jaws.position.z = 14; g.add(jaws);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.4, 260, 16), ghost(0xbcc4cc)); shaft.rotation.x = Math.PI / 2; shaft.position.z = 14 + 130; g.add(shaft);
  return {
    group: g,
    aim(tip, port) { g.position.copy(tip); g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), port.clone().sub(tip).normalize()); },
  };
}

/** a silk tie around a vessel: a torus in the plane perpendicular to the vessel */
export function tie(at: THREE.Vector3, dir: THREE.Vector3, radius: number): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.75, 10, 36), new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.55, metalness: 0.1 }));
  m.position.copy(at); m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.clone().normalize());
  m.renderOrder = 12;
  return m;
}

/** knot pusher: thin rod with a forked tip, used while the ties are cinched */
export function knotPusher(tip: THREE.Vector3, port: THREE.Vector3): THREE.Group {
  const g = new THREE.Group();
  g.add(rod(tip, tip.clone().addScaledVector(port.clone().sub(tip).normalize(), 240), 1.6, ghost(0xbcc4cc)));
  const fork = new THREE.Mesh(new THREE.TorusGeometry(2.2, 0.6, 8, 16, Math.PI * 1.4), metal());
  fork.position.copy(tip); fork.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), port.clone().sub(tip).normalize()); g.add(fork);
  return g;
}

/** a short run of staples laid along `dir`: two rows either side of the cut */
export function stapleRun(a: THREE.Vector3, b: THREE.Vector3, normal: THREE.Vector3): THREE.Group {
  const g = new THREE.Group();
  const d = b.clone().sub(a); const len = d.length(); const u = d.clone().normalize();
  const side = new THREE.Vector3().crossVectors(u, normal).normalize();
  const mat = new THREE.MeshStandardMaterial({ color: 0xeef2f5, metalness: 1, roughness: 0.18, emissive: 0x222831 });
  const n = Math.max(3, Math.round(len / 2.6));
  for (const s of [-1.6, 1.6]) for (let i = 0; i <= n; i++) {
    const p = a.clone().addScaledVector(u, (i / n) * len).addScaledVector(side, s);
    const st = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 2.2), mat);
    st.position.copy(p); st.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), u); g.add(st);
  }
  return g;
}

/** diathermy hook: insulated shaft, bare L-shaped tip; tip at the origin, aimed at the port */
export function hook(): { group: THREE.Group; aim(tip: THREE.Vector3, port: THREE.Vector3): void } {
  const g = new THREE.Group();
  const bare = metal();
  const up = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 7, 10), bare); up.position.set(0, 3.5, 0); g.add(up);         // the hook's short limb
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 9, 10), bare); neck.rotation.x = Math.PI / 2; neck.position.z = 4.5; g.add(neck);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(2.5, 2.5, 260, 16), ghost(0x2e333b)); shaft.rotation.x = Math.PI / 2; shaft.position.z = 9 + 130; g.add(shaft);
  return {
    group: g,
    aim(tip, port) { g.position.copy(tip); g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), port.clone().sub(tip).normalize()); },
  };
}

/**
 * Rib spreader (Finochietto type): two blades hooked over the ribs either side of the intercostal space, on arms
 * that slide along a toothed rack outside the chest. `c` is the centre of the space, `out` points out of the chest,
 * `along` runs along the ribs; the blades open along `sep` (roughly head-to-foot).
 */
export function ribSpreader(c: THREE.Vector3, out: THREE.Vector3, along: THREE.Vector3, sep: THREE.Vector3): { group: THREE.Group; setGap(mm: number): void } {
  const g = new THREE.Group();
  const O = out.clone().normalize(); const S = sep.clone().sub(O.clone().multiplyScalar(sep.dot(O))).normalize(); const A = new THREE.Vector3().crossVectors(S, O);
  const frame = new THREE.Group(); frame.matrixAutoUpdate = false; frame.matrix.makeBasis(A, S, O).setPosition(c); g.add(frame);
  const steel = metal();
  const blade = () => {
    const b = new THREE.Group();
    const plate = new THREE.Mesh(new THREE.BoxGeometry(64, 3, 26), steel); plate.position.z = -4; b.add(plate);          // goes into the chest
    const lip = new THREE.Mesh(new THREE.BoxGeometry(64, 8, 3), steel); lip.position.set(0, 0, 9); b.add(lip);            // hooks over the rib
    const arm = new THREE.Mesh(new THREE.BoxGeometry(10, 5, 34), steel); arm.position.set(-38, 0, 20); b.add(arm);        // up to the rack
    return b;
  };
  const b1 = blade(), b2 = blade(); frame.add(b1, b2);
  const rack = new THREE.Mesh(new THREE.BoxGeometry(8, 150, 5), steel); rack.position.set(-38, 0, 38); frame.add(rack);
  for (let k = -12; k <= 12; k++) { const t = new THREE.Mesh(new THREE.BoxGeometry(9, 1.2, 2), steel); t.position.set(-38, k * 5.5, 41); frame.add(t); }
  const crank = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 22, 12), dark()); crank.rotation.z = Math.PI / 2; crank.position.set(-26, 0, 44); frame.add(crank);
  const setGap = (mm: number) => { b1.position.y = mm / 2; b2.position.y = -mm / 2; crank.position.y = -mm / 2; };
  setGap(4);
  return { group: g, setGap };
}

/**
 * Vascular clamp (Satinsky / DeBakey type): two long atraumatic jaws closing across a vessel or the whole hilum.
 * Built like the stapler: jaws along `jaw`, opening along `sep`, the shaft leaving toward `port`. It stays on.
 */
export function vascularClamp(at: THREE.Vector3, jaw: THREE.Vector3, sep: THREE.Vector3, port: THREE.Vector3, radius: number, jawLen = 60): Stapler {
  const g = new THREE.Group();
  const Z = jaw.clone().normalize(); const X = sep.clone().sub(Z.clone().multiplyScalar(sep.dot(Z))).normalize(); const Y = new THREE.Vector3().crossVectors(Z, X);
  const frame = new THREE.Group(); frame.matrixAutoUpdate = false; frame.matrix.makeBasis(X, Y, Z).setPosition(at); g.add(frame);
  const back = radius + 8;
  const steel = metal();
  const mkJaw = () => {
    const piv = new THREE.Group(); piv.position.set(0, 0, -back);
    // slightly curved jaw: three short segments bending toward +Y at the tip (Satinsky curve)
    let z = 0; let y = 0;
    for (const [len, bend] of [[jawLen * 0.45, 0], [jawLen * 0.35, 0.18], [jawLen * 0.2, 0.42]] as [number, number][]) {
      const b = new THREE.Mesh(new THREE.BoxGeometry(3.4, 6, len), steel); b.rotation.x = -bend; b.position.set(0, y + Math.sin(bend) * len / 2, z + Math.cos(bend) * len / 2); piv.add(b);
      y += Math.sin(bend) * len; z += Math.cos(bend) * len;
    }
    frame.add(piv); return piv;
  };
  const j1 = mkJaw(), j2 = mkJaw();
  const box = new THREE.Mesh(new THREE.BoxGeometry(9, 9, 9), steel); box.position.set(0, 0, -back - 4); frame.add(box);
  const base = new THREE.Vector3(0, 0, -back - 4).applyMatrix4(frame.matrix);
  const toPort = port.clone().sub(base); const L = Math.max(80, toPort.length() + 40); const dir = toPort.normalize();
  for (const s of [-1, 1]) { const off = X.clone().multiplyScalar(s * 3); g.add(rod(base.clone().add(off), base.clone().add(off).addScaledVector(dir, L), 2.2, ghost(0xbcc4cc))); }
  // ratchet handles outside the chest, see-through so they never hide the field
  const end = base.clone().addScaledVector(dir, L);
  for (const s of [-1, 1]) { const ring = new THREE.Mesh(new THREE.TorusGeometry(7, 1.6, 10, 24), ghost(0xbcc4cc)); ring.position.copy(end).addScaledVector(X, s * 12); ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), Y); g.add(ring); }
  const setClamp = (f: number) => {
    const open = THREE.MathUtils.degToRad(16) * (1 - f); const gap = 2.4 + radius * 0.9 * (1 - f);
    j1.position.x = gap; j1.rotation.y = open; j2.position.x = -gap; j2.rotation.y = -open;
  };
  setClamp(0);
  return { group: g, setClamp, setFire: () => undefined, dispose: () => g.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); } }) };
}

/**
 * One pledgeted horizontal mattress stitch across a wound: two felt pledgets either side of the wound, joined by a
 * polypropylene loop over the top. `c` is the stitch centre, `across` runs across the wound, `n` out of the surface.
 */
export function pledgetStitch(c: THREE.Vector3, across: THREE.Vector3, n: THREE.Vector3): THREE.Group {
  const g = new THREE.Group();
  const A = across.clone().normalize(); const N = n.clone().normalize(); const B = new THREE.Vector3().crossVectors(N, A);
  const felt = new THREE.MeshStandardMaterial({ color: 0xf4f1ea, roughness: 0.95 });
  const thread = new THREE.MeshStandardMaterial({ color: 0x2d3fa0, roughness: 0.4 });
  for (const s of [-1, 1]) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(3.2, 7, 1.2), felt);
    p.position.copy(c).addScaledVector(A, s * 7).addScaledVector(N, 0.8);
    p.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(A, B, N)); g.add(p);
  }
  // the two limbs of the mattress, bridging the wound
  for (const s of [-1.8, 1.8]) {
    const pts = [-7, -3.5, 0, 3.5, 7].map((t) => c.clone().addScaledVector(A, t).addScaledVector(B, s).addScaledVector(N, 1.6 + 1.6 * Math.cos((t / 7) * Math.PI / 2)));
    g.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 16, 0.45, 6), thread));
  }
  return g;
}

/** curved needle in a needle holder: needle tip at `tip`, holder shaft toward `port` */
export function needleDriver(tip: THREE.Vector3, port: THREE.Vector3, n: THREE.Vector3): THREE.Group {
  const g = new THREE.Group();
  const u = port.clone().sub(tip).normalize();
  const nd = new THREE.Mesh(new THREE.TorusGeometry(6, 0.5, 8, 24, Math.PI), metal());
  nd.position.copy(tip).addScaledVector(n.clone().normalize(), 1); nd.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3().crossVectors(u, n).normalize()); g.add(nd);
  g.add(rod(tip.clone().addScaledVector(u, 6), tip.clone().addScaledVector(u, 200), 2.4, ghost(0xbcc4cc)));
  return g;
}

/** oscillating sternal saw blade (or a Gigli saw) across the sternum: a thin toothed plate */
export function sawBlade(): THREE.Group {
  const g = new THREE.Group();
  const b = new THREE.Mesh(new THREE.BoxGeometry(34, 0.8, 14), metal()); g.add(b);
  for (let k = -8; k <= 8; k++) { const t = new THREE.Mesh(new THREE.ConeGeometry(0.9, 2, 4), metal()); t.position.set(k * 2, 0, -8); t.rotation.x = Math.PI; g.add(t); }
  const body = new THREE.Mesh(new THREE.CylinderGeometry(9, 9, 60, 20), ghost(0x2e333b)); body.position.set(0, 0, 38); body.rotation.x = Math.PI / 2; g.add(body);
  return g;
}
