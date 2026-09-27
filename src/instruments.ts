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
