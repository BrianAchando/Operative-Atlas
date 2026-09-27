import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import type { Vec3 } from './ctview.ts';
import { stapler, peanut, hook, tie, knotPusher, stapleRun, RELOAD } from './instruments.ts';
import type { Action } from './procedure.ts';

export interface StructureMeta {
  id: string; name: string; group: string; colour: string; opacity: number; file: string;
  centroid: Vec3; bbox: [Vec3, Vec3]; label?: number; schematic?: boolean; visible?: boolean;
  /** staple line: point on the vessel/bronchus and the direction from proximal (kept) to distal (specimen) */
  division?: { point: Vec3; dir: Vec3; radius: number };
  note?: string;
}

interface Item { meta: StructureMeta; mesh: THREE.Mesh; mat: THREE.MeshPhysicalMaterial; hull?: THREE.Mesh; home: THREE.Vector3; ghost?: THREE.Mesh; distal?: THREE.Mesh; distalPlane?: THREE.Plane; staple?: THREE.Group; ties?: THREE.Mesh[] }

/**
 * In-vivo palette: what the tissues look like through a thoracoscope, kept distinct enough to teach with
 * (pulmonary artery dusky blue-violet, pulmonary veins deep red, bronchus white cartilage, lung pink with carbon
 * speckles, pericardial fat pale yellow, adult hilar nodes anthracotic grey-black).
 */
const REAL: Record<string, string> = {
  arteries: '#3b52a8', 'lul-intra-a': '#3b52a8', veins: '#8c1f3c', airway: '#e6dcc6', lungs: '#dc9d92', nerves: '#efe1a8', nodes: '#4d4845',
  aorta: '#b8382e', bct: '#b8382e', lcca: '#b8382e', lsca: '#b8382e', svc: '#5e6a92', lbcv: '#5e6a92', heart: '#d6b577', laa: '#b0645a',
  esophagus: '#d49a88', fissure: '#f1dfa0', skin: '#d9b8a4', 'lig-art': '#d8c9ae', bone: '#e9dec6',
};
function realColour(m: StructureMeta): string {
  if (REAL[m.id]) return REAL[m.id]!;
  if (m.group === 'lul-intra' || m.group === 'lll-intra') return m.colour.toLowerCase() === '#3f6fd8' ? REAL['arteries']! : m.colour.toLowerCase() === '#c2476f' ? REAL['veins']! : REAL['airway']!;
  if (m.group === 'chest-wall') return m.id === 'skin' ? REAL['skin']! : REAL['bone']!;
  if (m.id.startsWith('port-')) return m.colour;
  return REAL[m.group] ?? m.colour;
}

/** lung surface: soft mottling and scattered anthracotic speckles, in world space so retraction does not swim */
function lungShader(mat: THREE.MeshPhysicalMaterial): void {
  mat.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;').replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>
varying vec3 vW;
float h3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float vn(vec3 p) { vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(h3(i), h3(i + vec3(1,0,0)), f.x), mix(h3(i + vec3(0,1,0)), h3(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(h3(i + vec3(0,0,1)), h3(i + vec3(1,0,1)), f.x), mix(h3(i + vec3(0,1,1)), h3(i + vec3(1,1,1)), f.x), f.y), f.z); }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
float mott = vn(vW * 0.08) * 0.6 + vn(vW * 0.25) * 0.4;
diffuseColor.rgb *= 0.86 + 0.24 * mott;
float sp = vn(vW * 0.9);
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.16, 0.13, 0.14), smoothstep(0.83, 0.9, sp) * 0.75);`);
  };
}

/** see-through organs drawn as silhouettes: nearly clear face-on, denser at the edges, so the outline reads without a veil */
function rimShader(mat: THREE.MeshPhysicalMaterial): void {
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (sh, r) => {
    prev?.call(mat, sh, r);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nfloat rimF = 1.0 - abs(dot(normalize(normal), normalize(vViewPosition)));')
      .replace('#include <opaque_fragment>', 'diffuseColor.a = min(1.0, diffuseColor.a * (0.22 + 2.6 * pow(rimF, 2.2)));\n#include <opaque_fragment>');
  };
}

/** a slightly inflated back-face shell: the glowing outline for "working on" and "protect" */
function hullMaterial(): THREE.MeshBasicMaterial {
  const m = new THREE.MeshBasicMaterial({ color: 0x46c2c7, side: THREE.BackSide, transparent: true, opacity: 0.9, depthWrite: false, toneMapped: false });
  m.onBeforeCompile = (sh) => { sh.vertexShader = sh.vertexShader.replace('#include <begin_vertex>', 'vec3 transformed = vec3(position) + normal * 0.6;'); };
  return m;
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * The 3D side: the structures, the CT plane mirrored from the 2D viewer, a focus marker, floating labels, and the
 * operative effects (retraction, highlight, danger, stapled division, specimen removal). World units are RAS mm.
 */
export class Scene3D {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
  readonly controls: OrbitControls;
  readonly items = new Map<string, Item>();
  private plane: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>;
  private planeTex: THREE.CanvasTexture | null = null;
  private planeVersion = -1;
  private marker: THREE.Mesh;
  private labelsHost: HTMLElement;
  private labels = new Map<string, HTMLElement>();
  private raycaster = new THREE.Raycaster();
  private tween: { from: [THREE.Vector3, THREE.Vector3]; to: [THREE.Vector3, THREE.Vector3]; t0: number; ms: number } | null = null;
  private moves: { item: Item; from: THREE.Vector3; to: THREE.Vector3; op0: number; op1: number; t0: number; ms: number }[] = [];
  highlight = new Set<string>();
  danger = new Set<string>();
  /** render on demand: until this time (ms) the scene animates; after it, frames are drawn only when invalidated */
  private activeUntil = 0;
  private dirty = true;
  /** keep drawing for `ms` (animations, the highlight pulse after a step change) */
  invalidate(ms = 0): void { this.dirty = true; this.activeUntil = Math.max(this.activeUntil, performance.now() + ms); }
  onPick: (id: string | null, point: Vec3 | null) => void = () => undefined;
  onHover: (id: string | null) => void = () => undefined;
  planeSource: (() => { canvas: HTMLCanvasElement; corners: [Vec3, Vec3, Vec3, Vec3]; version: number; visible: boolean }) | null = null;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.localClippingEnabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 0.95;
    this.renderer.setClearColor(0x0b0f13);
    host.append(this.renderer.domElement);
    this.labelsHost = document.createElement('div'); this.labelsHost.className = 'labels3d'; host.append(this.labelsHost);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.12; this.controls.zoomToCursor = true;
    this.controls.addEventListener('change', () => this.invalidate(300));
    const hemi = new THREE.HemisphereLight(0xf3f6ff, 0x30343a, 0.8); hemi.position.set(0, 0, 1); this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.5); key.position.set(-300, 400, 500); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xbfd8ff, 0.6); rim.position.set(300, -300, 100); this.scene.add(rim);
    // head lamp, like a thoracoscope's light: follows the camera
    const lamp = new THREE.PointLight(0xfff1e0, 1.1, 0, 0); this.camera.add(lamp); this.scene.add(this.camera);
    const pm = new THREE.PMREMGenerator(this.renderer); this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture; pm.dispose();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Array(12).fill(0), 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 1, 1, 1, 0, 0, 1, 0], 2));
    g.setIndex([0, 2, 1, 2, 3, 1]);
    this.plane = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.92, depthWrite: true, toneMapped: false }));
    this.plane.renderOrder = -1; this.plane.frustumCulled = false; this.scene.add(this.plane);
    this.marker = new THREE.Mesh(new THREE.SphereGeometry(1.6, 20, 14), new THREE.MeshBasicMaterial({ color: 0x46c2c7, depthTest: false, transparent: true, opacity: 0.9 }));
    this.marker.renderOrder = 30; this.scene.add(this.marker);
    new ResizeObserver(() => this.resize()).observe(host);
    let down: [number, number] | null = null;
    const el = this.renderer.domElement;
    el.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; this.controls.autoRotate = false; });
    el.addEventListener('pointerup', (e) => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) this.pick(e, true); down = null; });
    el.addEventListener('pointermove', (e) => { if (!e.buttons) this.pick(e, false); });
    this.resize();
    this.renderer.setAnimationLoop((t) => this.tick(t));
  }

  resize(): void {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.invalidate();
  }

  async load(base: string, metas: StructureMeta[], onProgress: (f: number) => void): Promise<void> {
    const loader = new GLTFLoader(); loader.setMeshoptDecoder(MeshoptDecoder);
    let done = 0;
    await Promise.all(metas.map(async (m) => {
      const gltf = await loader.loadAsync(base + m.file);
      let geo: THREE.BufferGeometry | null = null;
      gltf.scene.updateMatrixWorld(true);
      let world = new THREE.Matrix4();
      gltf.scene.traverse((o) => { if ((o as THREE.Mesh).isMesh && !geo) { geo = (o as THREE.Mesh).geometry; world = o.matrixWorld.clone(); } });
      if (!geo) return;
      // meshopt ships quantised, normalised int16 positions with the scale in the node matrix; expand to float
      // before baking the matrix in, or setXYZ re-normalises and clamps every vertex to the unit cube
      const g = new THREE.BufferGeometry();
      const src = geo as THREE.BufferGeometry;
      const pos = src.getAttribute('position');
      const f = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) { f[i * 3] = pos.getX(i); f[i * 3 + 1] = pos.getY(i); f[i * 3 + 2] = pos.getZ(i); }
      g.setAttribute('position', new THREE.BufferAttribute(f, 3));
      if (src.index) g.setIndex(Array.from(src.index.array));
      g.applyMatrix4(world);
      geo = g;
      geo.computeVertexNormals();
      const wet = !['chest-wall', 'nodes', 'lungs'].includes(m.group) && !m.id.startsWith('port-');
      const mat = new THREE.MeshPhysicalMaterial({ color: realColour(m), roughness: m.group === 'lungs' ? 0.55 : 0.38, metalness: 0.0, clearcoat: wet ? 0.4 : 0, clearcoatRoughness: 0.25,
        transparent: m.opacity < 1, opacity: m.opacity, depthWrite: m.opacity >= 0.6, side: THREE.DoubleSide, envMapIntensity: m.group === 'lungs' ? 0.08 : 0.28 });
      if (m.schematic) mat.roughness = 0.55;
      if (m.group === 'lungs' && m.id !== 'fissure') lungShader(mat);
      if (m.group === 'lungs' || m.id === 'heart' || m.id === 'skin') rimShader(mat);
      // lungs and skin: front faces only, so a camera inside them (a retracted lobe, the chest wall) sees through
      if (m.group === 'lungs' || m.id === 'skin') mat.side = THREE.FrontSide;
      const mesh = new THREE.Mesh(geo, mat); mesh.name = m.id; mesh.userData['id'] = m.id; mesh.renderOrder = m.opacity < 1 ? 5 : 0;
      mesh.visible = m.visible !== false;
      this.scene.add(mesh);
      this.items.set(m.id, { meta: m, mesh, mat, home: new THREE.Vector3() });
      onProgress(++done / metas.length);
    }));
  }

  // ---------------------------------------------------------------- camera
  flyTo(eye: Vec3, target: Vec3, ms = 1100): void {
    this.tween = { from: [this.camera.position.clone(), this.controls.target.clone()], to: [new THREE.Vector3(...eye), new THREE.Vector3(...target)], t0: performance.now(), ms };
    this.invalidate(ms + 100);
  }

  /** frame a set of structures from a direction */
  frame(ids: string[] | null, dir: Vec3, pad = 1.35, ms = 900): void {
    const box = new THREE.Box3();
    for (const [id, it] of this.items) if ((!ids && it.mesh.visible) || ids?.includes(id)) box.union(new THREE.Box3(new THREE.Vector3(...it.meta.bbox[0]), new THREE.Vector3(...it.meta.bbox[1])));
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3()); const r = box.getSize(new THREE.Vector3()).length() / 2;
    const d = r * pad / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const eye = c.clone().add(new THREE.Vector3(...dir).normalize().multiplyScalar(d));
    this.flyTo([eye.x, eye.y, eye.z], [c.x, c.y, c.z], ms);
  }

  // ---------------------------------------------------------------- state
  setVisible(id: string, v: boolean): void { this.invalidate(); const it = this.items.get(id); if (it) { it.mesh.visible = v; if (it.distal) it.distal.visible = v && it.distal.visible; } }
  setOpacity(id: string, op: number): void {
    const it = this.items.get(id); if (!it) return;
    this.dirty = true;
    for (const m of [it.mat, it.distal?.material as THREE.MeshPhysicalMaterial | undefined]) if (m) { m.opacity = op; m.transparent = op < 1; m.depthWrite = op >= 0.6; m.needsUpdate = true; }
    it.mesh.renderOrder = op < 1 ? 5 : 0;
  }
  setFocus(p: Vec3): void { this.marker.position.set(...p); this.invalidate(); }

  /** move a structure away from home (retraction) with an eased animation */
  retract(id: string, offset: Vec3, opacity: number, ms = 900): void {
    const it = this.items.get(id); if (!it) return;
    this.moves.push({ item: it, from: it.mesh.position.clone(), to: new THREE.Vector3(...offset), op0: it.mat.opacity, op1: opacity, t0: performance.now(), ms });
    this.invalidate(ms + 100);
  }

  /**
   * Divide at the staple line: the proximal side stays (clipped), the distal side becomes a separate mesh that moves
   * with the specimen. `staple` lays two staple rows across the cut; `tie` expects ligatures placed by tieOff().
   */
  divide(id: string, animate = true, style: 'staple' | 'tie' = 'staple'): void {
    const it = this.items.get(id); const d = it?.meta.division; if (!it || !d || it.distal) return;
    const n = new THREE.Vector3(...d.dir).normalize(); const p = new THREE.Vector3(...d.point);
    const gap = style === 'tie' ? 1.2 : 2.2;   // mm between the two cut faces
    const keep = new THREE.Plane().setFromNormalAndCoplanarPoint(n.clone().negate(), p.clone().addScaledVector(n, -gap / 2));
    const go = new THREE.Plane().setFromNormalAndCoplanarPoint(n, p.clone().addScaledVector(n, gap / 2));
    it.mat.clippingPlanes = [keep]; it.mat.needsUpdate = true;
    const dm = it.mat.clone(); dm.clippingPlanes = [go.clone()];
    const distal = new THREE.Mesh(it.mesh.geometry, dm); distal.userData['id'] = id; distal.position.copy(it.mesh.position);
    this.scene.add(distal); it.distal = distal; it.distalPlane = go;
    if (style === 'staple') {
      const staple = new THREE.Group();
      const r = Math.max(3, d.radius * 1.35);
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
      for (const s of [-1, 1]) {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(r * 2.3, r * 0.55, 0.9), new THREE.MeshStandardMaterial({ color: 0xc9d1d9, metalness: 0.9, roughness: 0.25 }));
        bar.position.copy(p).addScaledVector(n, s * gap / 2); bar.quaternion.copy(q);
        for (let k = -3; k <= 3; k++) {
          const tick = new THREE.Mesh(new THREE.BoxGeometry(0.35, r * 0.8, 1.2), new THREE.MeshStandardMaterial({ color: 0xeef2f5, metalness: 1, roughness: 0.15 }));
          tick.position.set(k * r * 0.3, 0, 0); bar.add(tick);
        }
        staple.add(bar);
      }
      // the distal row belongs to the specimen: parent it to the distal stump so it travels with it
      const distalBar = staple.children[1]!; staple.remove(distalBar); distalBar.position.sub(distal.position); distal.add(distalBar);
      this.scene.add(staple); it.staple = staple;
      if (animate) { staple.scale.setScalar(0.01); const t0 = performance.now(); const grow = () => { const f = Math.min(1, (performance.now() - t0) / 350); staple.scale.setScalar(ease(f)); this.invalidate(); if (f < 1) requestAnimationFrame(grow); }; grow(); }
    }
    // open the cut a little so the two ends read as divided
    const sep = n.clone().multiplyScalar(style === 'tie' ? 3.5 : 3);
    if (animate) { const t0 = performance.now(); const from = distal.position.clone(); const stepf = () => { const f = Math.min(1, (performance.now() - t0) / 500); this.shiftDistal(it, from.clone().addScaledVector(sep, ease(f))); this.invalidate(); if (f < 1) requestAnimationFrame(stepf); }; stepf(); }
    else this.shiftDistal(it, distal.position.clone().add(sep));
    this.invalidate(600);
  }

  /** place the distal stump (its clipping plane lives in world space, so it moves with it) */
  private shiftDistal(it: Item, pos: THREE.Vector3): void {
    if (!it.distal || !it.distalPlane) return;
    it.distal.position.copy(pos);
    (it.distal.material as THREE.Material).clippingPlanes = [it.distalPlane.clone().translate(pos)];
  }

  /** ligatures: two on the kept side, one on the specimen side (moves with the distal stump) */
  tieOff(id: string): { meshes: THREE.Mesh[]; points: THREE.Vector3[] } {
    const it = this.items.get(id); const d = it?.meta.division; if (!it || !d) return { meshes: [], points: [] };
    const n = new THREE.Vector3(...d.dir).normalize(); const p = new THREE.Vector3(...d.point); const r = Math.max(2.2, d.radius * 0.95);
    const pts = [p.clone().addScaledVector(n, -6), p.clone().addScaledVector(n, -3.2), p.clone().addScaledVector(n, 2.6)];
    const meshes = pts.map((q) => tie(q, n, r));
    for (const m of meshes) this.scene.add(m);
    it.ties = meshes;
    return { meshes, points: pts };
  }

  undivide(id: string): void {
    const it = this.items.get(id); if (!it) return;
    if (it.distal) { this.scene.remove(it.distal); (it.distal.material as THREE.Material).dispose(); it.distal = undefined; it.distalPlane = undefined; }
    if (it.staple) { this.scene.remove(it.staple); it.staple = undefined; }
    for (const t of it.ties ?? []) { this.scene.remove(t); t.parent?.remove(t); }
    it.ties = undefined;
    it.mat.clippingPlanes = null; it.mat.needsUpdate = true;
    this.invalidate();
  }

  /** move every specimen part (the lobe and the distal ends of divided structures) along `offset` */
  moveSpecimen(ids: string[], offset: Vec3, opacity: number, ms = 1600): void {
    const to = new THREE.Vector3(...offset);
    this.invalidate(ms + 200);
    for (const id of ids) this.retract(id, offset, opacity, ms);
    for (const it of this.items.values()) if (it.distal) {
      const from = it.distal.position.clone(); const dest = from.clone().add(to); const t0 = performance.now();
      const step = () => {
        const f = ms <= 1 ? 1 : Math.min(1, (performance.now() - t0) / ms); const e = ease(f);
        const pos = from.clone().lerp(dest, e); this.shiftDistal(it, pos);
        this.invalidate(); if (f < 1) requestAnimationFrame(step);
      };
      step();
    }
  }

  resetOperative(): void {
    this.seq++;
    for (const [id, it] of this.items) { this.undivide(id); it.mesh.position.set(0, 0, 0); this.setOpacity(id, it.meta.opacity); it.mesh.visible = it.meta.visible !== false; }
    this.moves = [];
    this.clearTools(); this.clearExtras();
    this.controls.autoRotate = false;
  }

  // ---------------------------------------------------------------- instruments and actions
  private tools = new THREE.Group();
  private extras = new THREE.Group();
  private seq = 0;
  private toolsAdded = false;
  clearTools(): void {
    if (!this.toolsAdded) { this.scene.add(this.tools, this.extras); this.toolsAdded = true; }
    for (const c of [...this.tools.children]) { this.tools.remove(c); c.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); } }); }
    this.invalidate();
  }
  private clearExtras(): void { if (!this.toolsAdded) { this.scene.add(this.tools, this.extras); this.toolsAdded = true; } for (const c of [...this.extras.children]) this.extras.remove(c); }

  /** slow motion (1 = real time); also handy for teaching a step frame by frame */
  timeScale = 1;
  private anim(ms: number, fn: (e: number) => void, token: number): Promise<boolean> {
    ms = ms / this.timeScale;
    return new Promise((res) => {
      const t0 = performance.now();
      const f = () => {
        if (token !== this.seq) { res(false); return; }
        const x = Math.min(1, (performance.now() - t0) / ms); fn(ease(x)); this.invalidate(50);
        if (x < 1) requestAnimationFrame(f); else res(true);
      };
      f();
    });
  }
  private wait(ms: number, token: number): Promise<boolean> { return this.anim(ms, () => undefined, token); }

  /** geometry of a stapler clamped across structure `id`, coming through `port` */
  private stapleGeom(id: string, port: THREE.Vector3): { p: THREE.Vector3; jaw: THREE.Vector3; sep: THREE.Vector3; r: number } | null {
    const d = this.items.get(id)?.meta.division; if (!d) return null;
    const p = new THREE.Vector3(...d.point); const n = new THREE.Vector3(...d.dir).normalize();
    const u = port.clone().sub(p).normalize();
    let jaw = u.clone().negate().addScaledVector(n, u.dot(n)); if (jaw.lengthSq() < 1e-4) jaw = new THREE.Vector3(0, 0, 1).cross(n);
    jaw.normalize();
    return { p, jaw, sep: new THREE.Vector3().crossVectors(n, jaw).normalize(), r: d.radius };
  }

  /** show the instrument waiting at the start of the action */
  ready(a: Action, port: Vec3): void {
    this.clearTools(); const P = new THREE.Vector3(...port);
    // staplers arrive only when fired, so the waiting view shows the structure unobstructed
    if ((a.kind === 'dissect' || a.kind === 'open-fissure') && a.path?.length) {
      const pn = a.tool === 'hook' ? hook() : peanut(); pn.aim(new THREE.Vector3(...a.path[0]!), P); this.tools.add(pn.group);
    } else if (a.kind === 'ligate' && a.ids?.length) {
      const d = this.items.get(a.ids[0]!)?.meta.division; if (!d) return;
      const tip = new THREE.Vector3(...d.point).addScaledVector(new THREE.Vector3(...d.dir), -4.5);
      this.tools.add(knotPusher(tip.clone().addScaledVector(P.clone().sub(tip).normalize(), 25), P));
    }
    this.invalidate(300);
  }

  /** play the action; resolves true when it ran to the end (false if the step changed) */
  async play(a: Action, port: Vec3): Promise<boolean> {
    const token = ++this.seq; const P = new THREE.Vector3(...port);
    this.clearTools();
    if (a.kind === 'staple') {
      for (const id of a.ids ?? []) {
        const g = this.stapleGeom(id, P); if (!g) continue;
        const st = stapler(g.p, g.jaw, g.sep, P, g.r, RELOAD[a.reload ?? 'vascular']); this.tools.add(st.group);
        const back = g.jaw.clone().multiplyScalar(-30);
        if (!await this.anim(650, (e) => st.group.position.copy(back.clone().multiplyScalar(1 - e)), token)) return false;
        if (!await this.anim(450, (e) => st.setClamp(e), token)) return false;
        if (!await this.anim(380, (e) => st.setFire(Math.sin(e * Math.PI)), token)) return false;
        this.divide(id, true, 'staple');
        if (!await this.wait(250, token)) return false;
        if (!await this.anim(300, (e) => st.setClamp(1 - e), token)) return false;
        if (!await this.anim(550, (e) => st.group.position.copy(back.clone().multiplyScalar(e)), token)) return false;
        this.clearTools();
      }
    } else if (a.kind === 'ligate') {
      for (const id of a.ids ?? []) {
        const { meshes, points } = this.tieOff(id);
        for (const m of meshes) m.visible = false;
        for (let k = 0; k < meshes.length; k++) {
          const m = meshes[k]!; const q = points[k]!;
          const pusher = knotPusher(q.clone().addScaledVector(P.clone().sub(q).normalize(), 3), P); this.tools.add(pusher);
          m.visible = true;
          if (!await this.anim(520, (e) => m.scale.setScalar(2.4 - 1.4 * e), token)) return false;
          this.clearTools();
        }
        if (!await this.wait(200, token)) return false;
        this.divide(id, true, 'tie');
        // the specimen-side tie travels with the distal stump
        const it = this.items.get(id)!; const t2 = meshes[2]!;
        if (it.distal) { t2.position.sub(it.distal.position); it.distal.add(t2); }
        if (!await this.wait(500, token)) return false;
      }
    } else if (a.kind === 'dissect' || a.kind === 'open-fissure') {
      const path = (a.path ?? []).map((v) => new THREE.Vector3(...v)); if (!path.length) return true;
      const pn = a.tool === 'hook' ? hook() : peanut(); this.tools.add(pn.group);
      const curve = path.length > 1 ? new THREE.CatmullRomCurve3(path) : null;
      for (const s of a.spread ?? []) for (const id of s.ids) { const it = this.items.get(id); if (it) this.retract(id, s.offset, it.mat.opacity, 2600); }
      const ms = 2600;
      const ok = await this.anim(ms, (e) => {
        const tip = curve ? curve.getPointAt(e) : path[0]!.clone();
        const tan = curve ? curve.getTangentAt(e) : new THREE.Vector3(1, 0, 0);
        const side = new THREE.Vector3().crossVectors(tan, P.clone().sub(tip).normalize()).normalize();
        tip.addScaledVector(side, Math.sin(e * Math.PI * 9) * (a.tool === 'hook' ? 1.2 : 3.2));   // short sweeping strokes
        pn.aim(tip, P);
      }, token);
      if (!ok) return false;
      await this.wait(250, token); this.clearTools();
      for (const id of a.remove ?? []) this.fadeOut(id, 500);
    } else if (a.kind === 'staple-fissure') {
      const path = (a.path ?? []).map((v) => new THREE.Vector3(...v)); const N = new THREE.Vector3(...(a.normal ?? [0, 1, 0]));
      for (let i = 0; i < path.length - 1; i++) {
        const A = path[i]!, B = path[i + 1]!; const dir = B.clone().sub(A).normalize();
        const st = stapler(A, dir, N, P, 6, RELOAD[a.reload ?? 'tissue'], Math.min(60, A.distanceTo(B) + 12)); this.tools.add(st.group);
        const back = dir.clone().multiplyScalar(-30);
        if (!await this.anim(550, (e) => st.group.position.copy(back.clone().multiplyScalar(1 - e)), token)) return false;
        if (!await this.anim(400, (e) => st.setClamp(e), token)) return false;
        if (!await this.anim(350, (e) => st.setFire(Math.sin(e * Math.PI)), token)) return false;
        this.extras.add(stapleRun(A, B, N));
        if (!await this.anim(250, (e) => st.setClamp(1 - e), token)) return false;
        if (!await this.anim(450, (e) => st.group.position.copy(back.clone().multiplyScalar(e)), token)) return false;
        this.clearTools();
      }
      for (const s of a.spread ?? []) for (const id of s.ids) { const it = this.items.get(id); if (it) this.retract(id, s.offset, it.mat.opacity, 900); }
    }
    this.invalidate(400);
    return true;
  }

  private fadeOut(id: string, ms: number): void {
    const it = this.items.get(id); if (!it) return;
    const op0 = it.mat.opacity; const t0 = performance.now();
    const f = () => { const x = Math.min(1, (performance.now() - t0) / ms); this.setOpacity(id, op0 * (1 - x)); this.invalidate(); if (x < 1) requestAnimationFrame(f); else it.mesh.visible = false; };
    f();
  }

  /** the end state of an action already done (rebuilding the scene when the reader jumps between steps) */
  applyDone(a: Action): void {
    for (const id of a.remove ?? []) this.setVisible(id, false);
    if (a.kind === 'staple') for (const id of a.ids ?? []) this.divide(id, false, 'staple');
    else if (a.kind === 'ligate') for (const id of a.ids ?? []) {
      const { meshes } = this.tieOff(id); this.divide(id, false, 'tie');
      const it = this.items.get(id)!; const t2 = meshes[2]!; if (it.distal) { t2.position.sub(it.distal.position); it.distal.add(t2); }
    } else if (a.kind === 'open-fissure' || a.kind === 'staple-fissure') {
      for (const s of a.spread ?? []) for (const id of s.ids) { const it = this.items.get(id); if (it) this.retract(id, s.offset, it.mat.opacity, 1); }
      if (a.kind === 'staple-fissure') { const path = (a.path ?? []).map((v) => new THREE.Vector3(...v)); const N = new THREE.Vector3(...(a.normal ?? [0, 1, 0])); for (let i = 0; i < path.length - 1; i++) this.extras.add(stapleRun(path[i]!, path[i + 1]!, N)); }
    }
  }

  /** slow turntable for the anatomy overview */
  spin(on: boolean): void { this.controls.autoRotate = on; this.controls.autoRotateSpeed = 0.7; this.invalidate(); }

  // ---------------------------------------------------------------- labels
  setLabels(ids: string[], kind: (id: string) => 'hi' | 'danger' | 'plain'): void {
    for (const [id, el] of this.labels) if (!ids.includes(id)) { el.remove(); this.labels.delete(id); }
    for (const id of ids) {
      const it = this.items.get(id); if (!it) continue;
      let el = this.labels.get(id);
      if (!el) { el = document.createElement('div'); el.textContent = it.meta.name.replace(/ \(.*\)$/, ''); this.labelsHost.append(el); this.labels.set(id, el); }
      el.className = `lab3d ${kind(id)}`;
    }
    this.invalidate();
  }

  // ---------------------------------------------------------------- picking
  private pick(e: PointerEvent, click: boolean): void {
    const r = this.renderer.domElement.getBoundingClientRect();
    const m = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(m, this.camera);
    const meshes = [...this.items.values()].flatMap((it) => [it.mesh, ...(it.distal ? [it.distal] : [])]).filter((o) => o.visible && (o.material as THREE.Material).opacity > 0.15);
    const hits = this.raycaster.intersectObjects(meshes, false).filter((h) => {
      const planes = (h.object as THREE.Mesh).material as THREE.Material; return !planes.clippingPlanes?.some((pl) => pl.distanceToPoint(h.point) < 0);
    });
    const h = hits[0];
    const id = h ? String(h.object.userData['id']) : null;
    if (click) this.onPick(id, h ? [h.point.x, h.point.y, h.point.z] : null);
    else { this.onHover(id); this.renderer.domElement.style.cursor = id ? 'pointer' : ''; }
  }

  // ---------------------------------------------------------------- loop
  private tick(t: number): void {
    const now = performance.now();
    const src = this.planeSource?.();
    if (src && src.version !== this.planeVersion) this.dirty = true;
    if (!this.dirty && !this.tween && !this.moves.length && !this.controls.autoRotate && now > this.activeUntil) return;
    this.dirty = false;
    if (this.tween) {
      const f = Math.min(1, (performance.now() - this.tween.t0) / this.tween.ms); const e = ease(f);
      this.camera.position.lerpVectors(this.tween.from[0], this.tween.to[0], e);
      this.controls.target.lerpVectors(this.tween.from[1], this.tween.to[1], e);
      if (f >= 1) this.tween = null;
    }
    this.moves = this.moves.filter((mv) => {
      const f = Math.min(1, (performance.now() - mv.t0) / mv.ms); const e = ease(f);
      mv.item.mesh.position.lerpVectors(mv.from, mv.to, e);
      this.setOpacity(mv.item.meta.id, mv.op0 + (mv.op1 - mv.op0) * e);
      return f < 1;
    });
    this.controls.update();
    // highlight: a cyan glowing outline; danger: a red one. The tissue keeps its own colour.
    const pulse = now < this.activeUntil ? 0.5 + 0.5 * Math.sin(t / 380) : 1;
    for (const [id, it] of this.items) {
      const on = this.highlight.has(id), dg = this.danger.has(id);
      const sheet = it.meta.group === 'pleura' || it.meta.id === 'fissure';   // thin sheets glow instead of taking an outline
      if ((on || dg) && it.mesh.visible && !sheet) {
        if (!it.hull) { it.hull = new THREE.Mesh(it.mesh.geometry, hullMaterial()); it.hull.renderOrder = 4; it.mesh.add(it.hull); }
        const hm = it.hull.material as THREE.MeshBasicMaterial;
        hm.color.set(on ? 0x46e0e6 : 0xff4a55); hm.opacity = (on ? 0.5 : 0.35) + 0.35 * pulse; hm.clippingPlanes = it.mat.clippingPlanes;
        it.hull.visible = true;
      } else if (it.hull) it.hull.visible = false;
      const em = on ? 0.06 + 0.06 * pulse : 0;
      if (sheet && (on || dg)) it.mat.emissive.setRGB(on ? 0.05 : 0.35, on ? 0.3 + 0.15 * pulse : 0.05, on ? 0.32 + 0.15 * pulse : 0.07);
      else it.mat.emissive.setRGB(em, em, em);
      if (it.distal) (it.distal.material as THREE.MeshPhysicalMaterial).emissive.setRGB(em, em, em);
    }
    this.syncPlane();
    this.renderer.render(this.scene, this.camera);
    this.placeLabels();
  }

  private syncPlane(): void {
    const src = this.planeSource?.(); if (!src) return;
    this.plane.visible = src.visible;
    if (!src.visible) return;
    if (!this.planeTex || this.planeTex.image !== src.canvas) { this.planeTex?.dispose(); this.planeTex = new THREE.CanvasTexture(src.canvas); this.planeTex.colorSpace = THREE.SRGBColorSpace; this.plane.material.map = this.planeTex; this.plane.material.needsUpdate = true; }
    if (src.version !== this.planeVersion) {
      this.planeVersion = src.version; this.planeTex.needsUpdate = true;
      const pos = this.plane.geometry.getAttribute('position') as THREE.BufferAttribute;
      src.corners.forEach((c, i) => pos.setXYZ(i, c[0], c[1], c[2])); pos.needsUpdate = true;
    }
  }

  private placeLabels(): void {
    const w = this.host.clientWidth, h = this.host.clientHeight; const v = new THREE.Vector3();
    const placed: { x: number; y: number; el: HTMLElement; w: number }[] = [];
    for (const [id, el] of this.labels) {
      const it = this.items.get(id); if (!it) continue;
      const c = it.meta.division ? it.meta.division.point : it.meta.centroid;
      v.set(...c).add(it.mesh.position).project(this.camera);
      const vis = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 && it.mesh.visible;
      el.style.display = vis ? '' : 'none';
      if (vis) placed.push({ x: ((v.x + 1) / 2) * w, y: ((1 - v.y) / 2) * h, el, w: el.offsetWidth || 120 });
    }
    // greedy de-overlap: walk down the screen and push each label below any it would cover
    placed.sort((a, b) => a.y - b.y);
    const boxes: { x: number; y: number; w: number }[] = [];
    for (const p of placed) {
      let y = p.y;
      for (let guard = 0; guard < 20; guard++) {
        const hit = boxes.find((b) => Math.abs(b.y - y) < 20 && p.x < b.x + b.w + 6 && b.x < p.x + p.w + 6);
        if (!hit) break; y = hit.y + 21;
      }
      const x = Math.max(4, Math.min(p.x, w - p.w - 8));
      boxes.push({ x, y, w: p.w });
      p.el.style.transform = `translate(${x}px, ${y}px)`;
      p.el.classList.toggle('shifted', y !== p.y);
    }
  }
}
