import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import type { Vec3 } from './ctview.ts';
import { stapler, peanut, hook, tie, knotPusher, stapleRun, ribSpreader, vascularClamp, pledgetStitch, needleDriver, sawBlade, RELOAD } from './instruments.ts';
import type { Action } from './procedure.ts';

export interface StructureMeta {
  id: string; name: string; group: string; colour: string; opacity: number; file: string;
  centroid: Vec3; bbox: [Vec3, Vec3]; label?: number; schematic?: boolean; visible?: boolean;
  /** staple line: point on the vessel/bronchus and the direction from proximal (kept) to distal (specimen) */
  division?: { point: Vec3; dir: Vec3; radius: number };
  note?: string;
  /** left or right hilum: an operation shows only its own side */
  side?: 'left' | 'right';
  /** for right-side structures: visible when the right side is shown */
  sideVisible?: boolean;
}

interface Item { meta: StructureMeta; mesh: THREE.Mesh; mat: THREE.MeshPhysicalMaterial; hull?: THREE.Mesh; home: THREE.Vector3; ghost?: THREE.Mesh; distal?: THREE.Mesh; distalPlane?: THREE.Plane; keepPlane?: THREE.Plane; staple?: THREE.Group; ties?: THREE.Mesh[] }

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
  private moves: { item: Item; from: THREE.Vector3; to: THREE.Vector3; op0: number; op1: number; t0: number; ms: number; keepOp?: boolean }[] = [];
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
    this.moves.push({ item: it, from: it.mesh.position.clone(), to: new THREE.Vector3(...offset), op0: it.mat.opacity, op1: opacity < 0 ? it.mat.opacity : opacity, t0: performance.now(), ms, keepOp: opacity < 0 });
    this.invalidate(ms + 100);
  }

  /**
   * Divide at the staple line: the proximal side stays (clipped), the distal side becomes a separate mesh that moves
   * with the specimen. `staple` lays two staple rows across the cut; `tie` expects ligatures placed by tieOff().
   */
  divide(id: string, animate = true, style: 'staple' | 'tie' | 'cut' = 'staple', over?: { point: Vec3; dir: Vec3; radius: number }): void {
    const it = this.items.get(id); const d = over ?? it?.meta.division; if (!it || !d || it.distal) return;
    const n = new THREE.Vector3(...d.dir).normalize(); const p = new THREE.Vector3(...d.point);
    const gap = style === 'tie' ? 1.2 : style === 'cut' ? 1.5 : 2.2;   // mm between the two cut faces
    const keep = new THREE.Plane().setFromNormalAndCoplanarPoint(n.clone().negate(), p.clone().addScaledVector(n, -gap / 2));
    const go = new THREE.Plane().setFromNormalAndCoplanarPoint(n, p.clone().addScaledVector(n, gap / 2));
    it.mat.clippingPlanes = [keep]; it.mat.needsUpdate = true; it.keepPlane = keep; it.keepPlane = keep;
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
    const sep = n.clone().multiplyScalar(style === 'tie' ? 3.5 : style === 'cut' ? 1.5 : 3);
    if (animate) { const t0 = performance.now(); const from = distal.position.clone(); const stepf = () => { const f = Math.min(1, (performance.now() - t0) / 500); this.shiftDistal(it, from.clone().addScaledVector(sep, ease(f))); this.invalidate(); if (f < 1) requestAnimationFrame(stepf); }; stepf(); }
    else this.shiftDistal(it, distal.position.clone().add(sep));
    this.invalidate(600);
  }

  /** move the kept side of a divided structure (its clipping plane moves with it) */
  private shiftKept(it: Item, pos: THREE.Vector3): void {
    if (!it.keepPlane) return;
    it.mesh.position.copy(pos); it.mat.clippingPlanes = [it.keepPlane.clone().translate(pos)];
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
    it.mat.clippingPlanes = null; it.mat.needsUpdate = true; it.keepPlane = undefined;
    this.invalidate();
  }

  /** move every specimen part (the lobe and the distal ends of divided structures) along `offset` */
  moveSpecimen(ids: string[], offset: Vec3, opacity: number, ms = 1600, only?: string[]): void {
    const to = new THREE.Vector3(...offset);
    this.invalidate(ms + 200);
    for (const id of ids) this.retract(id, offset, opacity, ms);
    for (const it of this.items.values()) if (it.distal && (!only || only.includes(it.meta.id))) {
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
    for (const [id, it] of this.items) { this.undivide(id); it.mesh.position.set(0, 0, 0); it.mesh.quaternion.identity(); it.mesh.scale.setScalar(1); this.setOpacity(id, it.meta.opacity); it.mesh.visible = it.meta.visible !== false; }
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
  private stapleGeom(id: string, port: THREE.Vector3, over?: { point: Vec3; dir: Vec3; radius: number }): { p: THREE.Vector3; jaw: THREE.Vector3; sep: THREE.Vector3; r: number } | null {
    const d = over ?? this.items.get(id)?.meta.division; if (!d) return null;
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
    const over = a.at && a.kind !== 'clamp' ? { point: a.at, dir: a.axis ?? [0, 0, 1] as Vec3, radius: a.radius ?? 8 } : undefined;
    if (a.kind === 'staple') {
      for (const id of a.ids ?? []) {
        const g = this.stapleGeom(id, P, over); if (!g) continue;
        const st = stapler(g.p, g.jaw, g.sep, P, g.r, RELOAD[a.reload ?? 'vascular']); this.tools.add(st.group);
        const back = g.jaw.clone().multiplyScalar(-30);
        if (!await this.anim(650, (e) => st.group.position.copy(back.clone().multiplyScalar(1 - e)), token)) return false;
        if (!await this.anim(450, (e) => st.setClamp(e), token)) return false;
        if (!await this.anim(380, (e) => st.setFire(Math.sin(e * Math.PI)), token)) return false;
        this.divide(id, true, 'staple', over);
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
        if (a.keep) continue;
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
    } else if (a.kind === 'thoracotomy') {
      for (const id of a.show ?? []) { this.setVisible(id, true); this.setOpacity(id, 0.5); }
      if (a.incision) { this.setVisible(a.incision, true); const it = this.items.get(a.incision); if (it) { if (!await this.anim(700, (e) => this.setOpacity(a.incision!, e), token)) return false; } }
      const sp = this.spreader(P, a); if (!sp) return true;
      if (!await this.wait(300, token)) return false;
      for (const [k, id] of (a.ribs ?? []).entries()) { const it = this.items.get(id); if (it) this.retract(id, sp.shift(k), -1, 1600); }
      if (!await this.anim(1600, (e) => sp.set(4 + 66 * e), token)) return false;
    } else if (a.kind === 'saw') {
      const id = a.ids?.[0]; const d = over ?? (id ? this.items.get(id)?.meta.division : undefined);
      if (d) {
        const p = new THREE.Vector3(...d.point); const blade = sawBlade(); this.tools.add(blade);
        // blade lies in the cut plane, teeth down into the bone, body out of the chest
        const n = new THREE.Vector3(...d.dir).normalize(); const out = new THREE.Vector3(0, 1, 0).addScaledVector(n, -n.y).normalize();
        blade.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3().crossVectors(n, out), n, out));
        if (!await this.anim(1800, (e) => { blade.position.copy(p).addScaledVector(out, 10 - 16 * e).addScaledVector(new THREE.Vector3().crossVectors(n, out), Math.sin(e * Math.PI * 28) * 3); }, token)) return false;
        this.clearTools(); this.divide(id!, true, 'cut', over);
        if (!await this.wait(650, token)) return false;
        if (a.open) { const o = a.open; if (!await this.anim(1400, (e) => this.openHalves(id!, a, o * e), token)) return false; }
      }
      if (a.hinge) { const h = a.hinge; if (!await this.anim(1800, (e) => this.turn(h, e), token)) return false; }
    } else if (a.kind === 'twist') {
      if (a.hinge) { const h = a.hinge; if (!await this.anim(2600, (e) => this.turn(h, e), token)) return false; }
    } else if (a.kind === 'clamp') {
      const c = this.clampAt(a, P); if (!c) return true;
      const dir = new THREE.Vector3(...(a.at ?? [0, 0, 0])).sub(P).normalize().multiplyScalar(-40);
      if (!await this.anim(800, (e) => c.group.position.copy(dir.clone().multiplyScalar(1 - e)), token)) return false;
      if (!await this.anim(600, (e) => c.setClamp(e), token)) return false;
    } else if (a.kind === 'suture') {
      for (const s of this.stitches(a)) {
        const nd = needleDriver(s.c.clone().addScaledVector(s.across, 7), P, s.n); this.tools.add(nd);
        if (!await this.anim(600, (e) => nd.position.copy(s.across.clone().multiplyScalar(-14 * e)), token)) return false;
        this.clearTools();
        const st = pledgetStitch(s.c, s.across, s.n); st.scale.setScalar(0.01); this.extras.add(st);
        if (!await this.anim(350, (e) => { st.scale.setScalar(Math.max(0.01, e)); st.position.copy(s.c.clone().multiplyScalar(1 - Math.max(0.01, e))); }, token)) return false;
      }
      for (const id of a.remove ?? []) this.fadeOut(id, 600);
    } else if (a.kind === 'layers') {
      for (const L of a.layers ?? []) {
        const it = this.items.get(L.id); if (!it) continue;
        this.setVisible(L.id, true); this.highlight = new Set([L.id]); this.setLabels([L.id], () => 'hi');
        if (!await this.wait(700, token)) return false;
        if (!await this.layer(L, true, token)) return false;
        if (!await this.wait(350, token)) return false;
      }
      this.highlight = new Set();
    } else if (a.kind === 'sternotomy') {
      const id = a.ids?.[0] ?? 'sternum'; const it = this.items.get(id);
      if (it && a.at && a.axis && a.path && a.path.length > 1) {
        const n = new THREE.Vector3(...a.axis).normalize(); const out = new THREE.Vector3(0, 1, 0);
        const blade = sawBlade(); this.tools.add(blade);
        blade.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3().crossVectors(n, out), n, out));
        const A = new THREE.Vector3(...a.path[0]!), B = new THREE.Vector3(...a.path[a.path.length - 1]!);
        if (!await this.anim(2600, (e) => { blade.position.copy(A).lerp(B, e).addScaledVector(out, 4 + Math.sin(e * Math.PI * 40) * 2); }, token)) return false;
        this.clearTools(); this.divide(id, true, 'cut', { point: a.at, dir: a.axis, radius: 30 });
        if (!await this.wait(600, token)) return false;
        const sp = this.sternalRetractor(a); const w = a.radius ?? 70;
        if (!await this.anim(1600, (e) => { this.openSternum(it, n, (w / 2) * e); sp?.setGap(20 + w * e); }, token)) return false;
      }
    } else if (a.kind === 'reveal') {
      for (const id of a.ids ?? []) { const it = this.items.get(id); if (!it) continue; this.setVisible(id, true); const op = it.meta.opacity; if (!await this.anim(1200, (e) => this.setOpacity(id, Math.max(0.02, op * e)), token)) return false; }
    } else if (a.kind === 'decorticate') {
      for (const id of a.ids ?? []) this.fadeOut(id, 2200 / this.timeScale);
      const ex = a.expand;
      if (ex) { if (!await this.anim(2600, (e) => this.scaleAbout(ex.ids, ex.pivot, ex.from + (1 - ex.from) * e), token)) return false; }
    } else if (a.kind === 'massage') {
      const id = a.ids?.[0] ?? 'heart'; const it = this.items.get(id);
      if (it) {
        const c = new THREE.Vector3(...it.meta.centroid);
        const ok = await this.anim(3600, (e) => { const s = 1 - 0.09 * Math.max(0, Math.sin(e * Math.PI * 12)); it.mesh.scale.set(s, s, s); it.mesh.position.copy(c).multiplyScalar(1 - s); }, token);
        it.mesh.scale.setScalar(1); it.mesh.position.set(0, 0, 0);
        if (!ok) return false;
      }
    }
    for (const id of a.show ?? []) this.setVisible(id, true);
    this.invalidate(400);
    return true;
  }

  /** the rib spreader seated in the thoracotomy at `P`, with the rib offsets it produces when opened */
  private spreader(P: THREE.Vector3, a: Action): { set(mm: number): void; shift(k: number): Vec3; ghost(): void } | null {
    const [up, lo] = (a.ribs ?? []).map((id) => this.items.get(id)); if (!up || !lo) return null;
    const cu = new THREE.Vector3(...up.meta.centroid), cl = new THREE.Vector3(...lo.meta.centroid);
    const sep = cu.clone().sub(cl).normalize();                          // lower rib -> upper rib
    const out = new THREE.Vector3(P.x, P.y, 0).normalize();               // away from the midline, in the axial plane
    const along = new THREE.Vector3().crossVectors(sep, out);
    const sp = ribSpreader(P, out, along, sep); this.extras.add(sp.group);
    const ghost = () => sp.group.traverse((o) => { const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined; if (m) { m.transparent = true; m.opacity = 0.28; m.depthWrite = false; } });
    return { ghost, set: (mm) => { sp.setGap(mm); this.invalidate(); }, shift: (k) => { const v = sep.clone().multiplyScalar(k === 0 ? 30 : -30); return [v.x, v.y, v.z]; } };
  }

  /** a sawn bone opened by a retractor: each half moves `mm / 2` away from the cut; the retractor sits between them */
  private openHalves(id: string, a: Action, mm: number): void {
    const it = this.items.get(id); if (!it?.distal) return;
    const n = new THREE.Vector3(...(a.axis ?? it.meta.division?.dir ?? [1, 0, 0])).normalize();
    this.shiftKept(it, n.clone().multiplyScalar(-mm / 2));
    const d0 = it.distalPlane!; it.distal.position.copy(n.clone().multiplyScalar(mm / 2)); (it.distal.material as THREE.Material).clippingPlanes = [d0.clone().translate(it.distal.position)];
    let r = this.extras.getObjectByName('sternal-retractor') as THREE.Group | undefined;
    if (!r && a.at) {
      const c = new THREE.Vector3(...a.at).add(new THREE.Vector3(0, 14, 0));
      const sp = ribSpreader(c, new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), n); sp.group.name = 'sternal-retractor'; this.extras.add(sp.group);
      (sp.group as unknown as { setGap: (m: number) => void }).setGap = sp.setGap; r = sp.group;
    }
    (r as unknown as { setGap?: (m: number) => void })?.setGap?.(Math.max(4, mm));
    this.invalidate(50);
  }

  /** one layer of the chest wall: cut across and opened, split, retracted, passed through or spared */
  private async layer(L: NonNullable<Action['layers']>[number], animate: boolean, token: number): Promise<boolean> {
    const it = this.items.get(L.id); if (!it) return true;
    if ((L.fate === 'divide' || L.fate === 'split') && L.point && L.dir) {
      this.divide(L.id, animate, 'cut', { point: L.point, dir: L.dir, radius: 40 });
      const open = new THREE.Vector3(...L.dir).normalize().multiplyScalar(L.open ?? (L.fate === 'split' ? 8 : 16));
      if (it.distal) {
        const from = it.distal.position.clone();
        if (animate) { if (!await this.anim(900, (e) => this.shiftDistal(it, from.clone().addScaledVector(open, e)), token)) return false; }
        else this.shiftDistal(it, from.clone().add(open));
      }
    } else if (L.fate === 'retract' && L.offset) {
      this.retract(L.id, L.offset, -1, animate ? 1000 / this.timeScale : 1);
      if (animate && !await this.wait(1000, token)) return false;
    } else if (L.fate === 'through') {
      this.setOpacity(L.id, Math.min(it.mat.opacity, 0.35));
    }
    return true;
  }

  /** the two halves of a divided sternum pushed apart by `half` mm each */
  private openSternum(it: Item, n: THREE.Vector3, half: number): void {
    if (it.distal && it.distalPlane) { const pos = n.clone().multiplyScalar(half + 0.75); this.shiftDistal(it, pos); }
    if (it.keepPlane) { it.mesh.position.copy(n.clone().multiplyScalar(-half)); it.mat.clippingPlanes = [it.keepPlane.clone().translate(it.mesh.position)]; }
    this.invalidate(50);
  }

  /** a sternal retractor seated in the sternotomy (the rib spreader turned on its side) */
  private sternalRetractor(a: Action): { setGap(mm: number): void } | null {
    if (!a.path || a.path.length < 2 || !a.axis) return null;
    const A = new THREE.Vector3(...a.path[0]!), B = new THREE.Vector3(...a.path[a.path.length - 1]!);
    const c = A.clone().lerp(B, 0.55).add(new THREE.Vector3(0, 6, 0));
    const sp = ribSpreader(c, new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(...a.axis));
    this.extras.add(sp.group); return sp;
  }

  /** scale structures about a point (a trapped lung, drawn smaller about its hilum) */
  scaleAbout(ids: string[], pivot: Vec3, s: number): void {
    const P = new THREE.Vector3(...pivot);
    for (const id of ids) { const it = this.items.get(id); if (!it) continue; it.mesh.scale.setScalar(s); it.mesh.position.copy(P).multiplyScalar(1 - s); }
    this.invalidate(50);
  }

  // ---------------------------------------------------------------- patient position
  private stage = new THREE.Group();
  private stageAdded = false;
  /** lateral decubitus, operated side up: the camera's up turns to that side, and the table, roll and arm rests appear */
  setPose(pose: 'lateral' | undefined, side: 'left' | 'right' | 'both'): void {
    if (!this.stageAdded) { this.scene.add(this.stage); this.stageAdded = true; }
    for (const c of [...this.stage.children]) { this.stage.remove(c); c.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); } }); }
    const lat = pose === 'lateral' && side !== 'both';
    const s = side === 'right' ? 1 : -1;                                   // the side that is up
    this.camera.up.set(lat ? s : 0, 0, lat ? 0 : 1);
    const ctl = this.controls as unknown as { _quat: THREE.Quaternion; _quatInverse: THREE.Quaternion };
    ctl._quat.setFromUnitVectors(this.camera.up, new THREE.Vector3(0, 1, 0)); ctl._quatInverse.copy(ctl._quat).invert();
    const skin = this.items.get('skin');
    const ribX = (sd: 'l' | 'r') => { let v = sd === 'l' ? 0 : 0; for (const [id, it] of this.items) if (id.startsWith('rib-') && id.endsWith(`-${sd}`)) v = sd === 'l' ? Math.min(v, it.meta.bbox[0][0]) : Math.max(v, it.meta.bbox[1][0]); return v; };
    const xl = ribX('l') - 42, xr = ribX('r') + 42;
    if (skin) { skin.mat.clippingPlanes = lat ? [new THREE.Plane(new THREE.Vector3(1, 0, 0), -xl), new THREE.Plane(new THREE.Vector3(-1, 0, 0), xr)] : null; skin.mat.needsUpdate = true; }
    this.invalidate(300);
    if (!lat) return;
    // the table under the dependent side, broken (flexed) below the costal margin to open the spaces on the upper side
    const down = -s; const xt = down > 0 ? xr + 6 : xl - 6;                // table surface
    const mat = (c: number, o = 1) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.7, metalness: 0.15, transparent: o < 1, opacity: o });
    const zb = -250; const tilt = THREE.MathUtils.degToRad(12);
    for (const [z0, z1, sign] of [[zb, 380, 1], [-720, zb, -1]] as [number, number, number][]) {
      const g = new THREE.Group(); g.position.set(xt, 0, zb);
      const len = Math.abs(z1 - z0); const mid = (z0 + z1) / 2 - zb;
      const pad = new THREE.Mesh(new THREE.BoxGeometry(70, 540, len), mat(0x38414d)); pad.position.set(down * 35, 0, mid); g.add(pad);
      const top = new THREE.Mesh(new THREE.BoxGeometry(6, 540, len), mat(0x5b6b80)); top.position.set(down * 2, 0, mid); g.add(top);
      g.rotation.y = down * sign * tilt; this.stage.add(g);
    }
    // axillary roll: under the dependent chest, a hand's breadth below the axilla, keeping weight off the brachial plexus
    const r3 = this.items.get(`rib-4-${down > 0 ? 'r' : 'l'}`); const za = r3 ? r3.meta.bbox[0][2] - 10 : -80;
    const roll = new THREE.Mesh(new THREE.CylinderGeometry(32, 32, 300, 28), mat(0x5aa2c8, 0.9)); roll.position.set(xt - down * 30, 10, za); this.stage.add(roll);
    // arms forward: the dependent arm on an arm board, the upper arm on a rest above it, both shoulders flexed about 90 degrees
    const skinM = (o: number) => new THREE.MeshStandardMaterial({ color: 0xd9b8a4, roughness: 0.8, transparent: true, opacity: o, depthWrite: false });
    const hum = (sd: 'l' | 'r') => { const h = this.items.get(`humerus-${sd}`); return h ? new THREE.Vector3(...h.meta.centroid) : new THREE.Vector3(sd === 'l' ? -170 : 170, -10, 40); };
    const arm = (sh: THREE.Vector3, lift: number) => {
      const a = sh.clone(); const b = sh.clone().add(new THREE.Vector3(-down * lift, 300, 30)); const c = b.clone().add(new THREE.Vector3(0, 250, 40));
      for (const [p, q, r] of [[a, b, 34], [b, c, 28]] as [THREE.Vector3, THREE.Vector3, number][]) {
        const d = q.clone().sub(p); const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, d.length(), 8, 16), skinM(0.3));
        m.position.copy(p).addScaledVector(d, 0.5); m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize()); this.stage.add(m);
      }
      const board = new THREE.Mesh(new THREE.BoxGeometry(12, 380, 120), mat(0x38414d, 0.7)); board.position.copy(b).add(new THREE.Vector3(down * 40, 110, 20)); this.stage.add(board);
    };
    arm(hum(down > 0 ? 'r' : 'l').setX(xt - down * 70), 0);
    arm(hum(down > 0 ? 'l' : 'r'), 90);
    // head on a pillow, level with the spine
    const head = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 18), skinM(0.35)); head.scale.set(75, 95, 110); head.position.set(0, 10, 230); this.stage.add(head);
    const pillow = new THREE.Mesh(new THREE.BoxGeometry(110, 260, 260), mat(0x4d6e8a, 0.95)); pillow.position.set(xt - down * 55 + (down > 0 ? 0 : 0), 10, 230); this.stage.add(pillow);
  }

  /** turn structures (or the divided distal part, e.g. the upper sternum) by `f` of the hinge angle about its axis */
  private turn(h: NonNullable<Action['hinge']>, f: number): void {
    const pivot = new THREE.Vector3(...h.pivot);
    const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(...h.axis).normalize(), THREE.MathUtils.degToRad(h.angle) * f);
    for (const id of h.ids) {
      const it = this.items.get(id); if (!it) continue;
      const o = it.distal ?? it.mesh;
      o.quaternion.copy(q); o.position.copy(pivot).sub(pivot.clone().applyQuaternion(q));
      if (it.distal && it.distalPlane) { o.updateMatrixWorld(true); (it.distal.material as THREE.Material).clippingPlanes = [it.distalPlane.clone().applyMatrix4(o.matrixWorld)]; }
      this.moves = this.moves.filter((m) => m.item !== it || it.distal);
    }
    this.invalidate(50);
  }

  /** a vascular clamp closed across the action's point (kept on until the end of the operation) */
  private clampAt(a: Action, P: THREE.Vector3): ReturnType<typeof vascularClamp> | null {
    if (!a.at) return null;
    const p = new THREE.Vector3(...a.at); const n = new THREE.Vector3(...(a.axis ?? [0, 0, 1])).normalize();
    const u = P.clone().sub(p).normalize();
    let jaw = u.clone().negate().addScaledVector(n, u.dot(n)); if (jaw.lengthSq() < 1e-4) jaw = new THREE.Vector3(0, 0, 1).cross(n);
    jaw.normalize();
    const c = vascularClamp(p, jaw, new THREE.Vector3().crossVectors(n, jaw), P, a.radius ?? 10, a.jawLen ?? 60);
    this.extras.add(c.group); return c;
  }

  /** stitch centres along a wound, with the direction across it */
  private stitches(a: Action): { c: THREE.Vector3; across: THREE.Vector3; n: THREE.Vector3 }[] {
    const n = new THREE.Vector3(...(a.normal ?? [0, 1, 0])).normalize(); const ax = new THREE.Vector3(...(a.axis ?? [1, 0, 0])).normalize();
    const across = new THREE.Vector3().crossVectors(n, ax).normalize();
    return (a.path ?? []).map((v) => ({ c: new THREE.Vector3(...v), across, n }));
  }

  private fadeOut(id: string, ms: number): void {
    const it = this.items.get(id); if (!it) return;
    const op0 = it.mat.opacity; const t0 = performance.now();
    const f = () => { const x = Math.min(1, (performance.now() - t0) / ms); this.setOpacity(id, op0 * (1 - x)); this.invalidate(); if (x < 1) requestAnimationFrame(f); else it.mesh.visible = false; };
    f();
  }

  /** the end state of an action already done (rebuilding the scene when the reader jumps between steps) */
  applyDone(a: Action, port?: Vec3): void {
    const over = a.at && a.kind !== 'clamp' ? { point: a.at, dir: a.axis ?? [0, 0, 1] as Vec3, radius: a.radius ?? 8 } : undefined;
    for (const id of a.remove ?? []) this.setVisible(id, false);
    for (const id of a.show ?? []) { this.setVisible(id, true); if (a.kind === 'thoracotomy') this.setOpacity(id, 0.5); }
    if (a.kind === 'thoracotomy' && port) {
      const sp = this.spreader(new THREE.Vector3(...port), a);
      if (sp) { sp.set(70); sp.ghost(); for (const [k, id] of (a.ribs ?? []).entries()) { const it = this.items.get(id); if (it) this.retract(id, sp.shift(k), -1, 1); } }
    }
    if (a.kind === 'sternotomy') {
      const it = this.items.get(a.ids?.[0] ?? 'sternum');
      if (it && a.at && a.axis) { this.divide(it.meta.id, false, 'cut', { point: a.at, dir: a.axis, radius: 30 }); const w = a.radius ?? 70; this.openSternum(it, new THREE.Vector3(...a.axis).normalize(), w / 2); this.sternalRetractor(a)?.setGap(20 + w); }
    } else if (a.kind === 'reveal') { for (const id of a.ids ?? []) this.setVisible(id, true); }
    else if (a.kind === 'decorticate') { for (const id of a.ids ?? []) this.setVisible(id, false); }
    else if (a.kind === 'layers') { for (const L of a.layers ?? []) { this.setVisible(L.id, true); void this.layer(L, false, this.seq); } }
    else if (a.kind === 'saw') { for (const id of a.ids ?? []) { this.divide(id, false, 'cut'); } if (a.hinge) this.turn(a.hinge, 1); }
    else if (a.kind === 'twist') { if (a.hinge) this.turn(a.hinge, 1); }
    else if (a.kind === 'clamp' && port) { this.clampAt(a, new THREE.Vector3(...port))?.setClamp(1); }
    else if (a.kind === 'suture') { for (const s of this.stitches(a)) this.extras.add(pledgetStitch(s.c, s.across, s.n)); }
    else if (a.kind === 'staple') for (const id of a.ids ?? []) this.divide(id, false, 'staple', a.at && a.axis ? { point: a.at, dir: a.axis, radius: a.radius ?? 8 } : undefined);
    else if (a.kind === 'ligate') for (const id of a.ids ?? []) {
      const { meshes } = this.tieOff(id); if (a.keep) continue; this.divide(id, false, 'tie');
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
      if (!mv.keepOp) this.setOpacity(mv.item.meta.id, mv.op0 + (mv.op1 - mv.op0) * e);
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
      it.mesh.updateMatrixWorld(); v.set(...c).applyMatrix4(it.mesh.matrixWorld).project(this.camera);
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
