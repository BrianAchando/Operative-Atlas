import * as THREE from 'three';

/**
 * Procedural tissue surfaces: every structure gets a surface that reads as the real tissue under an operating light,
 * with no image textures to download. Each tissue class adds, in object space (so a retracted organ keeps its
 * pattern): a colour variation, a characteristic pattern (muscle fibres, fat lobules, vasa vasorum, bone pores,
 * cartilage rings, a woven graft), micro-relief as a bump on the normal, and wet or matte highlights.
 */
export type Tissue = 'muscle' | 'myocardium' | 'artery' | 'pulm-artery' | 'vein' | 'bone' | 'cartilage' | 'fat' | 'skin' | 'organ' | 'liver'
  | 'valve' | 'nerve' | 'node' | 'metal' | 'fabric' | 'suture' | 'plastic' | 'calcium' | 'plain';

interface Meta { id: string; group: string; schematic?: boolean }

const has = (id: string, ...xs: string[]) => xs.some((x) => id === x || id.startsWith(x));

export function tissueOf(m: Meta): Tissue {
  const id = m.id, g = m.group;
  if (has(id, 'incision', 'port-', 'uni-', 'bi-', 'line-', 'lm-', 'hemi-cut', 'aortotomy', 'pa-harvest', 'root-distal', 'pericardiotomy', 'tract-', 'wound-', 'koch', 'av-node', 'tv-avnode', 'his-bundle',
    'la-incision', 'septal-incision', 'ra-incision', 'cs-ostium', 'ipl', 'isp-', 'fissure', 'peel-', 'empyema-', 'trach-steno')) return 'plain';
  if (g.startsWith('ports') || g === 'landmarks' || g === 'incisions' || g === 'pleura') return 'plain';
  if (has(id, 'can-', 'snares', 'ett-')) return 'plastic';
  if (has(id, 'stay-sutures', 'tv-devega')) return 'suture';
  if (id === 'mv-prosthesis') return 'metal';
  if (has(id, 'cvg', 'tv-ring')) return 'fabric';
  if (has(id, 'av-prosthesis', 'tv-prosthesis', 'mv-ant-leaflet', 'mv-post-leaflet', 'tv-septal', 'tv-anterior', 'tv-posterior', 'av-cusp-', 'chordae',
    'mitral-annulus', 'tricuspid-annulus', 'aortic-annulus', 'stj')) return 'valve';
  if (id === 'av-calcium') return 'calcium';
  if (has(id, 'la', 'lv', 'ra', 'rv', 'myocardium', 'laa', 'papillary', 'lvot')) return 'myocardium';
  if (has(id, 'pa-trunk', 'pa-root', 'homograft', 'autograft-ao') || g === 'arteries') return 'pulm-artery';
  if (has(id, 'aorta', 'bct', 'lcca', 'lsca', 'rcca', 'coronaries', 'circumflex', 'rca-groove', 'ostium-', 'button-', 'lga', 'rgea', 'ima-', 'septal-perforator', 'root-aneurysm', 'trach-vessels', 'cusp-')) return 'artery';
  if (has(id, 'svc', 'lbcv', 'ivc', 'azygos', 'thymic-vein', 'coronary-sinus') || g === 'veins') return 'vein';
  if (id === 'heart' || id === 'thymus') return 'fat';
  if (id === 'skin') return 'skin';
  if (id === 'liver' || id === 'spleen') return 'liver';
  if (has(id, 'esophagus', 'stomach', 'duodenum', 'pancreas', 'conduit', 'anast-', 'thyroid')) return 'organ';
  if (g === 'airway' || id === 'cartilages' || has(id, 'cricoid', 'thyroid-cart')) return 'cartilage';
  if (g === 'muscles' || has(id, 'erector')) return 'muscle';
  if (g === 'chest-wall') return 'bone';
  if (g === 'nerves' || has(id, 'thoracic-duct', 'cisterna')) return 'nerve';
  if (g === 'nodes') return 'node';
  return 'plain';
}

/** surface parameters per tissue: base physical properties, and the procedural pattern */
interface Look {
  rough: number; clear: number; clearRough: number; sheen?: number; sheenColor?: string; metal?: number; env?: number;
  /** pattern: 0 none, 1 fibres, 2 lobules, 3 vasa vasorum, 4 pores, 5 rings, 6 weave, 7 mottle only */
  pat: number; freq: number; bump: number; mottle: number; accent?: string;
}
const LOOK: Record<Tissue, Look> = {
  myocardium: { rough: 0.45, clear: 0.8, clearRough: 0.14, sheen: 0.25, sheenColor: '#ff9a8a', pat: 1, freq: 1.1, bump: 0.22, mottle: 0.14 },
  muscle: { rough: 0.5, clear: 0.55, clearRough: 0.2, sheen: 0.35, sheenColor: '#ff9f90', pat: 1, freq: 1.3, bump: 0.3, mottle: 0.12 },
  artery: { rough: 0.48, clear: 0.45, clearRough: 0.18, sheen: 0.15, sheenColor: '#ffd6c8', pat: 3, freq: 0.22, bump: 0.08, mottle: 0.1, accent: '#7a1418' },
  'pulm-artery': { rough: 0.46, clear: 0.45, clearRough: 0.18, sheen: 0.2, sheenColor: '#c8c8ff', pat: 3, freq: 0.22, bump: 0.08, mottle: 0.1, accent: '#5a1830' },
  vein: { rough: 0.38, clear: 0.55, clearRough: 0.14, sheen: 0.3, sheenColor: '#b0a0ff', pat: 7, freq: 0.25, bump: 0.12, mottle: 0.12 },
  fat: { rough: 0.35, clear: 0.95, clearRough: 0.06, sheen: 0.2, sheenColor: '#fff4c8', pat: 2, freq: 0.28, bump: 0.8, mottle: 0.12 },
  skin: { rough: 0.55, clear: 0.15, clearRough: 0.5, sheen: 0.5, sheenColor: '#ffd2c0', pat: 4, freq: 1.4, bump: 0.25, mottle: 0.08 },
  bone: { rough: 0.72, clear: 0.05, clearRough: 0.6, pat: 4, freq: 1.1, bump: 0.45, mottle: 0.14 },
  calcium: { rough: 0.85, clear: 0.0, clearRough: 0.8, pat: 4, freq: 1.6, bump: 1.0, mottle: 0.22 },
  cartilage: { rough: 0.46, clear: 0.4, clearRough: 0.2, sheen: 0.3, sheenColor: '#fff0ea', pat: 5, freq: 0.32, bump: 0.5, mottle: 0.08 },
  organ: { rough: 0.35, clear: 0.9, clearRough: 0.08, sheen: 0.3, sheenColor: '#ffc0b0', pat: 3, freq: 0.3, bump: 0.2, mottle: 0.12, accent: '#b8404a' },
  liver: { rough: 0.3, clear: 0.95, clearRough: 0.06, sheen: 0.15, pat: 7, freq: 0.5, bump: 0.12, mottle: 0.16 },
  valve: { rough: 0.35, clear: 0.7, clearRough: 0.12, sheen: 0.5, sheenColor: '#fff6e6', pat: 1, freq: 1.2, bump: 0.25, mottle: 0.06 },
  nerve: { rough: 0.45, clear: 0.6, clearRough: 0.15, sheen: 0.4, sheenColor: '#fff8d8', pat: 1, freq: 1.6, bump: 0.35, mottle: 0.06 },
  node: { rough: 0.55, clear: 0.4, clearRough: 0.25, pat: 7, freq: 0.6, bump: 0.3, mottle: 0.25 },
  metal: { rough: 0.22, clear: 0.6, clearRough: 0.05, metal: 1.0, env: 1.2, pat: 0, freq: 0, bump: 0, mottle: 0 },
  fabric: { rough: 0.8, clear: 0.0, clearRough: 0.6, sheen: 0.8, sheenColor: '#ffffff', pat: 6, freq: 2.2, bump: 0.6, mottle: 0.04 },
  suture: { rough: 0.25, clear: 0.8, clearRough: 0.05, pat: 0, freq: 0, bump: 0, mottle: 0 },
  plastic: { rough: 0.15, clear: 1.0, clearRough: 0.03, env: 0.8, pat: 0, freq: 0, bump: 0, mottle: 0.02 },
  plain: { rough: 0.5, clear: 0.0, clearRough: 0.4, pat: 0, freq: 0, bump: 0, mottle: 0 },
};

const NOISE = `
varying vec3 vObj;
uniform vec3 uFib;
float tsH(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float tsN(vec3 p) { vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(tsH(i), tsH(i + vec3(1,0,0)), f.x), mix(tsH(i + vec3(0,1,0)), tsH(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(tsH(i + vec3(0,0,1)), tsH(i + vec3(1,0,1)), f.x), mix(tsH(i + vec3(0,1,1)), tsH(i + vec3(1,1,1)), f.x), f.y), f.z); }
float tsF(vec3 p) { return 0.55 * tsN(p) + 0.3 * tsN(p * 2.03 + 7.1) + 0.15 * tsN(p * 4.01 + 3.3); }
/* cellular: distance to the nearest and second-nearest feature point (fat lobules) */
vec2 tsW(vec3 p) { vec3 i = floor(p), f = fract(p); float d1 = 8.0, d2 = 8.0;
  for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec3 o = vec3(float(x), float(y), float(z)); vec3 r = o + vec3(tsH(i + o), tsH(i + o + 19.1), tsH(i + o + 47.7)) - f;
    float d = dot(r, r); if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d; }
  return vec2(sqrt(d1), sqrt(d2)); }
/* the tissue's height field and a colour weight for its pattern */
vec2 tsPattern(vec3 p) {
  float h = 0.0, c = 0.0;
#if TS_PAT == 1
  vec3 a = normalize(uFib); vec3 q = p - a * dot(p, a) * 0.85;                 /* stretch along the fibres */
  float s = tsF(q * TS_FREQ * vec3(1.0)); float fib = abs(sin((dot(q, cross(a, vec3(0.31, 0.72, 0.62))) + s * 3.0) * TS_FREQ * 5.0));
  fib = mix(fib, s, 0.55); h = 0.5 * fib + 0.5 * s; c = fib * 0.5;
#elif TS_PAT == 2
  vec2 w = tsW(p * TS_FREQ); float edge = smoothstep(0.0, 0.18, w.y - w.x);
  h = edge * (0.6 + 0.4 * (1.0 - w.x)); c = 1.0 - edge;
#elif TS_PAT == 3
  float v = abs(tsF(p * TS_FREQ) - 0.5); float vas = (1.0 - smoothstep(0.0, 0.012, v)) * smoothstep(0.45, 0.7, tsN(p * 0.15 + 5.0));
  h = tsF(p * TS_FREQ * 3.0) * 0.6 - vas * 0.4; c = vas;
#elif TS_PAT == 4
  float pr = smoothstep(0.78, 0.9, tsN(p * TS_FREQ * 2.5)); h = tsF(p * TS_FREQ) - pr * 0.8; c = pr;
#elif TS_PAT == 5
  float ring = 0.5 + 0.5 * sin(dot(p, normalize(uFib)) * TS_FREQ * 6.2832 / 1.0 + tsN(p * 0.2) * 2.0); h = smoothstep(0.3, 0.7, ring); c = 1.0 - h;
#elif TS_PAT == 6
  vec3 a = normalize(uFib); vec3 b = normalize(cross(a, vec3(0.3, 0.7, 0.64)));
  float wv = sin(dot(p, a) * TS_FREQ * 6.2832) * sin(dot(p, b) * TS_FREQ * 6.2832); h = 0.5 + 0.5 * wv; c = 0.0;
#elif TS_PAT == 7
  h = tsF(p * TS_FREQ); c = 0.0;
#endif
  return vec2(h, c);
}`;

export function applyTissue(mat: THREE.MeshPhysicalMaterial, kind: Tissue, fibre: THREE.Vector3, keySuffix = ''): void {
  const L = LOOK[kind];
  mat.roughness = L.rough; mat.clearcoat = L.clear; mat.clearcoatRoughness = L.clearRough; mat.metalness = L.metal ?? 0;
  mat.envMapIntensity = L.env ?? 0.45;
  if (L.sheen) { mat.sheen = L.sheen; mat.sheenRoughness = 0.5; mat.sheenColor = new THREE.Color(L.sheenColor ?? '#ffffff'); }
  if (kind === 'metal') mat.color.set('#3a3d42');
  if (kind === 'fabric') mat.color.set('#efeee6');
  if (L.pat === 0 && L.mottle === 0) return;
  const accent = new THREE.Color(L.accent ?? '#000000');
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (sh, r) => {
    prev?.call(mat, sh, r);
    sh.uniforms['uFib'] = { value: fibre.clone().normalize() };
    sh.defines = { ...(sh.defines ?? {}), TS_PAT: L.pat, TS_FREQ: L.freq.toFixed(3) };
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vObj;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvObj = position;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + NOISE)
      .replace('#include <color_fragment>', `#include <color_fragment>
vec2 tsP = tsPattern(vObj);
float tsM = tsF(vObj * 0.09);
diffuseColor.rgb *= (1.0 - ${L.mottle.toFixed(3)}) + ${(2 * L.mottle).toFixed(3)} * tsM;
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(${accent.r.toFixed(3)}, ${accent.g.toFixed(3)}, ${accent.b.toFixed(3)}), ${L.accent ? '0.55' : '0.0'} * tsP.y);
#if TS_PAT == 1
diffuseColor.rgb *= 0.9 + 0.12 * tsP.y;
#elif TS_PAT == 2
diffuseColor.rgb *= 0.86 + 0.2 * (1.0 - tsP.y);
#elif TS_PAT == 4
diffuseColor.rgb *= 1.0 - 0.35 * tsP.y;
#elif TS_PAT == 5
diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * vec3(0.93, 0.8, 0.8), 0.6 * tsP.y);
#endif`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
roughnessFactor = clamp(roughnessFactor * (0.75 + 0.5 * tsF(vObj * 0.35 + 11.0)), 0.04, 1.0);`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
{
  float tsh = tsPattern(vObj).x;
  vec3 dpx = dFdx(-vViewPosition), dpy = dFdy(-vViewPosition);
  float dhx = dFdx(tsh), dhy = dFdy(tsh);
  vec3 r1 = cross(dpy, normal), r2 = cross(normal, dpx); float det = dot(dpx, r1);
  vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
  normal = normalize(abs(det) * normal - grad * ${(L.bump * 1.2).toFixed(3)});
}`);
  };
  const key = `tissue-${kind}${keySuffix}`;
  mat.customProgramCacheKey = () => key;
}

/** the main direction of a structure (its longest axis), used for fibre, ring and weave orientation */
export function mainAxis(geo: THREE.BufferGeometry): THREE.Vector3 {
  const p = geo.getAttribute('position'); const n = p.count; const step = Math.max(1, Math.floor(n / 400));
  const c = new THREE.Vector3(); let k = 0;
  for (let i = 0; i < n; i += step) { c.x += p.getX(i); c.y += p.getY(i); c.z += p.getZ(i); k++; }
  c.divideScalar(Math.max(1, k));
  const m = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < n; i += step) {
    const x = p.getX(i) - c.x, y = p.getY(i) - c.y, z = p.getZ(i) - c.z;
    m[0] += x * x; m[1] += x * y; m[2] += x * z; m[3] += y * y; m[4] += y * z; m[5] += z * z;
  }
  let v = new THREE.Vector3(1, 0.7, 0.4).normalize();
  for (let it = 0; it < 12; it++) {
    v = new THREE.Vector3(m[0]! * v.x + m[1]! * v.y + m[2]! * v.z, m[1]! * v.x + m[3]! * v.y + m[4]! * v.z, m[2]! * v.x + m[4]! * v.y + m[5]! * v.z);
    if (v.lengthSq() < 1e-9) return new THREE.Vector3(0, 0, 1);
    v.normalize();
  }
  return v;
}
