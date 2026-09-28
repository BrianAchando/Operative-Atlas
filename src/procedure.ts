import type { Vec3, Plane } from './ctview.ts';

export interface Choice { text: string; correct: boolean; why: string }

/** what the surgeon does in a step, played in 3D with the instrument coming through a port */
export interface Action {
  kind: 'staple' | 'ligate' | 'dissect' | 'open-fissure' | 'staple-fissure' | 'thoracotomy' | 'saw' | 'clamp' | 'twist' | 'suture' | 'massage' | 'layers' | 'sternotomy' | 'reveal' | 'decorticate' | 'anastomose';
  /** ligate: tie without dividing (mass ligation of the thoracic duct) */
  keep?: boolean;
  /** decorticate: the trapped lung re-expands from this scale about `pivot` as the peel comes off */
  expand?: { ids: string[]; pivot: Vec3; from: number };
  /** layers: the chest wall taken one layer at a time; each divided (cut across `dir` at `point`, the far side opened by `open` mm),
   *  split along its fibres (same, narrower), retracted by `offset`, passed through, or spared */
  layers?: { id: string; label: string; fate: 'divide' | 'split' | 'retract' | 'through' | 'spare'; point?: Vec3; dir?: Vec3; open?: number; offset?: Vec3 }[];
  /** clamp: where the jaws close, the axis of the vessel (or hilum) they cross, its radius and the jaw length.
   *  staple and saw: a cut at `at` across `axis` instead of the structure's own division (e.g. the oesophagus in the neck, a median sternotomy) */
  at?: Vec3; axis?: Vec3; radius?: number; jawLen?: number;
  /** saw: the two halves opened this far apart (a median sternotomy retractor) */
  open?: number;
  /** saw (the chest-wall lid of a clamshell) and twist (the lung about its hilum): structures turned about an axis */
  hinge?: { ids: string[]; pivot: Vec3; axis: Vec3; angle: number };
  /** thoracotomy: the incision to draw and the two ribs either side of the space (upper first) */
  incision?: string;
  ribs?: [string, string];
  /** structures made visible once the action is done, and kept visible (e.g. the ribs around a thoracotomy) */
  show?: string[];
  /** dissecting instrument: peanut (blunt) or diathermy hook */
  tool?: 'peanut' | 'hook';
  /** structures taken down by the action (e.g. the pulmonary ligament), hidden once it is done */
  remove?: string[];
  /** button text, e.g. "Fire the stapler" */
  label: string;
  /** structures divided (staple, ligate) */
  ids?: string[];
  /** landmark id of the port the instrument comes through */
  port: string;
  /** peanut tip path (dissect, open-fissure) or the staple line (staple-fissure), world mm */
  path?: Vec3[];
  /** lobes pushed apart as the fissure opens */
  spread?: { ids: string[]; offset: Vec3 }[];
  reload?: 'vascular' | 'tissue';
  /** staple-fissure: the jaws close along this direction (the fissure normal) */
  normal?: Vec3;
}

export interface Retract { ids: string[]; offset: Vec3; opacity: number }

export interface Step {
  id: string;
  title: string;
  /** the phase shown above the title: Anatomy, Setup, Fissure, Artery, Bronchus, Vein, Close */
  phase: string;
  body: string;                          // HTML, short paragraphs
  view: { eye: Vec3; target: Vec3 } | { frame: string[]; dir: Vec3; pad?: number };
  highlight?: string[];
  danger?: string[];
  labels?: string[];                     // extra structures to name in 3D
  show?: string[];                       // make visible for this step
  hide?: string[];
  opacity?: Record<string, number>;      // e.g. lobes made translucent
  retract?: Retract | Retract[];
  action?: Action;
  /** the specimen moved out; `distal` limits which divided structures' far ends go with it (default: all) */
  specimen?: { ids: string[]; offset: Vec3; distal?: string[] };
  spin?: boolean;                        // slow turntable (anatomy overview)
  ct?: { focus: Vec3 | string; plane: Plane; window?: string };
  /** 'lateral': the patient turned on the other side, operated side up, on the table */
  pose?: 'lateral';
  /** a trapped lung, drawn smaller about its hilum (empyema before decortication) */
  shrink?: { ids: string[]; pivot: Vec3; scale: number };
  ask?: { question: string; choices: Choice[] };
  pearl?: string;
  /** the step's place in the approach's sequence strip */
  seq?: number;
}

export interface Procedure {
  id: string; name: string; approach: string; summary: string;
  /** the operation this approach belongs to, e.g. 'lul', and its display name */
  op: string; opName: string;
  /** menu group: Lobectomy, Pneumonectomy, Segmentectomy, … */
  group?: string;
  /** which hilum the operation is on */
  side: 'left' | 'right' | 'both';
  /** the order to remember, shown as a strip: e.g. Fissure → A2 → Truncus → Bronchus → Vein */
  sequence: { label: string; kind: 'artery' | 'vein' | 'bronchus' | 'fissure' | 'other' }[];
  ports: { id: string; name: string; at: Vec3; note: string }[];
  steps: Step[];
  sources: { title: string; url: string }[];
}
