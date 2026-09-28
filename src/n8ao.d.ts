declare module 'n8ao' {
  import type { Pass } from 'three/addons/postprocessing/Pass.js';
  import type * as THREE from 'three';
  export class N8AOPass extends Pass {
    constructor(scene: THREE.Scene, camera: THREE.Camera, width?: number, height?: number);
    configuration: Record<string, unknown>;
    setSize(w: number, h: number): void;
  }
}
