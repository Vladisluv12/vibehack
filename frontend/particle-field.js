/**
 * ParticleField — decorative "Flow" view: a fine luminous orbital disk rendered
 * with GPU-analytic particle motion, plus stable clickable markers for the real
 * knowledge nodes in the same projection.
 *
 * Design notes
 * - Particle positions are computed in the vertex shader as a closed-form
 *   function of a per-particle random vector and a single time uniform. There is
 *   no CPU simulation, no state feedback and no per-frame attribute upload.
 * - Particles are decoration. They never encode notes, counts or activity.
 * - Real nodes are a separate InstancedMesh (max 500) with deterministic,
 *   id-derived placement so markers stay put across setData() calls.
 * - Caller data is never mutated: setData() clones what it needs.
 *
 * API
 *   new ParticleField({ container, onSelect, onStatus })
 *   setData({ nodes, links })      rebuild markers + sparse link lines
 *   focus(id)                      move camera to a marker
 *   fit()                          frame the whole disk (narrow containers safe)
 *   setQuality('auto'|'static'|'low'|'balanced')
 *   setGlow(0..2)                  bloom strength, observed only
 *   setPreset('galaxy'|'blackhole')
 *   setActivity({running,tasks})    actual running task beacons
 *   pushEvents([{seq,type,...}])    observed event trails, bounded and deduplicated
 *   setMotion(0..1)                orbital time scale
 *   setPaused(bool) / setVisible(bool)
 *   resize() / dispose()
 *
 * onStatus callback receives { quality, particles, fps, paused, error? }:
 *   quality   effective quality actually rendered (in 'auto' mode this may be
 *             downgraded from 'low' to 'static')
 *   particles decorative particle count currently drawn (never node count)
 *   fps       integer measured from real RAF frame deltas, or null when nothing
 *             was animated since the last start (static/paused/hidden/reduced)
 *   error     string, only present on a render/context problem
 *
 * Requires the pinned three@0.180 package plus its addons: OrbitControls,
 * EffectComposer, RenderPass, UnrealBloomPass, OutputPass.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/** Decorative particle budgets per quality level. */
const PARTICLES = { static: 1200, low: 16384, balanced: 65536 };
const MAX_PARTICLES = PARTICLES.balanced;
const MAX_MARKERS = 500;
const MAX_LINK_LINES = 180;
const MAX_PROJECTS = 8;
const MAX_AGENTS = 16;
const MAX_EFFECTS = 64;
const EFFECT_LIFETIME = 2600;
const EVENT_COLORS = {
  'task.start': '#79aaff', 'task.complete': '#ffd47d', 'task.error': '#ff6c83',
  'tool.start': '#bd8dff', 'tool.finish': '#ccadff',
  'memory.search': '#68e5d6', 'memory.read': '#68e5d6', 'memory.draft': '#8ae8be',
  'agent.message': '#ffafe4',
};
const DISK_RADIUS = 250;
const DISK_THICKNESS = 26;
/** Deterministic seed for placement and particle scatter (valid hex). */
const SEED_HEX = '50415254';
const BASE_SEED = Number.parseInt(SEED_HEX, 16) >>> 0;
const KIND_COLOR = {
  project: '#ffc978',
  research: '#ae96ff',
  source: '#6dafff',
  session: '#64d9d2',
  decision: '#f5a5c6',
  concept: '#a5d89c',
  note: '#bdc9df',
};
const KIND_SIZE = { project: 5.4, decision: 3.6 };
const TAU = Math.PI * 2;
const DOWNGRADE_FPS = 24;

const PARTICLE_VERTEX = /** glsl */ `
#define TAU 6.28318530718
uniform float uTime;
uniform float uRadius;
uniform float uThickness;
uniform float uSize;
uniform float uScale;
uniform float uBlackHole;
uniform float uClusterCount;
uniform vec4 uClusters[9];
attribute vec4 aRand;
attribute float aCluster;
varying vec3 vColor;
varying float vAlpha;

vec3 coreColor(float k) {
  return mix(vec3(1.0, 0.96, 0.88), vec3(1.0, 0.78, 0.36), k) * 1.7;
}

vec3 armColor(float arm) {
  float k = mod(arm, 3.0);
  vec3 violet = vec3(0.58, 0.32, 1.0);
  vec3 cyan = vec3(0.28, 0.86, 1.0);
  vec3 pink = vec3(1.0, 0.42, 0.74);
  return k < 1.0 ? violet : (k < 2.0 ? cyan : pink);
}

void main() {
  float a = aRand.x, b = aRand.y, c = aRand.z, d = aRand.w;
  bool isCore = a < 0.15;
  bool isHaze = !isCore && a < 0.27;
  float arm = floor(b * 5.0);
  float rNorm;
  float theta;

  if (isCore) {
    rNorm = 0.02 + 0.16 * pow(c, 2.5);
    theta = d * TAU + uTime * (1.6 / (0.25 + rNorm));
  } else if (isHaze) {
    rNorm = 0.14 + 0.86 * sqrt(d);
    theta = c * TAU + uTime * (0.5 / (0.30 + rNorm));
  } else {
    rNorm = 0.10 + 0.90 * sqrt(mix(d, d * d, 0.5));
    theta = arm * (TAU / 5.0) + rNorm * 3.1 + (c - 0.5) * 0.30
          + uTime * (0.5 / (0.28 + rNorm));
  }

  // Each decorative particle belongs to one bounded analytic disk. No CPU
  // per-particle work or cross-product with the knowledge node count.
  float clusterIndex = floor(aCluster * uClusterCount);
  vec4 cluster = uClusters[0];
  for (int i = 0; i < 9; i++) {
    if (float(i) == clusterIndex) cluster = uClusters[i];
  }
  if (uBlackHole > 0.5) rNorm = 0.13 + rNorm * 0.87;
  float radius = rNorm * uRadius * cluster.w;
  float height = isCore ? 1.7 : (1.0 - 0.72 * rNorm);
  vec3 local = cluster.xyz + vec3(cos(theta) * radius, (b - 0.5) * uThickness * height * cluster.w, sin(theta) * radius);
  vec4 view = modelViewMatrix * vec4(local, 1.0);
  gl_Position = projectionMatrix * view;

  float sizeFactor = isCore ? (1.1 + d * 1.3) : (0.55 + c * 0.95);
  // Clamp: drivers cap point size, and huge additive sprites cost fill rate.
  gl_PointSize = clamp(uSize * sizeFactor * (uScale / max(1.0, -view.z)), 1.0, 42.0);

  vec3 hazeColor = vec3(0.34, 0.44, 0.86);
  vColor = isCore ? coreColor(pow(d, 0.6))
                  : mix(coreColor(0.35), isHaze ? hazeColor : armColor(arm), smoothstep(0.12, 0.52, rNorm));
  float alpha = isCore ? 0.55 : (isHaze ? 0.15 : (0.34 + 0.30 * (1.0 - rNorm)));
  vAlpha = alpha * mix(1.0, 0.25, smoothstep(0.80, 1.0, rNorm));
}
`;

const PARTICLE_FRAGMENT = /** glsl */ `
varying vec3 vColor;
varying float vAlpha;

void main() {
  vec2 p = gl_PointCoord - 0.5;
  float d2 = dot(p, p);
  if (d2 > 0.25) discard;
  float falloff = exp(-d2 * 13.0) * 1.15;
  gl_FragColor = vec4(vColor * falloff, falloff * vAlpha);
}
`;

/** mulberry32: small deterministic PRNG, used once per geometry build. */
function makeRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a over id + salt, normalised to [0,1). Stable regardless of node order. */
function hashUnit(value, salt) {
  let h = 2166136261 >>> 0;
  const text = `${salt}:${value}`;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 1000003) / 1000003;
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function linkEndId(end) {
  if (end && typeof end === 'object') return String(end.id ?? '');
  return end === undefined || end === null ? '' : String(end);
}

export class ParticleField {
  /**
   * @param {object} options
   * @param {HTMLElement} options.container element the canvas is mounted into
   * @param {(node: object|null) => void} [options.onSelect] called with a cloned
   *   node record on marker click, and with null on a background click
   * @param {(status: object) => void} [options.onStatus] observed renderer info
   */
  constructor({ container, onSelect = null, onStatus = null }) {
    if (!container) throw new Error('ParticleField requires a container element.');
    this._container = container;
    this._onSelect = onSelect;
    this._onStatus = onStatus;

    this._disposed = false;
    this._contextLost = false;
    this._renderFailed = false;
    this._lastError = null;
    this._visible = true;
    this._paused = false;
    this._mode = 'auto';
    this._quality = 'low';
    this._glow = 1;
    this._motion = 1;
    this._running = 0;
    this._time = 0;
    this._raf = null;
    this._lastFrame = null;
    this._frameCount = 0;
    this._windowStart = null;
    this._fps = null;
    this._slowWindows = 0;
    this._tween = null;
    this._selected = null;
    this._nodes = [];
    this._nodeById = new Map();
    this._links = [];
    this._pointerDown = null;
    this._preset = 'galaxy';
    this._clusters = [{id: null, position: new THREE.Vector3(), scale: 1}];
    this._projectClusters = new Map();
    this._tasks = [];
    this._activitySignature = '';
    this._agentPositions = new Map();
    this._effects = [];
    this._seenEvents = new Set();
    this._clusterSprites = [];

    this._motionQuery = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)') ?? null;
    this._reducedMotion = Boolean(this._motionQuery?.matches);
    this._motionOverride = false;

    try {
      this._initThree();
    } catch (error) {
      this._contextLost = true;
      this._renderFailed = true;
      this._emitStatus(`WebGL недоступен: ${error?.message || error}`);
      this.dispose();
      return;
    }
    if (this._contextLost) return;

    this._buildParticles();
    this._bindEvents();
    this.resize();
    this.fit(true);
    this._emitStatus();
  }

  // ---------------------------------------------------------------- lifecycle

  _initThree() {
    const { width, height } = this._size();

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color('#050a14');
    this._labelScene = new THREE.Scene();
    this._labels = new THREE.Group();
    this._labelScene.add(this._labels);

    this._camera = new THREE.PerspectiveCamera(52, width / height, 1, 6000);
    this._camera.position.set(0, 340, 470);

    this._renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this._renderer.debug.onShaderError = () => {
      this._renderFailed = true;this._stopLoop();
      this._emitStatus('Shader Flow не скомпилирован. Используйте Graph или каталог.');
    };
    this._renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 1.5));
    this._renderer.setSize(width, height, false);
    this._renderer.domElement.style.display = 'block';
    this._renderer.domElement.style.width = '100%';
    this._renderer.domElement.style.height = '100%';
    this._renderer.domElement.setAttribute('aria-hidden', 'true');
    this._container.appendChild(this._renderer.domElement);

    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.08;
    this._controls.rotateSpeed = 0.55;
    this._controls.autoRotateSpeed = 0.28 * this._motion;
    this._controls.minDistance = 60;
    this._controls.maxDistance = 3200;
    this._controls.target.set(0, 0, 0);

    this._disk = new THREE.Group();
    this._disk.rotation.set(-0.20, 0, 0.34);
    this._scene.add(this._disk);

    // A soft luminous core is decoration, independent of knowledge markers.
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.12, 'rgba(255,252,229,1)');
    gradient.addColorStop(0.30, 'rgba(255,206,122,0.7)');
    gradient.addColorStop(0.65, 'rgba(255,170,77,0.09)');
    gradient.addColorStop(1, 'rgba(255,161,65,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 128, 128);
    this._coreTexture = new THREE.CanvasTexture(canvas);
    this._coreMaterial = new THREE.SpriteMaterial({map:this._coreTexture,
      color:new THREE.Color(1.5,1.3,1.0), blending:THREE.AdditiveBlending,
      transparent:true, depthWrite:false, depthTest:false});
    this._core = new THREE.Sprite(this._coreMaterial);
    this._core.scale.set(68,68,1);
    this._core.renderOrder = 1;
    this._disk.add(this._core);

    // Dark core and photon-ring inspired outline: visual metaphors only.
    this._holeGeometry = new THREE.CircleGeometry(28, 64);
    this._holeMaterial = new THREE.MeshBasicMaterial({color:'#01030a', transparent:true,
      opacity:1, depthTest:false, depthWrite:false, side:THREE.DoubleSide});
    this._hole = new THREE.Mesh(this._holeGeometry, this._holeMaterial);
    this._hole.rotation.x = -Math.PI / 2;
    this._hole.renderOrder = 20;
    this._hole.visible = false;
    this._disk.add(this._hole);
    this._ringGeometry = new THREE.TorusGeometry(31, 0.8, 8, 96);
    this._ringMaterial = new THREE.MeshBasicMaterial({color:'#ffc67a', transparent:true,
      opacity:0.9, blending:THREE.AdditiveBlending, depthTest:false, depthWrite:false});
    this._ring = new THREE.Mesh(this._ringGeometry, this._ringMaterial);
    this._ring.rotation.x = Math.PI / 2;
    this._ring.renderOrder = 21;
    this._ring.visible = false;
    this._disk.add(this._ring);

    this._agentGeometry = new THREE.OctahedronGeometry(1, 0);
    this._agentMaterial = new THREE.MeshBasicMaterial({color:'#ffffff', transparent:true,
      opacity:0.9, blending:THREE.AdditiveBlending, depthWrite:false, depthTest:false});
    this._agents = new THREE.InstancedMesh(this._agentGeometry, this._agentMaterial, MAX_AGENTS);
    this._agents.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this._agents.count = 0;
    this._agents.frustumCulled = false;
    this._agents.renderOrder = 23;
    this._disk.add(this._agents);
    this._agentLinkGeometry = new THREE.BufferGeometry();
    this._agentLinkGeometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(MAX_AGENTS*6),3));
    this._agentLinkGeometry.setAttribute('color',new THREE.BufferAttribute(new Float32Array(MAX_AGENTS*6),3));
    this._agentLinkMaterial = new THREE.LineBasicMaterial({vertexColors:true,transparent:true,
      opacity:0.3,blending:THREE.AdditiveBlending,depthWrite:false,depthTest:false});
    this._agentLinks = new THREE.LineSegments(this._agentLinkGeometry,this._agentLinkMaterial);
    this._agentLinks.frustumCulled = false;
    this._agentLinks.renderOrder = 22;
    this._agentLinkGeometry.setDrawRange(0,0);
    this._disk.add(this._agentLinks);
    this._effectsGroup = new THREE.Group();
    this._effectsGroup.renderOrder = 24;
    this._disk.add(this._effectsGroup);

    this._markerGeometry = new THREE.SphereGeometry(1, 16, 12);
    // Real knowledge markers remain readable above the decorative dark core.
    this._markerMaterial = new THREE.MeshBasicMaterial({color:'#ffffff',
      transparent:true, opacity:1, depthTest:false, depthWrite:false});
    this._haloMaterial = new THREE.MeshBasicMaterial({
      color: '#ffffff',
      transparent: true,
      opacity: 0.28,
      side: THREE.BackSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this._halo = new THREE.Mesh(this._markerGeometry, this._haloMaterial);
    this._halo.visible = false;
    this._halo.renderOrder = 27;
    this._disk.add(this._halo);

    this._linkGeometry = new THREE.BufferGeometry();
    this._linkGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_LINK_LINES * 6), 3));
    this._linkGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX_LINK_LINES * 6), 3));
    this._linkMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this._links3d = new THREE.LineSegments(this._linkGeometry, this._linkMaterial);
    this._links3d.frustumCulled = false;
    this._links3d.renderOrder = -1;
    this._disk.add(this._links3d);

    this._composer = new EffectComposer(this._renderer);
    this._composer.addPass(new RenderPass(this._scene, this._camera));
    this._bloom = new UnrealBloomPass(new THREE.Vector2(width, height), 0.9, 0.65, 0.30);
    this._composer.addPass(this._bloom);
    this._composer.addPass(new OutputPass());
    this._applyGlow();

    this._raycaster = new THREE.Raycaster();
    this._ndc = new THREE.Vector2();
  }

  _buildParticles() {
    const geometry = new THREE.BufferGeometry();
    const rand = new Float32Array(MAX_PARTICLES * 4);
    const clusterSeeds = new Float32Array(MAX_PARTICLES);
    const next = makeRandom(BASE_SEED);
    for (let i = 0; i < MAX_PARTICLES; i += 1) {
      rand[i * 4] = next();
      rand[i * 4 + 1] = next();
      rand[i * 4 + 2] = next();
      rand[i * 4 + 3] = next();
      clusterSeeds[i] = next();
    }
    geometry.setAttribute('aRand', new THREE.BufferAttribute(rand, 4));
    geometry.setAttribute('aCluster', new THREE.BufferAttribute(clusterSeeds, 1));
    // Three derives the vertex count from the position attribute even though the
    // shader ignores it, so the zeroed buffer has to exist.
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3));
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), DISK_RADIUS * 1.4);
    geometry.setDrawRange(0, PARTICLES[this._quality]);

    this._particleUniforms = {
      uTime: { value: 0 },
      uRadius: { value: DISK_RADIUS },
      uThickness: { value: DISK_THICKNESS },
      uSize: { value: 2.4 },
      uScale: { value: 300 },
      uBlackHole: { value: 0 },
      uClusterCount: { value: 1 },
      uClusters: { value: Array.from({length:9}, () => new THREE.Vector4(0,0,0,1)) },
    };
    const material = new THREE.ShaderMaterial({
      uniforms: this._particleUniforms,
      vertexShader: PARTICLE_VERTEX,
      fragmentShader: PARTICLE_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    this._particles = new THREE.Points(geometry, material);
    this._particles.frustumCulled = false;
    // Decorative haze is below all actual knowledge markers and event trails.
    this._particles.renderOrder = -2;
    this._disk.add(this._particles);
    this._particleGeometry = geometry;
    this._particleMaterial = material;
  }

  _bindEvents() {
    this._onCanvasLost = (event) => {
      event.preventDefault();
      this._contextLost = true;
      this._stopLoop();
      this._emitStatus('Контекст WebGL потерян: анимация остановлена, частицы недоступны — используйте каталог заметок.');
    };
    this._onCanvasRestored = () => {
      this._contextLost = false;
      this._renderFailed = false;
      this._lastError = null;
      this._emitStatus();
      this.resize();
      this._syncLoop();
      this.renderOnce();
    };
    this._onVisibility = () => {
      this._syncLoop();
      this.renderOnce();
    };
    this._onMotionQuery = (event) => {
      this._reducedMotion = event.matches;
      this._syncLoop();
      this.renderOnce();
    };
    this._onPointerDown = (event) => {
      this._pointerDown = { x: event.clientX, y: event.clientY };
    };
    this._onPointerUp = (event) => {
      const down = this._pointerDown;
      this._pointerDown = null;
      if (!down || this._contextLost || this._disposed) return;
      // Orbit drag and a click are mutually exclusive by travel distance only.
      if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 6) return;
      this._pick(event);
    };
    this._onControlsChange = () => {
      if (!this._shouldAnimate()) this.renderOnce();
    };

    const canvas = this._renderer.domElement;
    canvas.addEventListener('webglcontextlost', this._onCanvasLost);
    canvas.addEventListener('webglcontextrestored', this._onCanvasRestored);
    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointerup', this._onPointerUp);
    document.addEventListener('visibilitychange', this._onVisibility);
    this._controls.addEventListener('change', this._onControlsChange);
    this._motionQuery?.addEventListener('change', this._onMotionQuery);

    this._observer = new ResizeObserver(() => this.resize());
    this._observer.observe(this._container);
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._stopLoop();

    const canvas = this._renderer?.domElement;
    if (canvas) {
      canvas.removeEventListener('webglcontextlost', this._onCanvasLost);
      canvas.removeEventListener('webglcontextrestored', this._onCanvasRestored);
      canvas.removeEventListener('pointerdown', this._onPointerDown);
      canvas.removeEventListener('pointerup', this._onPointerUp);
    }
    document.removeEventListener('visibilitychange', this._onVisibility);
    this._controls?.removeEventListener('change', this._onControlsChange);
    this._motionQuery?.removeEventListener('change', this._onMotionQuery);
    this._observer?.disconnect();
    this._observer = null;

    this._releaseMarkers();
    for (const effect of [...this._effects]) this._releaseEffect(effect);
    this._clearClusterSprites();
    this._agents?.dispose();
    this._agentGeometry?.dispose();
    this._agentMaterial?.dispose();
    this._agentLinkGeometry?.dispose();
    this._agentLinkMaterial?.dispose();
    this._holeGeometry?.dispose();
    this._holeMaterial?.dispose();
    this._ringGeometry?.dispose();
    this._ringMaterial?.dispose();
    this._seenEvents.clear();
    this._agentPositions.clear();
    this._particleGeometry?.dispose();
    this._particleMaterial?.dispose();
    this._coreTexture?.dispose();
    this._coreMaterial?.dispose();
    this._markerGeometry?.dispose();
    this._markerMaterial?.dispose();
    this._haloMaterial?.dispose();
    this._linkGeometry?.dispose();
    this._linkMaterial?.dispose();
    for (const pass of this._composer?.passes || []) pass.dispose?.();
    this._bloom?.materialHighPassFilter?.dispose();
    this._composer?.dispose();
    this._controls?.dispose();
    this._renderer?.dispose();
    if (canvas?.parentElement === this._container) this._container.removeChild(canvas);

    this._nodes = [];
    this._nodeById.clear();
    this._links = [];
  }

  // ------------------------------------------------------------------- public

  /**
   * Replace the knowledge projection. Caller objects are never mutated or kept.
   * @param {{nodes?: Array<object>, links?: Array<object>}} [data]
   */
  setData(data = {}) {
    if (this._contextLost || this._disposed) return;
    const incoming = Array.isArray(data.nodes) ? data.nodes : [];
    const incomingLinks = Array.isArray(data.links) ? data.links : [];

    const hadData = this._nodes.length > 0;
    this._nodes = incoming.slice(0, MAX_MARKERS).map((node) => this._record(node));
    this._nodeById = new Map(this._nodes.map((node) => [node.id, node]));
    if (!this._nodeById.has(this._selected)) this._selected = null;
    this._buildClusters();
    this._placeNodes();
    this._rebuildAgents();
    this._links = this._resolveLinks(incomingLinks);

    this._rebuildMarkers();
    this._rebuildLinkLines();
    if (!hadData && this._clusters.length > 1) this.fit(true);
    this._applySelection();
    this._emitStatus();
    this._syncLoop();
    this.renderOnce();
  }

  /**
   * Point the camera at a node marker. Silent no-op for unknown ids.
   * @param {string|number} id
   */
  focus(id) {
    if (this._contextLost || this._disposed) return;
    if (id === null) { this._select(null); return; }
    const node = this._nodeById.get(String(id));
    if (!node) return;
    this._select(node.id);
    const target = this._markerWorldPosition(node);
    const offset = new THREE.Vector3().subVectors(this._camera.position, this._controls.target);
    const distance = offset.lengthSq() > 1e-6 ? THREE.MathUtils.clamp(offset.length(), 500, 1200) : 700;
    this._moveTo(target.clone().add(offset.setLength(distance)), target, 900);
  }

  /** Frame the full disk; distance accounts for narrow container aspect. */
  fit(immediate = false) {
    if (this._contextLost || this._disposed) return;
    this._disk.updateMatrixWorld(true);
    const radius = this._clusters.length > 1 ? 475 : DISK_RADIUS * 1.18;
    const vertical = THREE.MathUtils.degToRad(this._camera.fov) / 2;
    const horizontal = Math.atan(Math.tan(vertical) * this._camera.aspect);
    const distance = THREE.MathUtils.clamp(
      (radius / Math.sin(Math.min(vertical, horizontal))) * 1.08,
      this._controls.minDistance,
      this._controls.maxDistance,
    );
    const direction = new THREE.Vector3().subVectors(this._camera.position, this._controls.target);
    if (direction.lengthSq() < 1e-6) direction.set(0, 0.42, 0.91);
    this._moveTo(direction.setLength(distance), new THREE.Vector3(), immediate ? 0 : 800);
  }

  /**
   * @param {'auto'|'static'|'low'|'balanced'} quality unknown values keep current
   */
  setQuality(quality) {
    if ((quality !== 'auto' && !PARTICLES[quality]) || this._disposed || this._contextLost) return;
    this._mode = quality;
    this._quality = quality === 'auto' ? 'low' : quality;
    this._slowWindows = 0;
    this._particleGeometry.setDrawRange(0, PARTICLES[this._quality]);
    this._fps = null;
    this._emitStatus();
    this._syncLoop();
    this.renderOnce();
  }

  /** @param {number} value 0..2 bloom strength */
  setGlow(value) {
    this._glow = clampNumber(value, 0, 2, this._glow);
    if (this._contextLost) return;
    this._applyGlow();
    this.renderOnce();
  }

  /** Running snapshots create beacons only for actual running/cancelling tasks. */
  setActivity({running = 0, tasks = []} = {}) {
    if (this._disposed || this._contextLost) return;
    const next = Math.floor(clampNumber(running, 0, 100, 0));
    const records = (Array.isArray(tasks) ? tasks : []).filter(task =>
      task?.id != null && ['running','cancelling'].includes(task.status)
    ).slice(0, MAX_AGENTS).map(task => ({id:String(task.id), provider:String(task.provider || ''),
      project_id:String(task.project_id || ''), status:task.status}));
    records.sort((a,b) => a.id.localeCompare(b.id));
    const signature = JSON.stringify(records);
    if (signature === this._activitySignature && next === this._running) return;
    this._tasks = records;
    this._activitySignature = signature;
    this._running = next;
    this._rebuildAgents();
    this._updateEffects(Date.now(), !this._shouldAnimate());
    this._applyGlow();
    this._emitStatus();
    this.renderOnce();
  }

  /** Select the ambient visual metaphor; knowledge markers stay unchanged. */
  setPreset(preset) {
    if (!['galaxy','blackhole'].includes(preset) || this._disposed || this._contextLost) return;
    this._preset = preset;
    this._particleUniforms.uBlackHole.value = preset === 'blackhole' ? 1 : 0;
    this._hole.visible = this._ring.visible = preset === 'blackhole';
    this._core.visible = preset !== 'blackhole';
    this._emitStatus();
    this.renderOnce();
  }

  /** Caller delivers only fresh observed events. IDs are deduplicated, bounded. */
  pushEvents(events = []) {
    if (this._disposed || this._contextLost || !Array.isArray(events)) return;
    const now = Date.now();
    this._updateEffects(now, !this._shouldAnimate());
    for (const event of events.slice(-MAX_EFFECTS)) {
      const key = event?.id ?? event?.seq;
      const color = event?.failed ? '#ff8b9c' : EVENT_COLORS[event?.type];
      if (key == null || !color || this._seenEvents.has(String(key))) continue;
      this._seenEvents.add(String(key));
      if (this._seenEvents.size > 512) this._seenEvents.delete(this._seenEvents.values().next().value);
      if (this._effects.length >= MAX_EFFECTS) this._releaseEffect(this._effects[0]);
      const project = this._projectClusters.get(String(event.project_id || ''));
      const source = (this._agentPositions.get(String(event.task_id || '')) || project?.position || new THREE.Vector3()).clone();
      const targetNode = this._nodeById.get(String(event.target_id || ''));
      const targetProject = this._projectClusters.get(String(event.target_id || ''));
      const memory = event.type.startsWith('memory.');
      const target = (targetNode?.position || targetProject?.position || (memory ? new THREE.Vector3() : project?.position) || source).clone();
      this._createEffect({source, target, color, now, target_id:targetNode?.id ?? null});
    }
    this._updateEffects(now, !this._shouldAnimate());
    this._emitStatus();
    this.renderOnce();
  }

  /** @param {number} value 0..1 orbital speed */
  setMotion(value) {
    const next = clampNumber(value, 0, 1, this._motion);
    if (next === this._motion) return;
    this._motion = next;
    if (this._contextLost || this._disposed) return;
    this._controls.autoRotateSpeed = 0.28 * this._motion;
    this._syncLoop();
    this.renderOnce();
  }

  /** Explicit per-page opt-in; never changes the operating system preference. */
  setMotionOverride(enabled) {
    this._motionOverride = Boolean(enabled);
    this._syncLoop();
    this._emitStatus();
    this.renderOnce();
  }

  /** @param {boolean} paused stop all animation and rendering */
  setPaused(paused) {
    this._paused = Boolean(paused);
    if (this._paused) this._tween = null;
    this._syncLoop();
    this._emitStatus();
    this.renderOnce();
  }

  /** @param {boolean} visible false drops the render loop and skips rendering */
  setVisible(visible) {
    this._visible = Boolean(visible);
    if (this._visible) this.resize();
    this._syncLoop();
    this.renderOnce();
  }

  /** Sync camera, renderer and post-processing to the container box. */
  resize() {
    if (this._contextLost || this._disposed) return;
    const { width, height } = this._size();
    this._camera.aspect = width / height;
    // fov is vertical here, so a narrow container needs extra vertical angle.
    this._camera.fov = this._camera.aspect < 1 ? 52 + (1 - this._camera.aspect) * 16 : 52;
    this._camera.updateProjectionMatrix();
    this._renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 1.5));
    this._renderer.setSize(width, height, false);
    this._composer.setPixelRatio?.(this._renderer.getPixelRatio());
    this._composer.setSize(width, height); // also resizes every pass, bloom included
    this._particleUniforms.uScale.value = (height * this._renderer.getPixelRatio()) / 2;
    this._syncLoop();
    this.renderOnce();
  }

  /** Draw a single frame if the field is idle and drawable. */
  renderOnce() {
    // A failed render is terminal until the GL context is restored: reporting it
    // again on every orbit tick would only spam the status channel.
    if (this._contextLost || this._renderFailed || this._disposed || !this._visible || document.hidden) return;
    if (this._shouldAnimate()) return; // the loop already owns rendering
    this._render();
  }

  // ------------------------------------------------------------------ internals

  _size() {
    const rect = this._container?.getBoundingClientRect?.();
    const width = Math.max(1, Math.floor(rect?.width || this._container?.clientWidth || 1));
    const height = Math.max(1, Math.floor(rect?.height || this._container?.clientHeight || 1));
    return { width, height };
  }

  _applyGlow() {
    this._bloom.enabled = this._glow > 0;
    this._bloom.strength = 1.05 * this._glow;
    if (this._coreMaterial) this._coreMaterial.opacity = Math.min(0.85, 0.35 + this._glow * 0.25);
  }

  _record(node) {
    const id = node?.id === undefined || node?.id === null ? '' : String(node.id);
    return {
      id,
      kind: typeof node?.kind === 'string' ? node.kind : '',
      title: typeof node?.title === 'string' ? node.title : '',
      project_id: node?.project_id == null ? '' : String(node.project_id),
      project_materials: Math.max(1,Number(node?.project_materials)||1),
      position: new THREE.Vector3(),
      size: KIND_SIZE[node?.kind] ?? 2.7,
    };
  }

  _clearClusterSprites() {
    for (const sprite of this._clusterSprites) {
      sprite.removeFromParent();
      if (sprite.userData.ownedMap) sprite.material.map?.dispose();
      sprite.material.dispose();
    }
    this._clusterSprites.length = 0;
  }

  _clusterLabel(title, position) {
    const canvas=document.createElement('canvas');canvas.width=512;canvas.height=96;
    const ctx=canvas.getContext('2d');ctx.font='40px "Segoe UI",sans-serif';
    ctx.textAlign='center';ctx.lineWidth=7;ctx.strokeStyle='#07101d';ctx.fillStyle='#cbd7f0';
    const text=String(title).slice(0,36);ctx.strokeText(text,256,57,480);ctx.fillText(text,256,57,480);
    const map=new THREE.CanvasTexture(canvas);
    const material=new THREE.SpriteMaterial({map,transparent:true,depthWrite:false,depthTest:false});
    const label=new THREE.Sprite(material);label.scale.set(260,49,1);label.position.copy(position);
    label.position.y+=52;label.renderOrder=28;label.userData.ownedMap=true;
    this._labels.add(label);this._clusterSprites.push(label);
  }

  _buildClusters() {
    this._clearClusterSprites();
    this._clusters = [{id:null, position:new THREE.Vector3(), scale:1}];
    this._projectClusters.clear();
    this._clusterLabel('Общая память',new THREE.Vector3());
    const projects = this._nodes.filter(node => node.kind === 'project').sort((a,b) => a.id.localeCompare(b.id)).slice(0,MAX_PROJECTS);
    const occupied = new Set();
    for (const node of projects) {
      let slot = Math.floor(hashUnit(node.id,'cluster') * MAX_PROJECTS);
      while (occupied.has(slot)) slot = (slot+1) % MAX_PROJECTS;
      occupied.add(slot);
      const angle = slot * TAU / MAX_PROJECTS;
      const cluster = {id:node.id, position:new THREE.Vector3(Math.cos(angle)*340, (hashUnit(node.id,'tilt')-0.5)*45, Math.sin(angle)*340), scale:Math.min(0.58,0.36+Math.log2(1+node.project_materials)*0.025)};
      this._clusters.push(cluster);
      this._projectClusters.set(node.id, cluster);
      if (node.project_id) this._projectClusters.set(node.project_id, cluster);
      const material = this._coreMaterial.clone();
      material.opacity = 0.5;
      const sprite = new THREE.Sprite(material);
      sprite.position.copy(cluster.position);
      sprite.scale.set(38*cluster.scale/0.34,38*cluster.scale/0.34,1);
      sprite.renderOrder = 1;
      this._disk.add(sprite);
      this._clusterSprites.push(sprite);
      this._clusterLabel(`${node.title} · ${node.project_materials} материалов`,cluster.position);
    }
    this._particleUniforms.uClusterCount.value = this._clusters.length;
    this._clusters.forEach((cluster,index) => this._particleUniforms.uClusters.value[index].set(cluster.position.x,cluster.position.y,cluster.position.z,cluster.scale));
  }

  _placeNodes() {
    for (const node of this._nodes) {
      const cluster = this._projectClusters.get(node.kind === 'project' ? node.id : node.project_id) || this._clusters[0];
      if (node.kind === 'project' && cluster.id === node.id) {
        node.position.copy(cluster.position);
        continue;
      }
      const spin = hashUnit(node.id, 'angle');
      const arm = Math.floor(hashUnit(node.id, 'arm') * 5);
      const rNorm = 0.16 + 0.72 * Math.sqrt(hashUnit(node.id, 'r'));
      const theta = arm * (TAU / 5) + rNorm * 3.1 + (spin - 0.5) * 0.5;
      const radius = rNorm * DISK_RADIUS * cluster.scale;
      node.position.set(Math.cos(theta)*radius,
        (hashUnit(node.id,'z')-0.5)*DISK_THICKNESS*(1-0.7*rNorm)*cluster.scale,
        Math.sin(theta)*radius).add(cluster.position);
    }
  }

  _rebuildAgents() {
    this._agentPositions.clear();
    const matrix = new THREE.Matrix4();
    const scale = new THREE.Vector3(5,7,5);
    const linkPositions = this._agentLinkGeometry.getAttribute('position');
    const linkColors = this._agentLinkGeometry.getAttribute('color');
    const providerColors = {codex:'#91baff', qoder:'#bd98ff', antigravity:'#85e8cd'};
    this._tasks.forEach((task,index) => {
      const cluster = this._projectClusters.get(task.project_id) || this._clusters[0];
      const angle = hashUnit(task.id,'beacon')*TAU;
      const radius = cluster.id ? 45 : 105;
      const position = new THREE.Vector3(Math.cos(angle)*radius,25,Math.sin(angle)*radius).add(cluster.position);
      this._agentPositions.set(task.id,position);
      matrix.compose(position,new THREE.Quaternion(),scale);
      this._agents.setMatrixAt(index,matrix);
      const color = new THREE.Color(task.status === 'cancelling' ? '#ff9c86' : providerColors[task.provider] || '#cbd8ff');
      this._agents.setColorAt(index,color);
      linkPositions.setXYZ(index*2,position.x,position.y,position.z);
      linkPositions.setXYZ(index*2+1,cluster.position.x,cluster.position.y,cluster.position.z);
      linkColors.setXYZ(index*2,color.r,color.g,color.b);
      linkColors.setXYZ(index*2+1,color.r,color.g,color.b);
    });
    linkPositions.needsUpdate = linkColors.needsUpdate = true;
    this._agentLinkGeometry.setDrawRange(0,this._tasks.length*2);
    this._agents.count = this._tasks.length;
    this._agents.instanceMatrix.needsUpdate = true;
    if (this._agents.instanceColor) this._agents.instanceColor.needsUpdate = true;
  }

  _createEffect({source,target,color,now,target_id}) {
    const control = source.clone().lerp(target,0.5);
    control.y += 28 + Math.min(70,source.distanceTo(target)*0.18);
    const curve = new THREE.QuadraticBezierCurve3(source,control,target);
    const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
    const lineMaterial = new THREE.LineBasicMaterial({color,transparent:true,opacity:0.55,
      blending:THREE.AdditiveBlending,depthWrite:false,depthTest:false});
    const line = new THREE.Line(geometry,lineMaterial);
    line.renderOrder = 24;
    const material = new THREE.MeshBasicMaterial({color,transparent:true,opacity:1,
      blending:THREE.AdditiveBlending,depthWrite:false,depthTest:false});
    const head = new THREE.Mesh(this._markerGeometry,material);
    head.scale.setScalar(4);
    head.renderOrder = 25;
    this._effectsGroup.add(line,head);
    const staticEffect = !this._shouldAnimate();
    head.position.copy(staticEffect ? target : source);
    if (staticEffect) head.scale.setScalar(5);
    this._effects.push({curve,line,head,geometry,lineMaterial,material,start:now,target_id,target,staticEffect});
  }

  _releaseEffect(effect) {
    this._effectsGroup?.remove(effect.line,effect.head);
    effect.geometry.dispose();
    effect.lineMaterial.dispose();
    effect.material.dispose();
    const index = this._effects.indexOf(effect);
    if (index >= 0) this._effects.splice(index,1);
  }

  _updateEffects(now,staticFrame = false) {
    for (const effect of [...this._effects]) {
      const age = Math.max(0,now-effect.start);
      if (age >= EFFECT_LIFETIME) { this._releaseEffect(effect); continue; }
      // Existing pulses freeze where they are when paused. Events received in
      // reduced/static mode are endpoint highlights, never travelling flashes.
      if (staticFrame && !effect.staticEffect) continue;
      const staticHighlight = effect.staticEffect;
      const progress = staticHighlight ? 1 : Math.min(1,age/(EFFECT_LIFETIME*0.75));
      effect.head.position.copy(effect.curve.getPoint(progress));
      effect.head.scale.setScalar(staticHighlight ? 5 : 3+Math.sin(progress*Math.PI)*3+Math.max(0,progress-0.75)*14);
      effect.lineMaterial.opacity = staticHighlight ? 0.45 : 0.5*(1-age/EFFECT_LIFETIME);
      effect.material.opacity = staticHighlight ? 0.9 : Math.max(0.1,1-age/EFFECT_LIFETIME);
    }
  }

  _resolveLinks(links) {
    const stride = Math.max(1, Math.ceil(links.length / MAX_LINK_LINES));
    const resolved = [];
    for (let i = 0; i < links.length; i += stride) {
      const source = this._nodeById.get(linkEndId(links[i]?.source));
      const target = this._nodeById.get(linkEndId(links[i]?.target));
      if (source && target && source !== target) resolved.push([source, target]);
    }
    return resolved;
  }

  _rebuildMarkers() {
    this._releaseMarkers();
    if (!this._nodes.length) return;
    const mesh = new THREE.InstancedMesh(this._markerGeometry, this._markerMaterial, this._nodes.length);
    mesh.renderOrder = 26;
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    const matrix = new THREE.Matrix4();
    const scale = new THREE.Vector3();
    const color = new THREE.Color();
    this._nodes.forEach((node, index) => {
      matrix.compose(node.position, new THREE.Quaternion(), scale.setScalar(node.size));
      mesh.setMatrixAt(index, matrix);
      mesh.setColorAt(index, color.set(KIND_COLOR[node.kind] || KIND_COLOR.note));
      node.index = index;
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    this._markers = mesh;
    this._disk.add(mesh);
  }

  _releaseMarkers() {
    if (!this._markers) return;
    this._disk.remove(this._markers);
    this._markers.dispose();
    this._markers = null;
  }

  _rebuildLinkLines() {
    const position = this._linkGeometry.getAttribute('position');
    for (let i = 0; i < this._links.length; i += 1) {
      const [a, b] = this._links[i];
      position.setXYZ(i * 2, a.position.x, a.position.y, a.position.z);
      position.setXYZ(i * 2 + 1, b.position.x, b.position.y, b.position.z);
    }
    position.needsUpdate = true;
    this._linkGeometry.setDrawRange(0, this._links.length * 2);
    this._applyLinkColors();
  }

  _applyLinkColors() {
    const color = this._linkGeometry.getAttribute('color');
    const base = new THREE.Color('#2f4c7e');
    const hot = new THREE.Color('#8fb4ff');
    this._links.forEach(([a, b], i) => {
      const on = this._selected !== null && (a.id === this._selected || b.id === this._selected);
      const c = on ? hot : base;
      color.setXYZ(i * 2, c.r, c.g, c.b);
      color.setXYZ(i * 2 + 1, c.r, c.g, c.b);
    });
    color.needsUpdate = true;
  }

  _markerWorldPosition(node) {
    this._disk.updateMatrixWorld(true);
    return node.position.clone().applyMatrix4(this._disk.matrixWorld);
  }

  _pick(event) {
    if (!this._markers) return;
    // Instance matrices can be newer than the last rendered frame when paused.
    this._disk.updateMatrixWorld(true);
    const rect = this._renderer.domElement.getBoundingClientRect();
    this._ndc.set(
      ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1,
      -((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1,
    );
    this._raycaster.setFromCamera(this._ndc, this._camera);
    const hits = this._raycaster.intersectObject(this._markers, false);
    const hit = hits.find((entry) => entry.instanceId !== undefined && entry.instanceId !== null);
    if (!hit) {
      this._select(null);
      this._onSelect?.(null);
      return;
    }
    const node = this._nodes[hit.instanceId];
    if (!node) return;
    this._select(node.id);
    this._onSelect?.({ ...node });
  }

  _select(id) {
    const next = this._nodeById.has(id) ? id : null;
    if (next === this._selected) return;
    this._selected = next;
    this._applySelection();
  }

  _applySelection() {
    const color = new THREE.Color();
    if (this._markers) {
      for (const node of this._nodes) {
        color.set(KIND_COLOR[node.kind] || KIND_COLOR.note);
        if (node.id === this._selected) color.multiplyScalar(1.5);
        this._markers.setColorAt(node.index, color);
      }
      if (this._markers.instanceColor) this._markers.instanceColor.needsUpdate = true;
    }
    const node = this._selected ? this._nodeById.get(this._selected) : null;
    this._halo.visible = Boolean(node);
    if (node) {
      this._halo.position.copy(node.position);
      this._halo.scale.setScalar(node.size * 2.6);
    }
    this._applyLinkColors();
    this.renderOnce();
  }

  _moveTo(position, target, duration) {
    if (!this._shouldAnimate() || duration <= 0 || this._paused) {
      this._tween = null;
      this._camera.position.copy(position);
      this._controls.target.copy(target);
      this._controls.update();
      this.renderOnce();
      return;
    }
    this._tween = {
      fromPosition: this._camera.position.clone(),
      toPosition: position.clone(),
      fromTarget: this._controls.target.clone(),
      toTarget: target.clone(),
      duration,
      elapsed: 0,
    };
    this._startLoop();
  }

  _updateCamera(deltaMs) {
    if (!this._tween) return;
    this._tween.elapsed += deltaMs;
    const k = Math.min(1, this._tween.elapsed / this._tween.duration);
    const eased = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    this._camera.position.lerpVectors(this._tween.fromPosition, this._tween.toPosition, eased);
    this._controls.target.lerpVectors(this._tween.fromTarget, this._tween.toTarget, eased);
    if (k >= 1) this._tween = null;
  }

  /** True when continuous animation is allowed right now. */
  _shouldAnimate() {
    return (
      !this._disposed
      && !this._contextLost
      && !this._renderFailed
      && !document.hidden
      && this._visible
      && !this._paused
      && (!this._reducedMotion || this._motionOverride)
      && this._motion > 0
      && this._quality !== 'static'
    );
  }

  _syncLoop() {
    // No controls means _initThree() never finished: WebGL is unavailable.
    if (this._disposed || !this._controls) return;
    const animate = this._shouldAnimate();
    this._updateEffects(Date.now(), !animate);
    this._controls.enableDamping = animate;
    this._controls.autoRotate = animate;
    this._controls.autoRotateSpeed = 0.28 * this._motion;
    if (animate) this._startLoop();
    else this._stopLoop();
  }

  _startLoop() {
    if (this._raf !== null || !this._shouldAnimate()) return;
    this._lastFrame = null;
    this._frameCount = 0;
    this._windowStart = null;
    this._fps = null;
    this._raf = requestAnimationFrame(this._step);
  }

  _stopLoop() {
    if (this._raf === null) return;
    cancelAnimationFrame(this._raf);
    this._raf = null;
    this._lastFrame = null;
    this._windowStart = null;
    this._frameCount = 0;
  }

  _step = (now) => {
    if (this._disposed) { this._raf = null; return; }
    if (!this._shouldAnimate()) { this._stopLoop(); this._emitStatus(); return; }
    this._raf = requestAnimationFrame(this._step);
    const delta = this._lastFrame === null ? 0 : Math.min(64, now - this._lastFrame);
    this._lastFrame = now;

    this._time += delta * 0.0001 * this._motion;
    this._particleUniforms.uTime.value = this._time;
    this._updateCamera(delta);
    this._controls.update(delta / 1000);
    if (!this._render()) { this._stopLoop(); return; }
    this._measure(now, delta);
  };

  _render() {
    this._updateEffects(Date.now(), !this._shouldAnimate());
    try {
      this._composer.render();
      // Text is an explanatory overlay, kept out of the bloom pass.
      this._labels.quaternion.copy(this._disk.quaternion);
      const autoClear = this._renderer.autoClear;
      try {
        this._renderer.autoClear = false;
        this._renderer.clearDepth();
        this._renderer.render(this._labelScene, this._camera);
      } finally { this._renderer.autoClear = autoClear; }
      return true;
    } catch (error) {
      this._renderFailed = true;
      this._emitStatus(`Постобработка недоступна: ${error?.message || error}`);
      return false;
    }
  }

  /** RAF cadence measures animation scheduling, not GPU presentation or frame time. */
  _measure(now, delta) {
    if (this._windowStart === null) this._windowStart = now;
    this._frameCount += 1;
    const elapsed = now - this._windowStart;
    if (elapsed < 1000 || delta === 0) return;
    const measured = Math.round((this._frameCount * 1000) / elapsed);
    this._fps = measured;
    this._windowStart = now;
    this._frameCount = 0;
    this._emitStatus();

    if (this._mode !== 'auto') return;
    if (measured >= DOWNGRADE_FPS) {
      this._slowWindows = 0;
      return;
    }
    this._slowWindows += 1;
    if (this._slowWindows < 2 || this._quality === 'static') return;
    // One-way downgrade: auto never upgrades on its own.
    this._quality = 'static';
    this._particleGeometry.setDrawRange(0, PARTICLES.static);
    this._stopLoop();
    this._fps = null;
    this._emitStatus(`Авто: измерено ${measured} fps, понижение до статичного режима.`);
    this.renderOnce();
  }

  _emitStatus(error) {
    if (error) this._lastError = String(error);
    if (typeof this._onStatus !== 'function') return;
    const animated = this._raf !== null;
    this._onStatus({
      quality: this._quality,
      particles: this._contextLost ? 0 : PARTICLES[this._quality],
      fps: animated ? this._fps : null,
      paused: this._paused,
      reducedMotion: this._reducedMotion && !this._motionOverride,
      running: this._running,
      agents: this._tasks.length,
      effects: this._effects.length,
      clusters: this._clusters.length,
      preset: this._preset,
      ...((error || ((this._contextLost || this._renderFailed) && this._lastError)) ? { error: this._lastError } : {}),
    });
  }
}

export default ParticleField;
