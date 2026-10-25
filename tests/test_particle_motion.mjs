import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {ParticleField} from '../frontend/particle-field.js';

// Exercise the real constructor, RAF callback, uniforms and OrbitControls. Only
// browser event wiring and GPU rendering are substituted: this is not a WebGL
// performance or visual-acceptance test.
let reduced = false;
let nextFrame = 0;
const pending = new Map();
globalThis.document = {hidden: false, removeEventListener() {}};
globalThis.matchMedia = () => ({matches: reduced, addEventListener() {}, removeEventListener() {}});
globalThis.requestAnimationFrame = callback => {
  pending.set(++nextFrame, callback);
  return nextFrame;
};
globalThis.cancelAnimationFrame = id => pending.delete(id);

class MotionField extends ParticleField {
  _initThree() {
    this._camera = new THREE.PerspectiveCamera(52, 1, 1, 6000);
    this._camera.position.set(0, 340, 470);
    this._controls = new OrbitControls(this._camera, null);
    this._controls.dampingFactor = 0.08;
    this._controls.minDistance = 60;
    this._controls.maxDistance = 3200;
    // No DOM was connected by this harness, so there are no listeners to remove.
    this._controls.dispose = () => {};
    this._disk = new THREE.Group();
    this._labels = new THREE.Group();
    this._labelScene = new THREE.Scene();
    this._labelScene.add(this._labels);
    this._renderer = {
      autoClear: true, setPixelRatio() {}, setSize() {}, getPixelRatio: () => 1,
      clearDepth() {}, render() {}, dispose() {},
    };
    this._composer = {passes: [], setPixelRatio() {}, setSize() {}, render() {}, dispose() {}};
    this._bloom = {};
  }
  _bindEvents() {}
  // Projection/resource behavior has its own test_particle_field.mjs harness.
  _buildClusters() {}
  _placeNodes() {}
  _rebuildAgents() {}
  _rebuildMarkers() {}
  _rebuildLinkLines() {}
  _applySelection() {}
}

const box = {getBoundingClientRect: () => ({width: 700, height: 700})};

function createField(prefersReduced = false) {
  assert.equal(pending.size, 0, 'previous field must release its scheduled RAF');
  document.hidden = false;
  reduced = prefersReduced;
  const statuses = [];
  const field = new MotionField({container: box, onStatus: status => statuses.push(status)});
  field.setData({nodes: [{id: 'test-note', kind: 'note'}]});
  field.setQuality('low');
  return {field, statuses};
}

function frame(now) {
  assert.equal(pending.size, 1, 'exactly one RAF owns rendering');
  const [id, callback] = pending.entries().next().value;
  pending.delete(id);
  callback(now);
}

test('real lifecycle advances orbital time, camera rotation and measured RAF FPS', () => {
  const {field, statuses} = createField();
  try {
    const theta = field._controls.getAzimuthalAngle();
    for (let i = 0; i <= 90; i += 1) frame(i * 1000 / 60);
    assert.ok(Math.abs(field._particleUniforms.uTime.value - 0.15) < 1e-10);
    assert.ok(Math.abs(field._controls.getAzimuthalAngle() - theta) > 0.003);
    assert.ok(statuses.some(status => status.fps >= 59 && status.fps <= 62));
    assert.equal(pending.size, 1);
  } finally {
    field.dispose();
  }
  assert.equal(pending.size, 0);
});

test('rotation uses elapsed seconds rather than a fixed increment per frame', () => {
  function measure(hz) {
    const {field} = createField();
    try {
      // Remove damping lag from the comparison, so this checks the delta passed
      // into the actual OrbitControls rotation math at different RAF cadences.
      field._controls.enableDamping = false;
      const before = field._controls.getAzimuthalAngle();
      for (let i = 0; i <= hz; i += 1) frame(i * 1000 / hz);
      return Math.abs(field._controls.getAzimuthalAngle() - before);
    } finally {
      field.dispose();
    }
  }
  const at60 = measure(60);
  const at30 = measure(30);
  assert.ok(at60 > 0.02);
  assert.ok(Math.abs(at60 - at30) < 1e-10, `rotation differs: ${at60} vs ${at30}`);
});

test('reduced-motion requires explicit override and pause/hidden/static still prevail', () => {
  const {field, statuses} = createField(true);
  try {
    assert.equal(pending.size, 0);
    assert.equal(field._shouldAnimate(), false);
    assert.equal(statuses.at(-1).fps, null);
    field.setMotionOverride(true);
    assert.equal(field._shouldAnimate(), true);
    frame(0);
    frame(20);
    assert.ok(field._particleUniforms.uTime.value > 0);

    const stoppedTime = field._time;
    field.setPaused(true);
    assert.equal(pending.size, 0);
    assert.equal(statuses.at(-1).fps, null);
    assert.equal(field._time, stoppedTime);
    field.setPaused(false);
    assert.equal(pending.size, 1);
    document.hidden = true;
    field._syncLoop();
    assert.equal(pending.size, 0);
    assert.equal(field._shouldAnimate(), false);
    document.hidden = false;
    field._syncLoop();
    assert.equal(pending.size, 1);
    field.setVisible(false);
    assert.equal(pending.size, 0);
    field.setVisible(true);
    assert.equal(pending.size, 1);
    field.setQuality('static');
    assert.equal(pending.size, 0);
    field.setQuality('low');
    assert.equal(pending.size, 1);
    field.setMotionOverride(false);
    assert.equal(pending.size, 0);
    assert.equal(field._shouldAnimate(), false);
  } finally {
    document.hidden = false;
    field.dispose();
  }
  assert.equal(pending.size, 0);
});
