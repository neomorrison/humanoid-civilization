// Procedural low-poly townsfolk: a jointed rig built from primitives, dressed by job, scaled by
// age, and posed procedurally (walk, sit, eat, hoe, pick, milk, carry, serve, chat, lie...).
import * as THREE from "three";
import { JOB_COLORS } from "./replay.js";
import { hash, pick, clamp, lerp } from "./util.js";

const SKIN = [0xf3cfae, 0xe8b893, 0xd49a70, 0xb57a52, 0x8d5a3b, 0x654030];
const HAIR = [0x2a1c14, 0x3b2616, 0x5a3a22, 0x8a5a2e, 0xc79a58, 0xe0c27a, 0x17130f, 0x8c3b22, 0x6b4f3a];
const PANTS = [0x2f3f5c, 0x3d4a3a, 0x4a3b30, 0x2d2f36, 0x55607a, 0x5e4b3c];
const DRESS = [0x7a4f8a, 0x3f6f8f, 0x9b4a4a, 0x4f7a5a, 0x8a6f3f];
const PACKS = [0xe0533b, 0x3d7fd1, 0xf2b631, 0x3ea675, 0xa25fd0];

const matCache = new Map();
export function mat(hex, opts) {
  const key = hex + (opts ? JSON.stringify(opts) : "");
  let m = matCache.get(key);
  if (!m) { m = new THREE.MeshLambertMaterial({ color: hex, ...(opts || {}) }); matCache.set(key, m); }
  return m;
}

const cap = (r, len, rs = 7) => new THREE.CapsuleGeometry(r, len, 2, rs);
const G = {
  thigh: cap(0.052, 0.13).translate(0, -0.115, 0),
  shin: cap(0.045, 0.14).translate(0, -0.11, 0),
  foot: new THREE.BoxGeometry(0.075, 0.045, 0.13).translate(0, -0.215, 0.025),
  pelvis: new THREE.CylinderGeometry(0.105, 0.1, 0.13, 10).scale(1, 1, 0.78).translate(0, 0.01, 0),
  torso: new THREE.CylinderGeometry(0.118, 0.1, 0.28, 10).scale(1, 1, 0.74).translate(0, 0.2, 0),
  shoulders: new THREE.SphereGeometry(0.118, 10, 6, 0, Math.PI * 2, 0, Math.PI / 2).scale(1, 0.42, 0.74).translate(0, 0.338, 0),
  neck: new THREE.CylinderGeometry(0.033, 0.036, 0.07, 7).translate(0, 0.37, 0),
  upper: cap(0.04, 0.1).translate(0, -0.08, 0),
  fore: cap(0.035, 0.1).translate(0, -0.075, 0),
  hand: new THREE.SphereGeometry(0.036, 7, 5).translate(0, -0.165, 0.005),
  head: new THREE.SphereGeometry(0.1, 14, 10).scale(0.94, 1.04, 0.98).translate(0, 0.1, 0),
  eyes: (() => {
    const e = new THREE.SphereGeometry(0.0135, 6, 4);
    const a = e.clone().translate(0.036, 0.108, 0.088), b = e.clone().translate(-0.036, 0.108, 0.088);
    return mergeTwo(a, b);
  })(),
  nose: new THREE.SphereGeometry(0.016, 5, 4).scale(0.9, 1, 1.2).translate(0, 0.085, 0.1),
  capHair: new THREE.SphereGeometry(0.108, 14, 8, 0, Math.PI * 2, 0, Math.PI * 0.56).rotateX(-0.28).translate(0, 0.105, -0.006),
  longHair: new THREE.BoxGeometry(0.19, 0.2, 0.06).translate(0, 0.03, -0.075),
  bun: new THREE.SphereGeometry(0.047, 8, 6).translate(0, 0.17, -0.085),
  pony: cap(0.03, 0.1, 6).rotateX(0.35).translate(0, 0.04, -0.12),
  curls: (() => {
    const parts = [];
    for (let k = 0; k < 9; k++) {
      const a = (k / 9) * Math.PI * 2;
      parts.push(new THREE.SphereGeometry(0.04, 6, 4).translate(Math.cos(a) * 0.085, 0.17 + (k % 2) * 0.02, Math.sin(a) * 0.075 - 0.01));
    }
    return mergeMany(parts);
  })(),
  skirt: new THREE.CylinderGeometry(0.1, 0.165, 0.25, 12, 1, true).translate(0, -0.1, 0),
  belly: new THREE.SphereGeometry(0.095, 10, 8).scale(0.95, 0.9, 1).translate(0, 0.12, 0.075),
  strawBrim: new THREE.CylinderGeometry(0.19, 0.2, 0.014, 14).translate(0, 0.175, 0),
  strawTop: new THREE.CylinderGeometry(0.085, 0.1, 0.075, 12).translate(0, 0.215, 0),
  toque: mergeTwo(new THREE.CylinderGeometry(0.085, 0.08, 0.12, 12).translate(0, 0.23, 0), new THREE.SphereGeometry(0.1, 10, 6).scale(1, 0.6, 1).translate(0, 0.3, 0)),
  capHat: mergeTwo(new THREE.CylinderGeometry(0.108, 0.11, 0.06, 12).translate(0, 0.175, 0), new THREE.BoxGeometry(0.15, 0.012, 0.09).translate(0, 0.152, 0.1)),
  badge: new THREE.BoxGeometry(0.035, 0.035, 0.01).translate(0.05, 0.25, 0.09),
  apron: new THREE.BoxGeometry(0.17, 0.3, 0.012).translate(0, 0.02, 0.1),
  pack: new THREE.BoxGeometry(0.16, 0.18, 0.08).translate(0, 0.2, -0.11),
  // props
  hoe: mergeTwo(new THREE.CylinderGeometry(0.012, 0.012, 0.62, 5).translate(0, 0, 0), new THREE.BoxGeometry(0.1, 0.012, 0.07).translate(0, -0.31, 0.03)),
  crate: new THREE.BoxGeometry(0.22, 0.16, 0.18),
  basket: new THREE.CylinderGeometry(0.09, 0.07, 0.09, 10),
  pail: new THREE.CylinderGeometry(0.06, 0.05, 0.1, 10),
  bundle: new THREE.SphereGeometry(0.08, 10, 8).scale(1.35, 0.85, 0.85),
  babyHead: new THREE.SphereGeometry(0.052, 10, 8),
  bowl: new THREE.SphereGeometry(0.05, 10, 6, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2),
  tray: new THREE.BoxGeometry(0.2, 0.015, 0.14),
  blob: new THREE.CircleGeometry(0.2, 18).rotateX(-Math.PI / 2),
  pickProxy: new THREE.CylinderGeometry(0.16, 0.16, 1.0, 6).translate(0, 0.5, 0),
};
function mergeTwo(a, b) { return mergeMany([a, b]); }
function mergeMany(list) {
  // minimal merge for geometries with the same attributes (position, normal, uv), all indexed
  let nv = 0, ni = 0;
  list.forEach((g) => { nv += g.attributes.position.count; ni += g.index ? g.index.count : g.attributes.position.count; });
  const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), idx = new Uint32Array(ni);
  let ov = 0, oi = 0;
  list.forEach((g) => {
    pos.set(g.attributes.position.array, ov * 3); nor.set(g.attributes.normal.array, ov * 3);
    const n = g.attributes.position.count;
    if (g.index) { for (let k = 0; k < g.index.count; k++) idx[oi + k] = g.index.array[k] + ov; oi += g.index.count; }
    else { for (let k = 0; k < n; k++) idx[oi + k] = ov + k; oi += n; }
    ov += n;
  });
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  out.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  out.setIndex(new THREE.BufferAttribute(idx, 1));
  return out;
}

const blobMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22, depthWrite: false });
const pickMat = new THREE.MeshBasicMaterial({ visible: false });

function mesh(geo, material, parent) { const m = new THREE.Mesh(geo, material); parent.add(m); return m; }

// joint targets
const ZERO = { y: 0, lie: 0, sp: 0, hx: 0, hy: 0, hl: 0, hr: 0, kl: 0, kr: 0, sl: 0, sr: 0, szl: 0.08, szr: -0.08, el: -0.15, er: -0.15 };
const KEYS = Object.keys(ZERO);

export class Human {
  constructor(pid, sex, seed) {
    this.pid = pid; this.sex = sex;
    const u = (k) => hash(pid, seed, k);
    this.skin = pick(SKIN, u(1));
    this.hairCol = pick(HAIR, u(2));
    this.pants = pick(PANTS, u(3));
    this.dressCol = pick(DRESS, u(4));
    this.dress = sex === 0 && u(5) < 0.45;
    this.hairStyle = sex === 0 ? pick(["long", "bun", "pony", "long", "short", "curly"], u(6)) : pick(["short", "short", "curly", "buzz", "short"], u(6));
    this.pack = pick(PACKS, u(7));
    this.phase = u(8) * 100;
    this.gait = 0.9 + u(9) * 0.2;

    const root = (this.root = new THREE.Group());
    const body = (this.body = new THREE.Group()); root.add(body);
    this.blob = mesh(G.blob, blobMat, root); this.blob.position.y = 0.012; this.blob.renderOrder = 1;
    this.proxy = mesh(G.pickProxy, pickMat, root); this.proxy.userData.human = this;
    const hips = (this.hips = new THREE.Group()); hips.position.y = 0.5; body.add(hips);
    const spine = (this.spine = new THREE.Group()); hips.add(spine);
    this.legs = [0.07, -0.07].map((x) => {
      const hip = new THREE.Group(); hip.position.set(x, 0, 0); hips.add(hip);
      const thigh = mesh(G.thigh, null, hip);
      const knee = new THREE.Group(); knee.position.y = -0.235; hip.add(knee);
      const shin = mesh(G.shin, null, knee);
      const foot = mesh(G.foot, null, knee);
      return { hip, knee, thigh, shin, foot };
    });
    this.pelvis = mesh(G.pelvis, null, hips);
    this.skirt = mesh(G.skirt, null, hips);
    this.torso = mesh(G.torso, null, spine);
    this.shoulderCap = mesh(G.shoulders, null, spine);
    this.belly = mesh(G.belly, null, spine);
    this.apron = mesh(G.apron, mat(0xfafaf5), spine); this.apron.position.set(0, 0.17, 0.0);
    this.backpack = mesh(G.pack, null, spine);
    this.badge = mesh(G.badge, mat(0xf2c94c), spine);
    this.neck = mesh(G.neck, null, spine);
    const head = (this.head = new THREE.Group()); head.position.y = 0.39; spine.add(head);
    this.headMesh = mesh(G.head, null, head);
    this.eyes = mesh(G.eyes, mat(0x1b1b22), head);
    this.nose = mesh(G.nose, null, head);
    this.hair = new THREE.Group(); head.add(this.hair);
    this.hat = new THREE.Group(); head.add(this.hat);
    this.arms = [0.142, -0.142].map((x) => {
      const sh = new THREE.Group(); sh.position.set(x, 0.315, 0); spine.add(sh);
      const upper = mesh(G.upper, null, sh);
      const el = new THREE.Group(); el.position.y = -0.165; sh.add(el);
      const fore = mesh(G.fore, null, el);
      const hand = mesh(G.hand, null, el);
      return { sh, el, upper, fore, hand };
    });
    // props
    const R = this.arms[1].el;
    this.hoe = mesh(G.hoe, mat(0x8a6a44), R); this.hoe.position.set(0, -0.17, 0.02); this.hoe.rotation.x = 1.2;
    this.crate = mesh(G.crate, mat(0xb58750), spine); this.crate.position.set(0, 0.15, 0.2);
    this.basket = mesh(G.basket, mat(0xa47a44), root); this.basket.position.set(0.18, 0.045, 0.1);
    this.pail = mesh(G.pail, mat(0xb9c0c6), root); this.pail.position.set(0.02, 0.05, 0.2);
    this.tray = mesh(G.tray, mat(0xd9d2c3), this.arms[0].el); this.tray.position.set(0, -0.19, 0.05);
    this.bowl = mesh(G.bowl, mat(0xf0e6d2), this.arms[1].el); this.bowl.position.set(0, -0.2, 0.02);
    this.baby = new THREE.Group(); spine.add(this.baby); this.baby.position.set(0, 0.2, 0.15);
    mesh(G.bundle, mat(0xf4e6f0), this.baby); const bh = mesh(G.babyHead, mat(this.skin), this.baby); bh.position.set(-0.085, 0.03, 0.01);

    // a baby basket set down beside a working parent
    this.cot = new THREE.Group(); root.add(this.cot); this.cot.position.set(-0.26, 0, 0.12);
    const cb = mesh(new THREE.CylinderGeometry(0.1, 0.08, 0.08, 12, 1, true), mat(0xb08650, { side: THREE.DoubleSide }), this.cot); cb.scale.set(1.4, 1, 0.8); cb.position.y = 0.04;
    const cbase = mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.01, 12), mat(0x9a7444), this.cot); cbase.scale.set(1.4, 1, 0.8); cbase.position.y = 0.005;
    const cbun = mesh(G.bundle, mat(0xcfe3f7), this.cot); cbun.scale.setScalar(0.8); cbun.position.y = 0.07;
    const chd = mesh(G.babyHead, mat(this.skin), this.cot); chd.scale.setScalar(0.8); chd.position.set(-0.07, 0.09, 0);
    this.cur = { ...ZERO }; this.tgt = { ...ZERO };
    this.yaw = 0; this.yawT = 0; this.walkPh = u(10) * 6;
    this.job = null; this.age = -1; this.police = null; this.preg = null;
    this.opacity = 1;
    this.setHair(false);
    this.setJob("none", false);
    this.all = [];
    root.traverse((o) => { if (o.isMesh && o !== this.blob && o !== this.proxy) { o.castShadow = false; this.all.push(o); } });
    this.setProps({});
  }

  setHair(grey) {
    this.hair.clear();
    const col = grey ? 0xd4d2cc : this.hairCol, m = mat(col);
    const style = this.hairStyle;
    if (style === "buzz") { const c = mesh(G.capHair, m, this.hair); c.scale.setScalar(0.97); return; }
    if (grey && this.sex === 1 && hash(this.pid, 77) < 0.4) {   // balding
      const c = mesh(G.capHair, m, this.hair); c.scale.set(1.02, 0.55, 1.02); c.position.y = -0.02; return;
    }
    mesh(G.capHair, m, this.hair);
    if (style === "long") mesh(G.longHair, m, this.hair);
    if (style === "bun") mesh(G.bun, m, this.hair);
    if (style === "pony") mesh(G.pony, m, this.hair);
    if (style === "curly") mesh(G.curls, m, this.hair);
  }

  setJob(job, police) {
    if (job === this.job && police === this.police) return;
    this.job = job; this.police = police;
    const shirt = mat(JOB_COLORS[job] ?? JOB_COLORS.none);
    const skin = mat(this.skin);
    const lower = this.dress ? mat(this.dressCol) : mat(job === "dairy" ? 0xe7e7e0 : this.pants);
    this.torso.material = shirt; this.shoulderCap.material = shirt; this.belly.material = shirt;
    this.arms.forEach((a) => { a.upper.material = shirt; a.fore.material = job === "dairy" || job === "warehouse" ? shirt : skin; a.hand.material = skin; });
    this.pelvis.material = lower; this.skirt.material = lower;
    this.legs.forEach((l) => { l.thigh.material = this.dress ? skin : lower; l.shin.material = this.dress ? skin : lower; l.foot.material = mat(0x2b2522); });
    this.neck.material = skin; this.headMesh.material = skin; this.nose.material = skin;
    this.skirt.visible = this.dress;
    this.apron.visible = job === "canteen";
    this.badge.visible = !!police;
    this.hat.clear();
    if (police) { mesh(G.capHat, mat(0x24407a), this.hat); }
    else if (job === "farm") { mesh(G.strawBrim, mat(0xe2c36b), this.hat); mesh(G.strawTop, mat(0xd8b65a), this.hat); }
    else if (job === "canteen") mesh(G.toque, mat(0xfbfbf7), this.hat);
    else if (job === "dairy") mesh(G.capHat, mat(0xf4f4ef), this.hat);
    else if (job === "warehouse") mesh(G.capHat, mat(0x6b4526), this.hat);
    else if (job === "orchard") { const c = mesh(G.capHat, mat(0xd9731d), this.hat); c.scale.set(1, 0.8, 1); }
    this.hat.traverse((o) => { if (o.isMesh) this.all && this.all.push(o); });
  }

  setAge(age) {
    const a = Math.floor(age * 2) / 2;
    if (a === this.age) return;
    this.age = a;
    const h = age >= 17 ? 1 : 0.36 + 0.64 * Math.pow(clamp(age / 17, 0, 1), 0.8);
    this.scale = h * 0.7;
    this.body.scale.setScalar(this.scale);
    this.blob.scale.setScalar(h * 1.1);
    const headK = 1 + 0.55 * (1 - h);
    this.head.scale.setScalar(headK);
    this.child = age < 12;
    this.backpack.visible = age >= 5 && age < 16;
    this.backpack.material = mat(this.pack);
    const grey = age >= 58;
    if (grey !== this.grey) { this.grey = grey; this.setHair(grey); }
    this.stoop = age >= 65 ? 0.14 : 0;
    this.height = 1.08 * this.scale * (age < 12 ? 1.05 : 1);
  }

  setPregnant(p) { this.belly.visible = !!p; }

  setProps({ hoe = false, crate = false, basket = false, pail = false, tray = false, bowl = false, baby = false, cot = false } = {}) {
    this.hoe.visible = hoe; this.crate.visible = crate; this.basket.visible = basket; this.pail.visible = pail;
    this.tray.visible = tray; this.bowl.visible = bowl; this.baby.visible = baby; this.cot.visible = cot;
  }

  // fade in/out by shrinking into the ground: cheap, keeps materials shared
  setOpacity(o) { this.opacity = o; }

  // pose targets; t = seconds of animation clock
  pose(name, t, o = {}) {
    const T = this.tgt, ph = t + this.phase;
    Object.assign(T, ZERO);
    T.sp = this.stoop || 0;
    const sin = Math.sin;
    switch (name) {
      case "walk": case "run": {
        const k = name === "run" ? 1.45 : 1, w = this.walkPh;
        T.hl = -0.5 * k * sin(w); T.hr = 0.5 * k * sin(w);
        T.kl = 0.12 + 0.55 * k * Math.max(0, sin(w + 1.3)); T.kr = 0.12 + 0.55 * k * Math.max(0, sin(w + 1.3 + Math.PI));
        T.sl = 0.42 * k * sin(w); T.sr = -0.42 * k * sin(w);
        T.el = T.er = name === "run" ? -1.2 : -0.35;
        T.y = 0.018 * k * Math.abs(Math.cos(w));
        T.sp += name === "run" ? 0.16 : 0.05;
        break;
      }
      case "sit": case "lesson": case "eat": case "jail": {
        T.hl = T.hr = -1.5; T.kl = T.kr = 1.5; T.y = -(0.5 - (o.seat ?? 0.27));
        T.sl = T.sr = -0.35; T.el = T.er = -0.9;
        if (name === "eat") {
          const b = Math.max(0, sin(ph * 2.2));
          T.sr = -0.55 - 0.35 * b; T.er = -1.25 - 0.6 * b; T.hx = 0.12;
        } else if (name === "lesson") {
          if (sin(ph * 0.45) > 0.86) { T.sr = -2.9; T.er = -0.1; } else { T.sl = T.sr = -0.55; T.el = T.er = -1.2; }
          T.hx = 0.05 * sin(ph * 0.7);
        } else if (name === "jail") { T.sp += 0.35; T.hx = 0.45; T.sl = T.sr = -0.5; T.el = T.er = -1.1; }
        else { T.hy = 0.3 * sin(ph * 0.4); }
        break;
      }
      case "ground": case "picnic": {
        T.hl = -1.45; T.hr = -1.3; T.kl = 0.25; T.kr = 0.55; T.y = -0.43;
        T.sl = T.sr = 0.35; T.el = T.er = -0.1; T.sp -= 0.12;
        if (name === "picnic") { const b = Math.max(0, sin(ph * 2)); T.sr = -0.6 - 0.35 * b; T.er = -1.3 - 0.6 * b; T.hx = 0.1; T.sp += 0.15; }
        break;
      }
      case "hoe": {
        const c = sin(ph * 2.6);
        T.sp += 0.32 + 0.12 * c; T.sl = T.sr = -0.95 + 0.55 * c; T.el = T.er = -0.35; T.szl = -0.12; T.szr = 0.12;
        T.hl = -0.35; T.kl = 0.35; T.hr = 0.25; T.kr = 0.1; T.hx = 0.2;
        break;
      }
      case "pick": {
        const c = sin(ph * 2.1);
        T.sl = -2.55 + 0.3 * c; T.sr = -2.35 - 0.3 * c; T.el = T.er = -0.35; T.hx = -0.4; T.y = 0.01 + 0.01 * c;
        break;
      }
      case "milk": {
        T.hl = T.hr = -1.75; T.kl = T.kr = 2.3; T.y = -0.3; T.sp += 0.3;
        T.sl = -0.9 + 0.2 * sin(ph * 6); T.sr = -0.9 + 0.2 * sin(ph * 6 + Math.PI); T.el = T.er = -0.5; T.hx = 0.25;
        break;
      }
      case "carry": {
        T.sl = T.sr = -0.85; T.el = T.er = -0.95; T.szl = -0.12; T.szr = 0.12;
        if (o.moving) {
          const w = this.walkPh;
          T.hl = -0.45 * sin(w); T.hr = 0.45 * sin(w); T.kl = 0.12 + 0.5 * Math.max(0, sin(w + 1.3)); T.kr = 0.12 + 0.5 * Math.max(0, sin(w + 1.3 + Math.PI));
          T.y = 0.015 * Math.abs(Math.cos(w));
        }
        break;
      }
      case "serve": {
        T.sr = -1.0 + 0.35 * sin(ph * 2.4); T.er = -1.1 + 0.3 * sin(ph * 3.1); T.sl = -0.75; T.el = -1.2;
        T.hx = 0.1 + 0.08 * sin(ph * 1.3); T.hy = 0.25 * sin(ph * 0.5);
        break;
      }
      case "work": {
        const c = Math.max(0, sin(ph * 3.4));
        T.sr = -1.3 - 0.8 * c; T.er = -0.9 + 0.4 * c; T.sl = -0.6; T.el = -0.9; T.sp += 0.2;
        break;
      }
      case "talk": {
        T.sr = -0.45 + 0.28 * sin(ph * 1.7); T.er = -1.25 + 0.35 * sin(ph * 2.3);
        T.sl = -0.1 + 0.12 * sin(ph * 1.1 + 1); T.el = -0.3;
        T.hx = 0.08 * sin(ph * 2.1); T.hy = 0.2 * sin(ph * 0.6);
        break;
      }
      case "buy": {
        const c = sin(ph * 1.6);
        T.sr = -1.15 + 0.25 * c; T.er = -0.35; T.hx = 0.25; T.sp += 0.08;
        break;
      }
      case "play": {
        const w = this.walkPh;
        T.hl = -0.7 * sin(w); T.hr = 0.7 * sin(w); T.kl = 0.2 + 0.8 * Math.max(0, sin(w + 1.3)); T.kr = 0.2 + 0.8 * Math.max(0, sin(w + 1.3 + Math.PI));
        T.sl = -1.6 + 0.8 * sin(w); T.sr = -1.6 - 0.8 * sin(w); T.el = T.er = -0.4; T.y = 0.06 * Math.abs(sin(w));
        break;
      }
      case "lie": {
        T.lie = 1; T.sl = 0.1; T.sr = 0.1; T.szl = 0.4; T.szr = -0.4; T.hl = 0.05; T.hr = -0.05; T.hy = 0.5;
        break;
      }
      case "baby": {   // a toddler sitting on the floor, waving its arms
        T.hl = T.hr = -1.5; T.kl = T.kr = 0.2; T.y = -0.44; T.sl = -0.8 + 0.4 * sin(ph * 3); T.sr = -0.8 - 0.4 * sin(ph * 3); T.el = T.er = -0.5;
        break;
      }
      default: {   // stand
        T.sp += 0.02 * sin(ph * 1.3); T.hy = 0.35 * sin(ph * 0.31) * sin(ph * 0.17); T.hx = 0.05 * sin(ph * 0.5);
        T.sl = 0.05 * sin(ph * 0.9); T.sr = -0.05 * sin(ph * 0.9);
      }
    }
    if (o.cradle) { T.sl = T.sr = -0.6; T.el = T.er = -1.55; T.szl = -0.25; T.szr = 0.25; }
    if (this.belly.visible && T.sp > 0.1) T.sp = 0.1;
  }

  update(dt) {
    const k = 1 - Math.exp(-dt * 11), C = this.cur, T = this.tgt;
    for (const key of KEYS) C[key] += (T[key] - C[key]) * k;
    const [L, R] = this.legs, [AL, AR] = this.arms;
    L.hip.rotation.x = C.hl; R.hip.rotation.x = C.hr; L.knee.rotation.x = C.kl; R.knee.rotation.x = C.kr;
    AL.sh.rotation.x = C.sl; AR.sh.rotation.x = C.sr; AL.sh.rotation.z = C.szl; AR.sh.rotation.z = C.szr;
    AL.el.rotation.x = C.el; AR.el.rotation.x = C.er;
    this.spine.rotation.x = C.sp; this.head.rotation.x = C.hx; this.head.rotation.y = C.hy;
    this.hips.position.y = 0.5 + C.y;
    this.body.rotation.x = -C.lie * Math.PI / 2;
    const o = this.opacity;
    this.body.position.set(0, C.lie * 0.1 * this.scale - (1 - o) * 0.3 * this.scale, C.lie * 0.5 * this.scale);
    this.body.scale.setScalar(this.scale * (0.3 + 0.7 * o));
    this.blob.visible = o > 0.3;
    let d = this.yawT - this.yaw; d = Math.atan2(Math.sin(d), Math.cos(d));
    this.yaw += d * (1 - Math.exp(-dt * 9));
    this.root.rotation.y = this.yaw;
  }
}
