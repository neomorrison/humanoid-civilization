// The town, built procedurally from the replay's config: terrain, roads, houses, workplaces,
// trees, sky, seasons and night lighting. Static geometry is merged (vertex colours) so the
// whole town costs a handful of draw calls; repeated living things are instanced.
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { rng, hash, clamp, lerp, smooth, fbm, textCanvas, pick } from "./util.js";

// ------------------------------------------------------------------ snow shader patch
export const SNOW = { value: 0 };
const SNOW_COLOR = { value: new THREE.Color(0xe4ebf2) };
export function snowy(m, k = 1) {
  const K = { value: k };
  m.userData.snowK = K;
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uSnow = SNOW; sh.uniforms.uSnowK = K; sh.uniforms.uSnowColor = SNOW_COLOR;
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vSnowUp;")
      .replace("#include <beginnormal_vertex>", `#include <beginnormal_vertex>
        #ifdef USE_INSTANCING
          vSnowUp = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * objectNormal).y;
        #else
          vSnowUp = normalize(mat3(modelMatrix) * objectNormal).y;
        #endif`);
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vSnowUp; uniform float uSnow; uniform float uSnowK; uniform vec3 uSnowColor;")
      .replace("#include <color_fragment>", "#include <color_fragment>\n  diffuseColor.rgb = mix(diffuseColor.rgb, uSnowColor, clamp(uSnow * uSnowK * smoothstep(0.3, 0.75, vSnowUp), 0.0, 1.0));");
  };
  m.customProgramCacheKey = () => "snowy1";
  return m;
}

// ------------------------------------------------------------------ geometry helpers
const _e = new THREE.Euler(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3();
function TRS(x, y, z, ry = 0, sx = 1, sy = 1, sz = 1, rx = 0, rz = 0) {
  _e.set(rx, ry, rz, "YXZ"); _q.setFromEuler(_e);
  return new THREE.Matrix4().compose(_p.set(x, y, z), _q, _s.set(sx, sy, sz));
}
const U = {
  box: new THREE.BoxGeometry(1, 1, 1),
  cyl6: new THREE.CylinderGeometry(1, 1, 1, 6),
  cyl8: new THREE.CylinderGeometry(1, 1, 1, 8),
  cyl12: new THREE.CylinderGeometry(1, 1, 1, 12),
  cyl16: new THREE.CylinderGeometry(1, 1, 1, 18),
  cone8: new THREE.ConeGeometry(1, 1, 8),
  cone12: new THREE.ConeGeometry(1, 1, 12),
  sph: new THREE.SphereGeometry(1, 10, 7),
  sph6: new THREE.SphereGeometry(1, 7, 5),
  hemi: new THREE.SphereGeometry(1, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2),
  ico0: new THREE.IcosahedronGeometry(1, 0),
  ico1: new THREE.IcosahedronGeometry(1, 1),
  disc: new THREE.CircleGeometry(1, 24).rotateX(-Math.PI / 2),
  plane: new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
};
// a prism: profile [[a, y], ...] in the plane across the ridge, extruded along x by `len`
function prism(profile, len) {
  const sh = new THREE.Shape(profile.map(([a, y]) => new THREE.Vector2(a, y)));
  const g = new THREE.ExtrudeGeometry(sh, { depth: len, bevelEnabled: false });
  g.rotateY(Math.PI / 2); g.translate(-len / 2, 0, 0);
  return g;
}
const gable = (w, d, h) => prism([[-d / 2, 0], [d / 2, 0], [0, h]], w);

class Batch {
  constructor() { this.geos = []; }
  add(geo, color, m) {
    let g = geo.index ? geo.toNonIndexed() : geo.clone();
    for (const k of Object.keys(g.attributes)) if (k !== "position" && k !== "normal") g.deleteAttribute(k);
    if (m) g.applyMatrix4(m);
    const c = new THREE.Color(color), n = g.attributes.position.count, a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { a[i * 3] = c.r; a[i * 3 + 1] = c.g; a[i * 3 + 2] = c.b; }
    g.setAttribute("color", new THREE.BufferAttribute(a, 3));
    this.geos.push(g);
    return this;
  }
  build(material, { cast = true, receive = true } = {}) {
    if (!this.geos.length) return null;
    const g = mergeGeometries(this.geos, false);
    this.geos.forEach((x) => x.dispose());
    const m = new THREE.Mesh(g, material); m.castShadow = cast; m.receiveShadow = receive; m.matrixAutoUpdate = false;
    return m;
  }
}
// local frame: origin (ox, oz), rotation ry; returns a function giving world matrices
function frame(ox, oy, oz, ry = 0) {
  const F = TRS(ox, oy, oz, ry);
  return (x, y, z, r = 0, sx = 1, sy = 1, sz = 1, rx = 0, rz = 0) => F.clone().multiply(TRS(x, y, z, r, sx, sy, sz, rx, rz));
}
function localToWorld(ox, oz, ry, x, z) {
  const c = Math.cos(ry), s = Math.sin(ry);
  return { x: ox + x * c + z * s, z: oz - x * s + z * c };
}

// ------------------------------------------------------------------ palettes
const WALLS = [0xf3e3c3, 0xe9b894, 0xcfe0c3, 0xc2d7e8, 0xf2d98a, 0xe7a99b, 0xf7f3ea, 0xd9c4e3, 0xb9dcd3];
const ROOFS = [0xb5523b, 0x6d5a4f, 0x4f6475, 0x8a3a2c, 0x3f6b53, 0x9b6a3c, 0x5b4a6b];
const DOORS = [0x7a4b2a, 0x2f5d7c, 0x8c2f2f, 0x3f6b3f, 0x5b3a1e, 0x264653];
const FLOWERS = [0xe84a5f, 0xf7b538, 0xf2f2f2, 0xa05cd6, 0xff8fb1, 0xf26b38, 0x6fa8ff];

const SEASON_KEYS = ["spring", "summer", "autumn", "winter"];
function seasonIndex(name, k) {
  const n = (name || "").toLowerCase();
  if (n.startsWith("spr")) return 0; if (n.startsWith("sum")) return 1; if (n.startsWith("aut") || n.startsWith("fall")) return 2; if (n.startsWith("win")) return 3;
  return k % 4;
}
const SEASON = {
  grass: [0x86c95b, 0x78b44c, 0xa7a552, 0xb7c2bd].map((h) => new THREE.Color(h)),
  far: [0x7fb85a, 0x6ea34a, 0x9a9650, 0xcfd6d0].map((h) => new THREE.Color(h)),
  snow: [0, 0, 0, 1],
  sunMax: [1.15, 1.45, 1.0, 0.72],
};

// ------------------------------------------------------------------ the world
export class World {
  constructor(scene, cfg, { quality = 1 } = {}) {
    this.scene = scene; this.cfg = cfg; this.quality = quality;
    this.W = cfg.width; this.H = cfg.height;
    this.glow = 0; this.dark = 0; this.daylight = 1; this.season = 0; this.ctx = {};
    this.root = new THREE.Group(); scene.add(this.root);
    this.anchors = { houses: [] };
    this.labels = [];
    this.dyn = {};
    this.houses = [];
    this.buildings = [];              // footprints people should walk around: {x0, x1, z0, z1}
    this.static = new Batch();
    this.flat = new Batch();          // ground overlays (no shadows cast)
    this.roadB = new Batch();
    this.lampHeads = new Batch();
    this.treeList = [];               // {x, z, s, kind: 'round'|'pine'|'fruit', tint}
    this.bushList = [];
    this.flowerList = [];
    this.pools = [];                  // light pools on the ground
    this.grid = new Uint16Array(this.W * this.H);    // 0 grass, 1 road, 2 path, 3 blocked, 10+h house, 100+p place
    this.placeNames = Object.keys(cfg.places || {});
    this.materials();
    this.classify();
    this.buildSky();
    this.buildTerrain();
    this.buildRoads();
    (cfg.houses || []).forEach((b, h) => this.buildHouse(h, b));
    const P = cfg.places || {};
    for (const [name, box] of Object.entries(P)) {
      const fn = { farm: this.buildFarm, orchard: this.buildOrchard, dairy: this.buildDairy, warehouse: this.buildWarehouse,
        canteen: this.buildCanteen, market: this.buildMarket, school: this.buildSchool, station: this.buildStation }[name];
      if (fn) fn.call(this, box); else this.buildGeneric(name, box);
    }
    this.buildParks();
    this.buildLamps();
    this.buildVegetation();
    this.finish();
  }

  // ---------------------------------------------------------------- coordinates
  wx(x) { return x - (this.W - 1) / 2; }
  wz(y) { return (this.H - 1) / 2 - y; }
  cell(x, y) { return { x: this.wx(x), z: this.wz(y) }; }
  rect(b) {
    const x0 = this.wx(b[0]) - 0.5, x1 = this.wx(b[2]) + 0.5, z0 = this.wz(b[3]) - 0.5, z1 = this.wz(b[1]) + 0.5;
    return { x0, x1, z0, z1, cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };
  }
  toCell(x, z) { return [Math.round(x + (this.W - 1) / 2), Math.round((this.H - 1) / 2 - z)]; }
  inGrid(x, y) { return x >= 0 && y >= 0 && x < this.W && y < this.H; }
  inBuilding(x, y) { const c = this.cell(x, y); return this.buildings.some((b) => c.x > b.x0 && c.x < b.x1 && c.z > b.z0 && c.z < b.z1); }
  g(x, y) { return this.inGrid(x, y) ? this.grid[y * this.W + x] : 0; }
  isRoad(x, y) { return this.g(x, y) === 1; }
  cellInfo(x, y) {
    const v = this.g(x, y);
    if (v >= 100) return { type: "place", name: this.placeNames[v - 100] };
    if (v >= 10) return { type: "house", h: v - 10 };
    return { type: ["grass", "road", "path", "blocked"][v] || "grass" };
  }

  classify() {
    const { W, H, cfg } = this;
    const fill = (b, v) => { for (let x = b[0]; x <= b[2]; x++) for (let y = b[1]; y <= b[3]; y++) if (this.inGrid(x, y)) this.grid[y * W + x] = v; };
    (cfg.roads || []).forEach((b) => fill(b, 1));
    (cfg.houses || []).forEach((b, h) => fill(b, 10 + h));
    this.placeNames.forEach((n, i) => fill(cfg.places[n], 100 + i));
    // entrance paths: from each lot to the nearest road on its best sides
    this.paths = [];
    const addPath = (cells) => cells.forEach(([x, y]) => { if (this.g(x, y) === 0) { this.grid[y * W + x] = 2; this.paths.push([x, y]); } });
    (cfg.houses || []).forEach((b, h) => { const s = this.sides(b)[0]; if (s && s.k <= 4) addPath(s.cells); this.anchors.houses[h] = { side: s ? s.side : "S" }; });
    this.placeNames.forEach((n) => { this.sides(cfg.places[n]).filter((s, i) => s.k <= 3 && (i === 0 || s.k <= 2)).forEach((s) => addPath(s.cells)); });
  }
  // the lot's sides ordered by distance to a road along the side's middle
  sides(b) {
    const cx = Math.round((b[0] + b[2]) / 2), cy = Math.round((b[1] + b[3]) / 2);
    const out = [];
    const probe = (side, dx, dy, sx, sy) => {
      const cells = [];
      for (let k = 1; k <= 6; k++) {
        const x = sx + dx * k, y = sy + dy * k;
        if (!this.inGrid(x, y)) return;
        const v = this.g(x, y);
        if (v === 1) { out.push({ side, k, cells }); return; }
        if (v >= 10 && v < 100 || v >= 100) return;
        cells.push([x, y]);
      }
    };
    probe("S", 0, -1, cx, b[1]); probe("E", 1, 0, b[2], cy); probe("W", -1, 0, b[0], cy); probe("N", 0, 1, cx, b[3]);
    const pref = { S: 0, E: 1, W: 2, N: 3 };
    out.sort((a, c) => a.k - c.k || pref[a.side] - pref[c.side]);
    return out;
  }

  materials() {
    this.M = {
      static: snowy(new THREE.MeshLambertMaterial({ vertexColors: true }), 1),
      flat: snowy(new THREE.MeshLambertMaterial({ vertexColors: true }), 0.7),
      road: snowy(new THREE.MeshLambertMaterial({ vertexColors: true }), 0.12),
      terrain: snowy(new THREE.MeshLambertMaterial({ vertexColors: true, color: 0x86c95b }), 0.86),
      lamp: new THREE.MeshLambertMaterial({ color: 0xfff1c9, emissive: 0xffd27a, emissiveIntensity: 0 }),
      trunk: snowy(new THREE.MeshLambertMaterial({ color: 0x7a5534 }), 0.6),
      crown: snowy(new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true }), 1),
      pine: snowy(new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true }), 1),
    };
  }

  // ---------------------------------------------------------------- sky, lights
  buildSky() {
    const scene = this.scene;
    this.skyU = {
      top: { value: new THREE.Color(0x5d9fd6) }, horizon: { value: new THREE.Color(0xcfe5f0) }, bottom: { value: new THREE.Color(0x9fb8a0) },
      sunDir: { value: new THREE.Vector3(0, 1, 0) }, sunCol: { value: new THREE.Color(0xfff2d0) }, glow: { value: 0.5 },
    };
    const sky = new THREE.Mesh(new THREE.SphereGeometry(320, 32, 16), new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false, uniforms: this.skyU,
      vertexShader: "varying vec3 vP; void main(){ vP = (modelMatrix*vec4(position,1.0)).xyz - cameraPosition; gl_Position = projectionMatrix*viewMatrix*modelMatrix*vec4(position,1.0);} ",
      fragmentShader: `uniform vec3 top; uniform vec3 horizon; uniform vec3 bottom; uniform vec3 sunDir; uniform vec3 sunCol; uniform float glow; varying vec3 vP;
        void main(){ vec3 d = normalize(vP); float h = d.y;
          vec3 c = h > 0.0 ? mix(horizon, top, pow(clamp(h,0.0,1.0), 0.55)) : mix(horizon, bottom, clamp(-h*5.0,0.0,1.0));
          float s = max(dot(d, normalize(sunDir)), 0.0);
          c += sunCol * (pow(s, 900.0) * 3.0 + pow(s, 12.0) * 0.35 * glow + pow(s, 3.0) * 0.12 * glow);
          gl_FragColor = vec4(c, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }));
    sky.renderOrder = -10; sky.frustumCulled = false; scene.add(sky); this.sky = sky;
    // stars
    const r = rng(7), n = 1400, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = r() * 2 - 1, a = r() * Math.PI * 2, y = Math.abs(u) * 0.95 + 0.05, s = Math.sqrt(1 - y * y);
      pos.set([Math.cos(a) * s * 300, y * 300, Math.sin(a) * s * 300], i * 3);
    }
    const sg = new THREE.BufferGeometry(); sg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xffffff, size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0, fog: false, depthWrite: false }));
    this.stars.renderOrder = -9; scene.add(this.stars);
    // moon
    this.moon = new THREE.Mesh(new THREE.SphereGeometry(7, 20, 14), new THREE.MeshBasicMaterial({ color: 0xf2f0e6, fog: false, transparent: true }));
    this.moon.renderOrder = -8; scene.add(this.moon);
    // lights
    this.hemi = new THREE.HemisphereLight(0xdcefff, 0x6d7f55, 1.0); scene.add(this.hemi);
    this.amb = new THREE.AmbientLight(0x8090b0, 0.08); scene.add(this.amb);
    const sun = (this.sun = new THREE.DirectionalLight(0xfff4e0, 2.4));
    sun.castShadow = true;
    const sm = this.quality > 0.7 ? 2048 : 1024;
    sun.shadow.mapSize.set(sm, sm);
    const span = Math.max(this.W, this.H) * 0.62 + 2;
    Object.assign(sun.shadow.camera, { left: -span, right: span, top: span, bottom: -span, near: 1, far: 140 });
    sun.shadow.bias = -0.0006; sun.shadow.normalBias = 0.025;
    scene.add(sun); scene.add(sun.target);
    scene.fog = new THREE.Fog(0xcfe5f0, 70, 230);
    // clouds
    this.clouds = [];
    const cm = new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true, transparent: true, opacity: 0.95, fog: true });
    for (let k = 0; k < 11; k++) {
      const b = new Batch(), rr = rng(100 + k);
      const n2 = 4 + Math.floor(rr() * 4);
      for (let i = 0; i < n2; i++) b.add(U.ico1, 0xffffff, TRS((i - n2 / 2) * 1.6 + rr(), rr() * 0.8, rr() * 1.6 - 0.8, 0, 1.4 + rr() * 1.4, 0.9 + rr() * 0.7, 1.2 + rr()));
      const m = b.build(cm, { cast: false, receive: false }); m.matrixAutoUpdate = true;
      const ca = rr() * Math.PI * 2, cd = 70 + rr() * 90;
      m.position.set(Math.cos(ca) * cd, 55 + rr() * 25, Math.sin(ca) * cd - 30); m.scale.setScalar(2.5 + rr() * 2.5);
      scene.add(m); this.clouds.push(m);
    }
    // falling snow
    const ns = this.quality > 0.7 ? 3500 : 1500, sp = new Float32Array(ns * 3), sw = this.W + 20, sd = this.H + 20;
    const rs = rng(3);
    for (let i = 0; i < ns; i++) sp.set([rs() * sw - sw / 2, rs() * 16, rs() * sd - sd / 2], i * 3);
    const sgeo = new THREE.BufferGeometry(); sgeo.setAttribute("position", new THREE.BufferAttribute(sp, 3));
    this.snowU = { uTime: { value: 0 }, uOpacity: { value: 0 }, uScale: { value: 300 } };
    this.snowfall = new THREE.Points(sgeo, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, uniforms: this.snowU,
      vertexShader: `uniform float uTime; uniform float uScale; varying float vA;
        void main(){ vec3 p = position; p.y = mod(p.y - uTime * 1.1, 16.0); p.x += sin(uTime * 0.6 + position.z * 0.7) * 0.5; p.z += cos(uTime * 0.5 + position.x * 0.6) * 0.4;
          vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = uScale * 0.09 / -mv.z; vA = smoothstep(0.0, 2.0, p.y) * smoothstep(16.0, 13.0, p.y); }`,
      fragmentShader: `uniform float uOpacity; varying float vA; void main(){ vec2 c = gl_PointCoord - 0.5; float d = length(c); if (d > 0.5) discard; gl_FragColor = vec4(1.0, 1.0, 1.0, uOpacity * vA * smoothstep(0.5, 0.15, d)); }`,
    }));
    this.snowfall.frustumCulled = false; this.snowfall.visible = false; scene.add(this.snowfall);
  }

  // ---------------------------------------------------------------- terrain
  terrainH(x, z) {
    const hw = this.W / 2 + 1.5, hh = this.H / 2 + 1.5;
    const dx = Math.max(0, Math.abs(x) - hw), dz = Math.max(0, Math.abs(z) - hh), d = Math.hypot(dx, dz);
    if (d <= 0) return 0;
    let h = smooth(0, 16, d) * (fbm(x * 0.045, z * 0.045, 3) * 7 + d * 0.05) + smooth(40, 110, d) * fbm(x * 0.02, z * 0.02, 9) * 26;
    const rd = this.roadDist ? this.roadDist(x, z) : 99;
    h *= smooth(1.0, 5.0, rd);
    return h;
  }
  buildTerrain() {
    // roads that leave the grid continue into the countryside
    const exits = [];
    (this.cfg.roads || []).forEach((b) => {
      const horiz = b[2] - b[0] >= b[3] - b[1];
      if (horiz) {
        const z = this.wz((b[1] + b[3]) / 2);
        if (b[0] === 0) exits.push({ x0: -400, x1: this.wx(0) - 0.5, z0: z - 0.5 * (b[3] - b[1] + 1), z1: z + 0.5 * (b[3] - b[1] + 1) });
        if (b[2] === this.W - 1) exits.push({ x0: this.wx(this.W - 1) + 0.5, x1: 400, z0: z - 0.5 * (b[3] - b[1] + 1), z1: z + 0.5 * (b[3] - b[1] + 1) });
      } else {
        const x = this.wx((b[0] + b[2]) / 2), hw = 0.5 * (b[2] - b[0] + 1);
        if (b[1] === 0) exits.push({ x0: x - hw, x1: x + hw, z0: this.wz(0) + 0.5, z1: 400 });
        if (b[3] === this.H - 1) exits.push({ x0: x - hw, x1: x + hw, z0: -400, z1: this.wz(this.H - 1) - 0.5 });
      }
    });
    this.exits = exits;
    this.roadDist = (x, z) => {
      let d = 99;
      for (const e of exits) { const dx = Math.max(e.x0 - x, 0, x - e.x1), dz = Math.max(e.z0 - z, 0, z - e.z1); d = Math.min(d, Math.hypot(dx, dz)); }
      return d;
    };
    const size = 300, seg = this.quality > 0.7 ? 150 : 90;
    const geo = new THREE.PlaneGeometry(size, size, seg, seg); geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position, col = new Float32Array(pos.count * 3);
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      const h = this.terrainH(x, z);
      pos.setY(i, h);
      const n = fbm(x * 0.12, z * 0.12, 1), n2 = fbm(x * 0.5, z * 0.5, 5);
      const v = 0.86 + n * 0.2 + n2 * 0.06 - smooth(4, 20, h) * 0.06;
      c.setRGB(v * (0.97 + n2 * 0.05), v, v * (0.93 + n * 0.08));
      col.set([c.r, c.g, c.b], i * 3);
    }
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.computeVertexNormals();
    const t = new THREE.Mesh(geo, this.M.terrain); t.receiveShadow = true; t.matrixAutoUpdate = false; this.root.add(t);
    this.terrain = t;
    // distant blue mountains
    const mb = new Batch(), mr = rng(21);
    for (let k = 0; k < 26; k++) {
      const a = (k / 26) * Math.PI * 2 + mr() * 0.2, R = 190 + mr() * 40;
      mb.add(U.cone8, mixC(0x7f93a8, 0x9aa8b6, mr()), TRS(Math.cos(a) * R, -2, Math.sin(a) * R, mr() * 3, 30 + mr() * 30, 30 + mr() * 35, 30 + mr() * 30));
    }
    const mm = mb.build(new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }), { cast: false, receive: false });
    this.root.add(mm);
  }

  // ---------------------------------------------------------------- roads
  buildRoads() {
    const { W, H } = this, R = this.roadB, S = this.static, F = this.flat;
    const ASPH = 0x5c6268, CURB = 0xd6d1c4, LINE = 0xf1e7b5, ZEBRA = 0xf2f2ee;
    for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
      if (!this.isRoad(x, y)) continue;
      const { x: X, z: Z } = this.cell(x, y);
      R.add(U.box, ASPH, TRS(X, 0.01, Z, 0, 1.001, 0.02, 1.001));
      const n = { N: this.isRoad(x, y + 1) || y + 1 >= H, S: this.isRoad(x, y - 1) || y - 1 < 0, E: this.isRoad(x + 1, y) || x + 1 >= W, W: this.isRoad(x - 1, y) || x - 1 < 0 };
      // curbs toward non-road neighbours (paths get a dropped curb)
      const curb = (side) => {
        const [dx, dy] = { N: [0, 1], S: [0, -1], E: [1, 0], W: [-1, 0] }[side];
        if (this.g(x + dx, y + dy) === 2) return;
        const off = 0.44;
        if (dx) R.add(U.box, CURB, TRS(X + dx * off, 0.035, Z, 0, 0.12, 0.05, 1.0));
        else R.add(U.box, CURB, TRS(X, 0.035, Z - dy * off, 0, 1.0, 0.05, 0.12));
      };
      ["N", "S", "E", "W"].forEach((s) => { if (!n[s]) curb(s); });
      const straightH = n.E && n.W && !n.N && !n.S, straightV = n.N && n.S && !n.E && !n.W;
      const nearX = (dx, dy) => this.isRoad(x + dx, y + dy) && ((this.isRoad(x + dx + 1, y + dy) && this.isRoad(x + dx - 1, y + dy) && (this.isRoad(x + dx, y + dy + 1) || this.isRoad(x + dx, y + dy - 1))));
      if (straightH) {
        if (nearX(1, 0) || nearX(-1, 0)) for (let k = 0; k < 5; k++) R.add(U.box, ZEBRA, TRS(X, 0.022, Z - 0.36 + k * 0.18, 0, 0.5, 0.01, 0.09));
        else if ((x % 2) === 0) R.add(U.box, LINE, TRS(X, 0.022, Z, 0, 0.45, 0.01, 0.05));
      } else if (straightV) {
        if (nearX(0, 1) || nearX(0, -1)) for (let k = 0; k < 5; k++) R.add(U.box, ZEBRA, TRS(X - 0.36 + k * 0.18, 0.022, Z, 0, 0.09, 0.01, 0.5));
        else if ((y % 2) === 0) R.add(U.box, LINE, TRS(X, 0.022, Z, 0, 0.05, 0.01, 0.45));
      }
    }
    // roads out of town
    for (const e of this.exits) {
      const horiz = e.x1 - e.x0 > e.z1 - e.z0;
      const len = 90, step = 1;
      for (let k = 0; k < len; k += step) {
        let x, z;
        if (horiz) { const s = e.x0 < -100 ? -1 : 1; x = (s < 0 ? e.x1 : e.x0) + s * (k + 0.5); z = (e.z0 + e.z1) / 2; }
        else { const s = e.z0 < -100 ? -1 : 1; z = (s < 0 ? e.z1 : e.z0) + s * (k + 0.5); x = (e.x0 + e.x1) / 2; }
        const h = this.terrainH(x, z);
        R.add(U.box, ASPH, TRS(x, h + 0.012, z, 0, horiz ? 1.02 : 1.0, 0.02, horiz ? 1.0 : 1.02));
        if (k % 2 === 0) R.add(U.box, LINE, TRS(x, h + 0.024, z, 0, horiz ? 0.45 : 0.05, 0.01, horiz ? 0.05 : 0.45));
      }
    }
    // entrance paths (paving stones)
    for (const [x, y] of this.paths) {
      const { x: X, z: Z } = this.cell(x, y);
      F.add(U.box, 0xcdbfa6, TRS(X, 0.012, Z, 0, 0.62, 0.02, 1.0));
      F.add(U.box, 0xcdbfa6, TRS(X, 0.012, Z, 0, 1.0, 0.02, 0.62));
    }
  }

  // ---------------------------------------------------------------- houses
  buildHouse(h, b) {
    const R = this.rect(b), A = this.anchors.houses[h], side = A.side;
    const ry = { S: 0, N: Math.PI, E: Math.PI / 2, W: -Math.PI / 2 }[side];
    const lotW = side === "S" || side === "N" ? R.w : R.d, lotD = side === "S" || side === "N" ? R.d : R.w;
    const u = (k) => hash(h, 991, k);
    const L = frame(R.cx, 0, R.cz, ry), S = this.static, F = this.flat;
    const wall = WALLS[(h * 4 + Math.floor(u(1) * 3)) % WALLS.length], roof = pick(ROOFS, u(2)), door = pick(DOORS, u(3)), trim = 0xfbf8f1;
    const two = u(4) < 0.4;
    const bw = Math.min(lotW - 0.45, 1.75), bd = Math.min(lotD - 0.9, 1.4), wh = two ? 1.3 : 0.92, rh = 0.55 + u(6) * 0.12;
    const front = lotD / 2 - 0.62, bz = front - bd / 2;
    // yard: slightly different grass, a garden bed
    F.add(U.box, 0x8fc566, L(0, 0.004, 0, 0, lotW - 0.08, 0.008, lotD - 0.08));
    S.add(U.box, 0x9a8f86, L(0, 0.04, bz, 0, bw + 0.06, 0.08, bd + 0.06));
    S.add(U.box, wall, L(0, wh / 2 + 0.04, bz, 0, bw, wh, bd));
    // roof: ridge across (gable to the sides) or front-facing gable
    const ridgeX = u(5) < 0.62;
    const rg = ridgeX ? gable(bw + 0.26, bd + 0.24, rh) : gable(bd + 0.26, bw + 0.24, rh);
    S.add(rg, roof, L(0, wh + 0.04, bz, ridgeX ? 0 : Math.PI / 2));
    // gable wall triangles
    const tri = ridgeX ? gable(bw - 0.01, bd - 0.01, rh - 0.05) : gable(bd - 0.01, bw - 0.01, rh - 0.05);
    S.add(tri, wall, L(0, wh + 0.04, bz, ridgeX ? 0 : Math.PI / 2));
    // chimney
    const chx = (u(7) < 0.5 ? -1 : 1) * bw * 0.26, chz = bz - bd * 0.12;
    S.add(U.box, 0x9c5a43, L(chx, wh + rh * 0.62, chz, 0, 0.15, rh * 0.9 + 0.2, 0.15));
    S.add(U.box, 0x6e3f30, L(chx, wh + rh * 0.62 + rh * 0.45 + 0.11, chz, 0, 0.19, 0.04, 0.19));
    // door, frame, canopy, step
    const dx = two ? 0 : (u(8) - 0.5) * (bw - 0.7);
    const fz = front + 0.0;
    S.add(U.box, trim, L(dx, 0.34, fz + 0.012, 0, 0.36, 0.62, 0.02));
    S.add(U.box, door, L(dx, 0.33, fz + 0.024, 0, 0.28, 0.56, 0.02));
    S.add(U.sph6, 0xe0c060, L(dx + 0.09, 0.33, fz + 0.04, 0, 0.018, 0.018, 0.018));
    S.add(U.box, roof, L(dx, 0.7, fz + 0.1, 0, 0.46, 0.035, 0.22, -0.25));
    S.add(U.box, 0xb9b2a6, L(dx, 0.03, fz + 0.14, 0, 0.44, 0.06, 0.2));
    // windows (own material so they can glow)
    const WB = new Batch();
    const winAt = (x, y, z, rot, w = 0.24, hgt = 0.26) => {
      WB.add(U.box, 0xffffff, L(x, y, z, rot, w, hgt, 0.025));
      S.add(U.box, trim, L(x, y, z, rot, w + 0.07, hgt + 0.07, 0.018));
      const c = Math.cos(rot), s = Math.sin(rot);
      S.add(U.box, trim, L(x + s * 0.016, y, z + c * 0.016, rot, 0.022, hgt, 0.012));   // mullion
    };
    const rowsY = two ? [0.5, 1.02] : [0.52];
    rowsY.forEach((y, ri) => {
      const xs = ri === 0 && !two ? [dx - 0.47, dx + 0.47].filter((x) => Math.abs(x) < bw / 2 - 0.16) : [-bw * 0.3, bw * 0.3];
      xs.forEach((x) => {
        winAt(x, y, fz + 0.008, 0);
        if (ri === 0) {   // flower box
          S.add(U.box, 0x8a5a35, L(x, y - 0.19, fz + 0.06, 0, 0.3, 0.06, 0.08));
          for (let k = 0; k < 4; k++) S.add(U.sph6, FLOWERS[(h + k * 3 + ri) % FLOWERS.length], L(x - 0.1 + k * 0.066, y - 0.14, fz + 0.06, 0, 0.035, 0.035, 0.035));
        }
      });
      winAt(bw / 2 + 0.008, y, bz, Math.PI / 2); winAt(-bw / 2 - 0.008, y, bz, -Math.PI / 2);
      [-bw * 0.28, bw * 0.28].forEach((x) => winAt(x, y, bz - bd / 2 - 0.008, Math.PI));
    });
    const winMat = new THREE.MeshLambertMaterial({ color: 0x33475a, emissive: 0xffbf66, emissiveIntensity: 0 });
    const wm = WB.build(winMat, { cast: false, receive: false }); this.root.add(wm);
    // fence with a gate
    const fenceC = u(9) < 0.5 ? 0xf6f3ea : 0xa77d52;
    const fx = lotW / 2 - 0.06, fzb = lotD / 2 - 0.06;
    const post = (x, z) => S.add(U.box, fenceC, L(x, 0.11, z, 0, 0.035, 0.22, 0.035));
    for (let x = -fx; x <= fx + 0.01; x += 0.2) { if (Math.abs(x - dx) > 0.24) post(x, fzb); post(x, -fzb); }
    for (let z = -fzb; z <= fzb + 0.01; z += 0.2) { post(-fx, z); post(fx, z); }
    const rail = (x0, x1, z) => { if (x1 - x0 > 0.05) S.add(U.box, fenceC, L((x0 + x1) / 2, 0.15, z, 0, x1 - x0, 0.03, 0.02)); };
    rail(-fx, dx - 0.24, fzb); rail(dx + 0.24, fx, fzb); rail(-fx, fx, -fzb);
    S.add(U.box, fenceC, L(-fx, 0.15, 0, 0, 0.02, 0.03, lotD - 0.12)); S.add(U.box, fenceC, L(fx, 0.15, 0, 0, 0.02, 0.03, lotD - 0.12));
    // garden path, mailbox, bushes, a back-yard tree
    for (let z = fz + 0.28; z < lotD / 2; z += 0.17) F.add(U.box, 0xd9ccb4, L(dx + (Math.floor(z * 10) % 2 ? 0.02 : -0.02), 0.014, z, 0.1, 0.24, 0.02, 0.12));
    S.add(U.box, 0x5a4a3a, L(dx + 0.32, 0.12, lotD / 2 - 0.12, 0, 0.03, 0.24, 0.03));
    S.add(U.box, u(10) < 0.5 ? 0x3c6fb0 : 0xc0392b, L(dx + 0.32, 0.26, lotD / 2 - 0.12, 0, 0.09, 0.07, 0.13));
    const yard = [[-lotW / 2 + 0.25, lotD / 2 - 0.3], [lotW / 2 - 0.25, lotD / 2 - 0.3]];
    yard.forEach(([x, z], k) => { const p = localToWorld(R.cx, R.cz, ry, x, z); if (Math.abs(x - dx) > 0.35) this.bushList.push({ x: p.x, z: p.z, s: 0.16 + u(11 + k) * 0.06 }); });
    for (let k = 0; k < 6; k++) { const p = localToWorld(R.cx, R.cz, ry, (k % 2 ? 1 : -1) * (0.3 + (k >> 1) * 0.18) + dx * 0, lotD / 2 - 0.2); if (Math.abs(p.x) < 999 && Math.abs((k % 2 ? 1 : -1) * (0.3 + (k >> 1) * 0.18)) < fx - 0.1 && Math.abs((k % 2 ? 1 : -1) * (0.3 + (k >> 1) * 0.18) - dx) > 0.26) this.flowerList.push({ x: p.x, z: p.z, c: FLOWERS[(h + k) % FLOWERS.length] }); }
    if (lotD >= 2) { const p = localToWorld(R.cx, R.cz, ry, (u(12) - 0.5) * (lotW - 0.6), -lotD / 2 + 0.28); this.treeList.push({ x: p.x, z: p.z, s: 0.55 + u(13) * 0.2, kind: u(14) < 0.3 ? "fruit" : "round" }); }
    // anchors
    const doorW = localToWorld(R.cx, R.cz, ry, dx, fz + 0.3), top = localToWorld(R.cx, R.cz, ry, 0, bz);
    const chim = localToWorld(R.cx, R.cz, ry, chx, chz);
    Object.assign(A, { box: b, rect: R, door: { x: doorW.x, z: doorW.z, yaw: ry }, top: new THREE.Vector3(top.x, wh + rh + 0.25, top.z), chimney: new THREE.Vector3(chim.x, wh + rh * 0.62 + rh * 0.45 + 0.15, chim.z) });
    this.houses.push({ h, winMat, zzz: null, lit: 0 });
  }

  // ---------------------------------------------------------------- signs
  sign(text, x, z, ry = 0, { w = 1.7, h = 0.42, y = 0.75, bg = "#6b4a2e", fg = "#fff8e8", posts = true, sub = null } = {}) {
    const tex = textCanvas(text, { w: 512, h: Math.round(512 * h / w), bg, fg, font: `800 ${Math.round(512 * h / w * 0.5)}px Archivo, Arial, sans-serif`, sub });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshLambertMaterial({ map: tex, transparent: true }));
    m.position.set(x, y, z); m.rotation.y = ry; m.castShadow = false; this.root.add(m);
    const back = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshLambertMaterial({ color: 0x5a3d25 }));
    back.position.set(x - Math.sin(ry) * 0.012, y, z - Math.cos(ry) * 0.012); back.rotation.y = ry + Math.PI; this.root.add(back);
    if (posts) {
      const L = frame(x, 0, z, ry);
      [-w / 2 + 0.08, w / 2 - 0.08].forEach((px) => this.static.add(U.box, 0x5a3d25, L(px, (y - h / 2) / 2 + 0.02, -0.03, 0, 0.06, y - h / 2 + 0.04, 0.06)));
    }
    return m;
  }
  label(text, x, y, z, kind = "place") { this.labels.push({ text, pos: new THREE.Vector3(x, y, z), kind }); }

  // ---------------------------------------------------------------- workplaces
  buildFarm(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    F.add(U.box, 0x8a6a45, TRS(R.cx, 0.008, R.cz, 0, R.w - 0.1, 0.012, R.d - 0.1));
    // barn in the north-west corner, door facing the field
    const bx = R.x0 + 1.5, bz = R.z0 + 1.35, BW = 2.2, BD = 2.0, BH = 1.05;
    const L = frame(bx, 0, bz, 0);
    S.add(U.box, 0xa8352c, L(0, BH / 2, 0, 0, BW, BH, BD));
    S.add(prism([[-BW / 2 - 0.12, 0], [-BW / 2 + 0.05, 0.55], [-0.45, 0.95], [0.45, 0.95], [BW / 2 - 0.05, 0.55], [BW / 2 + 0.12, 0]], BD + 0.24), 0x4b3a36, L(0, BH, 0, Math.PI / 2));
    S.add(prism([[-BW / 2 + 0.01, 0], [-BW / 2 + 0.14, 0.5], [-0.42, 0.9], [0.42, 0.9], [BW / 2 - 0.14, 0.5], [BW / 2 - 0.01, 0]], BD - 0.02), 0xa8352c, L(0, BH, 0, Math.PI / 2));
    // big door with a white X
    S.add(U.box, 0x7d241e, L(0, 0.42, BD / 2 + 0.01, 0, 0.9, 0.84, 0.02));
    S.add(U.box, 0xf4efe6, L(0, 0.42, BD / 2 + 0.025, 0, 0.94, 0.05, 0.015)); S.add(U.box, 0xf4efe6, L(0, 0.84, BD / 2 + 0.025, 0, 0.94, 0.05, 0.015));
    S.add(U.box, 0xf4efe6, L(0, 0.42, BD / 2 + 0.03, 0, 1.1, 0.05, 0.012, 0, 0.75)); S.add(U.box, 0xf4efe6, L(0, 0.42, BD / 2 + 0.03, 0, 1.1, 0.05, 0.012, 0, -0.75));
    S.add(U.box, 0xf4efe6, L(-0.45, 0.42, BD / 2 + 0.025, 0, 0.05, 0.88, 0.015)); S.add(U.box, 0xf4efe6, L(0.45, 0.42, BD / 2 + 0.025, 0, 0.05, 0.88, 0.015));
    S.add(U.box, 0xf4efe6, L(0, 1.35, BD / 2 + 0.13, 0, 0.3, 0.26, 0.02));   // hay loft window
    // silo
    S.add(U.cyl16, 0xb9c3c7, L(BW / 2 + 0.45, 1.15, -0.3, 0, 0.42, 2.3, 0.42));
    S.add(U.hemi, 0x8e9aa0, L(BW / 2 + 0.45, 2.3, -0.3, 0, 0.43, 0.35, 0.43));
    for (let k = 0; k < 5; k++) S.add(U.cyl16, 0x9aa6ab, L(BW / 2 + 0.45, 0.3 + k * 0.45, -0.3, 0, 0.43, 0.03, 0.43));
    // hay bales
    [[BW / 2 + 0.3, 0.75], [BW / 2 + 0.72, 0.8], [BW / 2 + 0.5, 0.75, 0.3]].forEach(([x, z, y = 0]) => S.add(U.cyl12, 0xe0b95b, L(x, 0.17 + y, z, 0, 0.17, 0.32, 0.17, 0, Math.PI / 2)));
    // field rows
    const fx0 = R.x0 + 0.35, fx1 = R.x1 - 0.35, fz0 = R.z0 + 0.35, fz1 = R.z1 - 0.35;
    const barnZone = { x0: R.x0, x1: bx + BW / 2 + 1.0, z0: R.z0, z1: bz + BD / 2 + 0.45 };
    this.buildings.push({ x0: bx - BW / 2, x1: bx + BW / 2 + 0.9, z0: bz - BD / 2, z1: bz + BD / 2 });
    const inBarn = (x, z) => x < barnZone.x1 && z < barnZone.z1;
    const plants = [];
    for (let z = fz0; z <= fz1; z += 0.34) {
      const rowX0 = z < barnZone.z1 ? barnZone.x1 + 0.1 : fx0;
      if (rowX0 < fx1) F.add(U.box, 0x6e5234, TRS((rowX0 + fx1) / 2, 0.018, z, 0, fx1 - rowX0, 0.012, 0.12));
      for (let x = rowX0; x <= fx1; x += 0.25) plants.push([x + (hash(Math.round(x * 10), Math.round(z * 10)) - 0.5) * 0.06, z]);
    }
    const cropGeo = mergeGeometries([0, 1, 2].map((k) => { const g = new THREE.ConeGeometry(0.035, 1, 4); g.translate(0, 0.5, 0); g.rotateZ((k - 1) * 0.28); g.rotateY(k * 2.1); g.deleteAttribute("uv"); return g; }));
    const crops = new THREE.InstancedMesh(cropGeo, snowy(new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true }), 0.5), plants.length);
    crops.userData.pos = plants; crops.castShadow = false; crops.receiveShadow = true;
    this.root.add(crops); this.dyn.crops = crops;
    // scarecrow
    const sx = (barnZone.x1 + fx1) / 2 + 0.6, sz = (fz0 + fz1) / 2 + 1.2;
    S.add(U.box, 0x6b4a2e, TRS(sx, 0.4, sz, 0, 0.04, 0.8, 0.04)); S.add(U.box, 0x6b4a2e, TRS(sx, 0.62, sz, 0, 0.5, 0.035, 0.035));
    S.add(U.box, 0x3d6aa8, TRS(sx, 0.55, sz, 0, 0.18, 0.2, 0.1)); S.add(U.sph, 0xe8d49a, TRS(sx, 0.75, sz, 0, 0.075, 0.075, 0.075));
    S.add(U.cyl12, 0xd9b65a, TRS(sx, 0.81, sz, 0, 0.14, 0.012, 0.14)); S.add(U.cyl12, 0xcfa94f, TRS(sx, 0.85, sz, 0, 0.06, 0.07, 0.06));
    // low wooden fence around the field edge, with gaps
    this.railFence(R, 0x9b7a55, 0.3, [0.35]);
    this.sign("FARM", bx + 1.7, R.z1 + 0.05, 0, { w: 1.3 });
    this.anchors.farm = { rect: R, field: { x0: fx0, x1: fx1, z0: fz0, z1: fz1 }, barn: barnZone, barnDoor: { x: bx, z: bz + BD / 2 + 0.3 } };
    this.label("Farm", R.cx + 1, 1.2, R.cz);
  }

  railFence(R, color, h, gapsFrac = []) {
    const S = this.static;
    const edges = [[R.x0, R.z1, R.x1, R.z1], [R.x0, R.z0, R.x1, R.z0], [R.x0, R.z0, R.x0, R.z1], [R.x1, R.z0, R.x1, R.z1]];
    edges.forEach(([x0, z0, x1, z1]) => {
      const len = Math.hypot(x1 - x0, z1 - z0), n = Math.max(2, Math.round(len / 0.6));
      for (let k = 0; k <= n; k++) {
        const u = k / n, x = lerp(x0, x1, u), z = lerp(z0, z1, u);
        const [cx, cy] = this.toCell(x, z);
        const nearGate = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]].some(([dx, dy]) => this.g(cx + dx, cy + dy) === 2);
        if (nearGate) continue;
        S.add(U.box, color, TRS(x, h / 2, z, 0, 0.05, h, 0.05));
        if (k < n) {
          const x2 = lerp(x0, x1, (k + 1) / n), z2 = lerp(z0, z1, (k + 1) / n);
          const [c2x, c2y] = this.toCell(x2, z2);
          if ([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]].some(([dx, dy]) => this.g(c2x + dx, c2y + dy) === 2)) continue;
          const mx = (x + x2) / 2, mz = (z + z2) / 2, l = Math.hypot(x2 - x, z2 - z), ry = Math.atan2(x2 - x, z2 - z);
          S.add(U.box, color, TRS(mx, h * 0.72, mz, ry, 0.025, 0.035, l));
          S.add(U.box, color, TRS(mx, h * 0.35, mz, ry, 0.025, 0.035, l));
        }
      }
    });
  }

  buildOrchard(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    F.add(U.box, 0x7dba55, TRS(R.cx, 0.006, R.cz, 0, R.w - 0.1, 0.01, R.d - 0.1));
    const nx = Math.max(1, Math.floor((R.w - 0.7) / 1.4) + 1), nz = Math.max(1, Math.floor((R.d - 0.6) / 1.35) + 1);
    const sx = (R.w - (nx - 1) * 1.4) / 2, sz = (R.d - (nz - 1) * 1.35) / 2;
    const trees = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < nz; j++) {
      const x = R.x0 + sx + i * 1.4 + (j % 2 ? 0.2 : -0.2) * (nx > 2 ? 1 : 0), z = R.z0 + sz + j * 1.35;
      trees.push({ x, z });
      F.add(U.disc, 0x6d5236, TRS(x, 0.014, z, 0, 0.36, 1, 0.36));
      this.treeList.push({ x, z, s: 0.78, kind: "orchard" });
    }
    // fruit instances: 9 per tree on the crown
    const fr = [];
    const cr = 0.46 * 0.78, cy = 0.5 + cr * 0.55;       // matches the orchard crowns in buildVegetation
    trees.forEach((t, ti) => {
      for (let k = 0; k < 9; k++) {
        const a = (k / 9) * Math.PI * 2 + hash(ti, k, 5) * 0.6, e = -0.35 + hash(ti, k, 6) * 1.0;
        fr.push([t.x + Math.cos(a) * Math.cos(e) * cr * 1.02, cy + Math.sin(e) * cr * 0.95, t.z + Math.sin(a) * Math.cos(e) * cr * 1.02]);
      }
    });
    const fruit = new THREE.InstancedMesh(new THREE.SphereGeometry(0.055, 8, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }), fr.length);
    fruit.userData.pos = fr; fruit.castShadow = false;
    fr.forEach((p, i) => fruit.setColorAt(i, new THREE.Color(hash(i, 3) < 0.7 ? 0xd8342a : 0xf08a24)));
    this.root.add(fruit); this.dyn.fruit = fruit;
    // fruit crates and a ladder
    const cx = R.x1 - 0.5, cz = R.z1 - 0.35;
    for (let k = 0; k < 3; k++) {
      S.add(U.box, 0xb58750, TRS(cx - k * 0.32, 0.09, cz, 0, 0.28, 0.18, 0.22));
      for (let a = 0; a < 5; a++) S.add(U.sph6, a % 2 ? 0xd8342a : 0xf08a24, TRS(cx - k * 0.32 - 0.08 + a * 0.04, 0.2, cz + (a % 2 ? 0.05 : -0.04), 0, 0.045, 0.045, 0.045));
    }
    if (trees[0]) {
      const t = trees[0];
      S.add(U.box, 0xa07a4a, TRS(t.x + 0.3, 0.45, t.z + 0.12, 0, 0.03, 0.95, 0.03, 0, -0.35)); S.add(U.box, 0xa07a4a, TRS(t.x + 0.3, 0.45, t.z + 0.28, 0, 0.03, 0.95, 0.03, 0, -0.35));
      for (let k = 0; k < 5; k++) S.add(U.box, 0xa07a4a, TRS(t.x + 0.43 - k * 0.07, 0.12 + k * 0.17, t.z + 0.2, 0, 0.03, 0.02, 0.18));
    }
    this.railFence(R, 0xa98962, 0.26);
    this.sign("ORCHARD", R.cx, R.z1 + 0.05, 0, { w: 1.6 });
    this.anchors.orchard = { rect: R, trees };
    this.label("Orchard", R.cx, 1.8, R.cz);
  }

  buildDairy(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    F.add(U.box, 0x9bd06d, TRS(R.cx, 0.006, R.cz, 0, R.w - 0.1, 0.01, R.d - 0.1));
    // barn on the east side, door facing west
    const BW = 1.7, BD = 2.4, BH = 0.95, bx = R.x1 - BW / 2 - 0.25, bz = R.z0 + BD / 2 + 0.3;
    const L = frame(bx, 0, bz, 0);
    this.buildings.push({ x0: bx - BW / 2, x1: bx + BW / 2, z0: bz - BD / 2, z1: bz + BD / 2 });
    S.add(U.box, 0xf1ece0, L(0, BH / 2, 0, 0, BW, BH, BD));
    S.add(gable(BD + 0.24, BW + 0.26, 0.6), 0xa8352c, L(0, BH, 0, Math.PI / 2));
    S.add(gable(BD - 0.02, BW - 0.02, 0.56), 0xf1ece0, L(0, BH, 0, Math.PI / 2));
    S.add(U.box, 0x8c2f2a, L(-BW / 2 - 0.01, 0.38, 0.2, 0, 0.02, 0.76, 0.75));
    S.add(U.box, 0xfaf6ee, L(-BW / 2 - 0.02, 0.38, 0.2, 0, 0.012, 0.05, 0.8));
    S.add(U.box, 0x333a40, L(-BW / 2 - 0.01, 0.62, -0.75, 0, 0.02, 0.2, 0.28));
    // trough and feeder
    S.add(U.box, 0x8d99a3, TRS(R.x0 + 1.0, 0.1, R.z0 + 0.55, 0, 0.9, 0.2, 0.26)); S.add(U.box, 0x5fa8c8, TRS(R.x0 + 1.0, 0.19, R.z0 + 0.55, 0, 0.8, 0.02, 0.18));
    S.add(U.box, 0x8a6a44, TRS(R.x0 + 2.6, 0.25, R.z0 + 0.5, 0, 0.6, 0.5, 0.3)); S.add(U.box, 0xe0b95b, TRS(R.x0 + 2.6, 0.45, R.z0 + 0.5, 0, 0.5, 0.18, 0.24));
    // milking stations in front of the barn door
    const stations = [];
    for (let k = 0; k < 3; k++) {
      const z = bz - 0.7 + k * 0.72, x = bx - BW / 2 - 0.75;
      stations.push({ cow: { x, z, yaw: -Math.PI / 2 }, x: x + 0.05, z: z + 0.32, yaw: Math.PI });
      S.add(U.cyl8, 0x6b4a2e, TRS(x + 0.05, 0.08, z + 0.34, 0, 0.06, 0.16, 0.06));
    }
    this.railFence(R, 0xb89a72, 0.36);
    this.buildCows(R, stations, { x0: R.x0 + 0.5, x1: bx - BW / 2 - 1.3, z0: R.z0 + 0.9, z1: R.z1 - 0.45 });
    this.sign("DAIRY", R.cx - 0.6, R.z1 + 0.05, 0, { w: 1.3 });
    this.anchors.dairy = { rect: R, stations, barnDoor: { x: bx - BW / 2 - 0.3, z: bz + 0.2 } };
    this.label("Dairy", R.cx - 0.5, 1.5, R.cz);
  }

  buildCows(R, stations, area) {
    // one cow = body instance + head instance; head pivots at the neck
    const body = new Batch(), head = new Batch();
    const W = 0xf5f2ea, B = 0x22201e, P = 0xe9a3a0;
    body.add(U.box, W, TRS(0, 0.34, 0, 0, 0.3, 0.26, 0.6));
    body.add(U.box, B, TRS(0.08, 0.4, 0.08, 0, 0.16, 0.2, 0.22)); body.add(U.box, B, TRS(-0.1, 0.33, -0.16, 0, 0.13, 0.22, 0.18));
    body.add(U.box, B, TRS(0.151, 0.34, -0.05, 0, 0.01, 0.14, 0.2)); body.add(U.box, B, TRS(-0.151, 0.38, 0.14, 0, 0.01, 0.12, 0.16));
    [[0.1, 0.22], [-0.1, 0.22], [0.1, -0.22], [-0.1, -0.22]].forEach(([x, z]) => { body.add(U.box, W, TRS(x, 0.12, z, 0, 0.07, 0.24, 0.07)); body.add(U.box, B, TRS(x, 0.02, z, 0, 0.075, 0.04, 0.075)); });
    body.add(U.box, P, TRS(0, 0.2, -0.12, 0, 0.12, 0.06, 0.1));
    body.add(U.box, W, TRS(0, 0.36, -0.32, 0, 0.03, 0.2, 0.03, 0.3));
    head.add(U.box, W, TRS(0, 0.02, 0.1, 0, 0.16, 0.16, 0.2)); head.add(U.box, P, TRS(0, -0.02, 0.21, 0, 0.14, 0.09, 0.04));
    head.add(U.box, B, TRS(0.1, 0.06, 0.06, 0, 0.08, 0.03, 0.05)); head.add(U.box, B, TRS(-0.1, 0.06, 0.06, 0, 0.08, 0.03, 0.05));
    head.add(U.box, 0xe8dcc0, TRS(0.05, 0.12, 0.07, 0, 0.02, 0.06, 0.02)); head.add(U.box, 0xe8dcc0, TRS(-0.05, 0.12, 0.07, 0, 0.02, 0.06, 0.02));
    head.add(U.box, 0x111111, TRS(0.075, 0.05, 0.17, 0, 0.02, 0.02, 0.02)); head.add(U.box, 0x111111, TRS(-0.075, 0.05, 0.17, 0, 0.02, 0.02, 0.02));
    const mat = snowy(new THREE.MeshLambertMaterial({ vertexColors: true }), 0);
    const n = 9 + stations.length;
    const bg = body.build(mat).geometry, hg = head.build(mat).geometry;
    const bodies = new THREE.InstancedMesh(bg, mat, n), heads = new THREE.InstancedMesh(hg, mat, n);
    bodies.castShadow = true; heads.castShadow = true;
    this.root.add(bodies); this.root.add(heads);
    const cows = [];
    const r = rng(55);
    for (let i = 0; i < n; i++) {
      const st = i >= 9 ? stations[i - 9] : null;
      const s = 1.4 + r() * 0.2;
      cows.push({ i, x: st ? st.cow.x : lerp(area.x0, area.x1, r()), z: st ? st.cow.z : lerp(area.z0, area.z1, r()), yaw: st ? st.cow.yaw : r() * 6.28,
        tx: 0, tz: 0, wait: r() * 5, graze: r() * 10, s, st, vis: 0, want: 0 });
    }
    this.dyn.cows = { bodies, heads, cows, area };
  }

  buildWarehouse(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    F.add(U.box, 0xbdb8ae, TRS(R.cx, 0.008, R.cz, 0, R.w - 0.1, 0.012, R.d - 0.1));
    for (let x = R.x0 + 0.5; x < R.x1; x += 1) F.add(U.box, 0xa9a49a, TRS(x, 0.016, R.cz, 0, 0.02, 0.006, R.d - 0.2));
    const SW = Math.min(R.w - 1.6, 4.8), SD = Math.min(R.d - 1.9, 2.8), SH = 1.35;
    const sx = R.x0 + 0.2 + SW / 2, sz = R.z0 + 0.2 + SD / 2;
    const L = frame(sx, 0, sz, 0);
    this.buildings.push({ x0: sx - SW / 2, x1: sx + SW / 2, z0: sz - SD / 2, z1: sz + SD / 2 });
    S.add(U.box, 0x8fa3b3, L(0, SH / 2, 0, 0, SW, SH, SD));
    for (let x = -SW / 2 + 0.1; x < SW / 2; x += 0.16) S.add(U.box, 0x7f93a3, L(x, SH / 2, SD / 2 + 0.01, 0, 0.05, SH, 0.02));
    // barrel roof: half cylinder, axis along the shed, bulging up
    const arch = new THREE.CylinderGeometry(1, 1, 1, 16, 1, false, -Math.PI / 2, Math.PI);
    S.add(arch, 0x6f8494, L(0, SH, 0, Math.PI / 2, (SD + 0.3) / 2, SW + 0.25, 0.55, -Math.PI / 2, 0));
    [SW / 2, -SW / 2].forEach((x) => S.add(arch, 0x8fa3b3, L(x, SH, 0, Math.PI / 2, SD / 2, 0.02, 0.52, -Math.PI / 2, 0)));
    for (let x = -SW / 2 + 0.4; x < SW / 2; x += 0.8) S.add(arch, 0x5f7383, L(x, SH + 0.005, 0, Math.PI / 2, (SD + 0.32) / 2, 0.04, 0.56, -Math.PI / 2, 0));
    // roll-up door and loading dock
    S.add(U.box, 0xd9dcdf, L(0.4, 0.5, SD / 2 + 0.025, 0, 1.3, 0.98, 0.02));
    for (let y = 0.1; y < 1.0; y += 0.1) S.add(U.box, 0xb3b8bd, L(0.4, y, SD / 2 + 0.04, 0, 1.3, 0.015, 0.01));
    S.add(U.box, 0x9c9a94, L(0.4, 0.12, SD / 2 + 0.35, 0, 1.8, 0.24, 0.6));
    S.add(U.box, 0xf2c230, L(0.4, 0.245, SD / 2 + 0.64, 0, 1.8, 0.012, 0.03));
    S.add(U.box, 0x6d5a3b, L(-SW / 2 + 0.55, 0.36, SD / 2 + 0.02, 0, 0.36, 0.7, 0.02));   // side door
    this.lampHeads.add(U.box, 0xffffff, L(0.4, 1.12, SD / 2 + 0.08, 0, 0.18, 0.06, 0.1));
    this.pools.push({ x: sx + 0.4, z: sz + SD / 2 + 0.6, r: 1.2 });
    // sign on the shed
    const tex = textCanvas("WAREHOUSE", { w: 512, h: 96, bg: "#23394d", fg: "#f5f0e0", font: "800 58px Archivo, Arial, sans-serif", border: "#15222e" });
    const sg = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 0.34), new THREE.MeshLambertMaterial({ map: tex }));
    sg.position.set(sx + 0.4, SH - 0.2, sz + SD / 2 + 0.03); this.root.add(sg);
    // pallets and crate stacks (instanced, count follows stock)
    const yard = { x0: R.x0 + 0.35, x1: sx + SW / 2 - 0.2, z0: sz + SD / 2 + 0.75, z1: R.z1 - 0.3 };
    const slots = [];
    for (let z = yard.z0; z <= yard.z1; z += 0.34) for (let x = yard.x0; x <= yard.x0 + 1.4; x += 0.34) for (let lv = 0; lv < 3; lv++) slots.push([x, 0.11 + lv * 0.21, z, hash(Math.round(x * 9), Math.round(z * 9), lv) * 0.3]);
    for (let z = yard.z0; z <= yard.z1; z += 0.68) for (let x = yard.x0; x <= yard.x0 + 1.4; x += 0.68) F.add(U.box, 0x9a7a52, TRS(x + 0.17, 0.02, z + 0.17, 0, 0.66, 0.04, 0.66));
    slots.sort((a, c) => a[1] - c[1] || a[2] - c[2] || a[0] - c[0]);
    const crates = new THREE.InstancedMesh(new THREE.BoxGeometry(0.3, 0.2, 0.3), snowy(new THREE.MeshLambertMaterial({ color: 0xffffff }), 0.8), slots.length);
    slots.forEach((s, i) => crates.setColorAt(i, new THREE.Color(pick([0xb58750, 0xa8784a, 0xc49a62, 0x9c6e3f], hash(i, 2)))));
    crates.userData.slots = slots; crates.castShadow = true; crates.receiveShadow = true;
    this.root.add(crates); this.dyn.crates = crates;
    // truck at the east side of the yard, facing the road
    const tx = R.x1 - 0.85, tz = R.z1 - 1.25;
    const T = frame(tx, 0, tz, Math.PI / 2);
    S.add(U.box, 0xeeeae0, T(0, 0.48, -0.35, 0, 0.62, 0.6, 1.1));
    S.add(U.box, 0x2f6fb0, T(0, 0.48, -0.35, 0, 0.63, 0.1, 1.11));
    S.add(U.box, 0xc8453a, T(0, 0.38, 0.45, 0, 0.6, 0.42, 0.45));
    S.add(U.box, 0xc8453a, T(0, 0.66, 0.4, 0, 0.58, 0.2, 0.34));
    S.add(U.box, 0x26323d, T(0, 0.66, 0.575, 0, 0.5, 0.16, 0.012));
    S.add(U.box, 0x333333, T(0, 0.2, 0.69, 0, 0.58, 0.08, 0.04));
    [[-0.3, 0.45], [0.3, 0.45], [-0.3, -0.1], [0.3, -0.1], [-0.3, -0.62], [0.3, -0.62]].forEach(([x, z]) => S.add(U.cyl12, 0x222222, T(x, 0.12, z, 0, 0.12, 0.08, 0.12, 0, Math.PI / 2)));
    this.lampHeads.add(U.box, 0xffffff, T(0.2, 0.3, 0.69, 0, 0.08, 0.05, 0.02)); this.lampHeads.add(U.box, 0xffffff, T(-0.2, 0.3, 0.69, 0, 0.08, 0.05, 0.02));
    const route = { a: { x: sx + 0.4, z: sz + SD / 2 + 0.8 }, b: { x: tx - 0.95, z: tz } };
    this.sign("WAREHOUSE", R.x0 + 1.4, R.z1 + 0.05, 0, { w: 1.9, h: 0.38 });
    this.anchors.warehouse = { rect: R, route, yard, shed: { x0: sx - SW / 2, x1: sx + SW / 2, z0: sz - SD / 2, z1: sz + SD / 2 } };
    this.label("Warehouse", sx, 2.1, sz);
  }

  buildCanteen(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    const BD = Math.min(1.75, R.d * 0.45), BH = 1.15, bx = R.cx, bz = R.z0 + 0.15 + BD / 2, BW = R.w - 0.3;
    F.add(U.box, 0xd9b48c, TRS(R.cx, 0.01, (bz + BD / 2 + R.z1) / 2, 0, R.w - 0.1, 0.014, R.z1 - (bz + BD / 2) - 0.05));
    for (let x = R.x0 + 0.25; x < R.x1; x += 0.5) F.add(U.box, 0xc9a37c, TRS(x, 0.019, (bz + BD / 2 + R.z1) / 2, 0, 0.015, 0.004, R.z1 - (bz + BD / 2) - 0.1));
    const L = frame(bx, 0, bz, 0);
    this.buildings.push({ x0: bx - BW / 2, x1: bx + BW / 2, z0: bz - BD / 2, z1: bz + BD / 2 });
    S.add(U.box, 0xf4e7cf, L(0, BH / 2, 0, 0, BW, BH, BD));
    S.add(U.box, 0xb0473a, L(0, 0.12, 0, 0, BW + 0.02, 0.24, BD + 0.02));
    S.add(U.box, 0xe9dcc2, L(0, BH + 0.05, 0, 0, BW + 0.12, 0.1, BD + 0.12));
    S.add(U.box, 0xd9cbb0, L(0, BH + 0.12, 0, 0, BW - 0.2, 0.05, BD - 0.2));
    // awning: red/white stripes, sloping over the counter
    const aw = BW - 0.1, stripes = Math.round(aw / 0.22);
    for (let k = 0; k < stripes; k++) S.add(U.box, k % 2 ? 0xfaf6ee : 0xd23b35, L(-aw / 2 + (k + 0.5) * aw / stripes, 0.98, BD / 2 + 0.3, 0, aw / stripes + 0.002, 0.025, 0.66, 0.38));
    for (let k = 0; k < stripes; k++) S.add(U.box, k % 2 ? 0xfaf6ee : 0xd23b35, L(-aw / 2 + (k + 0.5) * aw / stripes, 0.79, BD / 2 + 0.6, 0, aw / stripes + 0.002, 0.09, 0.02));
    // counter
    S.add(U.box, 0x8a5a35, L(0, 0.24, BD / 2 + 0.16, 0, BW - 0.6, 0.48, 0.24));
    S.add(U.box, 0xe9dcc2, L(0, 0.49, BD / 2 + 0.16, 0, BW - 0.55, 0.03, 0.3));
    for (let k = 0; k < 5; k++) S.add(U.cyl12, [0xd9a441, 0xc8453a, 0xf0e6d2, 0x8fbf5a, 0xe6b07a][k], L(-BW / 2 + 0.6 + k * (BW - 1.2) / 4, 0.54, BD / 2 + 0.16, 0, 0.07, 0.06, 0.07));
    // windows (glow when open) and door
    const WB = new Batch();
    [-BW / 2 + 0.45, BW / 2 - 0.45].forEach((x) => { WB.add(U.box, 0xffffff, L(x, 0.6, BD / 2 + 0.01, 0, 0.5, 0.42, 0.02)); S.add(U.box, 0xfbf8f1, L(x, 0.6, BD / 2 + 0.005, 0, 0.58, 0.5, 0.012)); });
    WB.add(U.box, 0xffffff, L(-BW / 2 - 0.01, 0.6, 0, 0, 0.02, 0.42, BD * 0.5)); WB.add(U.box, 0xffffff, L(BW / 2 + 0.01, 0.6, 0, 0, 0.02, 0.42, BD * 0.5));
    const winMat = new THREE.MeshLambertMaterial({ color: 0x33475a, emissive: 0xffc46b, emissiveIntensity: 0 });
    this.root.add(WB.build(winMat, { cast: false }));
    // sign board on the roof, OPEN/CLOSED plaque
    const tex = textCanvas("CANTEEN", { w: 512, h: 110, bg: "#b23a30", fg: "#fff6e6", font: "800 70px Archivo, Arial, sans-serif", border: "#7a2420" });
    const sg = new THREE.Mesh(new THREE.PlaneGeometry(2.0, 0.43), new THREE.MeshLambertMaterial({ map: tex, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 0 }));
    sg.position.set(bx, BH + 0.38, bz + BD / 2 + 0.02); this.root.add(sg);
    S.add(U.box, 0x5a3d25, L(-0.7, BH + 0.2, BD / 2 - 0.05, 0, 0.05, 0.3, 0.05)); S.add(U.box, 0x5a3d25, L(0.7, BH + 0.2, BD / 2 - 0.05, 0, 0.05, 0.3, 0.05));
    const openTex = textCanvas("OPEN", { w: 256, h: 96, bg: "#2e7d4a", fg: "#ffffff", font: "800 60px Archivo, Arial, sans-serif", border: "#1d5332" });
    const closedTex = textCanvas("CLOSED", { w: 256, h: 96, bg: "#7a2d2a", fg: "#ffe9e6", font: "800 52px Archivo, Arial, sans-serif", border: "#4d1a18" });
    const plaque = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.23), new THREE.MeshLambertMaterial({ map: closedTex, emissive: 0xffffff, emissiveMap: closedTex, emissiveIntensity: 0.15 }));
    plaque.position.set(R.x1 - 0.35, 0.55, R.z1 + 0.02); this.root.add(plaque);
    S.add(U.box, 0x5a3d25, TRS(R.x1 - 0.35, 0.22, R.z1 - 0.01, 0, 0.05, 0.44, 0.04));
    // terrace tables with parasols; seats face the tables
    const seats = [], tz0 = bz + BD / 2 + 0.95, tz1 = R.z1 - 0.45;
    const cols = Math.max(1, Math.floor((R.w - 0.4) / 1.35)), rows = Math.max(1, Math.floor((tz1 - tz0) / 1.05) + 1);
    const tableC = 0xf3eee4, chairC = 0x3f6f8f;
    for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
      const x = R.x0 + (R.w - (cols - 1) * 1.35) / 2 + i * 1.35 + (j % 2 ? 0.3 : -0.1), z = rows > 1 ? lerp(tz0, tz1, j / (rows - 1)) : (tz0 + tz1) / 2;
      S.add(U.cyl12, tableC, TRS(x, 0.37, z, 0, 0.28, 0.035, 0.28)); S.add(U.cyl6, 0x555555, TRS(x, 0.18, z, 0, 0.025, 0.36, 0.025)); S.add(U.cyl12, 0x555555, TRS(x, 0.01, z, 0, 0.1, 0.02, 0.1));
      S.add(U.cyl6, 0x777777, TRS(x, 0.66, z, 0, 0.015, 0.8, 0.015));
      const n = 10;
      for (let k = 0; k < n; k++) {
        const g = new THREE.ConeGeometry(0.46, 0.2, n, 1, true, (k / n) * Math.PI * 2, Math.PI * 2 / n);
        S.add(g, k % 2 ? 0xfaf6ee : (i + j) % 2 ? 0xd23b35 : 0x2e8b7a, TRS(x, 1.12, z));
      }
      for (let k = 0; k < 4; k++) {
        const a = k * Math.PI / 2 + Math.PI / 4, cx = x + Math.sin(a) * 0.43, cz = z + Math.cos(a) * 0.43, yaw = a + Math.PI;
        const C = frame(cx, 0, cz, yaw);
        S.add(U.box, chairC, C(0, 0.21, 0, 0, 0.19, 0.03, 0.19)); S.add(U.box, chairC, C(0, 0.35, -0.09, 0, 0.19, 0.26, 0.025));
        [[-0.08, -0.08], [0.08, -0.08], [-0.08, 0.08], [0.08, 0.08]].forEach(([lx, lz]) => S.add(U.box, 0x2d4f66, C(lx, 0.105, lz, 0, 0.02, 0.21, 0.02)));
        seats.push({ x: cx, z: cz, yaw, table: { x, z }, seat: 0.36 });
      }
    }
    // string lights along the terrace edge
    const bulbs = new Batch();
    [R.x0 + 0.1, R.x1 - 0.1].forEach((x) => S.add(U.box, 0x5a3d25, TRS(x, 0.66, R.z1 - 0.1, 0, 0.05, 1.32, 0.05)));
    for (let k = 0; k <= 16; k++) {
      const u = k / 16, x = lerp(R.x0 + 0.1, R.x1 - 0.1, u), y = 1.28 - Math.sin(u * Math.PI) * 0.16;
      bulbs.add(U.sph6, 0xffffff, TRS(x, y, R.z1 - 0.1, 0, 0.018, 0.024, 0.018));
    }
    const bulbMat = new THREE.MeshLambertMaterial({ color: 0xfff3d6, emissive: 0xffd27a, emissiveIntensity: 0 });
    this.root.add(bulbs.build(bulbMat, { cast: false }));
    // planters
    [[R.x0 + 0.2, R.z1 - 0.25], [R.x1 - 0.2, bz + BD / 2 + 0.55]].forEach(([x, z]) => { S.add(U.box, 0x9c5a43, TRS(x, 0.1, z, 0, 0.26, 0.2, 0.26)); this.bushList.push({ x, z, s: 0.15, y: 0.2 }); });
    // benches along the side wall for waiting
    const benches = [];
    const bxw = R.x1 - 0.2;
    S.add(U.box, 0x8a5a35, TRS(bxw, 0.16, bz + BD / 2 + 0.55, 0, 0.18, 0.04, 0.9)); S.add(U.box, 0x6b4a2e, TRS(bxw, 0.08, bz + BD / 2 + 0.55, 0, 0.14, 0.16, 0.85));
    for (let k = 0; k < 3; k++) benches.push({ x: bxw - 0.02, z: bz + BD / 2 + 0.25 + k * 0.3, yaw: -Math.PI / 2, seat: 0.2 });
    const staff = [], queue = [];
    const ns = Math.max(2, Math.floor((BW - 0.8) / 0.55));
    for (let k = 0; k < ns; k++) {
      const x = -BW / 2 + 0.6 + k * (BW - 1.2) / Math.max(1, ns - 1);
      staff.push({ x: bx + x, z: bz + BD / 2 - 0.12, yaw: 0 });
      queue.push({ x: bx + x, z: bz + BD / 2 + 0.55, yaw: Math.PI });
    }
    this.pools.push({ x: R.cx, z: (tz0 + tz1) / 2, r: 2.4, key: "canteen" });
    this.dyn.canteen = { winMat, sign: sg, plaque, openTex, closedTex, bulbMat, open: -1 };
    this.anchors.canteen = { rect: R, seats, staff, queue, benches, building: { x0: bx - BW / 2, x1: bx + BW / 2, z0: bz - BD / 2, z1: bz + BD / 2 } };
    this.label("Canteen", bx, 2.2, bz);
  }

  buildMarket(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    F.add(U.box, 0xcbbfa8, TRS(R.cx, 0.01, R.cz, 0, R.w - 0.06, 0.014, R.d - 0.06));
    for (let x = R.x0 + 0.25; x < R.x1; x += 0.25) F.add(U.box, 0xbcae94, TRS(x, 0.018, R.cz, 0, 0.012, 0.004, R.d - 0.1));
    for (let z = R.z0 + 0.25; z < R.z1; z += 0.25) F.add(U.box, 0xbcae94, TRS(R.cx, 0.018, z, 0, R.w - 0.1, 0.004, 0.012));
    // market hall: open-fronted, pitched roof on posts, counters with produce
    const HD = Math.min(1.8, R.d * 0.46), HW = R.w - 0.2, hx = R.cx, hz = R.z0 + 0.1 + HD / 2, HH = 1.05;
    const L = frame(hx, 0, hz, 0);
    this.buildings.push({ x0: hx - HW / 2, x1: hx + HW / 2, z0: hz - HD / 2, z1: hz + HD / 2 - 0.3 });
    S.add(U.box, 0xe8d6b0, L(0, HH / 2, -HD / 2 + 0.05, 0, HW, HH, 0.1));
    S.add(U.box, 0xe8d6b0, L(-HW / 2 + 0.05, HH / 2, 0, 0, 0.1, HH, HD)); S.add(U.box, 0xe8d6b0, L(HW / 2 - 0.05, HH / 2, 0, 0, 0.1, HH, HD));
    [-HW / 2 + 0.06, -HW / 6, HW / 6, HW / 2 - 0.06].forEach((x) => S.add(U.box, 0x7a4f2e, L(x, HH / 2, HD / 2 - 0.05, 0, 0.09, HH, 0.09)));
    S.add(gable(HW + 0.3, HD + 0.35, 0.55), 0xa4452f, L(0, HH, 0, 0));
    S.add(U.box, 0x7a4f2e, L(0, HH - 0.04, HD / 2 - 0.05, 0, HW, 0.08, 0.08));
    // counters
    const cz = HD / 2 - 0.42;
    S.add(U.box, 0x9a6a3e, L(0, 0.22, cz, 0, HW - 0.4, 0.44, 0.3));
    S.add(U.box, 0x7a4f2e, L(0, 0.45, cz, 0, HW - 0.35, 0.03, 0.34));
    const goods = (x, kind) => {
      if (kind === 0) for (let k = 0; k < 3; k++) S.add(U.sph, 0xe2cf9c, L(x - 0.15 + k * 0.15, 0.53, cz + (k % 2) * 0.05, 0, 0.07, 0.09, 0.07));
      if (kind === 1) { S.add(U.box, 0xb58750, L(x, 0.52, cz, 0, 0.36, 0.1, 0.22)); for (let k = 0; k < 7; k++) S.add(U.sph6, [0xd8342a, 0xf08a24, 0x8fbf3a][k % 3], L(x - 0.13 + (k % 4) * 0.085, 0.6, cz - 0.05 + Math.floor(k / 4) * 0.1, 0, 0.045, 0.045, 0.045)); }
      if (kind === 2) { for (let k = 0; k < 3; k++) S.add(U.cyl8, 0xf4f4ef, L(x - 0.12 + k * 0.12, 0.55, cz, 0, 0.04, 0.16, 0.04)); S.add(U.cyl12, 0xf2c94c, L(x + 0.2, 0.5, cz + 0.05, 0, 0.08, 0.06, 0.08)); }
    };
    const nG = Math.max(3, Math.floor((HW - 0.4) / 0.55));
    for (let k = 0; k < nG; k++) goods(-HW / 2 + 0.45 + k * (HW - 0.9) / Math.max(1, nG - 1), k % 3);
    // hanging lamps
    for (let k = 0; k < 3; k++) this.lampHeads.add(U.sph6, 0xffffff, L(-HW / 3 + k * HW / 3, HH - 0.2, 0, 0, 0.06, 0.06, 0.06));
    const tex = textCanvas("MARKET", { w: 512, h: 110, bg: "#2f5d4a", fg: "#fff6e0", font: "800 72px Archivo, Arial, sans-serif", border: "#1c3a2d", sub: null });
    const sg = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.34), new THREE.MeshLambertMaterial({ map: tex }));
    sg.position.set(hx, HH + 0.2, hz + HD / 2 + 0.14); this.root.add(sg);
    // square: picnic tables, a tree
    const seats = [], ty = R.z1 - 0.65;
    const nt = Math.max(1, Math.floor((R.w - 0.3) / 1.6));
    for (let k = 0; k < nt; k++) {
      const x = R.x0 + (R.w - (nt - 1) * 1.6) / 2 + k * 1.6 - 0.2;
      S.add(U.box, 0x9a6a3e, TRS(x, 0.3, ty, 0, 0.9, 0.04, 0.34)); S.add(U.box, 0x7a4f2e, TRS(x, 0.15, ty, 0, 0.8, 0.3, 0.06));
      [-1, 1].forEach((s) => {
        S.add(U.box, 0x9a6a3e, TRS(x, 0.18, ty + s * 0.34, 0, 0.9, 0.035, 0.14));
        for (let q = 0; q < 2; q++) seats.push({ x: x - 0.22 + q * 0.44, z: ty + s * 0.36, yaw: s > 0 ? Math.PI : 0, table: { x, z: ty }, seat: 0.21 });
      });
    }
    this.treeList.push({ x: R.x1 - 0.35, z: R.z1 - 0.3, s: 0.55, kind: "round" });
    const buy = [];
    const nb = Math.max(3, Math.floor((HW - 0.4) / 0.45));
    for (let k = 0; k < nb; k++) buy.push({ x: hx - HW / 2 + 0.4 + k * (HW - 0.8) / Math.max(1, nb - 1), z: hz + cz + 0.42, yaw: Math.PI });
    this.pools.push({ x: hx, z: hz + HD / 2, r: 1.8, key: "market" });
    this.dyn.market = { sign: sg };
    this.anchors.market = { rect: R, buy, seats, square: { x: R.cx, z: (hz + HD / 2 + R.z1) / 2 }, hall: { x0: hx - HW / 2, x1: hx + HW / 2, z0: hz - HD / 2, z1: hz + HD / 2 } };
    this.label("Market", hx, 2.0, hz);
  }

  buildSchool(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    const BD = Math.min(1.25, R.d * 0.42), BW = R.w - 0.3, BH = 1.05, bx = R.cx, bz = R.z0 + 0.1 + BD / 2;
    F.add(U.box, 0xd8cdb2, TRS(R.cx, 0.01, (bz + BD / 2 + R.z1) / 2, 0, R.w - 0.08, 0.014, R.z1 - bz - BD / 2 - 0.02));
    const L = frame(bx, 0, bz, 0);
    this.buildings.push({ x0: bx - BW / 2, x1: bx + BW / 2, z0: bz - BD / 2, z1: bz + BD / 2 });
    S.add(U.box, 0xc9553f, L(0, BH / 2, 0, 0, BW, BH, BD));
    for (let y = 0.1; y < BH; y += 0.12) S.add(U.box, 0xb84a36, L(0, y, BD / 2 + 0.004, 0, BW, 0.012, 0.004));
    S.add(gable(BW + 0.24, BD + 0.26, 0.5), 0x3e4b5a, L(0, BH, 0, 0));
    S.add(gable(BW - 0.02, BD - 0.02, 0.46), 0xc9553f, L(0, BH, 0, 0));
    // bell tower with a clock
    S.add(U.box, 0xf1ead8, L(0, BH + 0.55, 0.1, 0, 0.36, 0.5, 0.36));
    S.add(gable(0.44, 0.44, 0.3), 0x3e4b5a, L(0, BH + 0.8, 0.1, 0));
    S.add(U.cyl12, 0xd9a441, L(0, BH + 0.62, 0.1, 0, 0.07, 0.12, 0.07));
    S.add(U.cyl16, 0xfafaf5, L(0, BH + 0.35, 0.29, 0, 0.13, 0.02, 0.13, Math.PI / 2));
    S.add(U.box, 0x222222, L(0, BH + 0.39, 0.305, 0, 0.012, 0.09, 0.01)); S.add(U.box, 0x222222, L(0.03, BH + 0.35, 0.305, 0, 0.07, 0.012, 0.01));
    S.add(U.box, 0x2f5d7c, L(0, 0.3, BD / 2 + 0.01, 0, 0.3, 0.6, 0.02));
    const WB = new Batch();
    for (let k = 0; k < 6; k++) { const x = -BW / 2 + 0.35 + k * (BW - 0.7) / 5; if (Math.abs(x) < 0.3) continue; WB.add(U.box, 0xffffff, L(x, 0.6, BD / 2 + 0.012, 0, 0.26, 0.34, 0.02)); S.add(U.box, 0xfbf8f1, L(x, 0.6, BD / 2 + 0.006, 0, 0.32, 0.4, 0.012)); }
    const winMat = new THREE.MeshLambertMaterial({ color: 0x33475a, emissive: 0xffe0a0, emissiveIntensity: 0 });
    this.root.add(WB.build(winMat, { cast: false }));
    const tex = textCanvas("SCHOOL", { w: 512, h: 100, bg: "#f1ead8", fg: "#8c2f2a", font: "800 66px Archivo, Arial, sans-serif", border: "#c9b99a" });
    const sg = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 0.26), new THREE.MeshLambertMaterial({ map: tex }));
    sg.position.set(bx - BW / 4 - 0.2, BH - 0.2, bz + BD / 2 + 0.02); this.root.add(sg);
    // outdoor class: blackboard on the west, desks facing it
    const yz0 = bz + BD / 2 + 0.25, yz1 = R.z1 - 0.2;
    const boardX = R.x0 + 0.3;
    S.add(U.box, 0x2c4a3a, TRS(boardX, 0.55, (yz0 + yz1) / 2, 0, 0.04, 0.42, 0.8)); S.add(U.box, 0x8a6a44, TRS(boardX, 0.55, (yz0 + yz1) / 2, 0, 0.05, 0.48, 0.86));
    S.add(U.box, 0x8a6a44, TRS(boardX - 0.02, 0.25, (yz0 + yz1) / 2 - 0.35, 0, 0.04, 0.5, 0.04)); S.add(U.box, 0x8a6a44, TRS(boardX - 0.02, 0.25, (yz0 + yz1) / 2 + 0.35, 0, 0.04, 0.5, 0.04));
    for (let k = 0; k < 4; k++) S.add(U.box, 0xf4f4ef, TRS(boardX + 0.025, 0.5 + k * 0.06, (yz0 + yz1) / 2 - 0.2 + k * 0.08, 0, 0.005, 0.012, 0.22));
    const desks = [];
    const playX0 = R.x1 - Math.min(2.0, R.w * 0.4);
    const dcols = Math.max(1, Math.floor((playX0 - boardX - 0.6) / 0.5)), drows = Math.max(1, Math.floor((yz1 - yz0) / 0.42));
    for (let i = 0; i < dcols; i++) for (let j = 0; j < drows; j++) {
      const x = boardX + 0.75 + i * 0.5, z = yz0 + 0.2 + j * ((yz1 - yz0 - 0.3) / Math.max(1, drows - 1 || 1));
      S.add(U.box, 0xc49a62, TRS(x - 0.12, 0.2, z, 0, 0.14, 0.02, 0.24)); S.add(U.box, 0x8a6a44, TRS(x - 0.12, 0.1, z, 0, 0.1, 0.2, 0.2));
      S.add(U.box, 0x3f6f8f, TRS(x + 0.06, 0.1, z, 0, 0.1, 0.02, 0.14)); S.add(U.box, 0x2d4f66, TRS(x + 0.06, 0.05, z, 0, 0.06, 0.1, 0.1));
      desks.push({ x: x + 0.07, z, yaw: -Math.PI / 2, seat: 0.12 });
    }
    // playground: swings, slide, sandbox
    const px = (playX0 + R.x1) / 2, pz = (yz0 + yz1) / 2;
    F.add(U.box, 0xe8d49a, TRS(px + 0.35, 0.016, pz + 0.25, 0, 0.7, 0.012, 0.5));
    S.add(U.box, 0x9c5a43, TRS(px + 0.35, 0.04, pz + 0.25, 0, 0.74, 0.08, 0.54));
    const sw = frame(px - 0.3, 0, pz - 0.05, 0);
    [-0.35, 0.35].forEach((x) => { S.add(U.box, 0x3d7fd1, sw(x, 0.35, -0.12, 0, 0.035, 0.75, 0.035, 0.3)); S.add(U.box, 0x3d7fd1, sw(x, 0.35, 0.12, 0, 0.035, 0.75, 0.035, -0.3)); });
    S.add(U.box, 0x3d7fd1, sw(0, 0.7, 0, 0, 0.78, 0.035, 0.035));
    [-0.15, 0.15].forEach((x) => { S.add(U.box, 0x777777, sw(x - 0.05, 0.45, 0, 0, 0.008, 0.5, 0.008)); S.add(U.box, 0x777777, sw(x + 0.05, 0.45, 0, 0, 0.008, 0.5, 0.008)); S.add(U.box, 0xf2b631, sw(x, 0.2, 0, 0, 0.14, 0.02, 0.07)); });
    const sl = frame(px + 0.45, 0, pz - 0.2, -Math.PI / 2);
    S.add(U.box, 0xf2b631, sl(0, 0.5, 0.25, 0, 0.26, 0.02, 0.8, 0.62)); S.add(U.box, 0xe0533b, sl(0, 0.38, -0.2, 0, 0.26, 0.76, 0.24));
    const play = [];
    for (let k = 0; k < 8; k++) play.push({ x: px - 0.4 + (k % 4) * 0.33, z: pz - 0.2 + Math.floor(k / 4) * 0.45 });
    this.anchors.school = { rect: R, desks, play, teacher: { x: boardX + 0.3, z: (yz0 + yz1) / 2, yaw: Math.PI / 2 } };
    this.dyn.school = { winMat };
    this.label("School", bx, 2.2, bz);
  }

  buildStation(b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    F.add(U.box, 0xb9bcc0, TRS(R.cx, 0.01, R.cz, 0, R.w - 0.06, 0.014, R.d - 0.06));
    const BD = Math.min(1.7, R.d * 0.45), BW = R.w - 0.2, BH = 1.1, bx = R.cx, bz = R.z0 + 0.1 + BD / 2;
    const L = frame(bx, 0, bz, 0);
    this.buildings.push({ x0: bx - BW / 2, x1: bx + BW / 2, z0: bz - BD / 2, z1: bz + BD / 2 });
    S.add(U.box, 0xd4dde6, L(0, BH / 2, 0, 0, BW, BH, BD));
    S.add(U.box, 0x24407a, L(0, 0.86, BD / 2 + 0.005, 0, BW + 0.01, 0.1, 0.01));
    S.add(U.box, 0x24407a, L(0, BH + 0.05, 0, 0, BW + 0.12, 0.1, BD + 0.12));
    S.add(U.box, 0xbfc9d3, L(0, BH + 0.1, 0, 0, BW - 0.1, 0.04, BD - 0.1));
    S.add(U.box, 0x24407a, L(0, 0.32, BD / 2 + 0.01, 0, 0.34, 0.64, 0.02));
    S.add(U.box, 0x333a40, L(0, 0.68, BD / 2 + 0.12, 0, 0.5, 0.03, 0.26));
    const WB = new Batch();
    [-BW / 2 + 0.35, BW / 2 - 0.35].forEach((x) => { WB.add(U.box, 0xffffff, L(x, 0.5, BD / 2 + 0.012, 0, 0.36, 0.3, 0.02)); S.add(U.box, 0xfbf8f1, L(x, 0.5, BD / 2 + 0.006, 0, 0.42, 0.36, 0.012)); });
    const winMat = new THREE.MeshLambertMaterial({ color: 0x33475a, emissive: 0xfff0c8, emissiveIntensity: 0 });
    this.root.add(WB.build(winMat, { cast: false }));
    const tex = textCanvas("POLICE", { w: 512, h: 110, bg: "#24407a", fg: "#ffffff", font: "800 74px Archivo, Arial, sans-serif", border: "#15284d" });
    const sg = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 0.28), new THREE.MeshLambertMaterial({ map: tex, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 0 }));
    sg.position.set(bx, BH + 0.3, bz + BD / 2 + 0.04); this.root.add(sg);
    S.add(U.box, 0x333a40, L(-0.4, BH + 0.2, BD / 2 - 0.04, 0, 0.04, 0.2, 0.04)); S.add(U.box, 0x333a40, L(0.4, BH + 0.2, BD / 2 - 0.04, 0, 0.04, 0.2, 0.04));
    const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8), new THREE.MeshLambertMaterial({ color: 0x3a6fe0, emissive: 0x3a7bff, emissiveIntensity: 0.2 }));
    beacon.position.set(bx + BW / 2 - 0.25, BH + 0.22, bz); this.root.add(beacon);
    // holding cell with bars and a bench
    const cw = Math.min(1.4, R.w - 0.5), cd = 0.95, cx = R.x1 - 0.2 - cw / 2, cz = R.z1 - 0.2 - cd / 2;
    F.add(U.box, 0x9a9da2, TRS(cx, 0.02, cz, 0, cw, 0.02, cd));
    for (let k = 0; k <= Math.round(cw / 0.1); k++) { const x = cx - cw / 2 + k * cw / Math.round(cw / 0.1); S.add(U.cyl6, 0x59616b, TRS(x, 0.4, cz + cd / 2, 0, 0.01, 0.8, 0.01)); S.add(U.cyl6, 0x59616b, TRS(x, 0.4, cz - cd / 2, 0, 0.01, 0.8, 0.01)); }
    for (let k = 0; k <= Math.round(cd / 0.1); k++) { const z = cz - cd / 2 + k * cd / Math.round(cd / 0.1); S.add(U.cyl6, 0x59616b, TRS(cx - cw / 2, 0.4, z, 0, 0.01, 0.8, 0.01)); S.add(U.cyl6, 0x59616b, TRS(cx + cw / 2, 0.4, z, 0, 0.01, 0.8, 0.01)); }
    [[cx, cz - cd / 2, cw + 0.03, 0.03], [cx, cz + cd / 2, cw + 0.03, 0.03]].forEach(([x, z, w, d]) => S.add(U.box, 0x4a525c, TRS(x, 0.8, z, 0, w, 0.035, d)));
    [[cx - cw / 2, cz], [cx + cw / 2, cz]].forEach(([x, z]) => S.add(U.box, 0x4a525c, TRS(x, 0.8, z, 0, 0.035, 0.035, cd + 0.03)));
    S.add(U.box, 0x7a6a58, TRS(cx, 0.18, cz - cd / 2 + 0.15, 0, cw - 0.2, 0.04, 0.2));
    const jail = [];
    for (let k = 0; k < 4; k++) jail.push({ x: cx - cw / 2 + 0.25 + k * (cw - 0.5) / 3, z: cz - cd / 2 + 0.2, yaw: 0, seat: 0.2 });
    // flagpole
    S.add(U.cyl6, 0xd9d9d9, TRS(R.x0 + 0.25, 0.8, R.z1 - 0.3, 0, 0.02, 1.6, 0.02));
    S.add(U.box, 0x3a6fe0, TRS(R.x0 + 0.46, 1.45, R.z1 - 0.3, 0, 0.4, 0.25, 0.01));
    this.dyn.station = { winMat, sign: sg, beacon };
    this.anchors.station = { rect: R, jail, desk: { x: bx, z: bz + BD / 2 + 0.3, yaw: 0 } };
    this.label("Police", bx, 2.0, bz);
  }

  buildGeneric(name, b) {
    const R = this.rect(b), S = this.static, F = this.flat;
    F.add(U.box, 0xcbbfa8, TRS(R.cx, 0.01, R.cz, 0, R.w - 0.06, 0.014, R.d - 0.06));
    const BW = Math.min(R.w - 0.4, 3), BD = Math.min(R.d - 0.4, 1.6);
    S.add(U.box, 0xe0d6c2, TRS(R.cx, 0.5, R.z0 + 0.2 + BD / 2, 0, BW, 1.0, BD));
    S.add(gable(BW + 0.2, BD + 0.2, 0.45), 0x7a5a4a, TRS(R.cx, 1.0, R.z0 + 0.2 + BD / 2));
    this.sign(name.toUpperCase(), R.cx, R.z1 + 0.05, 0, { w: 1.6 });
    this.anchors[name] = { rect: R };
    this.label(name[0].toUpperCase() + name.slice(1), R.cx, 1.8, R.cz);
  }

  // ---------------------------------------------------------------- empty land: pond, cemetery, trees
  buildParks() {
    const { W, H } = this;
    const free = (x, y) => this.inGrid(x, y) && this.g(x, y) === 0;
    // largest empty rectangle not touching roads
    const ok = new Uint8Array(W * H);
    for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) ok[y * W + x] = free(x, y) && !this.isRoad(x + 1, y) && !this.isRoad(x - 1, y) && !this.isRoad(x, y + 1) && !this.isRoad(x, y - 1) ? 1 : 0;
    const best = (mask, minSide) => {
      let bb = null, ba = 0;
      for (let x0 = 0; x0 < W; x0++) for (let y0 = 0; y0 < H; y0++) {
        if (!mask[y0 * W + x0]) continue;
        let maxX = W - 1;
        for (let y1 = y0; y1 < H && mask[y1 * W + x0]; y1++) {
          let x1 = x0; while (x1 + 1 <= maxX && mask[y1 * W + x1 + 1]) x1++;
          maxX = x1;
          const w = maxX - x0 + 1, h = y1 - y0 + 1;
          if (w >= minSide && h >= minSide && w * h > ba) { ba = w * h; bb = [x0, y0, maxX, y1]; }
        }
      }
      return bb;
    };
    const pond = best(ok, 3);
    const S = this.static, F = this.flat;
    this.anchors.parks = [];
    if (pond) {
      for (let x = pond[0]; x <= pond[2]; x++) for (let y = pond[1]; y <= pond[3]; y++) { this.grid[y * W + x] = 3; ok[y * W + x] = 0; }
      const R = this.rect(pond);
      const rx = Math.min(R.w * 0.36, 3.2), rz = Math.min(R.d * 0.34, 2.2), cx = R.cx + (R.w > 6 ? R.w * 0.12 : 0), cz = R.cz;
      F.add(U.disc, 0xcdbd92, TRS(cx, 0.006, cz, 0, rx + 0.35, 1, rz + 0.3));
      const water = new THREE.Mesh(new THREE.CircleGeometry(1, 40).rotateX(-Math.PI / 2), new THREE.MeshPhongMaterial({ color: 0x4f9fc4, specular: 0xffffff, shininess: 90, transparent: true, opacity: 0.92 }));
      water.scale.set(rx, 1, rz); water.position.set(cx, 0.02, cz); water.receiveShadow = true; this.root.add(water);
      const r = rng(9);
      for (let k = 0; k < 16; k++) { const a = r() * 6.28; S.add(U.ico0, 0x9a978f, TRS(cx + Math.cos(a) * (rx + 0.18), 0.04, cz + Math.sin(a) * (rz + 0.14), r() * 3, 0.09 + r() * 0.08, 0.06, 0.09 + r() * 0.06)); }
      const pads = new Batch();
      for (let k = 0; k < 7; k++) { const a = r() * 6.28, d = 0.3 + r() * 0.55; pads.add(U.cyl12, 0x4c8a3a, TRS(cx + Math.cos(a) * rx * d, 0.03, cz + Math.sin(a) * rz * d, 0, 0.1 + r() * 0.06, 0.01, 0.1 + r() * 0.05)); }
      for (let k = 0; k < 18; k++) { const a = 2.2 + r() * 1.6, d = 0.85 + r() * 0.12; S.add(U.cone8, 0x5f8a3a, TRS(cx + Math.cos(a) * rx * d, 0.18, cz + Math.sin(a) * rz * d, 0, 0.02, 0.36 + r() * 0.2, 0.02)); }
      const padMesh = pads.build(new THREE.MeshLambertMaterial({ vertexColors: true }), { cast: false }); this.root.add(padMesh);
      // ducks
      const ducks = [];
      for (let k = 0; k < 3; k++) {
        const g = new THREE.Group();
        const bd = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8), new THREE.MeshLambertMaterial({ color: k ? 0xf5f1e6 : 0x8a6a44 })); bd.scale.set(1, 0.7, 1.4); bd.position.y = 0.05; g.add(bd);
        const hd = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), new THREE.MeshLambertMaterial({ color: k ? 0xf5f1e6 : 0x2f6b3a })); hd.position.set(0, 0.13, 0.1); g.add(hd);
        const bk = new THREE.Mesh(new THREE.ConeGeometry(0.02, 0.06, 5).rotateX(Math.PI / 2), new THREE.MeshLambertMaterial({ color: 0xf2a330 })); bk.position.set(0, 0.125, 0.16); g.add(bk);
        this.root.add(g); ducks.push({ g, a: k * 2.1, r: 0.45 + k * 0.12 });
      }
      // benches facing the water
      [[-0.6, 1], [0.7, 1]].forEach(([dx, s]) => { const x = cx + dx * rx, z = cz + (rz + 0.65) * s; S.add(U.box, 0x8a5a35, TRS(x, 0.16, z, 0, 0.6, 0.04, 0.16)); S.add(U.box, 0x8a5a35, TRS(x, 0.26, z + 0.08 * s, 0, 0.6, 0.14, 0.03)); S.add(U.box, 0x444444, TRS(x - 0.25, 0.08, z, 0, 0.03, 0.16, 0.14)); S.add(U.box, 0x444444, TRS(x + 0.25, 0.08, z, 0, 0.03, 0.16, 0.14)); });
      this.dyn.pond = { water, pads: padMesh, ducks, cx, cz, rx, rz };
      this.label("Pond", cx, 0.8, cz, "minor");
      // trees around the pond
      for (let k = 0; k < 9; k++) { const a = r() * 6.28; const x = cx + Math.cos(a) * (rx + 1.1 + r() * 0.8), z = cz + Math.sin(a) * (rz + 1.0 + r() * 0.6); if (x > R.x0 + 0.3 && x < R.x1 - 0.3 && z > R.z0 + 0.3 && z < R.z1 - 0.3) this.treeList.push({ x, z, s: 0.7 + r() * 0.4, kind: r() < 0.3 ? "pine" : "round" }); }
    }
    // cemetery in the next largest empty patch (gravestones appear as people die)
    const cem = best(ok, 2);
    if (cem && (cem[2] - cem[0] + 1) * (cem[3] - cem[1] + 1) >= 6) {
      for (let x = cem[0]; x <= cem[2]; x++) for (let y = cem[1]; y <= cem[3]; y++) { this.grid[y * W + x] = 3; ok[y * W + x] = 0; }
      const R = this.rect(cem);
      F.add(U.box, 0x7fae5a, TRS(R.cx, 0.006, R.cz, 0, R.w - 0.2, 0.01, R.d - 0.2));
      // low stone wall and a gate
      [[R.cx, R.z0 + 0.1, R.w - 0.2, 0.08], [R.cx, R.z1 - 0.1, R.w - 0.2, 0.08]].forEach(([x, z, w, d]) => S.add(U.box, 0xa8a49a, TRS(x, 0.1, z, 0, w, 0.2, d)));
      [[R.x0 + 0.1, R.cz], [R.x1 - 0.1, R.cz]].forEach(([x, z]) => S.add(U.box, 0xa8a49a, TRS(x, 0.1, z, 0, 0.08, 0.2, R.d - 0.2)));
      const slots = [];
      for (let z = R.z0 + 0.45; z < R.z1 - 0.3; z += 0.5) for (let x = R.x0 + 0.4; x < R.x1 - 0.3; x += 0.42) slots.push([x, z]);
      const g = mergeGeometries([new THREE.BoxGeometry(0.2, 0.26, 0.05).translate(0, 0.13, 0), new THREE.CylinderGeometry(0.1, 0.1, 0.05, 12, 1, false, -Math.PI / 2, Math.PI).rotateX(Math.PI / 2).translate(0, 0.26, 0)].map((q) => { q.deleteAttribute("uv"); return q; }));
      const stones = new THREE.InstancedMesh(g, snowy(new THREE.MeshLambertMaterial({ color: 0xb4b2ab }), 1), slots.length);
      const m = new THREE.Matrix4();
      slots.forEach(([x, z], i) => { m.compose(new THREE.Vector3(x, -1, z), new THREE.Quaternion(), new THREE.Vector3(0, 0, 0)); stones.setMatrixAt(i, m); });
      stones.castShadow = true; this.root.add(stones);
      this.dyn.graves = { stones, slots, shown: 0 };
      this.treeList.push({ x: R.x1 - 0.35, z: R.z0 + 0.35, s: 0.7, kind: "pine" });
      this.label("Cemetery", R.cx, 0.8, R.cz, "minor");
    }
    // scatter trees, bushes and flowers on the remaining free cells
    const r = rng(31);
    for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
      if (!free(x, y)) continue;
      const { x: X, z: Z } = this.cell(x, y);
      const byRoad = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dy]) => this.isRoad(x + dx, y + dy));
      const byLot = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dy]) => this.g(x + dx, y + dy) >= 10);
      const u = r();
      if (!byRoad && !byLot && u < 0.45) this.treeList.push({ x: X + (r() - 0.5) * 0.5, z: Z + (r() - 0.5) * 0.5, s: 0.6 + r() * 0.5, kind: r() < 0.3 ? "pine" : "round" });
      else if (byRoad && !byLot && u < 0.14) this.treeList.push({ x: X + (r() - 0.5) * 0.3, z: Z + (r() - 0.5) * 0.3, s: 0.5 + r() * 0.2, kind: "round" });
      else if (u < 0.35) this.bushList.push({ x: X + (r() - 0.5) * 0.6, z: Z + (r() - 0.5) * 0.6, s: 0.12 + r() * 0.1 });
      if (r() < 0.35) for (let k = 0; k < 3; k++) this.flowerList.push({ x: X + (r() - 0.5) * 0.8, z: Z + (r() - 0.5) * 0.8, c: FLOWERS[Math.floor(r() * FLOWERS.length)] });
    }
    // countryside: forest ring and meadows outside the grid
    const hw = W / 2, hh = H / 2;
    for (let k = 0; k < (this.quality > 0.7 ? 1300 : 650); k++) {
      const x = (r() - 0.5) * 170, z = (r() - 0.5) * 150;
      const dx = Math.max(0, Math.abs(x) - hw), dz = Math.max(0, Math.abs(z) - hh), d = Math.hypot(dx, dz);
      if (d < 1.2) continue;
      if (this.roadDist(x, z) < 1.4) continue;
      if (r() > smooth(1, 14, d) * 0.9 + 0.08) continue;
      const y = this.terrainH(x, z);
      this.treeList.push({ x, z, y, s: 0.7 + r() * 0.9 + smooth(10, 50, d) * 0.5, kind: r() < 0.45 ? "pine" : "round" });
    }
  }

  // ---------------------------------------------------------------- street lamps
  buildLamps() {
    const S = this.static, heads = this.lampHeads;
    const { W, H } = this;
    const put = (X, Z, face) => {
      const L = frame(X, 0, Z, face);
      S.add(U.cyl6, 0x3b4148, L(0, 0.62, 0, 0, 0.025, 1.24, 0.025));
      S.add(U.cyl8, 0x3b4148, L(0, 0.03, 0, 0, 0.06, 0.06, 0.06));
      S.add(U.box, 0x3b4148, L(0, 1.24, 0.14, 0, 0.03, 0.03, 0.3));
      S.add(U.box, 0x2e3439, L(0, 1.25, 0.28, 0, 0.14, 0.04, 0.12));
      heads.add(U.box, 0xffffff, L(0, 1.215, 0.28, 0, 0.1, 0.03, 0.08));
      const p = localToWorld(X, Z, face, 0, 0.3);
      this.pools.push({ x: p.x, z: p.z, r: 1.15 });
    };
    let k = 0;
    for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
      if (!this.isRoad(x, y)) continue;
      const h = this.isRoad(x - 1, y) && this.isRoad(x + 1, y) && !this.isRoad(x, y + 1) && !this.isRoad(x, y - 1);
      const v = this.isRoad(x, y - 1) && this.isRoad(x, y + 1) && !this.isRoad(x + 1, y) && !this.isRoad(x - 1, y);
      if (!h && !v) continue;
      if ((h ? x : y) % 4 !== 2) continue;
      const { x: X, z: Z } = this.cell(x, y);
      const side = (k++ % 2) ? 1 : -1;
      if (h) { const nz = Z + side * 0.56; const [cx, cy] = this.toCell(X, nz); if (this.g(cx, cy) >= 10) continue; put(X, nz, side > 0 ? Math.PI : 0); }
      else { const nx = X + side * 0.56; const [cx, cy] = this.toCell(nx, Z); if (this.g(cx, cy) >= 10) continue; put(nx, Z, side > 0 ? -Math.PI / 2 : Math.PI / 2); }
    }
  }

  // ---------------------------------------------------------------- vegetation (instanced)
  buildVegetation() {
    const trees = this.treeList;
    const round = trees.filter((t) => t.kind === "round" || t.kind === "orchard" || t.kind === "fruit");
    const pines = trees.filter((t) => t.kind === "pine");
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), s = new THREE.Vector3();
    const trunkG = new THREE.CylinderGeometry(0.06, 0.09, 1, 6).translate(0, 0.5, 0);
    const crownG = new THREE.IcosahedronGeometry(1, 1);
    const trunks = new THREE.InstancedMesh(trunkG, this.M.trunk, round.length + pines.length);
    const crowns = new THREE.InstancedMesh(crownG, this.M.crown, round.length);
    const pineG = mergeGeometries([0, 1, 2].map((k) => { const g = new THREE.ConeGeometry(0.55 - k * 0.13, 0.7, 7); g.translate(0, 0.55 + k * 0.38, 0); g.deleteAttribute("uv"); return g; }));
    const pineM = new THREE.InstancedMesh(pineG, this.M.pine, pines.length);
    let ti = 0;
    const r = rng(41);
    round.forEach((t, i) => {
      const y = t.y || 0, sc = t.s, th = t.kind === "orchard" ? 0.5 : 0.62 * sc + 0.1;
      q.setFromAxisAngle(v.set(0, 1, 0), r() * 6.28);
      m.compose(v.set(t.x, y, t.z), q, s.set(sc * 0.9, th, sc * 0.9)); trunks.setMatrixAt(ti++, m);
      const cr = t.kind === "orchard" ? 0.46 * sc : (0.42 + r() * 0.12) * sc;
      m.compose(v.set(t.x, y + th + cr * 0.55, t.z), q, s.set(cr * (1 + r() * 0.15), cr * (0.85 + r() * 0.25), cr)); crowns.setMatrixAt(i, m);
      t.tint = r(); t.crownY = y + th + cr * 0.55; t.cr = cr;
    });
    pines.forEach((t) => {
      const y = t.y || 0, sc = t.s;
      q.setFromAxisAngle(v.set(0, 1, 0), r() * 6.28);
      m.compose(v.set(t.x, y, t.z), q, s.set(sc * 0.7, sc * 0.35, sc * 0.7)); trunks.setMatrixAt(ti++, m);
      m.compose(v.set(t.x, y + sc * 0.15, t.z), q, s.set(sc, sc * (0.95 + r() * 0.3), sc)); pineM.setMatrixAt(pines.indexOf(t), m);
      t.tint = r();
    });
    trunks.count = ti;
    [trunks, crowns, pineM].forEach((x) => { x.castShadow = true; x.receiveShadow = true; this.root.add(x); });
    pines.forEach((t, i) => pineM.setColorAt(i, new THREE.Color(0x2f6b45).offsetHSL((t.tint - 0.5) * 0.03, 0, (t.tint - 0.5) * 0.08)));
    this.dyn.trees = { crowns, round, pines: pineM };
    // bushes
    const bushes = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0), this.M.crown, this.bushList.length);
    this.bushList.forEach((b, i) => { m.compose(v.set(b.x, (b.y || 0) + b.s * 0.7, b.z), q.setFromAxisAngle(v.set(0, 1, 0), i), s.set(b.s * 1.2, b.s, b.s * 1.2)); bushes.setMatrixAt(i, m); b.tint = r(); });
    bushes.castShadow = true; this.root.add(bushes); this.dyn.bushes = bushes;
    // flowers
    const fl = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.04, 0), new THREE.MeshLambertMaterial({ color: 0xffffff }), this.flowerList.length);
    this.flowerList.forEach((f, i) => { m.compose(v.set(f.x, 0.05, f.z), q.identity(), s.set(1, 1, 1)); fl.setMatrixAt(i, m); fl.setColorAt(i, new THREE.Color(f.c)); });
    this.root.add(fl); this.dyn.flowers = fl;
  }

  finish() {
    const st = this.static.build(this.M.static); if (st) this.root.add(st);
    const fl = this.flat.build(this.M.flat, { cast: false }); if (fl) this.root.add(fl);
    const rd = this.roadB.build(this.M.road, { cast: false }); if (rd) this.root.add(rd);
    const lh = this.lampHeads.build(this.M.lamp, { cast: false }); if (lh) this.root.add(lh);
    // glowing pools of light (additive)
    const c = document.createElement("canvas"); c.width = c.height = 64;
    const g = c.getContext("2d"), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, "rgba(255,210,140,1)"); grd.addColorStop(0.5, "rgba(255,190,110,0.35)"); grd.addColorStop(1, "rgba(255,180,100,0)");
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
    const tex = new THREE.CanvasTexture(c);
    const pm = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 });
    const pools = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), pm, this.pools.length);
    const m = new THREE.Matrix4();
    this.pools.forEach((p, i) => { m.compose(new THREE.Vector3(p.x, 0.03, p.z), new THREE.Quaternion(), new THREE.Vector3(p.r * 2, 1, p.r * 2)); pools.setMatrixAt(i, m); });
    pools.renderOrder = 2; this.root.add(pools); this.poolMesh = pools;
    // zzz sprites over houses, smoke puffs from chimneys
    const zc = document.createElement("canvas"); zc.width = 128; zc.height = 128;
    const zg = zc.getContext("2d"); zg.font = "800 44px Archivo, Arial, sans-serif"; zg.lineWidth = 7; zg.strokeStyle = "rgba(20,30,60,0.75)"; zg.fillStyle = "#e9efff";
    [["z", 20, 104, 34], ["z", 52, 74, 42], ["Z", 84, 42, 52]].forEach(([t, x, y, sz]) => { zg.font = `800 ${sz}px Archivo, Arial, sans-serif`; zg.strokeText(t, x, y); zg.fillText(t, x, y); });
    const ztex = new THREE.CanvasTexture(zc); ztex.colorSpace = THREE.SRGBColorSpace;
    this.anchors.houses.forEach((A, h) => {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: ztex, transparent: true, depthWrite: false, opacity: 0 }));
      sp.scale.set(0.8, 0.8, 1); sp.position.copy(A.top).add(new THREE.Vector3(0.35, 0.25, 0)); sp.visible = false; this.root.add(sp);
      this.houses[h].zzz = sp;
    });
    this.zzzTex = ztex;
    const puffs = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0), new THREE.MeshLambertMaterial({ color: 0xe6e6e6, transparent: true, opacity: 0.75, flatShading: true }), this.houses.length * 5);
    puffs.castShadow = false; this.root.add(puffs); this.dyn.puffs = puffs;
    this.houses.forEach((hs) => { hs.smoke = 0; });
  }

  // ---------------------------------------------------------------- per-frame updates
  // time of day and season; env = {hourF, day, season(dayIdx) fn}
  setTime(hourF, day, seasonOf, seasonNames) {
    const sName = (d) => seasonIndex(seasonNames[seasonOf(d)], seasonOf(d));
    let sA = sName(day), sB = sA, b = 0;
    if (hourF >= 22) { sB = sName(day + 1); b = smooth(22, 26, hourF); }
    else if (hourF < 2) { sA = sName(day - 1); b = smooth(22, 26, hourF + 24); }
    const mixS = (arr) => (arr[sA].isColor ? arr[sA].clone().lerp(arr[sB], b) : lerp(arr[sA], arr[sB], b));
    const snow = mixS(SEASON.snow);
    SNOW.value = snow;
    this.season = b < 0.5 ? sA : sB; this.seasonBlend = { sA, sB, b };
    this.M.terrain.color.copy(mixS(SEASON.grass));
    // sun path: rises 06:00, sets 21:00
    const h = hourF;
    let sunH;
    if (h >= 6 && h <= 21) sunH = Math.sin(Math.PI * (h - 6) / 15);
    else { const hh = h < 6 ? h + 24 : h; sunH = -Math.sin(Math.PI * (hh - 21) / 9) * 0.9; }
    const theta = Math.PI * (h - 6) / 15;
    const maxEl = mixS(SEASON.sunMax);
    const dir = new THREE.Vector3(Math.cos(theta) * 1.0, Math.max(-0.4, sunH) * maxEl + 0.02, 0.55).normalize();
    const day1 = smooth(-0.1, 0.3, sunH), twi = Math.exp(-Math.pow(sunH / 0.2, 2));
    this.daylight = day1; this.dark = 1 - smooth(0.02, 0.4, sunH);
    const C = (x) => new THREE.Color(x);
    const top = C(0x0b1a3a).lerp(C(sA === 3 || sB === 3 ? 0x7aa6cf : 0x4f94d6), day1).lerp(C(0x5f6fa8), twi * 0.35);
    const hor = C(0x1b2a4a).lerp(C(0xcfe5f0), day1).lerp(C(0xf5ae78), twi * 0.65);
    this.skyU.top.value.copy(top); this.skyU.horizon.value.copy(hor); this.skyU.bottom.value.copy(hor.clone().multiplyScalar(0.85));
    this.skyU.sunDir.value.copy(dir); this.skyU.glow.value = 0.4 + twi;
    this.skyU.sunCol.value.copy(C(0xffb070).lerp(C(0xfff4e0), smooth(0.05, 0.5, sunH))).multiplyScalar(sunH > -0.05 ? 1 : 0);
    this.scene.fog.color.copy(hor);
    // lights
    const center = new THREE.Vector3(0, 0, 0);
    const moonDir = new THREE.Vector3(-Math.cos(theta) * 0.8, 0.75, -0.35).normalize();
    if (sunH > -0.02) {
      this.sun.position.copy(center).addScaledVector(dir, 60);
      this.sun.color.copy(C(0xff9a55).lerp(C(0xfff4e0), smooth(0.0, 0.45, sunH)));
      this.sun.intensity = 2.5 * smooth(-0.02, 0.25, sunH) * (sA === 3 ? 0.85 : 1);
    } else {
      this.sun.position.copy(center).addScaledVector(moonDir, 60);
      this.sun.color.set(0xa9bbff); this.sun.intensity = 0.55 * smooth(-0.02, -0.2, sunH);
    }
    this.sun.target.position.copy(center);
    this.hemi.color.copy(C(0x3a4c85).lerp(C(0xdcefff), day1)); this.hemi.groundColor.copy(C(0x1c2436).lerp(C(0x6d7f55), day1));
    this.hemi.intensity = lerp(0.8, 1.05, day1) + twi * 0.1;
    this.amb.intensity = lerp(0.3, 0.05, day1);
    this.stars.material.opacity = smooth(0.1, -0.25, sunH) * 0.95;
    this.moon.position.copy(moonDir).multiplyScalar(250);
    this.moon.material.opacity = smooth(0.25, -0.1, sunH);
    // lamps and glow
    const glow = this.dark;
    this.M.lamp.emissiveIntensity = glow * 2.2;
    this.poolMesh.material.opacity = glow * 0.8;
    this.glow = glow;
    this.applyGlow();
    // seasonal colours of living things (only when the palette changes)
    const key = `${sA}|${sB}|${Math.round(b * 20)}`;
    if (key !== this._seasonKey) { this._seasonKey = key; this.paintSeason(sA, sB, b); }
    this.snowfall.visible = snow > 0.3 && hash(day, 12) < 0.65;
    this.snowU.uOpacity.value = snow * 0.9;
    if (this.dyn.pond) { const w = this.dyn.pond.water.material; w.color.set(0x4f9fc4).lerp(C(0xdfeef5), snow); w.shininess = lerp(90, 20, snow); this.dyn.pond.pads.visible = snow < 0.5; this.dyn.pond.ducks.forEach((d) => { d.g.visible = snow < 0.5; }); }
    this.dyn.flowers.visible = this.season === 0 || this.season === 1;
  }

  paintSeason(sA, sB, b) {
    const pal = (s, t, kind) => {
      const tint = t - 0.5;
      if (kind === "orchard") return new THREE.Color([0xf4b6c8, 0x4f9a3a, 0x8aa33a, 0x9a8a7a][s]).offsetHSL(tint * 0.02, 0, tint * 0.06);
      if (s === 0) return new THREE.Color(t < 0.18 ? 0xf2c4d6 : t < 0.26 ? 0xf7f0f2 : 0x7ccf52).offsetHSL(tint * 0.04, 0, tint * 0.08);
      if (s === 1) return new THREE.Color(0x4f9a3a).offsetHSL(tint * 0.05, 0, tint * 0.1);
      if (s === 2) return new THREE.Color([0xe0782c, 0xd9a132, 0xc9492f, 0xb5a03a, 0xe39b2e][Math.floor(t * 5)]).offsetHSL(0, 0, tint * 0.05);
      return new THREE.Color(0x8d8178).offsetHSL(0, 0, tint * 0.06);
    };
    const T = this.dyn.trees, m = new THREE.Matrix4(), v = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    T.round.forEach((t, i) => {
      const c = pal(sA, t.tint, t.kind).lerp(pal(sB, t.tint, t.kind), b);
      T.crowns.setColorAt(i, c);
      // bare, smaller crowns in winter
      const wf = (sA === 3 ? 1 - b : 0) + (sB === 3 ? b : 0);
      T.crowns.getMatrixAt(i, m); m.decompose(v, q, s);
      const base = t.cr, k = 1 - 0.3 * wf;
      s.set(base * k * (s.x / s.z) , base * k * (s.y / s.z), base * k);
      m.compose(v, q, s); T.crowns.setMatrixAt(i, m);
    });
    T.crowns.instanceColor.needsUpdate = true; T.crowns.instanceMatrix.needsUpdate = true;
    const B = this.dyn.bushes;
    this.bushList.forEach((bb, i) => { const c = new THREE.Color([0x5a9a3a, 0x3f7f35, 0x8a8a3a, 0x6b7a5a][sA]).lerp(new THREE.Color([0x5a9a3a, 0x3f7f35, 0x8a8a3a, 0x6b7a5a][sB]), b).offsetHSL((bb.tint - 0.5) * 0.04, 0, (bb.tint - 0.5) * 0.08); B.setColorAt(i, c); });
    B.instanceColor.needsUpdate = true;
  }

  // town state from the replay frame
  setTownState(w, ctx) {
    const d = this.dyn;
    const season = this.season;
    // crops: height from w.crops (0-9), colour from the season
    if (d.crops) {
      const lvl = w.crops ?? 0;
      if (lvl !== d.crops.userData.lvl || season !== d.crops.userData.season) {
        d.crops.userData.lvl = lvl; d.crops.userData.season = season;
        const m = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), s = new THREE.Vector3(), c = new THREE.Color();
        const hgt = lvl <= 0 ? 0 : 0.04 + 0.03 * lvl;
        const base = season === 2 || (season === 1 && lvl >= 8) ? 0xe2bf55 : season === 0 ? 0x86d25a : season === 3 ? 0xb9b08a : 0x5fae3e;
        d.crops.userData.pos.forEach(([x, z], i) => {
          const u = hash(i, 17);
          const hh = hgt * (0.8 + u * 0.4);
          m.compose(v.set(x, 0.02, z), q.setFromAxisAngle(v.set(0, 1, 0).clone(), u * 6.28), s.set(0.8 + hh * 1.6, Math.max(0.0001, hh), 0.8 + hh * 1.6));
          m.setPosition(x, 0.02, z);
          d.crops.setMatrixAt(i, m);
          c.set(base).offsetHSL((u - 0.5) * 0.03, 0, (u - 0.5) * 0.1 + (lvl < 3 ? 0.05 : 0));
          d.crops.setColorAt(i, c);
        });
        d.crops.instanceMatrix.needsUpdate = true; d.crops.instanceColor.needsUpdate = true;
        d.crops.visible = lvl > 0;
      }
    }
    if (d.fruit) {
      const lvl = Math.max(0, Math.min(9, w.fruit ?? 0));
      if (lvl !== d.fruit.userData.lvl) {
        d.fruit.userData.lvl = lvl;
        const m = new THREE.Matrix4();
        d.fruit.userData.pos.forEach((p, i) => { const k = i % 9; m.makeScale(k < lvl ? 1 : 0, k < lvl ? 1 : 0, k < lvl ? 1 : 0); m.setPosition(p[0], p[1], p[2]); d.fruit.setMatrixAt(i, m); });
        d.fruit.instanceMatrix.needsUpdate = true;
      }
    }
    if (d.cows) { const n = Math.max(0, Math.min(9, w.herd ?? 0)); d.cows.cows.forEach((c) => { if (!c.st) c.want = c.i < n ? 1 : 0; }); }
    if (d.crates) {
      const st = w.stock || [0, 0, 0], tot = st.reduce((a, b) => a + b, 0);
      const n = Math.min(d.crates.userData.slots.length, Math.round(tot / 3));
      if (n !== d.crates.userData.n) {
        d.crates.userData.n = n;
        const m = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), s = new THREE.Vector3(1, 1, 1);
        d.crates.userData.slots.forEach(([x, y, z, r], i) => { q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), r); m.compose(v.set(x, y, z), q, i < n ? s.set(1, 1, 1) : s.set(0, 0, 0)); d.crates.setMatrixAt(i, m); });
        d.crates.instanceMatrix.needsUpdate = true;
      }
    }
    if (d.canteen) {
      const open = ctx.canteenOpen ? 1 : 0;
      if (open !== d.canteen.open) {
        d.canteen.open = open;
        d.canteen.plaque.material.map = open ? d.canteen.openTex : d.canteen.closedTex;
        d.canteen.plaque.material.emissiveMap = d.canteen.plaque.material.map; d.canteen.plaque.material.needsUpdate = true;
      }
    }
    this.ctx = ctx;
    this.applyGlow();
    if (d.graves) {
      const n = Math.min(d.graves.slots.length, ctx.deaths || 0);
      if (n !== d.graves.shown) {
        d.graves.shown = n;
        const m = new THREE.Matrix4(), q = new THREE.Quaternion();
        d.graves.slots.forEach(([x, z], i) => { q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), (hash(i, 4) - 0.5) * 0.2); m.compose(new THREE.Vector3(x, 0, z), q, i < n ? new THREE.Vector3(1, 0.85 + hash(i, 5) * 0.3, 1) : new THREE.Vector3(0, 0, 0)); d.graves.stones.setMatrixAt(i, m); });
        d.graves.stones.instanceMatrix.needsUpdate = true;
      }
    }
    this.shopOpen = !!ctx.shopOpen;
  }

  // lights that depend on the time of day (called every rendered frame)
  applyGlow() {
    const d = this.dyn, g = this.glow || 0, ctx = this.ctx || {};
    if (d.canteen) {
      const open = d.canteen.open > 0;
      d.canteen.winMat.emissiveIntensity = open ? 0.35 + g * 1.2 : g * 0.15;
      d.canteen.bulbMat.emissiveIntensity = (open || (ctx.hourF ?? 0) < 22) ? g * 2.5 : 0;
      d.canteen.sign.material.emissiveIntensity = g * (open ? 0.9 : 0.35);
      d.canteen.plaque.material.emissiveIntensity = 0.15 + g * 0.8;
    }
    if (d.station) { d.station.winMat.emissiveIntensity = g * 1.1; d.station.sign.material.emissiveIntensity = g * 0.8; }
    if (d.school) d.school.winMat.emissiveIntensity = ctx.schoolOn ? 0.2 + g * 0.6 : 0;
    this.houses.forEach((hs) => { hs.winMat.emissiveIntensity = (hs.awake ? g : hs.asleep ? g * 0.06 : 0) * 1.7; });
  }

  // occupancy of each house: {awake, asleep}
  setHouses(occ) {
    this.houses.forEach((hs, h) => {
      const o = occ[h] || { awake: 0, asleep: 0 };
      hs.awake = o.awake; hs.asleep = o.asleep;
      hs.sleep = o.asleep > 0; hs.home = o.awake + o.asleep > 0;
      hs.zzz.visible = hs.sleep;
      hs.smoke = hs.home && (this.season === 3 || this.season === 2 || this.glow > 0.3) ? 1 : 0;
    });
  }

  // animation that runs every rendered frame (seconds of wall clock)
  tick(dt, clk, camera) {
    this.snowU.uTime.value = clk;
    if (camera) this.snowU.uScale.value = window.innerHeight * 0.9;
    this.clouds.forEach((c, k) => { c.position.x += dt * (0.25 + (k % 3) * 0.1); if (c.position.x > 170) c.position.x = -170; });
    // zzz bob
    this.houses.forEach((hs, h) => { if (hs.zzz.visible) { const u = (clk * 0.4 + h * 0.37) % 1; hs.zzz.material.opacity = Math.sin(u * Math.PI) * 0.95; hs.zzz.position.y = this.anchors.houses[h].top.y + 0.1 + u * 0.4; } });
    // chimney smoke
    const P = this.dyn.puffs, m = new THREE.Matrix4(), v = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    let pi = 0;
    this.houses.forEach((hs, h) => {
      const c = this.anchors.houses[h].chimney;
      for (let k = 0; k < 5; k++) {
        const u = (clk * 0.22 + k / 5 + h * 0.13) % 1, on = hs.smoke;
        const sc = on ? (0.05 + u * 0.13) * Math.sin(Math.min(1, u * 1.3) * Math.PI) : 0;
        m.compose(v.set(c.x + u * 0.35 + Math.sin(clk + k) * 0.04, c.y + u * 0.9, c.z - u * 0.1), q, s.set(sc, sc, sc));
        P.setMatrixAt(pi++, m);
      }
    });
    P.instanceMatrix.needsUpdate = true;
    // cows wander and graze
    const C = this.dyn.cows;
    if (C) {
      const A = C.area;
      C.cows.forEach((c) => {
        c.vis += ((c.st ? c.want : c.want) - c.vis) * Math.min(1, dt * 3);
        if (!c.st) {
          c.wait -= dt;
          if (c.wait <= 0) {
            if (c.tx === 0 && c.tz === 0 || Math.hypot(c.tx - c.x, c.tz - c.z) < 0.05) { c.tx = lerp(A.x0, A.x1, hash(c.i, Math.floor(clk / 7))); c.tz = lerp(A.z0, A.z1, hash(c.i + 50, Math.floor(clk / 7))); }
            const dx = c.tx - c.x, dz = c.tz - c.z, d = Math.hypot(dx, dz);
            if (d > 0.05) { const sp = Math.min(d, dt * 0.18); c.x += dx / d * sp; c.z += dz / d * sp; let dy = Math.atan2(dx, dz) - c.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy)); c.yaw += dy * Math.min(1, dt * 2); }
            else c.wait = 3 + hash(c.i, Math.floor(clk)) * 6;
          }
        }
        const graze = Math.sin(clk * 0.5 + c.i * 1.7) > -0.2 && !c.st;
        c.graze += ((graze ? 1 : 0) - c.graze) * Math.min(1, dt * 2);
        const sc = c.s * c.vis;
        q.setFromAxisAngle(v.set(0, 1, 0), c.yaw);
        m.compose(v.set(c.x, 0, c.z), q, s.set(sc, sc, sc)); C.bodies.setMatrixAt(c.i, m);
        const hx = c.x + Math.sin(c.yaw) * 0.3 * sc, hz = c.z + Math.cos(c.yaw) * 0.3 * sc;
        const e = new THREE.Euler(0.2 + c.graze * 0.75, c.yaw, 0, "YXZ");
        m.compose(v.set(hx, (0.4 - c.graze * 0.18) * sc, hz), q.setFromEuler(e), s.set(sc, sc, sc)); C.heads.setMatrixAt(c.i, m);
      });
      C.bodies.instanceMatrix.needsUpdate = true; C.heads.instanceMatrix.needsUpdate = true;
    }
    if (this.dyn.pond) {
      const p = this.dyn.pond;
      p.ducks.forEach((d, k) => { d.a += dt * (0.15 + k * 0.03); d.g.position.set(p.cx + Math.cos(d.a) * p.rx * d.r, 0.02 + Math.sin(clk * 2 + k) * 0.005, p.cz + Math.sin(d.a) * p.rz * d.r); d.g.rotation.y = -d.a; });
    }
    if (this.dyn.station) this.dyn.station.beacon.material.emissiveIntensity = 0.2 + this.glow * (0.6 + 0.6 * Math.max(0, Math.sin(clk * 3)));
  }

  setMilking(stationIdx, on) {
    const C = this.dyn.cows; if (!C) return;
    const c = C.cows.find((x) => x.st && C.cows.indexOf(x) - 9 === stationIdx);
    if (c) c.want = on ? 1 : 0;
  }
}

function mixC(a, b, t) { return new THREE.Color(a).lerp(new THREE.Color(b), t); }
