// Where each person stands in each frame, and how they get from one frame's spot to the next.
//
// A frame says "cell (x, y), activity a". The layout turns that into a spot in the 3D town
// (a canteen seat, a place at the market counter, a desk at school, a tree in the orchard,
// a circle of friends...) and a pose. Between frames people walk along the roads (A* on the
// grid, preferring roads and entrance paths, keeping to the right).
import { hash, clamp, lerp } from "./util.js";

const COST = { 0: 4.0, 1: 1.0, 2: 1.25, 3: 6.0 };   // grass, road, path, pond/cemetery
const LOT = 1.8;                                     // inside a house lot or a place

export class Layout {
  constructor(rep, world) {
    this.rep = rep; this.world = world;
    this.W = world.W; this.H = world.H;
    this.cost = new Float32Array(this.W * this.H);
    for (let x = 0; x < this.W; x++) for (let y = 0; y < this.H; y++) {
      const v = world.g(x, y);
      this.cost[y * this.W + x] = world.inBuilding(x, y) ? 14 : v >= 10 ? LOT : COST[v] ?? 4;
    }
    this.paths = new Map();
    this.frames = new Map();
    this.A = world.anchors;
    this.placeRects = {};
    for (const [k, v] of Object.entries(this.A)) if (v && v.rect) this.placeRects[k] = v.rect;
  }

  // ---------------------------------------------------------------- A* on the grid
  cellPath(x0, y0, x1, y1) {
    const key = ((x0 * 64 + y0) * 64 + x1) * 64 + y1;
    let p = this.paths.get(key);
    if (p) return p;
    const W = this.W, H = this.H, N = W * H;
    const clampC = (x, y) => [clamp(x, 0, W - 1), clamp(y, 0, H - 1)];
    [x0, y0] = clampC(x0, y0); [x1, y1] = clampC(x1, y1);
    const start = y0 * W + x0, goal = y1 * W + x1;
    const g = new Float32Array(N).fill(1e9), from = new Int32Array(N).fill(-1), closed = new Uint8Array(N);
    const heap = [];   // [f, node]
    const push = (f, n) => { heap.push([f, n]); let i = heap.length - 1; while (i > 0) { const pa = (i - 1) >> 1; if (heap[pa][0] <= heap[i][0]) break; [heap[pa], heap[i]] = [heap[i], heap[pa]]; i = pa; } };
    const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = i * 2 + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
    g[start] = 0; push(0, start);
    while (heap.length) {
      const [, n] = pop();
      if (closed[n]) continue; closed[n] = 1;
      if (n === goal) break;
      const x = n % W, y = (n / W) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = x + (d === 0 ? 1 : d === 1 ? -1 : 0), ny = y + (d === 2 ? 1 : d === 3 ? -1 : 0);
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const m = ny * W + nx;
        if (closed[m]) continue;
        // small penalty for turning keeps routes straight
        const pf = from[n], turn = pf >= 0 && ((pf % W === x) !== (nx === x)) ? 0.15 : 0;
        const ng = g[n] + this.cost[m] + turn;
        if (ng < g[m]) { g[m] = ng; from[m] = n; push(ng + (Math.abs(nx - x1) + Math.abs(ny - y1)), m); }
      }
    }
    const out = [];
    for (let n = goal; n >= 0; n = from[n]) { out.push([n % W, (n / W) | 0]); if (n === start) break; }
    out.reverse();
    if (this.paths.size > 30000) this.paths.clear();
    this.paths.set(key, out);
    return out;
  }

  // world-space polyline from spot a to spot b (keeps to the right on roads)
  route(a, b) {
    const w = this.world;
    const cells = this.cellPath(a.cx, a.cy, b.cx, b.cy);
    const inner = [];
    // skip cells inside the start and end lots (the spots are already at their doors/seats)
    let i0 = 0, i1 = cells.length - 1;
    const sameLot = (c, s) => s.lot != null && s.lot < 100 && w.g(c[0], c[1]) === s.lot;   // house lots only
    while (i0 < i1 && sameLot(cells[i0 + 1], a)) i0++;
    while (i1 > i0 && sameLot(cells[i1 - 1], b)) i1--;
    for (let i = i0 + 1; i < i1; i++) {
      const [x, y] = cells[i], p = w.cell(x, y);
      inner.push({ x: p.x, z: p.z, road: w.isRoad(x, y) });
    }
    const pts = [{ x: a.x, z: a.z }];
    // lane offset: keep right on road cells
    for (let i = 0; i < inner.length; i++) {
      const p = inner[i];
      if (p.road) {
        const prev = i > 0 ? inner[i - 1] : pts[0], next = i < inner.length - 1 ? inner[i + 1] : b;
        let dx = next.x - prev.x, dz = next.z - prev.z; const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
        pts.push({ x: p.x - dz * 0.24, z: p.z + dx * 0.24 });
      } else pts.push({ x: p.x, z: p.z });
    }
    pts.push({ x: b.x, z: b.z });
    // drop near-duplicates and measure
    const clean = [pts[0]];
    for (let i = 1; i < pts.length; i++) { const q = clean[clean.length - 1]; if (Math.hypot(pts[i].x - q.x, pts[i].z - q.z) > 0.03) clean.push(pts[i]); }
    const len = [0];
    for (let i = 1; i < clean.length; i++) len.push(len[i - 1] + Math.hypot(clean[i].x - clean[i - 1].x, clean[i].z - clean[i - 1].z));
    return { pts: clean, len, total: len[len.length - 1] };
  }
  static along(r, d) {
    const { pts, len } = r;
    if (pts.length === 1 || d <= 0) { const p = pts[0], q = pts[1] || p; return { x: p.x, z: p.z, dx: q.x - p.x, dz: q.z - p.z }; }
    if (d >= r.total) { const p = pts[pts.length - 1], q = pts[pts.length - 2]; return { x: p.x, z: p.z, dx: p.x - q.x, dz: p.z - q.z }; }
    let i = 1; while (len[i] < d) i++;
    const u = (d - len[i - 1]) / Math.max(1e-6, len[i] - len[i - 1]);
    const p = pts[i - 1], q = pts[i];
    return { x: lerp(p.x, q.x, u), z: lerp(p.z, q.z, u), dx: q.x - p.x, dz: q.z - p.z };
  }

  // ---------------------------------------------------------------- per-frame spots
  frame(k) {
    if (k < 0 || k >= this.rep.n) return null;
    let L = this.frames.get(k);
    if (L) return L;
    L = this.compute(k);
    if (this.frames.size > 200) this.frames.delete(this.frames.keys().next().value);
    this.frames.set(k, L);
    return L;
  }

  compute(k) {
    const rep = this.rep, w = this.world, A = this.A, C = rep.C, cfg = rep.cfg;
    const f = rep.frames[k], t = rep.tOf(k), hour = ((t % rep.daySteps) + rep.daySteps) % rep.daySteps;
    const S = rep.slots, spots = new Array(S).fill(null), list = [];
    for (let s = 0; s < S; s++) {
      const r = f.s[s];
      if (!r) continue;
      const x = r[C.x], y = r[C.y], info = w.cellInfo(x, y);
      const sp = {
        s, pid: r[C.pid], cx: x, cy: y, act: rep.acts[r[C.act]] || "idle", age: (r[C.age10] || 0) / 10, job: rep.jobs[r[C.job]] || "none",
        flags: r[C.flags] || 0, home: r[C.home], partner: r[C.partner], info,
        x: 0, z: 0, yaw: hash(r[C.pid], k >> 3) * 6.28, pose: "stand", hidden: false, lot: null, props: null, babies: 0,
      };
      spots[s] = sp; list.push(sp);
    }
    // babies ride with their mother
    for (const sp of list) {
      if (sp.age >= 2.5 || (sp.act !== "nursing" && sp.age >= 1)) continue;
      const mom = rep.people[sp.pid] ? rep.people[sp.pid].mother : 0;
      const ms = mom ? rep.pidSlot(k, mom) : -1, m = ms >= 0 ? spots[ms] : null;
      if (m && Math.abs(m.cx - sp.cx) + Math.abs(m.cy - sp.cy) <= 1) { sp.carriedBy = ms; m.babies++; }
    }
    const house = [];
    const groups = new Map();
    const add = (key, sp) => { let g = groups.get(key); if (!g) groups.set(key, (g = [])); g.push(sp); };
    for (const sp of list) {
      const info = sp.info, act = sp.act;
      if (info.type === "house") {
        const H = A.houses[info.h];
        sp.hidden = true; sp.lot = 10 + info.h; sp.house = info.h;
        sp.x = H.door.x; sp.z = H.door.z; sp.yaw = H.door.yaw + Math.PI;
        (house[info.h] = house[info.h] || { awake: 0, asleep: 0, pids: [] }).pids.push(sp.pid);
        if (act === "sleeping" || (sp.flags & 8)) house[info.h].asleep++; else house[info.h].awake++;
        continue;
      }
      if (sp.carriedBy != null) { sp.hidden = true; continue; }
      const place = info.type === "place" ? info.name : null;
      sp.place = place;
      if (place) sp.lot = w.g(sp.cx, sp.cy);
      if (act === "detained" || (sp.flags & 2)) { add(A.station ? "jail" : "cell", sp); continue; }
      if (act === "sleeping") { add("lie", sp); continue; }
      if (act === "walking" || act === "idle" && !place) { add("cell", sp); continue; }
      if (act === "socialising") { add("social", sp); continue; }
      if (act === "school") { add(place === "school" ? "school" : "cell", sp); continue; }
      if (act === "eating") {
        if (place === "canteen" && A.canteen) add("canteenSeat", sp);
        else if (place === "market" && A.market) add("marketSeat", sp);
        else add("picnic", sp);
        continue;
      }
      if (act === "buying") {
        if (place === "market" && A.market) add("marketBuy", sp);
        else if (place === "canteen" && A.canteen) add("canteenQueue", sp);
        else add("cell", sp);
        continue;
      }
      if (act === "working") {
        const j = place || sp.job;
        if (place === "canteen" && A.canteen) add("canteenStaff", sp);
        else if (place === "farm" && A.farm) add("farm", sp);
        else if (place === "orchard" && A.orchard) add("orchard", sp);
        else if (place === "dairy" && A.dairy) add("dairy", sp);
        else if (place === "warehouse" && A.warehouse) add("warehouse", sp);
        else { sp.pose = "work"; add("cell", sp); }
        void j; continue;
      }
      if (act === "idle" && place) {
        if (place === "canteen" && A.canteen) add("canteenWait", sp);
        else if (place === "market") add("cell", sp);
        else if (place === "school") add("school", sp);
        else add("rest", sp);
        continue;
      }
      if (act === "nursing") { sp.pose = "baby"; add("cell", sp); continue; }
      add("cell", sp);
    }
    // place each group
    const seatAssign = (people, seats, fn) => {
      const used = new Set();
      people.slice().sort((a, b) => a.pid - b.pid).forEach((sp) => {
        if (!seats.length) return;
        let i = Math.floor(hash(sp.pid, 11) * seats.length), n = 0;
        while (used.has(i) && n < seats.length) { i = (i + 1) % seats.length; n++; }
        if (n >= seats.length) { fn(sp, null); return; }
        used.add(i); fn(sp, seats[i], i);
      });
    };
    for (const [key, g] of groups) {
      switch (key) {
        case "canteenSeat": seatAssign(g, A.canteen.seats, (sp, st) => { if (!st) return this.cellSpot(sp, "picnic"); Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: "eat", seat: st.seat, props: { bowl: true } }); }); break;
        case "marketSeat": seatAssign(g, A.market.seats, (sp, st) => { if (!st) return this.cellSpot(sp, "picnic"); Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: "eat", seat: st.seat, props: { bowl: true } }); }); break;
        case "marketBuy": seatAssign(g, A.market.buy, (sp, st, i) => { if (!st) return this.cellSpot(sp, "stand"); Object.assign(sp, { x: st.x + (hash(sp.pid, 3) - 0.5) * 0.08, z: st.z, yaw: st.yaw, pose: "buy" }); }); break;
        case "canteenQueue": seatAssign(g, A.canteen.queue, (sp, st) => { if (!st) return this.cellSpot(sp, "stand"); Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: "buy" }); }); break;
        case "canteenStaff": seatAssign(g, A.canteen.staff, (sp, st) => { if (!st) return this.cellSpot(sp, "serve"); Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: "serve", props: { tray: true } }); }); break;
        case "canteenWait": seatAssign(g, A.canteen.benches.concat(A.canteen.queue), (sp, st) => { if (!st) return this.cellSpot(sp, "stand"); Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: st.seat ? "sit" : "stand", seat: st.seat }); }); break;
        case "jail": seatAssign(g, A.station.jail, (sp, st) => { if (!st) return this.cellSpot(sp, "stand"); Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: "jail", seat: st.seat }); }); break;
        case "school": {
          const s0 = (cfg.school_hours || [8, 15])[0];
          const recess = hour === 12 || hour === s0 + 2;
          const kids = g.filter((sp) => sp.act === "school" || sp.age < 16);
          const adults = g.filter((sp) => !kids.includes(sp));
          if (recess || g[0].act !== "school") seatAssign(kids, A.school.play, (sp, st) => { if (!st) return this.cellSpot(sp, "stand"); Object.assign(sp, { x: st.x, z: st.z, yaw: 0, pose: "play", play: true }); });
          else seatAssign(kids, A.school.desks, (sp, st) => { if (!st) return this.cellSpot(sp, "stand"); Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: "lesson", seat: st.seat }); });
          adults.forEach((sp) => { const T = A.school.teacher; Object.assign(sp, { x: T.x, z: T.z, yaw: T.yaw, pose: "talk" }); });
          break;
        }
        case "farm": {
          const F = A.farm, fld = F.field;
          g.forEach((sp) => {
            const c = w.cell(sp.cx, sp.cy);
            let x = clamp(c.x + (hash(sp.pid, 5) - 0.5) * 0.5, fld.x0, fld.x1), z = clamp(c.z + (hash(sp.pid, 6) - 0.5) * 0.3, fld.z0, fld.z1);
            if (x < F.barn.x1 && z < F.barn.z1) { if (F.barn.x1 - x < F.barn.z1 - z) x = F.barn.x1 + 0.2; else z = F.barn.z1 + 0.2; }
            z = fld.z0 + Math.round((z - fld.z0) / 0.34) * 0.34 + 0.17;    // between the crop rows
            Object.assign(sp, { x, z, yaw: hash(sp.pid, 7) < 0.5 ? Math.PI / 2 : -Math.PI / 2, pose: "hoe", props: { hoe: true } });
          });
          break;
        }
        case "orchard": {
          const T = A.orchard.trees, used = new Map();
          g.forEach((sp) => {
            const c = w.cell(sp.cx, sp.cy);
            let best = 0, bd = 1e9;
            T.forEach((tr, i) => { const d = Math.hypot(tr.x - c.x, tr.z - c.z) + (used.get(i) || 0) * 0.8; if (d < bd) { bd = d; best = i; } });
            const n = used.get(best) || 0; used.set(best, n + 1);
            const a = Math.PI * 0.5 + n * 2.1 + hash(sp.pid, 2) * 0.5, tr = T[best];
            const x = tr.x + Math.sin(a) * 0.42, z = tr.z + Math.cos(a) * 0.42;
            Object.assign(sp, { x, z, yaw: Math.atan2(tr.x - x, tr.z - z), pose: "pick", props: { basket: true } });
          });
          break;
        }
        case "dairy": {
          const D = A.dairy;
          seatAssign(g, D.stations, (sp, st, i) => {
            if (!st) { const c = w.cell(sp.cx, sp.cy); return Object.assign(sp, { x: c.x, z: c.z, pose: "carry", props: { pail: false } }); }
            Object.assign(sp, { x: st.x, z: st.z, yaw: st.yaw, pose: "milk", props: { pail: true }, milking: i });
          });
          break;
        }
        case "warehouse": {
          const Rt = A.warehouse.route;
          g.forEach((sp, i) => { Object.assign(sp, { x: Rt.a.x, z: Rt.a.z, pose: "carry", props: { crate: true }, route: Rt, routePh: hash(sp.pid, 9) + i * 0.37 }); });
          break;
        }
        case "social": this.socialCircles(g); break;
        case "lie": g.forEach((sp) => this.cellSpot(sp, "lie")); this.spread(g); break;
        case "picnic": g.forEach((sp) => this.cellSpot(sp, "picnic")); this.spread(g); break;
        case "rest": g.forEach((sp) => this.cellSpot(sp, "ground")); this.spread(g); break;
        default: g.forEach((sp) => this.cellSpot(sp, sp.pose === "work" || sp.pose === "baby" ? sp.pose : "stand")); this.spread(g);
      }
    }
    // mothers carry babies (only when their hands are free)
    for (const sp of list) if (sp.babies && !sp.hidden && ["stand", "buy", "talk", "walk"].includes(sp.pose)) sp.cradle = true;
    return { k, t, spots, house, list };
  }

  cellSpot(sp, pose) {
    const c = this.world.cell(sp.cx, sp.cy);
    sp.x = c.x; sp.z = c.z; sp.pose = pose;
    if (pose === "lie") sp.yaw = hash(sp.pid, 4) * 6.28;
  }

  // several people in one cell stand apart
  spread(g) {
    const byCell = new Map();
    g.forEach((sp) => { const key = sp.cx * 1000 + sp.cy; let a = byCell.get(key); if (!a) byCell.set(key, (a = [])); a.push(sp); });
    for (const a of byCell.values()) {
      a.sort((p, q) => p.pid - q.pid);
      const n = a.length;
      a.forEach((sp, i) => {
        if (n === 1) { sp.x += (hash(sp.pid, 21) - 0.5) * 0.2; sp.z += (hash(sp.pid, 22) - 0.5) * 0.2; return; }
        const ang = (i / n) * Math.PI * 2 + 0.6, r = sp.pose === "lie" ? 0.3 : 0.2 + 0.03 * n;
        sp.x += Math.cos(ang) * r; sp.z += Math.sin(ang) * r;
      });
    }
  }

  // people socialising near each other gather in circles, facing the middle
  socialCircles(g) {
    const n = g.length, parent = g.map((_, i) => i);
    const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (Math.max(Math.abs(g[i].cx - g[j].cx), Math.abs(g[i].cy - g[j].cy)) <= 2) parent[find(i)] = find(j);
    }
    const clusters = new Map();
    g.forEach((sp, i) => { const r = find(i); let c = clusters.get(r); if (!c) clusters.set(r, (c = [])); c.push(sp); });
    for (const c of clusters.values()) {
      c.sort((a, b) => a.pid - b.pid);
      let mx = 0, mz = 0;
      c.forEach((sp) => { const p = this.world.cell(sp.cx, sp.cy); mx += p.x; mz += p.z; });
      mx /= c.length; mz /= c.length;
      const r = c.length === 1 ? 0 : 0.2 + 0.07 * c.length, a0 = hash(c[0].pid, 31) * 6.28;
      c.forEach((sp, i) => {
        const a = a0 + (i / c.length) * Math.PI * 2;
        sp.x = mx + Math.cos(a) * r; sp.z = mz + Math.sin(a) * r;
        sp.yaw = c.length === 1 ? hash(sp.pid, 7) * 6.28 : Math.atan2(mx - sp.x, mz - sp.z);
        sp.pose = c.length === 1 ? "stand" : "talk";
      });
    }
  }

  // ---------------------------------------------------------------- interpolation
  // The display state of slot s at clock tau (absolute step, fractional).
  state(s, tau) {
    const rep = this.rep, k = rep.kOf(tau), f = clamp(tau - rep.tOf(k), 0, 1);
    const L1 = this.frame(k), b = L1 && L1.spots[s];
    const L0 = k > 0 ? this.frame(k - 1) : null, a0 = L0 && L0.spots[s];
    const a = a0 && b && a0.pid === b.pid ? a0 : null;
    if (!b) {
      // died (or the slot emptied) during this hour: fade out where they were
      if (a0) return { sp: a0, x: a0.x, z: a0.z, yaw: a0.yaw, pose: a0.pose, hidden: a0.hidden, fade: 1 - smoothstep(0.1, 0.8, f), dying: true };
      return null;
    }
    if (!a) return { sp: b, x: b.x, z: b.z, yaw: b.yaw, pose: b.pose, hidden: b.hidden, fade: k === 0 ? 1 : smoothstep(0.0, 0.5, f), born: k > 0 };
    const dist = Math.hypot(b.x - a.x, b.z - a.z) + (a.cx !== b.cx || a.cy !== b.cy ? 0.5 : 0);
    if (dist < 0.05 || (a.hidden && b.hidden && a.lot === b.lot) || b.carriedBy != null) {
      return { sp: b, x: b.x, z: b.z, yaw: b.yaw, pose: b.pose, hidden: b.hidden, fade: 1 };
    }
    // walk at the start of the hour, then do this hour's activity
    const key = `${k}:${s}`;
    let r = this._routeCache && this._routeCache.key === key ? this._routeCache.r : null;
    if (!r) {
      r = this.route(a, b);
      if (!this._routes) this._routes = new Map();
      const hit = this._routes.get(key);
      if (hit) r = hit; else { this._routes.set(key, r); if (this._routes.size > 800) this._routes.delete(this._routes.keys().next().value); }
    }
    const walking = b.act === "walking";
    const d = walking ? clamp(r.total / 14, 0.25, 1) : clamp(r.total / 24 + 0.04, 0.12, 0.45);
    if (f >= d) return { sp: b, x: b.x, z: b.z, yaw: b.yaw, pose: b.pose, hidden: b.hidden, fade: 1, arrived: true };
    const u = f / d, p = Layout.along(r, u * r.total);
    return { sp: b, x: p.x, z: p.z, yaw: Math.atan2(p.dx, p.dz), pose: "walk", hidden: false, fade: 1, speed: r.total / d, moving: true, route: r, u };
  }
}

function smoothstep(e0, e1, x) { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
