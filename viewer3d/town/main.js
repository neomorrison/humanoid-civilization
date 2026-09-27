// Town Life replay viewer: boot, playback loop, people, overlays, camera.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Replay, JOB_COLORS, QUIET } from "./replay.js";
import { World } from "./world.js";
import { Layout } from "./layout.js";
import { Human } from "./people.js";
import { UI, doing } from "./ui.js";
import { Live } from "./live.js";
import { clamp, lerp, fetchJSON, hash, esc, hexStr } from "./util.js";

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);

// show errors on the page instead of failing silently
function reportError(err) {
  console.error(err);
  const l = $("ltext"); if (!l) return;
  $("loading").hidden = false; l.classList.add("error");
  l.textContent = "Something went wrong: " + ((err && err.message) || err) + (Q.has("debug") && err && err.stack ? " @ " + err.stack.split("\n").slice(0, 4).join(" | ") : "");
}
window.addEventListener("error", (e) => reportError(e.error || e.message));
window.addEventListener("unhandledrejection", (e) => reportError(e.reason));

// ------------------------------------------------------------------ theme
const THEMES = ["auto", "light", "dark"];
let theme = Q.get("theme") || (() => { try { return localStorage.getItem("town-theme") || "auto"; } catch { return "auto"; } })();
function applyTheme() {
  if (theme === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = theme;
  $("theme").textContent = `Theme: ${theme}`;
  app.themeKey = (app.themeKey || 0) + 1;
}
$("theme").addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  try { localStorage.setItem("town-theme", theme); } catch { /* storage blocked */ }
  applyTheme(); app.ui && app.ui.panels(true);
});

// ------------------------------------------------------------------ renderer and scene
const view = $("view"), overlay = $("overlay");
const small = Math.min(window.innerWidth, window.innerHeight) < 700;
const quality = Q.has("q") ? +Q.get("q") : small ? 0.6 : 1;
const renderer = new THREE.WebGLRenderer({ antialias: !small, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(small ? 1.5 : 2, window.devicePixelRatio || 1));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.08;
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 16 / 10, 0.3, 700);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.maxPolarAngle = Math.PI * 0.44; controls.minDistance = 2.5; controls.maxDistance = 90;
controls.screenSpacePanning = false;

const selRing = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.27, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x19c0cf, transparent: true, opacity: 0.9, depthWrite: false }));
selRing.visible = false; selRing.renderOrder = 3; scene.add(selRing);
// harm-seekers (live sandbox): a red ring at their feet and a red marker over their head
const evilGeo = { ring: new THREE.RingGeometry(0.22, 0.31, 32).rotateX(-Math.PI / 2), mark: new THREE.OctahedronGeometry(0.075).scale(1, 1.5, 1) };
const evilMat = { ring: new THREE.MeshBasicMaterial({ color: 0xff2a1f, transparent: true, opacity: 0.9, depthWrite: false }), mark: new THREE.MeshBasicMaterial({ color: 0xff2a1f }) };
const evilMarks = new Map();
function evilMark(pid) {
  let m = evilMarks.get(pid);
  if (!m) {
    m = { ring: new THREE.Mesh(evilGeo.ring, evilMat.ring), mark: new THREE.Mesh(evilGeo.mark, evilMat.mark) };
    m.ring.renderOrder = 3; scene.add(m.ring); scene.add(m.mark); evilMarks.set(pid, m);
  }
  return m;
}

// ------------------------------------------------------------------ app state
const app = {
  index: null, stageIdx: 0, rep: null, world: null, layout: null, ui: null,
  tau: 0, k: 0, playing: false, speed: 1, sel: -1, following: false, labels: Q.get("labels") !== "0",
  humans: new Map(), canteenOpen: false, shopOpen: false, themeKey: 0, live: null,
  select(pid, { follow = false } = {}) {
    app.sel = pid;
    if (pid < 0) app.following = false; else if (follow) app.following = true;
    syncFollowBtn(); app.ui.panels(true);
  },
  setFollow(on) {
    if (on && app.sel < 0) { const L = app.layout.frame(app.k); const c = L.list.find((sp) => sp.job !== "none" && sp.age >= 16 && !sp.hidden) || L.list.find((sp) => sp.age >= 16) || L.list[0]; if (c) app.sel = c.pid; }
    app.following = on && app.sel >= 0; syncFollowBtn(); app.ui.panels(true);
  },
  seek(tau) {
    const rep = app.rep; if (!rep) return;
    app.tau = clamp(tau, rep.t0, rep.tEnd - 0.001); app.lastK = -2;
    floaters.forEach((f) => f.el.remove()); floaters.length = 0;
  },
};
applyTheme();
function syncFollowBtn() {
  const b = $("follow");
  b.setAttribute("aria-pressed", app.following ? "true" : "false");
  b.textContent = app.following && app.rep ? `Following ${app.rep.first(app.sel)}` : "Follow someone";
}
$("follow").addEventListener("click", () => app.setFollow(!app.following));
$("labels").addEventListener("click", () => { app.labels = !app.labels; $("labels").setAttribute("aria-pressed", app.labels ? "true" : "false"); });
$("labels").setAttribute("aria-pressed", app.labels ? "true" : "false");

// ------------------------------------------------------------------ playback controls
const SPEEDS = () => {
  const r = app.rep, day = r ? r.daySteps : 24, yr = r ? r.yearSteps : 96;
  return [["¼ h/s", 0.25], ["1 h/s", 1], ["3 h/s", 3], ["1 day / 2 s", day / 2], ["1 year / 4 s", yr / 4]];
};
function buildSpeeds() {
  const el = $("speed"); el.innerHTML = "";
  SPEEDS().forEach(([label, v]) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.dataset.v = v;
    b.setAttribute("aria-pressed", Math.abs(v - app.speed) < 1e-6 ? "true" : "false");
    b.addEventListener("click", () => { app.speed = v; el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false")); });
    el.appendChild(b);
  });
}
function setPlaying(p) { app.playing = p; $("play").textContent = p ? "Pause" : app.live ? "Resume" : (app.rep && app.tau >= app.rep.tEnd - 0.01 ? "Replay" : "Play"); }
app.setPlaying = setPlaying;
$("play").addEventListener("click", () => { if (!app.rep) return; if (!app.live && app.tau >= app.rep.tEnd - 0.01) app.seek(app.rep.t0); setPlaying(!app.playing); });
$("jumps").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-h]"); if (!b || !app.rep) return;
  const h = +b.dataset.h, ds = app.rep.daySteps, cur = Math.floor(app.tau) + 1;
  let t = Math.floor(cur / ds) * ds + h; if (t < cur) t += ds;
  if (t >= app.rep.tEnd) t -= ds * Math.ceil((t - app.rep.tEnd + 1) / ds);
  app.seek(t + 0.34);     // twenty minutes in: people have arrived and are doing it
});
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea, button, canvas#tl")) return;
  if (e.key === " ") { e.preventDefault(); $("play").click(); }
});

// ------------------------------------------------------------------ camera
function fitTown(az = -0.3, el = 0.9) {
  const w = app.world, aspect = camera.aspect;
  const vf = THREE.MathUtils.degToRad(camera.fov), hf = 2 * Math.atan(Math.tan(vf / 2) * aspect);
  const dW = (w.W / 2 + 1.5) / Math.tan(hf / 2), dH = (w.H / 2 + 1.5) * Math.sin(el) / Math.tan(vf / 2) + w.H * 0.25;
  const d = Math.max(dW, dH) * 1.02, tz = w.H * 0.08;
  controls.target.set(0, 0, tz);
  camera.position.set(Math.sin(az) * Math.cos(el) * d, Math.sin(el) * d, tz + Math.cos(az) * Math.cos(el) * d);
  controls.update();
}
function camPreset(name) {
  const w = app.world, A = w.anchors;
  const lookAt = (x, z, d = 9, az = -0.35, el = 0.72) => { controls.target.set(x, 0, z); camera.position.set(x + Math.sin(az) * Math.cos(el) * d, Math.sin(el) * d, z + Math.cos(az) * Math.cos(el) * d); controls.update(); };
  if (name === "houses") {
    // the biggest cluster of houses (houses within 5 units of each other)
    const hs = A.houses.filter((h) => h.door);
    let best = hs[0], bn = 0;
    hs.forEach((h) => { const n = hs.filter((q) => Math.hypot(q.top.x - h.top.x, q.top.z - h.top.z) < 5).length; if (n > bn) { bn = n; best = h; } });
    const near = hs.filter((q) => Math.hypot(q.top.x - best.top.x, q.top.z - best.top.z) < 5);
    const x = near.reduce((a, h) => a + h.top.x, 0) / near.length, z = near.reduce((a, h) => a + h.top.z, 0) / near.length;
    return lookAt(x + 1, z + 0.5, 11, 0.25, 0.62);
  }
  if (A[name] && A[name].rect) return lookAt(A[name].rect.cx, A[name].rect.cz + 0.4, name === "farm" ? 13 : 8.5);
  const m = /^(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)(?:,(-?[\d.]+),(-?[\d.]+))?$/.exec(name || "");
  if (m) { controls.target.set(+m[1], 0, +m[2]); const d = +m[3], az = m[4] != null ? +m[4] : -0.35, el = m[5] != null ? +m[5] : 0.72; camera.position.set(+m[1] + Math.sin(az) * Math.cos(el) * d, Math.sin(el) * d, +m[2] + Math.cos(az) * Math.cos(el) * d); controls.update(); return; }
  fitTown();
}
function resize() {
  const r = view.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height); camera.updateProjectionMatrix();
  app.vw = r.width; app.vh = r.height;
}
new ResizeObserver(resize).observe(view);
let inView = true;
new IntersectionObserver((es) => { inView = es[0].isIntersecting; }).observe(view);

// picking people in the 3D view
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6 || !app.rep) return;
  const r = renderer.domElement.getBoundingClientRect();
  const ray = new THREE.Raycaster(); ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const proxies = []; app.humans.forEach((h) => { if (h.root.visible) proxies.push(h.proxy); });
  const hit = ray.intersectObjects(proxies, false)[0];
  if (hit) app.select(hit.object.userData.human.pid, { follow: true });
});
controls.addEventListener("start", () => { if (app.following) { app.userOrbit = performance.now(); } });

// ------------------------------------------------------------------ overlays
const tags = new Map(), htags = [], ptags = [], floaters = [];
const _v = new THREE.Vector3();
function project(x, y, z) {
  _v.set(x, y, z).project(camera);
  return { x: (_v.x * 0.5 + 0.5) * app.vw, y: (-_v.y * 0.5 + 0.5) * app.vh, ok: _v.z < 1 && _v.z > -1 };
}
function tagFor(pid) {
  let t = tags.get(pid);
  if (!t) {
    const el = document.createElement("div"); el.className = "tag";
    el.addEventListener("click", () => app.select(pid, { follow: true }));
    overlay.appendChild(el); t = { el, key: "", w: 60, h: 30 }; tags.set(pid, t);
  }
  return t;
}
function buildOverlays() {
  overlay.innerHTML = ""; tags.clear(); htags.length = 0; ptags.length = 0; floaters.length = 0;
  app.world.anchors.houses.forEach((A, h) => { const el = document.createElement("div"); el.className = "htag"; overlay.appendChild(el); htags.push({ el, h, key: "" }); });
  app.world.labels.forEach((L) => { const el = document.createElement("div"); el.className = "ptag" + (L.kind === "minor" ? " minor" : ""); el.textContent = L.text; overlay.appendChild(el); ptags.push({ el, L, key: "" }); });
}
const ICONS = {
  coin: (c) => `<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="${c}" stroke="#7a5a0a" stroke-width="1.5"/><text x="10" y="14" text-anchor="middle" font-size="10" font-weight="800" fill="#7a5a0a" font-family="Archivo,Arial">¢</text></svg>`,
  heart: `<svg viewBox="0 0 20 20"><path d="M10 17s-7-4.4-7-9.2A3.8 3.8 0 0 1 10 5.6a3.8 3.8 0 0 1 7 2.2C17 12.6 10 17 10 17z" fill="#ff6fa3" stroke="#fff" stroke-width="1.2"/></svg>`,
  crib: `<svg viewBox="0 0 20 20"><rect x="3" y="8" width="14" height="7" rx="2" fill="#ffd1e3" stroke="#fff" stroke-width="1.2"/><circle cx="7" cy="8" r="2.6" fill="#f7c9a0" stroke="#fff" stroke-width="1"/><path d="M3 15v2M17 15v2" stroke="#fff" stroke-width="1.5"/></svg>`,
  grave: `<svg viewBox="0 0 20 20"><path d="M5 18V8a5 5 0 0 1 10 0v10z" fill="#c9ccd1" stroke="#fff" stroke-width="1.2"/><path d="M10 8v6M7.5 10.5h5" stroke="#6b7278" stroke-width="1.5"/></svg>`,
  badge: `<svg viewBox="0 0 20 20"><path d="M10 2l6 3v5c0 4-3 6.5-6 8-3-1.5-6-4-6-8V5z" fill="#4a7de0" stroke="#fff" stroke-width="1.2"/><path d="M10 6l1.2 2.5 2.7.3-2 1.8.6 2.6L10 11.9l-2.5 1.3.6-2.6-2-1.8 2.7-.3z" fill="#ffe27a"/></svg>`,
  grain: `<svg viewBox="0 0 20 20"><path d="M6 18c0-6 1-9 4-12 3 3 4 6 4 12z" fill="#e8cf7a" stroke="#fff" stroke-width="1.2"/><path d="M10 7v10" stroke="#b08a14" stroke-width="1.2"/></svg>`,
  fruit: `<svg viewBox="0 0 20 20"><circle cx="10" cy="11.5" r="6" fill="#e0452f" stroke="#fff" stroke-width="1.2"/><path d="M10 6c0-2 1-3 3-3" stroke="#5a8a2a" stroke-width="1.6" fill="none"/></svg>`,
  dairy: `<svg viewBox="0 0 20 20"><path d="M7 3h6v3l2 3v9H5V9l2-3z" fill="#f7f7f2" stroke="#9aa3a8" stroke-width="1.2"/><rect x="6" y="11" width="8" height="4" fill="#7fb2e5"/></svg>`,
  zzz: `<svg viewBox="0 0 20 20"><text x="3" y="16" font-size="12" font-weight="800" fill="#dfe6ff" stroke="#2a3a6a" stroke-width="0.8" font-family="Archivo,Arial">z</text><text x="9" y="10" font-size="9" font-weight="800" fill="#dfe6ff" stroke="#2a3a6a" stroke-width="0.6" font-family="Archivo,Arial">z</text></svg>`,
  house: `<svg viewBox="0 0 20 20"><path d="M3 10l7-6 7 6v8H3z" fill="#f2d98a" stroke="#fff" stroke-width="1.2"/><rect x="8" y="12" width="4" height="6" fill="#8a5a35"/></svg>`,
  job: `<svg viewBox="0 0 20 20"><rect x="3" y="7" width="14" height="10" rx="2" fill="#c49a62" stroke="#fff" stroke-width="1.2"/><path d="M7 7V5h6v2" stroke="#fff" stroke-width="1.4" fill="none"/></svg>`,
  theft: `<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="#e0453a" stroke="#fff" stroke-width="1.2"/><path d="M6 6l8 8M14 6l-8 8" stroke="#fff" stroke-width="2"/></svg>`,
};
function floatAt(pos, html, cls = "", dur = 2.2) {
  if (floaters.length > 14) { const f = floaters.shift(); f.el.remove(); }
  const el = document.createElement("div"); el.className = "fl " + cls; el.innerHTML = html; overlay.appendChild(el);
  floaters.push({ el, pos: pos.clone(), t: 0, dur });
}
function eventFloaters(k) {
  const rep = app.rep, fast = app.speed > 6, perPid = new Map();
  for (const ei of rep.evByFrame[k]) {
    const e = rep.events[ei], kind = e[1];
    if (QUIET.has(kind)) continue;
    const big = kind === "born" || kind.startsWith("died") || kind === "partnered" || kind === "theft" || kind === "detained" || kind === "citizens' arrest" || kind === "expecting" || kind === "hired" || kind === "moved in" || kind === "grieves" || kind.startsWith("arrived");
    if (fast && !big) continue;
    const n = perPid.get(e[2]) || 0; if (n >= 1 && !big) continue; perPid.set(e[2], n + 1);
    const p = personPos(kind === "born" ? e[3] : e[2]);
    if (!p) continue;
    const amt = Math.round(e[4] * 10) / 10;
    let html = "";
    if (kind === "earned") html = ICONS.coin("#ffd24a") + `<span style="color:#ffe27a">+${amt}</span>`;
    else if (kind === "docked") html = ICONS.coin("#ff8a7a") + `<span style="color:#ff9d90">−${amt} docked</span>`;
    else if (kind.startsWith("bought ")) html = (ICONS[kind.slice(7)] || ICONS.grain) + `<span>${amt}¢</span>`;
    else if (kind.startsWith("gave ")) html = (ICONS[kind.slice(5)] || ICONS.coin("#ffd24a")) + `<span>shared</span>`;
    else if (kind === "partnered") html = ICONS.heart + `<span style="color:#ffc2da">${esc(rep.first(e[2]))} ♥ ${esc(rep.first(e[3]))}</span>`;
    else if (kind === "expecting") html = ICONS.heart + `<span style="color:#ffc2da">expecting</span>`;
    else if (kind === "born") html = ICONS.crib + `<span style="color:#ffd6e6">${esc(rep.first(e[2]))} is born</span>`;
    else if (kind.startsWith("died")) html = ICONS.grave + `<span style="color:#e6e9ec">${esc(rep.first(e[2]))} ${kind === "died (old age)" ? "passed away" : kind.slice(6, -1)}</span>`;
    else if (kind === "theft") html = ICONS.theft + `<span style="color:#ffb0a8">stole ${amt}¢</span>`;
    else if (kind === "detained" || kind === "citizens' arrest") html = ICONS.badge + `<span style="color:#bcd0ff">${kind === "detained" ? "arrested " : "citizens' arrest: "}${esc(rep.first(e[3]))}</span>`;
    else if (kind === "hired") html = ICONS.job + `<span>new job: ${esc(rep.jobs[e[4]] || "")}</span>`;
    else if (kind === "moved in") html = ICONS.house + `<span>moved in</span>`;
    else if (kind.startsWith("fell asleep")) html = ICONS.zzz + `<span>dozed off</span>`;
    else if (kind === "enlisted") html = ICONS.badge + `<span>joined police</span>`;
    else if (kind === "grieves") html = ICONS.grave + `<span style="color:#e6e9ec">grieves for ${esc(rep.first(e[3]))}</span>`;
    else if (kind === "arrived") html = ICONS.house + `<span>${esc(rep.first(e[2]))} arrived</span>`;
    else if (kind === "arrived (evil)") html = ICONS.theft + `<span style="color:#ffb0a8">${esc(rep.first(e[2]))} arrived: harm-seeker</span>`;
    else continue;
    floatAt(p, html, big ? "big" : "", big ? 3.2 : 2.0);
  }
}
function personPos(pid) {
  const h = app.humans.get(pid);
  if (h && h.root.visible) return h.root.position.clone().setY(h.height + 0.3);
  const rep = app.rep, s = rep.pidSlot(app.k, pid);
  const sp = s >= 0 ? app.layout.frame(app.k).spots[s] : null;
  if (sp && sp.house != null) return app.world.anchors.houses[sp.house].top.clone();
  if (sp) return new THREE.Vector3(sp.x, 0.9, sp.z);
  return null;
}

function updateOverlays(dt) {
  const rep = app.rep, L = app.layout.frame(app.k), camD = camera.position.distanceTo(controls.target);
  const far = camD > 26, showTags = app.labels;
  const placed = [];
  const fits = (x, y, w, h) => { for (const r of placed) if (x < r.x + r.w && x + w > r.x && y < r.y + r.h && y + h > r.y) return false; return true; };
  // person tags, nearest first, the selected person always
  const cand = [];
  if (showTags) {
    app.humans.forEach((h, pid) => {
      if (!h.root.visible || h.opacity < 0.3) return;
      const p = h.root.position, top = h.cur.lie > 0.5 ? 0.35 : h.height + 0.12;
      const q = project(p.x, top, p.z);
      if (!q.ok || q.x < -40 || q.y < -20 || q.x > app.vw + 40 || q.y > app.vh + 40) return;
      cand.push({ pid, h, q, d: camera.position.distanceToSquared(p) - (pid === app.sel ? 1e6 : 0) - (rep.isEvil(pid) ? 5e5 : 0) });
    });
    cand.sort((a, b) => a.d - b.d);
  }
  const used = new Set();
  let shown = 0;
  const maxTags = app.vw < 500 ? 10 : 22;
  for (const c of cand) {
    const t = tagFor(c.pid), sel = c.pid === app.sel;
    if (!sel && shown >= maxTags) continue;
    const s = rep.pidSlot(app.k, c.pid), row = s >= 0 ? rep.row(app.k, s) : null;
    const sp = s >= 0 ? L.spots[s] : null;
    const hunger = rep.get(row, "hunger"), energy = rep.get(row, "energy");
    const age = sp ? Math.floor(sp.age) : 0, job = sp ? sp.job : "none";
    const compact = far && !sel;
    const fl = sp ? sp.flags : 0, held = sp && (sp.act === "detained" || fl & 2), evil = rep.isEvil(c.pid);
    const key = `${rep.first(c.pid)}|${age}|${Math.round(hunger / 5)}|${Math.round(energy / 5)}|${job}|${sel}|${compact}|${fl & 23}|${held}|${evil}`;
    if (key !== t.key) {
      t.key = key;
      const notes = [evil ? '<b class="evil">harm-seeker</b>' : "", fl & 4 ? "police" : "", fl & 1 ? "expecting" : "", held ? '<b class="held">held</b>' : fl & 16 ? '<b class="held">thief</b>' : ""].filter(Boolean).join(" · ");
      t.el.innerHTML = `${esc(rep.first(c.pid))}<small>${age}${notes ? " · " + notes : ""}</small><span class="b2"><i style="--v:${hunger}%;--c:${hunger > 85 ? "#ff6b5e" : "#f0a050"}" title="hunger"></i><i style="--v:${energy}%;--c:#6aa6e6" title="energy"></i></span>`;
      t.el.style.borderLeftColor = evil ? "#ff3b30" : hexStr(JOB_COLORS[job] ?? 0x999999);
      t.el.classList.toggle("sel", sel); t.el.classList.toggle("far", compact); t.el.classList.toggle("evil", evil);
      t.w = t.el.offsetWidth || 60; t.h = t.el.offsetHeight || 28;
    }
    let x = c.q.x, y = c.q.y, ok = false;
    for (let tries = 0; tries < 3; tries++) { if (fits(x - t.w / 2, y - t.h, t.w, t.h) || sel) { ok = true; break; } y -= t.h + 2; }
    if (!ok) continue;
    placed.push({ x: x - t.w / 2, y: y - t.h, w: t.w, h: t.h });
    t.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -100%)`;
    t.el.style.display = "";
    used.add(c.pid); shown++;
  }
  tags.forEach((t, pid) => { if (!used.has(pid)) t.el.style.display = "none"; });
  // houses: who is inside (filled = awake, faded = asleep), who lives there but is out (hollow)
  const residents = [];
  for (const sp of L.list) if (sp.home >= 0) (residents[sp.home] = residents[sp.home] || []).push(sp);
  htags.forEach((ht) => {
    const A = app.world.anchors.houses[ht.h], res = residents[ht.h] || [];
    const q = project(A.top.x, A.top.y + 0.15, A.top.z);
    if (!showTags || !res.length || !q.ok) { ht.el.style.display = "none"; return; }
    const occ = L.house[ht.h] || { pids: [] };
    const key = res.map((sp) => `${sp.pid}:${sp.info.type === "house" && sp.info.h === ht.h ? (sp.act === "sleeping" ? 2 : 1) : 0}`).join(",") + (far ? "f" : "");
    if (key !== ht.key) {
      ht.key = key;
      const sur = mostCommon(res.map((sp) => rep.sur(sp.pid)));
      const inside = res.filter((sp) => sp.info.type === "house" && sp.info.h === ht.h);
      const asleep = inside.filter((sp) => sp.act === "sleeping").length;
      ht.el.innerHTML = `${far ? "" : `<span>${esc(sur)}</span>`}` + res.slice(0, 6).map((sp) => {
        const inHere = sp.info.type === "house" && sp.info.h === ht.h, zz = inHere && sp.act === "sleeping";
        return `<i class="d ${inHere ? "" : "out"} ${zz ? "zz" : ""}${rep.isEvil(sp.pid) ? " ev" : ""}" style="background:${hexStr(rep.line[sp.pid] ?? 0x999999)}" title="${esc(rep.nm(sp.pid))}: ${inHere ? (zz ? "asleep" : "home") : "out"}"></i>`;
      }).join("") + (asleep ? `<em>z</em>` : "");
      ht.el.title = `House ${ht.h + 1}: ${inside.length} of ${res.length} home${asleep ? `, ${asleep} asleep` : ""}`;
      ht.w = ht.el.offsetWidth || 40; ht.hh = ht.el.offsetHeight || 16;
      void occ;
    }
    const x = q.x, y = q.y;
    if (!fits(x - ht.w / 2, y - ht.hh, ht.w, ht.hh)) { ht.el.style.display = "none"; return; }
    placed.push({ x: x - ht.w / 2, y: y - ht.hh, w: ht.w, h: ht.hh });
    ht.el.style.display = ""; ht.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -100%)`;
  });
  // place labels (fade in when zoomed out)
  const pa = clamp((camD - 11) / 8, 0, 1);
  ptags.forEach((pt) => {
    const q = project(pt.L.pos.x, pt.L.pos.y, pt.L.pos.z);
    if (pa <= 0.02 || !q.ok) { pt.el.style.display = "none"; return; }
    let extra = "";
    if (pt.L.text === "Canteen") extra = `<i class="${app.canteenOpen ? "open" : "closed"}">${app.canteenOpen ? "open" : "closed"}</i>`;
    if (pt.L.text === "Market") extra = `<i class="${app.shopOpen ? "open" : "closed"}">${app.shopOpen ? "shop open" : "closed"}</i>`;
    const key = pt.L.text + extra;
    if (key !== pt.key) { pt.key = key; pt.el.innerHTML = esc(pt.L.text) + extra; pt.w = 0; }
    const w = pt.w || (pt.w = pt.el.offsetWidth || 60), hh = 16;
    const clear = fits(q.x - w / 2, q.y - hh / 2, w, hh);
    pt.el.style.display = ""; pt.el.style.opacity = (pa * (clear ? 1 : 0.3)).toFixed(2);
    pt.el.style.transform = `translate(${q.x.toFixed(1)}px, ${q.y.toFixed(1)}px) translate(-50%, -50%)`;
  });
  // floaters rise and fade
  for (let i = floaters.length - 1; i >= 0; i--) {
    const f = floaters[i]; f.t += dt;
    const u = f.t / f.dur;
    if (u >= 1) { f.el.remove(); floaters.splice(i, 1); continue; }
    const q = project(f.pos.x, f.pos.y + u * 0.7, f.pos.z);
    f.el.style.display = q.ok ? "" : "none";
    f.el.style.opacity = (u < 0.1 ? u / 0.1 : 1 - Math.max(0, (u - 0.6) / 0.4)).toFixed(2);
    f.el.style.transform = `translate(${q.x.toFixed(1)}px, ${(q.y - 8).toFixed(1)}px) translate(-50%, -100%)`;
  }
}
function mostCommon(a) { const m = new Map(); let best = a[0] || "", n = 0; a.forEach((x) => { const c = (m.get(x) || 0) + 1; m.set(x, c); if (c > n) { n = c; best = x; } }); return best; }

// ------------------------------------------------------------------ people
function human(pid) {
  let h = app.humans.get(pid);
  if (!h) {
    const P = app.rep.people[pid] || {};
    h = new Human(pid, P.sex ?? (pid % 2), 7);
    scene.add(h.root); app.humans.set(pid, h);
  }
  return h;
}
function updatePeople(dt, clk) {
  const rep = app.rep, L = app.layout, seen = new Set(), marked = new Set();
  for (let s = 0; s < rep.slots; s++) {
    const st = L.state(s, app.tau);
    if (!st) continue;
    const sp = st.sp, h = human(sp.pid);
    seen.add(sp.pid);
    const vis = !st.hidden && st.fade > 0.02;
    h.root.visible = vis;
    if (!vis) { h.lastX = null; continue; }
    h.setJob(sp.job, !!(sp.flags & 4)); h.setAge(sp.age); h.setPregnant(!!(sp.flags & 1) && sp.age >= 14);
    let x = st.x, z = st.z, yaw = st.yaw, pose = st.pose, carryMoving = false, props = st.moving ? {} : (sp.props || {});
    if (!st.moving && sp.route && pose === "carry") {
      // warehouse: carry crates from the shed to the truck and walk back
      const R = sp.route, per = 14, u = ((clk / per + sp.routePh) % 1 + 1) % 1;
      const leg = u < 0.42 ? u / 0.42 : u < 0.5 ? 1 : u < 0.92 ? 1 - (u - 0.5) / 0.42 : 0;
      x = lerp(R.a.x, R.b.x, leg) + (hash(sp.pid, 3) - 0.5) * 0.3; z = lerp(R.a.z, R.b.z, leg) + (hash(sp.pid, 4) - 0.5) * 0.3;
      const going = u < 0.42, back = u >= 0.5 && u < 0.92;
      carryMoving = going || back;
      yaw = Math.atan2((R.b.x - R.a.x) * (going ? 1 : -1), (R.b.z - R.a.z) * (going ? 1 : -1));
      props = { crate: u < 0.46 || u > 0.96 };
      if (!props.crate) pose = carryMoving ? "walk" : "stand";
    }
    if (!st.moving && sp.play) {
      const a = clk * 1.3 + hash(sp.pid, 8) * 6.28, r = 0.22;
      x += Math.cos(a) * r; z += Math.sin(a) * r; yaw = Math.atan2(-Math.sin(a), Math.cos(a));
    }
    // legs follow the ground actually covered
    if (h.lastX != null) {
      const d = Math.hypot(x - h.lastX, z - h.lastZ);
      if (d < 2) {
        const stride = 0.34 * (h.scale / 0.64 + 0.2);
        h.walkPh += Math.min(dt * 19, (d / stride) * Math.PI);
        if (st.moving && d / Math.max(dt, 1e-3) > 3.2) pose = "run";
      }
    }
    if (sp.play) h.walkPh += dt * 9;
    h.lastX = x; h.lastZ = z;
    const cradle = !!sp.cradle || (st.moving && sp.babies > 0);
    h.setProps({ ...props, baby: cradle, cot: !cradle && sp.babies > 0 && !st.moving && pose !== "lie" });
    h.pose(pose, clk, { seat: sp.seat, cradle, moving: carryMoving });
    h.root.position.set(x, 0, z); h.yawT = yaw;
    if (st.born && st.fade < 1 || st.dying) h.setOpacity(st.fade); else h.setOpacity(1);
    h.update(dt);
    if (sp.milking != null) app.world.setMilking(sp.milking, !st.moving);
    if (rep.isEvil(sp.pid)) {
      const m = evilMark(sp.pid), k = (h.scale / 0.64) * (1 + 0.1 * Math.sin(clk * 5)), o = h.opacity;
      m.ring.visible = m.mark.visible = o > 0.3; marked.add(sp.pid);
      m.ring.position.set(x, 0.025, z); m.ring.scale.set(k, 1, k);
      m.mark.position.set(x, (h.cur.lie > 0.5 ? 0.45 : h.height + 0.32) + 0.05 * Math.sin(clk * 3), z); m.mark.rotation.y = clk * 2;
    }
  }
  app.humans.forEach((h, pid) => { if (!seen.has(pid)) { h.root.visible = false; h.lastX = null; } });
  evilMarks.forEach((m, pid) => { if (!marked.has(pid)) m.ring.visible = m.mark.visible = false; });
  // selection ring
  const h = app.sel >= 0 ? app.humans.get(app.sel) : null;
  selRing.visible = !!(h && h.root.visible);
  if (selRing.visible) { selRing.position.set(h.root.position.x, 0.02, h.root.position.z); const s = (h.scale / 0.64) * (1 + 0.08 * Math.sin(clk * 4)); selRing.scale.set(s, 1, s); }
}

// ------------------------------------------------------------------ frame changes
function onFrame(k, prevK) {
  const rep = app.rep, w1 = rep.w(k), w0 = rep.w(k - 1), L = app.layout.frame(k);
  // w is end-of-hour state: what was open during this hour is last frame's state
  app.canteenOpen = !!(k > 0 ? w0.canteen_open : w1.canteen_open) || L.list.some((sp) => sp.act === "working" && sp.place === "canteen");
  const c = rep.clock(app.tau);
  const sh = rep.cfg.shop_hours;
  app.shopOpen = w1.shop_open != null ? !!(k > 0 ? w0.shop_open : w1.shop_open) : sh ? c.hour >= sh[0] && c.hour < sh[1] : true;
  const deaths = rep.events.slice(0, rep.eventsBefore(k)).reduce((n, e) => n + (e[1].startsWith("died") ? 1 : 0), 0);
  const sch = rep.cfg.school_hours || [8, 15];
  app.world.setTownState(w1, { canteenOpen: app.canteenOpen, shopOpen: app.shopOpen, hourF: c.hourF, deaths, schoolOn: c.hour >= sch[0] && c.hour < sch[1] });
  const occ = []; L.house.forEach((o, h) => { if (o) occ[h] = o; });
  app.world.setHouses(occ);
  if (prevK >= 0 && k > prevK && k - prevK <= 2) for (let kk = prevK + 1; kk <= k; kk++) eventFloaters(kk);
  // dairy stations free when nobody milks
  if (app.world.dyn.cows) app.world.dyn.cows.cows.forEach((cw) => { if (cw.st) cw.want = 0; });
}
const SKY_ICONS = {
  sun: '<circle cx="12" cy="12" r="5" fill="#f5b82e"/><g stroke="#f5b82e" stroke-width="2" stroke-linecap="round"><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/></g>',
  moon: '<path d="M16.5 15.5A7 7 0 0 1 9 4a8 8 0 1 0 11 10.5 7 7 0 0 1-3.5 1z" fill="#4a5a9a"/><circle cx="18" cy="5" r="1" fill="#4a5a9a"/><circle cx="21" cy="9" r=".8" fill="#4a5a9a"/>',
  dawn: '<path d="M5 16a7 7 0 0 1 14 0z" fill="#f08a3a"/><path d="M2 19h20" stroke="#c0602a" stroke-width="2" stroke-linecap="round"/><g stroke="#f08a3a" stroke-width="2" stroke-linecap="round"><path d="M12 4v3M4.5 8.5l2 2M19.5 8.5l-2 2"/></g>',
};
function updateClock() {
  const rep = app.rep, c = rep.clock(app.tau);
  $("clock").textContent = rep.label(app.tau);
  const icon = c.hourF >= 21 || c.hourF < 6 ? "moon" : c.hourF < 7.5 || c.hourF >= 19.5 ? "dawn" : "sun";
  if (icon !== app.skyIcon) { app.skyIcon = icon; $("skyIcon").innerHTML = SKY_ICONS[icon]; }
  if (app.k !== app.clockK) {
    app.clockK = app.k;
    const L = app.layout.frame(app.k);
    const n = { home: 0, work: 0, school: 0, eat: 0, out: 0 };
    for (const sp of L.list) {
      if (sp.info.type === "house") n.home++; else if (sp.act === "working") n.work++; else if (sp.act === "school") n.school++; else if (sp.act === "eating") n.eat++; else n.out++;
    }
    const part = [[n.home, "at home"], [n.work, "working"], [n.eat, "eating"], [n.school, "at school"], [n.out, "out and about"]].filter(([v]) => v).map(([v, t]) => `${v} ${t}`).join(" · ");
    $("clockSub").textContent = part;
  }
}

// ------------------------------------------------------------------ main loop
const clock = new THREE.Clock();
let clk = 0;
function loop() {
  requestAnimationFrame(loop);
  if (document.hidden) { clock.getDelta(); return; }
  const dt = Math.min(0.1, clock.getDelta());
  clk += dt;
  const rep = app.rep;
  if (!rep) return;
  if (app.playing) {
    app.tau += dt * app.speed;
    if (app.tau >= rep.tEnd - 0.001) { app.tau = rep.tEnd - 0.001; if (!app.live) setPlaying(false); }   // live: wait for the next hour
  }
  if (app.live) app.live.tick();
  const k = rep.kOf(app.tau);
  const c = rep.clock(app.tau);
  if (inView || !app.playing) app.world.setTime(c.hourF, c.day, (d) => rep.seasonOfDay(d), rep.seasons);
  if (k !== app.lastK) { onFrame(k, app.lastK); app.lastK = k; }
  app.k = k;
  if (!inView && app.playing) { app.ui.panels(); if (app.live) app.live.panels(); return; }
  updatePeople(dt, clk);
  app.world.tick(dt, clk, camera);
  // follow camera
  if (app.following && app.sel >= 0) {
    const h = app.humans.get(app.sel);
    let p = h && h.root.visible ? h.root.position : null;
    if (!p) { const q = personPos(app.sel); if (q) p = q.setY(0); }
    if (p) {
      const off = camera.position.clone().sub(controls.target);
      const want = new THREE.Vector3(p.x, 0, p.z);
      controls.target.lerp(want, 1 - Math.exp(-dt * 3));
      const dist = off.length(), goal = performance.now() - (app.userOrbit || 0) < 4000 ? dist : clamp(dist, 5, 11);
      off.setLength(lerp(dist, goal, 1 - Math.exp(-dt * 1.5)));
      camera.position.copy(controls.target).add(off);
    }
  }
  controls.update();
  renderer.render(scene, camera);
  updateOverlays(dt);
  updateClock();
  app.ui.timeline();
  app.ui.panels();
  if (app.live) app.live.panels();
}

// ------------------------------------------------------------------ loading
function loadingText(t, frac) { $("loading").hidden = false; $("ltext").textContent = t; $("lbar").style.width = `${Math.round(clamp(frac, 0, 1) * 100)}%`; }
function sameTown(a, b) { const k = (c) => JSON.stringify([c.width, c.height, c.roads, c.houses, c.places]); return a && b && k(a) === k(b); }
async function selectStage(k) {
  app.stageIdx = k;
  const s = app.index.stages[k];
  document.querySelectorAll("#stages .chip").forEach((c, i) => c.setAttribute("aria-pressed", i === k ? "true" : "false"));
  setPlaying(false);
  loadingText("Downloading the replay…", 0);
  let raw;
  try {
    raw = await fetchJSON("replays/" + s.file, (got, total) => loadingText(`Downloading the replay… ${(got / 1e6).toFixed(1)}${total && got <= total ? ` / ${(total / 1e6).toFixed(1)}` : ""} MB`, total && got <= total ? got / total : 0.5));
  } catch (err) { loadingText(`Could not load replays/${s.file}: ${err.message}`, 0); return; }
  loadingText("Building the town…", 1);
  await new Promise((r) => setTimeout(r, 30));
  const tp0 = performance.now();
  const rep = new Replay(raw);
  app.perf = { replay: performance.now() - tp0 };
  install(rep);
  app.ui.summary(rep, s); app.ui.charts(app.index, k); app.ui.rules(rep); app.ui.legend();
  app.seek(rep.t0);
  $("loading").hidden = true;
}
function install(rep) {
  app.humans.forEach((h) => scene.remove(h.root)); app.humans.clear();
  evilMarks.forEach((m) => { scene.remove(m.ring); scene.remove(m.mark); }); evilMarks.clear();
  if (!app.world || !sameTown(app.world.cfg, rep.cfg)) {
    if (app.world) scene.remove(app.world.root);
    const tw0 = performance.now();
    app.world = new World(scene, rep.cfg, { quality });
    if (app.perf) app.perf.world = performance.now() - tw0;
    fitTown();
  }
  app.rep = rep; app.layout = new Layout(rep, app.world);
  buildOverlays(); buildSpeeds(); app.ui.reset();
  app.sel = -1; app.following = false; syncFollowBtn();
}
app.install = install;

// the live sandbox: the simulator itself runs in this page (town/live.js)
async function bootLive() {
  resize();
  app.ui = new UI(app);
  $("lede").textContent = "This town is running live in your browser: the real Python simulator steps it hour by hour and the trained policy decides what each person does. Pause it, take someone away, bring in a newcomer or someone who only wants to cause harm, or change the rules, and watch how the town reacts.";
  $("jumps").hidden = true; $("chartsSec").hidden = true;
  loadingText("Starting the live town…", 0.03);
  const live = (app.live = new Live(app, Q.get("livebase") || "live/"));
  live.onProgress = (t, f) => loadingText(t, f);
  let rep;
  try { rep = await live.start(Q.has("seed") ? +Q.get("seed") : Math.floor(Math.random() * 1e6)); }
  catch (err) { loadingText(`The live town could not start: ${err.message}`, 0); $("ltext").classList.add("error"); return; }
  app.speed = Q.has("speed") ? +Q.get("speed") : 3;
  install(rep);
  app.ui.rules(rep); app.ui.legend(); live.mount();
  app.seek(rep.t0);
  $("loading").hidden = true;
  if (Q.has("cam")) camPreset(Q.get("cam"));
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  setPlaying(!reduce && Q.get("paused") !== "1");
  loop();
}

async function boot() {
  if (Q.has("live")) return bootLive();
  resize();
  app.ui = new UI(app);
  loadingText("Loading…", 0.05);
  try { app.index = await fetchJSON("replays/index.json"); }
  catch (err) { loadingText("No replays found next to this page (replays/index.json).", 0); $("ltext").classList.add("error"); return; }
  if (!app.index.stages || !app.index.stages.length) { loadingText("replays/index.json lists no stages yet.", 0); return; }
  const stage = Q.has("stage") ? clamp(+Q.get("stage"), 0, app.index.stages.length - 1) : app.index.stages.length - 1;
  app.ui.stages(app.index, stage, (k) => selectStage(k));
  fetch("live/manifest.json").then((r) => {     // offer the live sandbox when the site has it
    if (!r.ok) return;
    const a = document.createElement("a"); a.className = "chip link-chip"; a.href = "?live=1"; a.textContent = "Live sandbox: run it yourself";
    $("stages").appendChild(a);
  }).catch(() => {});
  if (Q.has("speed")) app.speed = +Q.get("speed");
  await selectStage(stage);
  if (!app.rep) return;
  if (Q.has("t")) app.seek(+Q.get("t"));
  if (Q.has("cam")) camPreset(Q.get("cam"));
  if (Q.has("sel")) app.select(+Q.get("sel"), { follow: Q.get("follow") === "1" });
  if (Q.has("perf")) setTimeout(() => {    // testing aid: cost of laying out random moments
    const t0 = performance.now(); let n = 0;
    for (let i = 0; i < 300; i++) { const tau = app.rep.t0 + Math.random() * (app.rep.tEnd - app.rep.t0); for (let s = 0; s < app.rep.slots; s++) { app.layout.state(s, tau); n++; } }
    const per = (performance.now() - t0) / 300;
    const t1 = performance.now(); for (let i = 0; i < 20; i++) { app.ui.lastPanels = -1; app.ui.panels(true); } const ui = (performance.now() - t1) / 20;
    const pre = document.createElement("pre"); pre.id = "perf";
    pre.textContent = JSON.stringify({ ...app.perf, layoutPerRandomFrameMs: +per.toFixed(2), panelsMs: +ui.toFixed(2), frames: app.rep.n, calls: renderer.info.render.calls, tris: renderer.info.render.triangles });
    document.body.appendChild(pre);
  }, 800);
  if (Q.has("dump")) setTimeout(() => {    // testing aid: the current frame's layout as text
    const pre = document.createElement("pre"); pre.id = "dump";
    const L = app.layout.frame(app.k);
    pre.textContent = JSON.stringify({ tau: app.tau, k: app.k, spots: L.list.map((sp) => ({ pid: sp.pid, act: sp.act, cell: [sp.cx, sp.cy], at: [+sp.x.toFixed(2), +sp.z.toFixed(2)], pose: sp.pose, hidden: sp.hidden, place: sp.place, state: (() => { const st = app.layout.state(sp.s, app.tau); return st ? { x: +st.x.toFixed(2), z: +st.z.toFixed(2), pose: st.pose, moving: !!st.moving } : null; })() })) });
    document.body.appendChild(pre);
  }, 500);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce && Q.get("paused") !== "1") setPlaying(true);
  loop();
}
boot();
window.__town = app;
