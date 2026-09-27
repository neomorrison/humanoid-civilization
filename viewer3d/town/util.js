// Small helpers shared by the town viewer modules.
import * as THREE from "three";

export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
export const fract = (x) => x - Math.floor(x);

// deterministic pseudo-random numbers
export function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function hash(...n) {
  let h = 2166136261 >>> 0;
  for (const x of n) { h ^= (x | 0) & 0xffff; h = Math.imul(h, 16777619); h ^= (x | 0) >>> 16; h = Math.imul(h, 16777619); }
  h ^= h >>> 13; h = Math.imul(h, 0x5bd1e995); h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}
export const pick = (arr, u) => arr[Math.min(arr.length - 1, Math.floor(u * arr.length))];

export const col = (hex) => new THREE.Color(hex);
export function mixHex(a, b, t) { return new THREE.Color(a).lerp(new THREE.Color(b), t); }
export const hexStr = (c) => "#" + (typeof c === "number" ? c : c.getHex()).toString(16).padStart(6, "0");

// 1D value noise for terrain
export function noise2(x, y, seed = 0) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const s = (t) => t * t * (3 - 2 * t);
  const h = (i, j) => hash(i + 1013 * seed, j * 7 + 31);
  const a = h(xi, yi), b = h(xi + 1, yi), c = h(xi, yi + 1), d = h(xi + 1, yi + 1);
  return lerp(lerp(a, b, s(xf)), lerp(c, d, s(xf)), s(yf));
}
export function fbm(x, y, seed = 0) {
  let v = 0, amp = 0.5, f = 1;
  for (let k = 0; k < 4; k++) { v += amp * noise2(x * f, y * f, seed + k); f *= 2.03; amp *= 0.5; }
  return v;
}

// text drawn on a canvas, for signs
export function textCanvas(lines, { w = 512, h = 128, bg = "#6b4a2e", fg = "#fff8e8", font = "800 64px Archivo, Arial, sans-serif", border = "#3e2a19", radius = 18, sub = null } = {}) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  if (bg) {
    g.fillStyle = border; g.beginPath(); g.roundRect(0, 0, w, h, radius); g.fill();
    g.fillStyle = bg; g.beginPath(); g.roundRect(8, 8, w - 16, h - 16, radius - 6); g.fill();
    // wood grain
    g.globalAlpha = 0.08; g.fillStyle = "#000";
    for (let y = 14; y < h - 12; y += 11) g.fillRect(14, y, w - 28, 2);
    g.globalAlpha = 1;
  }
  g.fillStyle = fg; g.textAlign = "center"; g.textBaseline = "middle"; g.font = font;
  if (sub) {
    g.fillText(lines, w / 2, h * 0.42);
    g.font = "600 26px Archivo, Arial, sans-serif"; g.globalAlpha = 0.85; g.fillText(sub, w / 2, h * 0.78); g.globalAlpha = 1;
  } else g.fillText(lines, w / 2, h / 2 + 3);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  return tex;
}

// fetch JSON with a progress callback (bytes received, total or 0)
export function fetchJSON(url, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("GET", url);
    x.responseType = "text";
    x.onprogress = (e) => { if (onProgress) onProgress(e.loaded, e.lengthComputable ? e.total : 0); };
    x.onload = () => {
      if (x.status < 200 || x.status >= 300) { reject(new Error(`HTTP ${x.status}`)); return; }
      try { resolve(JSON.parse(x.responseText)); } catch (err) { reject(err); }
    };
    x.onerror = () => reject(new Error("network error"));
    x.send();
  });
}

export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
