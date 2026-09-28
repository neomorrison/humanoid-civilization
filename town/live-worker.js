// Live sandbox worker: runs the real simulator (civ/sandbox.py, numpy only) under Pyodide and hands
// the viewer frames in the docs/TOWN_REPLAY.md format. A classic worker, so it can importScripts
// Pyodide either from the copy vendored next to the page or from the CDN.
//
// In:  {id, op: "start", base, seed} | {id, op: "run", hours} | {id, op: "kill", pid}
//      {id, op: "spawn", sex, age, evil} | {id, op: "rule", key, value}
// Out: {op: "progress", text, frac} | {id, ok: true, ...result} | {id, ok: false, error}
"use strict";

const PYODIDE_VERSION = "0.29.5";
const LOCAL = new URL("../vendor/pyodide/", self.location.href).href;
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const HOME = "/home/pyodide";

let py = null, fn = null, source = "";

const progress = (text, frac) => postMessage({ op: "progress", text, frac });

async function bootPython() {
  let last = null;
  for (const url of [LOCAL, CDN]) {
    try {
      if (url === LOCAL) {     // vendored copy present? (importScripts would only say "NetworkError")
        const r = await fetch(url + "pyodide-lock.json");
        if (!r.ok) throw new Error(`no vendored Pyodide (HTTP ${r.status})`);
      }
      progress(`Downloading Python for the browser (Pyodide ${PYODIDE_VERSION}, about 15 MB, cached after the first visit)…`, 0.1);
      importScripts(url + "pyodide.js");
      py = await self.loadPyodide({ indexURL: url });
      source = url === LOCAL ? "vendored" : "CDN";
      return;
    } catch (err) { last = err; console.warn(`Pyodide from ${url} failed:`, err); }
  }
  throw new Error(`could not start Python in the browser: ${(last && last.message) || last}`);
}

async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return kind === "bytes" ? new Uint8Array(await r.arrayBuffer()) : kind === "json" ? r.json() : r.text();
}

// the few lines of Python between the worker and civ/sandbox.py; everything crosses as JSON text
const GLUE = `
import json, sys
sys.path.insert(0, "${HOME}")
from civ.sandbox import Sandbox
from civ.town import TownConfig

sb = None

def start(town, evil, seed):
    global sb
    # the era never runs out while you watch; the town only ends if everybody dies
    sb = Sandbox(town, evil_policy=evil or None, seed=int(seed), cfg=TownConfig(episode_years=10000.0))
    return json.dumps(dict(header=sb.header(), chunk=sb.advance(1), rules=sb.rules()), separators=(",", ":"))

def run(hours):
    return json.dumps(sb.advance(int(hours)), separators=(",", ":"))

def kill(pid):
    return json.dumps(dict(done=sb.kill(int(pid))))

def spawn(args):
    a = json.loads(args)
    sex = None if a.get("sex") in (None, "") else int(a["sex"])
    return json.dumps(dict(pid=sb.spawn(sex=sex, age=float(a.get("age", 25)), evil=bool(a.get("evil")))))

def rule(key, value):
    ok = sb.set_rule(key, json.loads(value))
    return json.dumps(dict(done=ok, rules=sb.rules()))
`;

async function start({ base, seed }) {
  if (!py) {
    await bootPython();
    progress("Loading numpy…", 0.55);
    await py.loadPackage("numpy", { messageCallback: () => {}, errorCallback: (m) => console.warn(m) });
  }
  progress("Loading the simulator and the trained policies…", 0.75);
  const man = await fetchOk(base + "manifest.json", "json");
  py.FS.mkdirTree(`${HOME}/civ`); py.FS.mkdirTree(`${HOME}/live`);
  for (const f of man.civ) py.FS.writeFile(`${HOME}/civ/${f}`, await fetchOk(base + "civ/" + f, "text"));
  const town = `${HOME}/live/${man.town}`;
  py.FS.writeFile(town, await fetchOk(base + man.town, "bytes"));
  let evil = "";
  if (man.evil) {
    try { py.FS.writeFile(`${HOME}/live/${man.evil}`, await fetchOk(base + man.evil, "bytes")); evil = `${HOME}/live/${man.evil}`; }
    catch (err) { console.warn("no harm-seeker policy:", err); }
  }
  progress("Founding the town…", 0.9);
  if (!fn) {
    py.runPython(GLUE);
    fn = Object.fromEntries(["start", "run", "kill", "spawn", "rule"].map((k) => [k, py.globals.get(k)]));
  }
  const t0 = performance.now();
  const out = JSON.parse(fn.start(town, evil, seed));
  return { ...out, ms: performance.now() - t0, python: py.runPython("import sys; sys.version.split()[0]"), pyodide: PYODIDE_VERSION, source };
}

const OPS = {
  start,
  run: ({ hours }) => { const t0 = performance.now(); const chunk = JSON.parse(fn.run(hours)); return { chunk, hours: chunk.frames.length, ms: performance.now() - t0 }; },
  kill: ({ pid }) => JSON.parse(fn.kill(pid)),
  spawn: (a) => JSON.parse(fn.spawn(JSON.stringify(a))),
  rule: ({ key, value }) => JSON.parse(fn.rule(key, JSON.stringify(value))),
};

// one message at a time, in order: an intervention lands between two runs, never inside one
let queue = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  queue = queue.then(async () => {
    try {
      if (!OPS[m.op]) throw new Error(`unknown op ${m.op}`);
      if (m.op !== "start" && !fn) throw new Error("the town has not started yet");
      postMessage({ id: m.id, ok: true, ...(await OPS[m.op](m)) });
    } catch (err) {
      postMessage({ id: m.id, ok: false, error: String((err && err.message) || err) });
    }
  });
};
