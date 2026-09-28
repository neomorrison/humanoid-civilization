// Live sandbox: the real simulator (civ/sandbox.py) runs in this browser under Pyodide, in a Web
// Worker (live-worker.js). Its hours are appended to a growing Replay that the viewer plays at the
// live edge, a few hours behind the simulation. Interventions (remove someone, add a newcomer or a
// harm-seeker, change a rule) land between two simulated hours.
import { Replay, evText, evClass } from "./replay.js";
import { esc, clamp } from "./util.js";

const $ = (id) => document.getElementById(id);
// replace an element's content only when it changed, so buttons under the pointer survive redraws
const setHTML = (el, html) => { if (el._html !== html) { el._html = html; el.innerHTML = html; } };
// what the short "Reactions" list shows: the sim's own responses to what happens in town
const REACT = (k) => k.startsWith("died") || k.startsWith("arrived") || k.startsWith("rule ") ||
  ["grieves", "theft", "detained", "citizens' arrest", "separated", "shunned", "born", "partnered"].includes(k);
const MAX_YEARS = 100;      // then the sandbox stops, to keep the page's memory in check
const RULE_UI = {
  wage: { label: "Wage", step: 0.1 }, dock: { label: "Docking", step: 0.25 }, lunch_break: { label: "Lunch break" }, police: { label: "Police" },
  gossip_rate: { label: "Gossip", step: 0.05 }, price_base: { label: "Food price", step: 0.5 }, export: { label: "Export pay", step: 0.1 },
  hunger_awake: { label: "Hunger rate", step: 0.005 }, conceive_prob: { label: "Conception", step: 0.005 },
};

export class Live {
  constructor(app, base) {
    this.app = app; this.base = new URL(base, location.href).href;
    this.seq = 0; this.calls = new Map(); this.busy = false; this.ended = false; this.failed = null;
    this.msPerHour = null; this.onProgress = null; this.wantSel = null; this.jumpTo = null; this.note = "";
    this.worker = new Worker(new URL("./live-worker.js", import.meta.url));
    this.worker.onmessage = (e) => this.message(e.data);
    this.worker.onerror = (e) => {
      const err = new Error(e.message || "the simulator worker failed to load");
      this.calls.forEach((c) => c.reject(err)); this.calls.clear(); this.fail(err);
    };
  }

  // ---------------------------------------------------------------- worker calls
  call(op, args = {}) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => { this.calls.set(id, { resolve, reject }); this.worker.postMessage({ id, op, ...args }); });
  }
  message(m) {
    if (m.op === "progress") { if (this.onProgress) this.onProgress(m.text, m.frac); return; }
    const c = this.calls.get(m.id); if (!c) return;
    this.calls.delete(m.id);
    if (m.ok) c.resolve(m); else c.reject(new Error(m.error));
  }
  fail(err) { console.error(err); this.failed = String((err && err.message) || err); this.panels(true); }

  // a new town; resolves to its Replay (the first hour already lived)
  async start(seed) {
    const r = await this.call("start", { base: this.base, seed });
    const h = (this.header = r.header);
    this.rules = r.rules; this.defaults = this.defaults || { ...r.rules };
    this.info = { python: r.python, pyodide: r.pyodide, source: r.source };
    this.seed = seed; this.ended = false; this.failed = null; this.busy = false; this.wantSel = null; this.jumpTo = null; this.note = "";
    this.counts = { born: 0, died: 0, theft: 0, arrest: 0 };
    const rep = new Replay({ kind: "town", version: 3, meta: (h.policies && h.policies.town) || {}, config: h.config, start_step: h.start_step, stride: 1,
      activities: h.activities, jobs: h.jobs, foods: h.foods, slot_cols: h.slot_cols, people: r.chunk.people, frames: r.chunk.frames, events: r.chunk.events, summary: {} });
    this.count(r.chunk.events);
    return rep;
  }

  // keep the simulation a little ahead of the playhead: about a third of a second of playback, so an
  // intervention shows almost at once (paused: the town waits too)
  tick() {
    const app = this.app, rep = app.rep;
    if (!rep || this.busy || this.ended || this.failed) return;
    if (rep.tEnd - rep.t0 >= MAX_YEARS * rep.yearSteps) { this.ended = "time"; this.panels(true); return; }
    const lead = 1.2 + app.speed * 0.35, ahead = rep.tEnd - app.tau, landing = this.jumpTo != null && rep.tEnd <= this.jumpTo;
    if (ahead >= lead && !landing) return;
    const hours = clamp(Math.ceil(lead - ahead), 1, 24);
    this.busy = true;
    this.call("run", { hours }).then((r) => {
      this.busy = false;
      if (r.hours) this.msPerHour = this.msPerHour == null ? r.ms / r.hours : this.msPerHour * 0.8 + 0.2 * (r.ms / r.hours);
      if (app.rep === rep) this.append(r.chunk);
    }).catch((err) => { this.busy = false; this.fail(err); });
  }
  append(chunk) {
    const app = this.app, rep = app.rep;
    this.count(chunk.events);
    rep.append(chunk);
    if (chunk.ended) { this.ended = true; this.panels(true); }
    if (this.jumpTo != null && rep.tEnd > this.jumpTo) {     // the hour an intervention took effect: show it now
      if (app.tau < this.jumpTo) { app.seek(this.jumpTo + 0.02); app.lastK = rep.kOf(this.jumpTo) - 1; }   // with its floaters
      this.jumpTo = null;
    }
    if (this.wantSel != null && rep.pidSlot(rep.n - 1, this.wantSel) >= 0) { app.select(this.wantSel, { follow: true }); this.wantSel = null; }
  }
  count(events) {
    const c = this.counts;
    for (const e of events) {
      const k = e[1];
      if (k === "born") c.born++; else if (k.startsWith("died")) c.died++; else if (k === "theft") c.theft++;
      else if (k === "detained" || k === "citizens' arrest") c.arrest++;
    }
  }

  // ---------------------------------------------------------------- interventions
  // an intervention takes effect in the next hour the simulator lives (t = tEnd, as all earlier hours have
  // arrived by the time the worker answers): play from there as soon as it has been lived
  landNext() { const app = this.app; this.jumpTo = app.rep.tEnd; if (!app.playing) app.setPlaying(true); this.tick(); }
  async kill(pid) {
    const app = this.app, name = app.rep.nm(pid);
    try {
      const r = await this.call("kill", { pid });
      this.note = r.done ? `${name} has been removed from the town.` : `${name} is no longer alive.`;
      if (r.done) this.landNext();
    } catch (err) { this.note = `Could not remove ${name}: ${err.message}`; }
    if (app.ui.cardParts) app.ui.cardParts.t = "";     // re-enable the card's buttons
    app.ui.panels(true); this.panels(true);
  }
  async spawn(evil) {
    const app = this.app, sex = $("lvSex").value, age = clamp(+$("lvAge").value || 25, evil ? 16 : 0, 90);
    try {
      const r = await this.call("spawn", { sex, age, evil });
      if (r.pid == null) this.note = "The town is full (every slot is taken): nobody else can move in until someone dies.";
      else {
        this.wantSel = r.pid; this.note = evil ? "A harm-seeker has arrived (red ring)." : "A newcomer has arrived.";
        this.landNext();
      }
    } catch (err) { this.note = `Could not add anyone: ${err.message}`; }
    this.panels(true);
  }
  async rule(key, value) {
    try {
      const r = await this.call("rule", { key, value });
      this.rules = r.rules;
      this.note = r.done ? `${(RULE_UI[key] || {}).label || key} changed from the next hour on.` : `Rule ${key} cannot be changed.`;
    } catch (err) { this.note = `Could not change ${key}: ${err.message}`; }
    this.renderRules(); this.panels(true);
  }
  async restart() {
    const app = this.app;
    const b = $("lvRestart"); if (b) b.disabled = true;
    try { const rep = await this.start(Math.floor(Math.random() * 1e6)); app.install(rep); this.mount(); app.seek(rep.t0); app.setPlaying(true); }
    catch (err) { this.fail(err); }
  }

  // ---------------------------------------------------------------- panel
  mount() {
    const app = this.app, h = this.header;
    $("livePanel").hidden = false;
    if (!this.bound) {
      this.bound = true;
      $("lvAdd").addEventListener("click", () => this.spawn(false));
      $("lvEvil").addEventListener("click", () => this.spawn(true));
      $("lvFeed").addEventListener("click", (e) => { const li = e.target.closest("li[data-pid]"); if (li && +li.dataset.pid) app.select(+li.dataset.pid, { follow: true }); });
      $("lvBtns").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-act]"); if (!b) return;
        if (b.dataset.act === "edge") app.seek(app.rep.tEnd - 0.999);
        if (b.dataset.act === "restart") this.restart();
      });
      $("lvRules").addEventListener("change", (e) => {
        const inp = e.target.closest("[data-rule]"); if (!inp) return;
        const v = inp.type === "checkbox" ? inp.checked : +inp.value;
        if (inp.type !== "checkbox" && !(Number.isFinite(v) && v >= 0)) { inp.value = this.rules[inp.dataset.rule]; return; }
        this.rule(inp.dataset.rule, v);
      });
    }
    const pol = h.policies || {}, ev = pol.evil;
    $("lvEvilNote").innerHTML = ev
      ? `Harm-seekers act with a policy trained only to make everyone else unhappy (iteration ${esc(ev.iteration ?? "?")}); they have the same actions as anyone else.`
      : `No trained harm-seeker policy yet: harm-seekers act with the townspeople's policy, but with no empathy and all greed and boldness.`;
    this.rulesBuilt = false; this.renderRules();
    const st = $("stages");
    st.innerHTML = `<span class="label">Live sandbox</span><span class="lvinfo">townspeople: policy it ${esc((pol.town || {}).iteration ?? "?")} · harm-seekers: ${ev ? `policy it ${esc(ev.iteration ?? "?")}` : "town policy, evil traits"} · Python ${esc(this.info.python)} (Pyodide ${esc(this.info.pyodide)})</span><a class="chip link-chip" href="${esc(location.pathname)}">Watch recorded replays</a>`;
    this.lastKey = ""; this.panels(true);
  }

  renderRules() {
    const el = $("lvRules"), docs = (this.header && this.header.rules) || {};
    if (!this.rulesBuilt) {
      this.rulesBuilt = true;
      el.innerHTML = Object.keys(docs).map((k) => {
        const v = this.rules[k], ui = RULE_UI[k] || {};
        const input = typeof v === "boolean" ? `<input type="checkbox" data-rule="${k}">`
          : `<input type="number" data-rule="${k}" min="0" step="${ui.step || "any"}" inputmode="decimal" aria-label="${esc(ui.label || k)}">`;
        return `<label class="rule"><span class="rn">${esc(ui.label || k)}<small>${esc(docs[k])}</small></span>${input}</label>`;
      }).join("");
    }
    el.querySelectorAll("[data-rule]").forEach((inp) => {
      const k = inp.dataset.rule, v = this.rules[k], d = this.defaults[k];
      if (document.activeElement !== inp) { if (inp.type === "checkbox") inp.checked = !!v; else inp.value = v; }
      const row = inp.closest(".rule");
      row.classList.toggle("changed", v !== d);
      row.title = `${(docs[k] || k)}; default ${typeof d === "boolean" ? (d ? "on" : "off") : d}`;
    });
  }

  panels(force = false) {
    const app = this.app, rep = app.rep;
    if (!rep || !this.header) return;
    const now = performance.now();
    const behind = rep.tEnd - app.tau > Math.max(4, app.speed * 3);
    const key = `${app.k}|${rep.n}|${this.ended}|${!!this.failed}|${behind}|${app.playing}|${this.note}`;
    if (!force && (key === this.lastKey || now - (this.lastAt || 0) < 250)) return;
    this.lastKey = key; this.lastAt = now;
    // status line
    const L = app.layout.frame(app.k), alive = L ? L.list.length : 0;
    const evil = L ? L.list.filter((sp) => rep.isEvil(sp.pid)).length : 0;
    const speed = this.msPerHour ? `${Math.round(this.msPerHour)} ms per simulated hour` : "";
    const over = this.ended === "time" ? `${MAX_YEARS} years have passed` : this.ended ? "Everybody has died" : "";
    let stat = `<span class="pulse ${app.playing && !this.ended && !this.failed ? "" : "off"}"></span><b>${over || (app.playing ? "Running" : "Paused")}</b>` +
      `<span>${alive} alive${evil ? ` · <span class="hs">${evil} harm-seeker${evil > 1 ? "s" : ""}</span>` : ""}</span>${speed ? `<span>${speed}</span>` : ""}`;
    if (this.failed) stat += `<span class="error">The simulator stopped: ${esc(this.failed)}</span>`;
    setHTML($("lvStat"), stat);
    setHTML($("lvBtns"), (behind && !this.ended ? `<button type="button" class="lbtn ghost" data-act="edge">Back to now</button>` : "") +
      (this.ended ? `<button type="button" class="lbtn" id="lvRestart" data-act="restart">Start a new town</button>` : ""));
    $("lvNote").textContent = this.note || (app.sel >= 0 ? "" : "Select someone (in the town or the list) to remove them.");
    $("lvAdd").disabled = $("lvEvil").disabled = this.ended || !!this.failed;
    // reactions up to the playhead, newest first
    const end = rep.eventsBefore(app.k), out = [];
    for (let i = end - 1; i >= 0 && out.length < 8; i--) if (REACT(rep.events[i][1])) out.push(rep.events[i]);
    setHTML($("lvFeed"), out.length ? out.map((e) => `<li class="${evClass(e[1])}" data-pid="${e[2]}"><span class="t">${rep.shortLabel(e[0])}</span><span>${esc(evText(rep, e))}</span></li>`).join("")
      : '<li class="empty">No deaths, crimes or arrivals yet.</li>');
    // header summary
    const c = rep.clock(app.tau), n = this.counts;
    setHTML($("summary"), [["year", c.year + 1], ["people", alive], ["births", n.born], ["deaths", n.died], ["thefts", n.theft], ["arrests", n.arrest]]
      .map(([a, v]) => `<span>${a} <b>${v}</b></span>`).join("") + `<span>seed <b>${this.seed}</b></span>`);
  }
}
