// Panels around the 3D view: people ledger, person card, town, families, feed, timeline, charts.
import { JOB_COLORS, JOB_LABEL, ACT_COLORS, ACT_LABEL, QUIET, evText, evClass } from "./replay.js";
import { esc, hexStr, clamp } from "./util.js";

const $ = (id) => document.getElementById(id);
const cssVar = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const coin = (v) => `${Math.round(v * 10) / 10}¢`;
const sexMark = (rep, pid) => (rep.people[pid] && rep.people[pid].sex === 0 ? "♀" : "♂");
const jobChip = (job) => `<span class="jchip" title="${esc(JOB_LABEL[job] || job)}"><i class="sq" style="background:${hexStr(JOB_COLORS[job] ?? JOB_COLORS.none)}"></i><span class="jl">${esc(JOB_LABEL[job] || job)}</span></span>`;
const lineCol = (rep, pid) => hexStr(rep.line[pid] ?? 0x999999);

export function placeWord(sp) {
  if (!sp) return "";
  if (sp.info && sp.info.type === "house") return sp.home === sp.info.h ? "home" : `house ${sp.info.h + 1}`;
  if (sp.place) return sp.place === "station" ? "police station" : sp.place;
  if (sp.info && sp.info.type === "road") return "the street";
  return "";
}
export function doing(sp) {
  if (!sp) return "";
  const a = sp.act, pl = placeWord(sp);
  if (sp.carriedBy != null) return "carried by mum";
  if (a === "sleeping") return pl === "home" ? "asleep at home" : `asleep${pl ? " at the " + pl : " outside"}`;
  if (a === "home") return pl === "home" ? "at home" : "resting";
  if (a === "working") return pl ? `working · ${pl}` : "working";
  if (a === "eating") return pl === "home" ? "eating at home" : pl ? `eating · ${pl}` : "eating";
  if (a === "buying") return pl ? `shopping · ${pl}` : "shopping";
  if (a === "idle") return pl === "home" ? "at home" : pl ? `idle · ${pl}` : "idle";
  if (a === "socialising") return pl ? `chatting · ${pl}` : "chatting";
  return (ACT_LABEL[a] || a) + (pl && a !== "walking" ? ` · ${pl}` : "");
}

export class UI {
  constructor(app) {
    this.app = app;
    this.sort = { k: "name", dir: 1 };
    document.querySelectorAll("th[data-k]").forEach((th) => th.addEventListener("click", () => {
      const k = th.dataset.k; this.sort = { k, dir: this.sort.k === k ? -this.sort.dir : 1 };
      this.panels(true);
    }));
    $("ledger").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-pid]"); if (tr) app.select(+tr.dataset.pid, { follow: true }); });
    $("families").addEventListener("click", (e) => { const b = e.target.closest("[data-pid]"); if (b) app.select(+b.dataset.pid, { follow: true }); });
    $("card").addEventListener("click", (e) => {
      const b = e.target.closest("[data-pid]"); if (b) { app.select(+b.dataset.pid, { follow: app.following }); return; }
      const a = e.target.closest("[data-act]"); if (!a) return;
      if (a.dataset.act === "follow") app.setFollow(!app.following);
      if (a.dataset.act === "close") app.select(-1);
    });
    $("feed").addEventListener("click", (e) => { const li = e.target.closest("li[data-t]"); if (li) { app.seek(+li.dataset.t + 0.02); if (li.dataset.pid) app.select(+li.dataset.pid, { follow: true }); } });
    this.bindTimeline();
  }

  reset() { this.lastPanels = -1; this.cardKey = ""; }

  // ---------------------------------------------------------------- header
  stages(index, current, onPick) {
    const el = $("stages");
    el.querySelectorAll(".chip").forEach((c) => c.remove());
    index.stages.forEach((s, k) => {
      const b = document.createElement("button"); b.className = "chip"; b.type = "button";
      const scripted = s.summary && s.summary.scripted;
      b.textContent = scripted ? "scripted" : s.it === 0 ? "untrained" : `it ${s.it}`;
      b.title = s.samples ? `${(s.samples / 1e6).toFixed(1)} M person-steps of practice` : "before any training";
      b.setAttribute("aria-pressed", k === current ? "true" : "false");
      b.addEventListener("click", () => onPick(k));
      el.appendChild(b);
    });
  }
  summary(rep, stage) {
    const m = rep.raw.summary || {}, ev = rep.events;
    const count = (f) => ev.reduce((n, e) => n + (f(e[1]) ? 1 : 0), 0);
    const pick = (...keys) => { for (const k of keys) if (typeof m[k] === "number") return m[k]; return null; };
    const years = (rep.tEnd - rep.t0) / rep.yearSteps;
    const births = count((k) => k === "born"), deaths = count((k) => k.startsWith("died"));
    const hires = count((k) => k === "hired");
    const jobsHeld = hires || pick("hires", "hired") || 0;
    const sold = count((k) => k.startsWith("bought ")) || (pick("meals_canteen") ?? 0) + (pick("groceries") ?? 0);
    const docked = pick("docked_hours") ?? count((k) => k === "docked");
    const thefts = count((k) => k === "theft") || pick("thefts") || 0;
    const arrests = count((k) => k === "detained" || k === "citizens' arrest") || (pick("detentions") ?? 0) + (pick("citizen_arrests") ?? 0);
    const items = [["years shown", years < 10 ? years.toFixed(1) : Math.round(years)], ["births", births], ["deaths", deaths], ["jobs held", jobsHeld],
      ["food sold", sold], ["docked hours", docked], ["thefts", thefts], ["arrests", arrests]];
    const meta = rep.raw.meta || {};
    const st = meta.scripted ? "scripted routine (not learned)" : stage.it === 0 ? "untrained" : `training iteration ${stage.it}`;
    $("summary").innerHTML = items.map(([a, v]) => `<span>${a} <b>${typeof v === "number" ? Math.round(v) : v}</b></span>`).join("") +
      `<span>stage <b>${st}</b>${stage.samples ? ` · ${(stage.samples / 1e6).toFixed(1)} M person-steps of practice` : ""}</span>`;
  }
  legend() {
    const jobs = ["farm", "orchard", "dairy", "warehouse", "canteen", "none"];
    $("legend").innerHTML = jobs.map((j) => `<span><i class="dot" style="background:${hexStr(JOB_COLORS[j])};box-shadow:inset 0 0 0 1px rgba(0,0,0,.2)"></i>${JOB_LABEL[j]}</span>`).join("") +
      `<span><i class="tick" style="background:var(--love)"></i>birth, couple</span><span><i class="tick" style="background:var(--death)"></i>death</span>` +
      `<span><i class="tick" style="background:var(--theft)"></i>theft</span><span><i class="tick" style="background:var(--police)"></i>arrest</span>`;
  }
  rules(rep) {
    const c = rep.cfg, sh = c.shifts || {}, hh = (h) => String(h).padStart(2, "0") + ":00";
    const shiftTxt = Object.entries(sh).map(([j, s]) => `${j} ${hh(s[0])}–${hh(s[1])}`).join(", ");
    const lunch = sh.farm ? `${hh(sh.farm[2])}–${hh(sh.farm[3])}` : "midday";
    const school = c.school_hours ? `${hh(c.school_hours[0])}–${hh(c.school_hours[1])}` : "in the day";
    const shop = c.shop_hours ? `${hh(c.shop_hours[0])}–${hh(c.shop_hours[1])}` : "in the day";
    $("rules").innerHTML = `
      <p><strong>Time.</strong> Every frame is one hour. A year has ${rep.dpy} days, ${rep.dpy === rep.seasons.length ? "one per season" : `${rep.dpy / rep.seasons.length} per season`}; nights run 21:00–06:00. Crops grow in spring and summer and are golden by autumn; fruit ripens through summer and autumn; the herd grazes except in winter.</p>
      <p><strong>Work.</strong> Adults can take one of five jobs, each with a shift and a paid break: ${esc(shiftTxt)}. Wages are paid by the town's businesses at the end of a shift; missing work is docked. The warehouse ships goods out of town.</p>
      <p><strong>Food.</strong> The canteen serves hot lunches (around ${lunch}) only while its staff are on shift. The shop in the market hall sells grain, fruit and dairy to take home (${shop}). Eating the same food again helps less, so people want all three.</p>
      <p><strong>Families.</strong> Couples live together, one house per family (${c.house_beds || 4} beds each). Babies nurse and go everywhere with their mother; children under 12 go to school (${school}) and cannot work. Temperament is inherited.</p>
      <p><strong>Trust.</strong> Anyone can steal; volunteers can join the police and detain a thief they saw, and witnesses together can make a citizens' arrest. <strong>Happiness</strong> is the only reward: fed, rested and healthy, a partner and thriving children, friends, useful work, and a little saved.</p>
      <p class="note">Everything in the 3D town is generated in the browser with three.js (MIT); no external models. Source: <a href="https://github.com/neomorrison/humanoid-civilization" rel="noopener">github.com/neomorrison/humanoid-civilization</a>.</p>`;
  }

  // ---------------------------------------------------------------- panels (throttled)
  panels(force = false) {
    const app = this.app, rep = app.rep, k = app.k;
    if (!rep) return;
    const now = performance.now();
    if (!force && k === this.lastPanels) return;
    if (!force && app.playing && now - (this.lastPanelTime || 0) < 220) return;
    this.lastPanels = k; this.lastPanelTime = now;
    const L = app.layout.frame(k);
    this.ledger(L); this.card(L); this.town(L); this.families(L); this.feed();
  }

  ledger(L) {
    const app = this.app, rep = app.rep, rows = [];
    for (const sp of L.list) {
      const partner = sp.partner >= 0 && L.spots[sp.partner] ? L.spots[sp.partner].pid : 0;
      rows.push({ sp, name: rep.nm(sp.pid), job: sp.job, money: rep.get(rep.row(L.k, sp.s), "money"), hunger: rep.get(rep.row(L.k, sp.s), "hunger"),
        energy: rep.get(rep.row(L.k, sp.s), "energy"), act: doing(sp), partner });
    }
    const { k, dir } = this.sort;
    rows.sort((a, b) => {
      const va = k === "partner" ? (a.partner ? rep.nm(a.partner) : "~") : k === "name" ? a.sp.pid : a[k], vb = k === "partner" ? (b.partner ? rep.nm(b.partner) : "~") : k === "name" ? b.sp.pid : b[k];
      return (va < vb ? -1 : va > vb ? 1 : a.sp.pid - b.sp.pid) * dir;
    });
    document.querySelectorAll("th[data-k]").forEach((th) => th.setAttribute("aria-sort", th.dataset.k === k ? (dir > 0 ? "ascending" : "descending") : "none"));
    $("pcount").textContent = `${rows.length} alive`;
    $("ledger").innerHTML = rows.map((r) => {
      const sp = r.sp, pid = sp.pid;
      const hc = r.hunger >= 90 ? "var(--theft)" : "var(--hunger)";
      return `<tr data-pid="${pid}" class="${pid === app.sel ? "sel" : ""}">
        <td><span class="who"><i class="dot" style="background:${lineCol(rep, pid)}"></i>${esc(r.name)} <small>${sexMark(rep, pid)}${Math.floor(sp.age)}</small></span></td>
        <td>${sp.age < 12 ? '<span class="muted">child</span>' : jobChip(sp.job)}</td>
        <td class="num">${Math.round(r.money)}</td>
        <td><div class="bars" title="hunger ${r.hunger}, energy ${r.energy}"><div class="bar"><i style="width:${r.hunger}%;background:${hc}"></i></div><div class="bar"><i style="width:${r.energy}%;background:var(--energy)"></i></div></div></td>
        <td><span class="act"><i class="sq" style="background:${ACT_COLORS[sp.act] || "#999"}"></i>${esc(r.act).replace(" · ", '<span class="pl"> · ') + (r.act.includes(" · ") ? "</span>" : "")}</span></td>
        <td class="c-partner">${r.partner ? esc(rep.first(r.partner)) : '<span class="muted">–</span>'}</td></tr>`;
    }).join("") || '<tr><td colspan="6" class="muted">Nobody is alive.</td></tr>';
  }

  card(L) {
    const app = this.app, rep = app.rep, pid = app.sel, el = $("card");
    if (pid < 0) {
      el.innerHTML = `<h2>Follow someone</h2><p class="empty">Click a person in the town or in the list to see their family, what they did all day, and what they earned. The camera follows them.</p>`;
      return;
    }
    const s = rep.pidSlot(L.k, pid), sp = s >= 0 ? L.spots[s] : null, row = s >= 0 ? rep.row(L.k, s) : null;
    const P = rep.people[pid] || {};
    const kids = (rep.kids[pid] || []).filter((c) => (rep.people[c].born_step ?? -1e9) <= L.t);
    const alive = (q) => rep.pidSlot(L.k, q) >= 0;
    const plink = (q) => `<button class="plink" data-pid="${q}">${esc(rep.first(q))}</button>${alive(q) ? "" : " †"}`;
    const partner = sp && sp.partner >= 0 && L.spots[sp.partner] ? L.spots[sp.partner].pid : 0;
    // today's schedule: hours 0..23 of the current day
    const c = rep.clock(app.tau), dayStart = c.day * rep.daySteps;
    const cells = [], seen = new Set();
    let earned = 0, docked = 0, bought = 0;
    for (let h = 0; h < rep.daySteps; h++) {
      const t = dayStart + h, k = Math.round((t - rep.t0) / rep.stride);
      let col = "", title = `${String(h).padStart(2, "0")}:00`;
      if (k >= 0 && k < rep.n && t <= L.t) {
        const ss = rep.pidSlot(k, pid);
        if (ss >= 0) { const a = rep.acts[rep.get(rep.row(k, ss), "act")]; col = ACT_COLORS[a] || ""; seen.add(a); title += " " + (ACT_LABEL[a] || a); }
        for (const ei of rep.evByFrame[k]) { const e = rep.events[ei]; if (e[2] !== pid) continue; if (e[1] === "earned") earned += e[4]; if (e[1] === "docked") docked += e[4]; if (e[1].startsWith("bought ")) bought += e[4]; }
      }
      cells.push(`<i class="${h === c.hour ? "now" : ""}" style="${col ? "background:" + col : ""}" title="${title}"></i>`);
    }
    const bar = (v, col) => `<div class="bar"><i style="width:${clamp(v, 0, 100)}%;background:${col}"></i></div>`;
    const born = P.born_step != null && P.born_step >= rep.t0 ? rep.clock(P.born_step) : null;
    const status = !sp ? `<span class="muted">${rep.events.some((e) => e[2] === pid && e[1].startsWith("died") && e[0] <= L.t) ? "has died" : "not born yet"}</span>` : esc(doing(sp));
    const flags = sp ? [sp.flags & 1 ? "expecting" : "", sp.flags & 4 ? "police volunteer" : "", sp.flags & 16 ? "seen stealing" : "", sp.flags & 64 ? "docked this hour" : ""].filter(Boolean) : [];
    el.innerHTML = `
      <div class="head"><div class="avatar" style="background:${lineCol(rep, pid)}">${esc(rep.first(pid)[0])}</div>
        <div><div class="name">${esc(rep.nm(pid))}</div><div class="sub">${sexMark(rep, pid)} ${sp ? Math.floor(sp.age) + " years" : ""}${sp ? " · " + (sp.age < 12 ? "child" : JOB_LABEL[sp.job] || sp.job) : ""}${sp && sp.home >= 0 ? ` · house ${sp.home + 1}` : ""}${born ? ` · born year ${born.year + 1}` : ""}</div></div></div>
      <div class="row"><span>Now: <b>${status}</b></span>${flags.length ? `<span>${flags.map(esc).join(" · ")}</span>` : ""}</div>
      <div class="row"><span>Partner: ${partner ? plink(partner) : '<span class="muted">none</span>'}</span>
        <span>Parents: ${[P.mother, P.father].filter(Boolean).map(plink).join(", ") || '<span class="muted">founders</span>'}</span>
        <span>Children: ${kids.length ? kids.map(plink).join(", ") : '<span class="muted">none</span>'}</span></div>
      ${row ? `<div class="stats"><div class="stat">hunger ${rep.get(row, "hunger")}${bar(rep.get(row, "hunger"), "var(--hunger)")}</div><div class="stat">energy ${rep.get(row, "energy")}${bar(rep.get(row, "energy"), "var(--energy)")}</div><div class="stat">health ${rep.get(row, "health")}${bar(rep.get(row, "health"), "var(--health)")}</div></div>
      <div class="row"><span>Money <b class="mono">${coin(rep.get(row, "money"))}</b></span><span>Carrying <b class="mono" style="color:var(--grain)">${rep.get(row, "g")}</b>·<b class="mono" style="color:var(--fruit)">${rep.get(row, "f")}</b>·<b class="mono" style="color:var(--dairy)">${rep.get(row, "d")}</b> <span class="muted">grain·fruit·dairy</span></span></div>` : ""}
      <div class="strip"><h2 style="margin:0 0 4px">Today, hour by hour</h2><div class="cells">${cells.join("")}</div><div class="hrs"><span>00</span><span>06</span><span>12</span><span>18</span><span>24</span></div>
        <div class="keys">${[...seen].map((a) => `<span><i class="sq" style="background:${ACT_COLORS[a] || "#999"}"></i> ${ACT_LABEL[a] || a}</span>`).join("")}</div></div>
      <div class="today">Today: earned <span class="up">+${coin(earned)}</span> · docked <span class="down">−${coin(docked)}</span> · spent on food <span class="mono">${coin(bought)}</span></div>
      <div class="btns"><button type="button" data-act="follow" class="${app.following ? "" : "primary"}">${app.following ? "Stop following" : "Follow with camera"}</button><button type="button" data-act="close">Close</button></div>`;
  }

  town(L) {
    const app = this.app, rep = app.rep, w = rep.w(L.k), wPrev = rep.w(L.k - 1);
    const st = w.stock || [0, 0, 0], pr = w.price || [];
    const open = app.canteenOpen, shop = app.shopOpen;
    const jobs = rep.jobs.filter((j) => j !== "none");
    const by = {}, on = {};
    jobs.forEach((j) => { by[j] = []; on[j] = []; });
    for (const sp of L.list) {
      if (!by[sp.job] || sp.age < 12) continue;
      by[sp.job].push(sp.pid);
      if (sp.act === "working" || (sp.flags & 32 && sp.act !== "sleeping" && sp.act !== "home")) on[sp.job].push(sp.pid);
    }
    const unemployed = L.list.filter((sp) => sp.job === "none" && sp.age >= 16).length;
    const onNames = jobs.flatMap((j) => on[j]).map((p) => rep.first(p));
    const food = rep.foods.map((f, i) => `<span style="color:var(--${f === "grain" ? "grain" : f === "fruit" ? "fruit" : "dairy"})">${Math.round(st[i] ?? 0)} ${esc(f)}</span>${pr[i] != null ? ` <span class="muted">@${coin(pr[i])}</span>` : ""}`).join(" · ");
    $("town").innerHTML = `<h2>Town</h2>
      <dl class="kv"><dt>Businesses' cash</dt><dd>${coin(w.treasury ?? 0)}</dd><dt>Food in stock</dt><dd>${food}</dd>
        <dt>Canteen</dt><dd><span class="badge ${open ? "on" : "off"}">${open ? "open" : "closed"}</span> <span class="muted" style="font-family:var(--font)">hot lunches while staff are on shift</span></dd>
        <dt>Market shop</dt><dd><span class="badge ${shop ? "on" : "off"}">${shop ? "open" : "closed"}</span></dd></dl>
      <div class="jobs">${jobs.map((j) => `<div class="jr">${jobChip(j)}<span class="pips" style="--c:${hexStr(JOB_COLORS[j])}">${by[j].map((p) => `<i class="${on[j].includes(p) ? "on" : ""}" title="${esc(rep.nm(p))}${on[j].includes(p) ? " (on shift)" : ""}"></i>`).join("")}</span><span class="mono muted">${on[j].length}/${by[j].length}</span></div>`).join("")}
        <div class="jr"><span class="muted">no job (16+)</span><span></span><span class="mono muted">${unemployed}</span></div></div>
      <div class="onshift">${onNames.length ? `On shift now: ${onNames.map(esc).join(", ")}` : "Nobody is on shift right now."}</div>`;
    void wPrev;
  }

  families(L) {
    const rep = this.app.rep, seen = new Set(), items = [];
    const alive = (pid) => rep.pidSlot(L.k, pid) >= 0;
    const plink = (q) => `<button class="plink" data-pid="${q}">${esc(rep.first(q))}</button>${alive(q) ? "" : " †"}`;
    for (const sp of L.list) {
      const pid = sp.pid, p = sp.partner;
      const kids = (rep.kids[pid] || []).filter((c) => (rep.people[c].born_step ?? -1e9) <= L.t);
      if (p >= 0 && L.spots[p]) {
        const q = L.spots[p].pid, key = [pid, q].sort((a, b) => a - b).join("-");
        if (seen.has(key)) continue; seen.add(key);
        const both = kids.filter((c) => rep.people[c].mother === q || rep.people[c].father === q);
        items.push(`<li><b><button class="plink" data-pid="${pid}">${esc(rep.nm(pid))}</button></b> ♥ <b><button class="plink" data-pid="${q}">${esc(rep.nm(q))}</button></b>${sp.home >= 0 ? ` <span class="muted">· house ${sp.home + 1}</span>` : ""}<div class="kids">${both.length ? "children: " + both.map(plink).join(", ") : "no children yet"}</div></li>`);
      } else if (kids.some(alive) && sp.age >= 16) {
        items.push(`<li><b><button class="plink" data-pid="${pid}">${esc(rep.nm(pid))}</button></b>${sp.home >= 0 ? ` <span class="muted">· house ${sp.home + 1}</span>` : ""}<div class="kids">children: ${kids.map(plink).join(", ")}</div></li>`);
      }
    }
    $("families").innerHTML = items.join("") || '<li class="kids">Nobody has paired up yet.</li>';
  }

  feed() {
    const app = this.app, rep = app.rep, end = rep.eventsBefore(app.k), out = [];
    for (let i = end - 1; i >= 0 && out.length < 60; i--) {
      const e = rep.events[i];
      if (QUIET.has(e[1]) || e[1] === "earned" && out.length > 30) continue;
      out.push(e);
    }
    $("feed").innerHTML = out.length ? out.map((e) => `<li class="${evClass(e[1])}" data-t="${e[0]}" data-pid="${e[2]}"><span class="t">${rep.shortLabel(e[0])}</span><span>${esc(evText(rep, e))}</span></li>`).join("")
      : '<li class="empty">Nothing has happened yet.</li>';
  }

  // ---------------------------------------------------------------- timeline
  bindTimeline() {
    const c = $("tl"), tip = $("tltip"), app = this.app;
    const tauAt = (e) => { const r = c.getBoundingClientRect(); const u = clamp((e.clientX - r.left) / r.width, 0, 1); return app.rep.t0 + u * (app.rep.tEnd - app.rep.t0); };
    let drag = false;
    c.addEventListener("pointerdown", (e) => { if (!app.rep) return; drag = true; c.setPointerCapture(e.pointerId); app.seek(tauAt(e)); });
    c.addEventListener("pointermove", (e) => {
      if (!app.rep) return;
      const tau = tauAt(e), r = c.getBoundingClientRect();
      tip.hidden = false; tip.textContent = app.rep.label(tau, { minutes: false }); tip.style.left = clamp(e.clientX - r.left, 60, r.width - 60) + "px";
      if (drag) app.seek(tau);
    });
    c.addEventListener("pointerup", () => { drag = false; });
    c.addEventListener("pointerleave", () => { tip.hidden = true; });
    c.addEventListener("keydown", (e) => {
      if (!app.rep) return;
      if (e.key === "ArrowRight") { app.seek(Math.floor(app.tau) + 1); e.preventDefault(); }
      if (e.key === "ArrowLeft") { app.seek(Math.floor(app.tau) - 1); e.preventDefault(); }
      if (e.key === "PageDown") { app.seek(app.tau + app.rep.daySteps); e.preventDefault(); }
      if (e.key === "PageUp") { app.seek(app.tau - app.rep.daySteps); e.preventDefault(); }
    });
  }
  // static part (bands, ticks, marks) drawn once per stage/size/theme; the playhead on top
  timeline() {
    const app = this.app, rep = app.rep, c = $("tl");
    if (!rep) return;
    const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const Wd = Math.max(1, Math.round(r.width * dpr)), Hd = Math.round(50 * dpr);
    const key = `${Wd}|${Hd}|${rep.n}|${document.documentElement.dataset.theme || ""}|${app.themeKey}`;
    if (key !== this.tlKey) {
      this.tlKey = key;
      if (c.width !== Wd || c.height !== Hd) { c.width = Wd; c.height = Hd; }
      const base = (this.tlBase = document.createElement("canvas")); base.width = Wd; base.height = Hd;
      const g = base.getContext("2d");
      const T0 = rep.t0, T1 = rep.tEnd, X = (t) => ((t - T0) / (T1 - T0)) * (Wd - 2) + 1;
      const bandY = 14 * dpr, bandH = 14 * dpr;
      const SEA = { spring: "#9fd67a", summer: "#e9c75a", autumn: "#e0894a", winter: "#b9cbe0" };
      // seasons, nights
      const d0 = Math.floor(T0 / rep.daySteps), d1 = Math.ceil(T1 / rep.daySteps);
      for (let d = d0; d < d1; d++) {
        const s = rep.seasons[rep.seasonOfDay(d)];
        g.fillStyle = SEA[(s || "").toLowerCase()] || "#ccc"; g.globalAlpha = 0.85;
        const xa = X(Math.max(T0, d * rep.daySteps)), xb = X(Math.min(T1, (d + 1) * rep.daySteps));
        g.fillRect(xa, bandY, Math.max(1, xb - xa), bandH);
        const pxDay = (Wd * rep.daySteps) / (T1 - T0);
        if (pxDay > 18) {   // night shading
          g.globalAlpha = 0.28; g.fillStyle = "#1b2440";
          const n0 = X(Math.max(T0, d * rep.daySteps)), n1 = X(Math.min(T1, d * rep.daySteps + 6)), n2 = X(Math.max(T0, d * rep.daySteps + 21)), n3 = X(Math.min(T1, (d + 1) * rep.daySteps));
          if (n1 > n0) g.fillRect(n0, bandY, n1 - n0, bandH); if (n3 > n2) g.fillRect(n2, bandY, n3 - n2, bandH);
        }
        if (pxDay > 5) { g.globalAlpha = 0.5; g.fillStyle = cssVar("--surface"); g.fillRect(Math.round(xa), bandY, Math.max(1, dpr * 0.6), bandH); }
      }
      g.globalAlpha = 1;
      // years
      const years = (T1 - T0) / rep.yearSteps, every = years > 40 ? 10 : years > 16 ? 5 : years > 6 ? 2 : 1;
      g.font = `500 ${10 * dpr}px IBM Plex Mono, monospace`; g.fillStyle = cssVar("--muted"); g.textBaseline = "top";
      for (let y = Math.ceil(T0 / rep.yearSteps); y * rep.yearSteps <= T1; y++) {
        const x = X(y * rep.yearSteps);
        g.fillStyle = cssVar("--muted"); g.fillRect(Math.round(x), bandY - 3 * dpr, dpr, bandH + 6 * dpr);
        if (y % every === 0 && x < Wd - 30 * dpr) g.fillText(`Y${y + 1}`, x + 3 * dpr, 1 * dpr);
      }
      // event marks below the band
      const COL = { born: "--love", partnered: "--love", theft: "--theft", detained: "--police", "citizens' arrest": "--police" };
      const cols = {}; ["--love", "--theft", "--police", "--death"].forEach((v) => { cols[v] = cssVar(v); });
      for (const e of rep.events) {
        let v = COL[e[1]]; if (!v && e[1].startsWith("died")) v = "--death";
        if (!v) continue;
        const x = X(e[0]);
        g.fillStyle = cols[v];
        const tall = e[1] === "born" || e[1].startsWith("died");
        g.fillRect(Math.round(x), bandY + bandH + 3 * dpr, Math.max(dpr, 1.5 * dpr), (tall ? 18 : 11) * dpr);
      }
    }
    const g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height); g.drawImage(this.tlBase, 0, 0);
    const x = ((app.tau - rep.t0) / (rep.tEnd - rep.t0)) * (c.width - 2) + 1;
    g.fillStyle = cssVar("--ink"); g.fillRect(Math.round(x) - dpr, 10 * dpr, 2 * dpr, c.height - 10 * dpr);
    g.beginPath(); g.arc(x, 10 * dpr, 4 * dpr, 0, Math.PI * 2); g.fill();
  }

  // ---------------------------------------------------------------- training curves
  charts(index, stageIdx) {
    const el = $("charts");
    if (!index || !index.curve || !index.curve.length) { el.innerHTML = '<p class="note">Training curves appear here once training has logged some eras.</p>'; $("curve-note").textContent = index && index.note ? index.note : ""; return; }
    const cv = index.curve.map((p) => ({ ...p, happy: p.person_steps ? 1000 * p.happiness / p.person_steps : p.happiness }));
    const known = [
      ["years", "Years the town lasted", "per era"], ["population", "Population at the end", "people alive"], ["births", "Births", "per era"], ["deaths", "Deaths", "per era"],
      ["starved", "Starvation deaths", "per era"], ["malnourished", "Malnutrition deaths", "per era"], ["work_hours", "Hours worked", "per era"], ["docked_hours", "Hours docked", "per era, missed work"],
      ["docked", "Docked", "per era, missed work"], ["meals_canteen", "Canteen lunches", "per era"], ["groceries", "Groceries bought", "per era"], ["meals", "Meals bought", "per era"],
      ["family_meals", "Family meals", "per era"], ["wages", "Wages paid", "coins per era"], ["earned", "Wages earned", "coins per era"], ["school_hours", "School hours", "per era"],
      ["sleep_home", "Nights slept at home", "person-hours per era"], ["sleep_rough", "Slept outside", "person-hours per era"], ["thefts", "Thefts", "per era"],
      ["detentions", "Arrests", "per era"], ["partnerships", "Partnerships", "per era"], ["happy", "Happiness", p0(cv) ? "per person, per 1,000 hours" : "per era"],
    ];
    const series = known.filter(([k]) => cv.some((p) => typeof p[k] === "number")).slice(0, 12);
    const sel = index.stages[stageIdx] ? index.stages[stageIdx].it : null;
    const Wc = 250, Hc = 110, L = 40, R = 12, T = 8, B = 20;
    const xs = cv.map((p) => p.it), xmax = Math.max(1, ...xs);
    const fmt = (v) => Math.abs(v) >= 10000 ? (v / 1000).toFixed(0) + "k" : Math.abs(v) >= 100 ? Math.round(v) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);
    el.innerHTML = series.map(([key, title, sub]) => {
      const ys = cv.map((p) => p[key] || 0);
      const lo = Math.min(0, ...ys), hi = Math.max(lo < 0 ? 1e-6 : 1, ...ys);
      const X = (v) => L + (v / xmax) * (Wc - L - R), Y = (v) => T + (1 - (v - lo) / Math.max(1e-6, hi - lo)) * (Hc - T - B);
      const pts = cv.map((p, k) => `${X(xs[k]).toFixed(1)},${Y(ys[k]).toFixed(1)}`).join(" ");
      const base = Y(Math.max(lo, 0));
      const area = `${X(xs[0]).toFixed(1)},${base} ${pts} ${X(xs[xs.length - 1]).toFixed(1)},${base}`;
      const ticks = [lo, (lo + hi) / 2, hi].map((v) => `<line x1="${L}" x2="${Wc - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/><text x="${L - 5}" y="${Y(v) + 3.5}" text-anchor="end" font-size="10" fill="var(--muted)" font-family="IBM Plex Mono, monospace">${fmt(v)}</text>`).join("");
      const mark = sel != null ? `<line x1="${X(sel)}" x2="${X(sel)}" y1="${T}" y2="${Hc - B}" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="3 3"/>` : "";
      return `<div class="chart"><h3>${title}</h3><p>${sub}</p><svg viewBox="0 0 ${Wc} ${Hc}" role="img" aria-label="${title} over training, latest ${fmt(ys[ys.length - 1])}">
        ${ticks}<polygon points="${area}" fill="var(--accent)" fill-opacity="0.12"/>
        <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>${mark}
        <circle cx="${X(xs[xs.length - 1])}" cy="${Y(ys[ys.length - 1])}" r="3.5" fill="var(--accent)"/>
        <text x="${L}" y="${Hc - 5}" font-size="10" fill="var(--muted)" font-family="IBM Plex Mono, monospace">it 0</text>
        <text x="${Wc - R}" y="${Hc - 5}" text-anchor="end" font-size="10" fill="var(--muted)" font-family="IBM Plex Mono, monospace">it ${xmax}</text></svg></div>`;
    }).join("") || '<p class="note">No known statistics in the training curve yet.</p>';
    $("curve-note").textContent = "Each point averages recent finished eras at that training iteration; the dashed line marks the stage being replayed." + (index.note ? " " + index.note : "");
  }
}
function p0(cv) { return cv.some((p) => p.person_steps); }
