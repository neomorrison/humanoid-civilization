// The replay data model: time, names, events and per-frame lookups (docs/TOWN_REPLAY.md).
//
// Timing convention: frame k (t = frames[k].t) is the hour that was just lived. Its `act` is
// what each person did during that hour and x,y is where they were at the end of it. So on
// the viewer's clock, during hour t a person walks from frame k-1's cell to frame k's cell and
// does frame k's activity. The town state `w` is also end-of-hour state.

export const FEM = ["Ada", "Bea", "Cleo", "Dina", "Esme", "Faye", "Gia", "Hana", "Iris", "Juno", "Kira", "Lena", "Mira", "Nora", "Opal", "Pia", "Rhea", "Sara", "Tess", "Uma", "Vera", "Wren", "Yara", "Zoe"];
export const MAS = ["Abe", "Ben", "Cal", "Dev", "Eli", "Finn", "Gus", "Hal", "Ivo", "Jon", "Kai", "Leo", "Max", "Ned", "Omar", "Pax", "Rex", "Sam", "Tom", "Ugo", "Vic", "Will", "Yuri", "Zed"];
export const SUR = ["Moss", "Reed", "Stone", "Vale", "Hart", "Fenn", "Brook", "Ash", "Lark", "Thorn", "Wells", "Gray", "Frost", "Hale", "Pike"];
export const LINE_COLORS = [0xe07a5f, 0x3d9970, 0x5b8ae0, 0xd4a017, 0x9b5de5, 0x00a6a6, 0xe56399, 0x7a9e3a, 0xf28f3b, 0x6c8ead];

export const JOB_COLORS = { none: 0x9aa3a8, farm: 0x4f9a3a, orchard: 0xe8892b, dairy: 0xf1f1ec, warehouse: 0x8a5a35, canteen: 0xd23b35 };
export const JOB_LABEL = { none: "no job", farm: "farm", orchard: "orchard", dairy: "dairy", warehouse: "warehouse", canteen: "canteen" };
export const ACT_COLORS = {
  sleeping: "#33406e", home: "#9a86c9", walking: "#a9b4bb", working: "#2f7fb8", eating: "#e39a2d", buying: "#b8863b",
  school: "#6bb05a", socialising: "#d9679a", idle: "#cfd3c9", detained: "#c2413a", nursing: "#f0a8c4",
};
export const ACT_LABEL = {
  sleeping: "asleep", home: "at home", walking: "walking", working: "working", eating: "eating", buying: "shopping",
  school: "at school", socialising: "chatting", idle: "idle", detained: "detained", nursing: "nursing",
};
export const QUIET = new Set(["talked", "courted", "ate grain", "ate fruit", "ate dairy", "family meal"]);

export class Replay {
  constructor(raw) {
    this.raw = raw;
    const cfg = (this.cfg = raw.config);
    this.frames = raw.frames;
    this.n = this.frames.length;
    this.C = {};
    (raw.slot_cols || []).forEach((c, i) => { this.C[c] = i; });
    this.acts = raw.activities || [];
    this.jobs = raw.jobs || ["none"];
    this.foods = raw.foods || ["grain", "fruit", "dairy"];
    this.stride = raw.stride || 1;
    this.t0 = this.frames.length && this.frames[0].t != null ? this.frames[0].t : (raw.start_step || 0);
    this.daySteps = cfg.day_steps || 24;
    this.dpy = cfg.days_per_year || 4;
    this.seasons = cfg.seasons || ["spring", "summer", "autumn", "winter"];
    this.yearSteps = this.daySteps * this.dpy;
    this.slots = cfg.slots || Math.max(0, ...this.frames.slice(0, 5).map((f) => f.s.length));
    this.people = raw.people || {};
    this.tEnd = this.tOf(this.n - 1) + this.stride;       // exclusive end of the replay clock
    this.events = (raw.events || []).slice().sort((a, b) => a[0] - b[0]);
    this.evByFrame = Array.from({ length: this.n }, () => []);
    this.events.forEach((e, i) => { this.evByFrame[this.kOf(e[0])].push(i); });
    this._pidSlot = new Map();
    this.nameAll();
  }

  tOf(k) { const f = this.frames[k]; return f && f.t != null ? f.t : this.t0 + k * this.stride; }
  kOf(tau) { return Math.max(0, Math.min(this.n - 1, Math.floor((tau - this.t0) / this.stride))); }
  row(k, s) { const f = this.frames[k]; return f ? f.s[s] || null : null; }
  w(k) { const f = this.frames[Math.max(0, Math.min(this.n - 1, k))]; return f ? f.w || {} : {}; }
  get(r, col) { const i = this.C[col]; return i == null || !r ? 0 : r[i]; }

  // clock for an absolute (fractional) step
  clock(tau) {
    const step = Math.floor(tau), ds = this.daySteps;
    const hour = ((step % ds) + ds) % ds;
    const day = Math.floor(step / ds);
    const dayOfYear = ((day % this.dpy) + this.dpy) % this.dpy;
    const season = Math.min(this.seasons.length - 1, Math.floor(dayOfYear * this.seasons.length / this.dpy));
    const year = Math.floor(day / this.dpy);
    const minute = Math.floor((tau - step) * 60);
    const perSeason = Math.max(1, this.dpy / this.seasons.length);
    const dayInSeason = Math.floor(dayOfYear - season * perSeason);
    return { step, hour, minute, day, dayOfYear, season, seasonName: this.seasons[season] || "", year, perSeason, dayInSeason, hourF: hour + (tau - step) };
  }
  seasonOfDay(day) {
    const d = ((day % this.dpy) + this.dpy) % this.dpy;
    return Math.min(this.seasons.length - 1, Math.floor(d * this.seasons.length / this.dpy));
  }
  label(tau, { minutes = true } = {}) {
    const c = this.clock(tau);
    const s = c.seasonName ? c.seasonName[0].toUpperCase() + c.seasonName.slice(1) : "";
    const day = c.perSeason > 1 ? ` day ${c.dayInSeason + 1}` : "";
    const hh = String(c.hour).padStart(2, "0"), mm = minutes ? String(Math.floor(c.minute / 10) * 10).padStart(2, "0") : "00";
    return `Year ${c.year + 1} · ${s}${day} · ${hh}:${mm}`;
  }
  shortLabel(t) {
    const c = this.clock(t);
    return `Y${c.year + 1} ${(c.seasonName || "").slice(0, 3)} ${String(c.hour).padStart(2, "0")}:00`;
  }

  // slot of a pid in frame k (-1 when not alive)
  pidSlot(k, pid) {
    let m = this._pidSlot.get(k);
    if (!m) {
      m = new Map();
      const f = this.frames[k];
      if (f) f.s.forEach((r, s) => { if (r) m.set(r[this.C.pid], s); });
      if (this._pidSlot.size > 400) this._pidSlot.clear();
      this._pidSlot.set(k, m);
    }
    const s = m.get(pid);
    return s == null ? -1 : s;
  }

  nameAll() {
    const people = this.people;
    this.name = {}; this.line = {}; this.kids = {};
    let founders = 0;
    Object.keys(people).map(Number).sort((a, b) => a - b).forEach((pid) => {
      const p = people[pid];
      const given = (p.sex === 0 ? FEM : MAS)[(pid * 7 + 3) % 24];
      let sur, line;
      if (!p.mother && !p.father) { sur = SUR[founders % SUR.length]; line = LINE_COLORS[founders % LINE_COLORS.length]; founders++; }
      else {
        const dad = this.name[p.father] ? p.father : p.mother;       // children take the father's surname and the mother's colour
        sur = this.name[dad] ? this.name[dad].split(" ")[1] : SUR[pid % SUR.length];
        line = this.line[p.mother] ?? this.line[p.father] ?? LINE_COLORS[pid % LINE_COLORS.length];
        [p.mother, p.father].forEach((q) => { if (q) (this.kids[q] = this.kids[q] || []).push(pid); });
      }
      this.name[pid] = `${given} ${sur}`; this.line[pid] = line;
    });
  }
  nm(pid) { return this.name[pid] || (pid ? `#${pid}` : "someone"); }
  first(pid) { return this.nm(pid).split(" ")[0]; }
  sur(pid) { return this.nm(pid).split(" ")[1] || ""; }

  // which events have happened by the end of frame k (index into this.events, exclusive)
  eventsBefore(k) {
    const t = this.tOf(k) + 0.5;
    let lo = 0, hi = this.events.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.events[m][0] < t) lo = m + 1; else hi = m; }
    return lo;
  }
}

export function evText(rep, e) {
  const [, kind, i, j, amt] = e, a = rep.nm(i), b = rep.nm(j), foods = rep.foods;
  const coins = (v) => `${Math.round(v * 10) / 10}¢`;
  if (kind.startsWith("bought ")) return j ? `${a} bought ${kind.slice(7)} from ${b} for ${coins(amt)}` : `${a} bought ${kind.slice(7)} for ${coins(amt)}`;
  if (kind.startsWith("gave ") && kind !== "gave coin") return `${a} gave ${b} some ${kind.slice(5)}`;
  if (kind.startsWith("ate ")) return `${a} ate ${kind.slice(4)}`;
  const jobName = rep.jobs[amt] || "a job";
  return ({
    "gave coin": `${a} gave ${b} a coin`, theft: `${a} stole ${coins(amt)} from ${b}`,
    detained: `${a} (police) detained ${b}${amt ? ` and returned ${coins(amt)}` : ""}`,
    "citizens' arrest": `${a} and other witnesses detained ${b}`, hired: `${a} started work at the ${jobName}`,
    earned: `${a} earned ${coins(amt)} for a shift`, docked: `${a} was docked ${coins(amt)} for missing work`,
    partnered: `${a} and ${b} became partners`, separated: `${a} left ${b}`, shunned: `${a} shuns ${b}`, forgave: `${a} forgave ${b}`,
    courted: `${a} courted ${b}`, talked: `${a} chatted with ${b}`, "family meal": `${a} ate with the family`,
    expecting: `${a} is expecting ${j ? rep.first(j) + "'s" : "a"} baby`, born: `${a} was born to ${b}`,
    "moved in": `${a} moved into house ${amt + 1}`, "fell asleep outside": `${a} fell asleep outside`, "fell asleep": `${a} dozed off`,
    "died (starved)": `${a} starved to death at ${Math.floor(amt)}`, "died (malnutrition)": `${a} died of malnutrition at ${Math.floor(amt)}`,
    "died (old age)": `${a} died of old age at ${Math.floor(amt)}`, enlisted: `${a} joined the police`, resigned: `${a} left the police`,
  })[kind] || `${a}: ${kind}${foods.length && 0 ? "" : ""}`;
}

export function evClass(k) {
  if (k === "theft") return "k-crime";
  if (k === "detained" || k === "citizens' arrest" || k === "enlisted") return "k-justice";
  if (k === "partnered" || k === "born" || k === "expecting" || k === "courted") return "k-love";
  if (k.startsWith("died")) return "k-death";
  if (k === "earned" || k === "hired") return "k-work";
  if (k === "docked") return "k-docked";
  return "";
}
