/* resolvent viz engine.
   A page's figures live in figs/<slug>.js (slug: path under /resolvent/, "/" -> "__",
   "" -> "index"), listed in figs/index.json. Each module exports
     figures = [{ id, anchor: "#heading-id", title, caption, source: ["file:line"], mount(el, api) }]
   The engine places a card after the anchor's heading, mounts it when it nears the viewport,
   and hands mount() a small api: theme tokens, reduced motion, svg helpers, controls. */

const BASE = "/resolvent/";
let manifest = null;

function slugOf(path) {
  let p = path.startsWith(BASE) ? path.slice(BASE.length) : path.replace(/^\//, "");
  p = p.replace(/index\.html$/, "").replace(/\/$/, "");
  return p ? p.replace(/\//g, "__") : "index";
}

function tokens() {
  const s = getComputedStyle(document.body), g = (n) => s.getPropertyValue(n).trim();
  return { ink: g("--rs-ink"), dim: g("--rs-dim"), faint: g("--rs-faint"), line: g("--rs-line"),
    line2: g("--rs-line2"), card: g("--rs-card"), panel: g("--rs-panel"), panel2: g("--rs-panel2"),
    primary: g("--rs-primary"), bar: g("--rs-bar"), violet: g("--rs-violet"), proved: g("--rs-proved"),
    kill: g("--rs-kill"), open: g("--rs-open"), unmet: g("--rs-unmet"), link: g("--rs-link") };
}

const NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function el(tag, attrs = {}, parent, text) {
  const e = document.createElement(tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function slider(parent, { label, min, max, step, value, fmt = (v) => v, oninput }) {
  const lab = el("label", { class: "vz-ctl" }, parent);
  el("span", {}, lab, label);
  const inp = el("input", { type: "range", min, max, step, value, "aria-label": label }, lab);
  const out = el("output", {}, lab, fmt(+value));
  inp.addEventListener("input", () => { out.textContent = fmt(+inp.value); oninput(+inp.value); });
  return inp;
}
function toggle(parent, { label, options, value, onchange }) {
  const g = el("div", { class: "vz-seg", role: "group", "aria-label": label }, parent);
  const bs = options.map(([v, t]) => {
    const b = el("button", { type: "button", "aria-pressed": String(v === value) }, g, t);
    b.addEventListener("click", () => { bs.forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true"); onchange(v); });
    return b;
  });
  return g;
}
function scale(d0, d1, r0, r1, log) {
  const f = log ? Math.log10 : (x) => x, a = f(d0), b = f(d1);
  return (x) => r0 + ((f(x) - a) / (b - a)) * (r1 - r0);
}
function fmt(x, p = 4) {
  if (!isFinite(x)) return String(x);
  const a = Math.abs(x);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? x.toExponential(2) : (+x.toPrecision(p)).toString();
}

/* Axes plot: series [{pts:[[x,y]], color, dash, label}], optional marks [{x|y, color, label}]. */
function plot(parent, { w = 640, h = 280, x, y, xlog, ylog, xlabel, ylabel, series = [], marks = [], title }) {
  const T = tokens(), m = { l: 56, r: 16, t: 22, b: 40 };
  const s = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": title || ylabel || "plot" }, parent);
  const X = scale(x[0], x[1], m.l, w - m.r, xlog), Y = scale(y[0], y[1], h - m.b, m.t, ylog);
  const ticks = (d, log) => { const out = [];
    if (log) { const lo = Math.ceil(Math.log10(d[0])), hi = Math.floor(Math.log10(d[1])), st = Math.max(1, Math.ceil((hi - lo) / 6));
      for (let e = lo; e <= hi; e += st) out.push(10 ** e); }
    else { const st = niceStep((d[1] - d[0]) / 5); for (let v = Math.ceil(d[0] / st) * st; v <= d[1] + 1e-12; v += st) out.push(+v.toFixed(10)); }
    return out; };
  for (const v of ticks(y, ylog)) { svg("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), stroke: T.line, "stroke-width": 1 }, s);
    svg("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "vz-t" }, s).textContent = fmt(v, 3); }
  for (const v of ticks(x, xlog)) svg("text", { x: X(v), y: h - m.b + 16, "text-anchor": "middle", class: "vz-t" }, s).textContent = fmt(v, 3);
  svg("line", { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: T.line2 }, s);
  if (xlabel) svg("text", { x: (m.l + w - m.r) / 2, y: h - 6, "text-anchor": "middle", class: "vz-t vz-lab" }, s).textContent = xlabel;
  if (ylabel) svg("text", { x: m.l + 6, y: m.t + 2, class: "vz-t vz-lab" }, s).textContent = ylabel;
  const g = svg("g", {}, s);
  const draw = (ser, mk) => {
    g.replaceChildren();
    for (const k of mk || marks) {
      if (k.x !== undefined) svg("line", { x1: X(k.x), x2: X(k.x), y1: m.t, y2: h - m.b, stroke: k.color || T.kill, "stroke-dasharray": "4 4" }, g);
      if (k.y !== undefined) svg("line", { x1: m.l, x2: w - m.r, y1: Y(k.y), y2: Y(k.y), stroke: k.color || T.kill, "stroke-dasharray": "4 4" }, g);
      if (k.label) svg("text", { x: k.x !== undefined ? X(k.x) + 5 : w - m.r - 4, y: k.y !== undefined ? Y(k.y) - 5 : m.t + 12,
        "text-anchor": k.x !== undefined ? "start" : "end", class: "vz-t", fill: k.color || T.kill }, g).textContent = k.label;
    }
    for (const r of ser || series) {
      const d = r.pts.filter((p) => isFinite(p[1])).map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(Math.min(Math.max(p[1], y[0]), y[1])).toFixed(1)).join("");
      svg("path", { d, fill: "none", stroke: r.color || T.bar, "stroke-width": r.width || 2, "stroke-dasharray": r.dash || "" }, g);
      for (const p of r.dots || []) svg("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: r.color || T.bar }, g);
      if (r.label) { const p = r.pts[r.pts.length - 1];
        svg("text", { x: X(p[0]) - 4, y: Y(Math.min(Math.max(p[1], y[0]), y[1])) - 6, "text-anchor": "end", class: "vz-t", fill: r.color || T.bar }, g).textContent = r.label; }
    }
  };
  draw();
  return { svg: s, X, Y, draw, g };
}
function niceStep(r) { const p = 10 ** Math.floor(Math.log10(r)), f = r / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }

/* Horizontal bars comparing measured values against a bar/control. rows: [{label, value, color, note}] */
function bars(parent, { rows, max, unit = "", mark, markLabel }) {
  const T = tokens(), w = 640, rh = 34, h = rows.length * rh + (mark !== undefined ? 22 : 8), l = 190;
  const s = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": rows.map((r) => r.label + " " + r.value + unit).join("; ") }, parent);
  const X = scale(0, max, l, w - 70);
  rows.forEach((r, i) => {
    const y = 6 + i * rh;
    svg("text", { x: l - 10, y: y + 17, "text-anchor": "end", class: "vz-t vz-row" }, s).textContent = r.label;
    svg("rect", { x: l, y: y + 4, width: w - 70 - l, height: 18, rx: 9, fill: T.panel2 }, s);
    svg("rect", { x: l, y: y + 4, width: Math.max(3, X(Math.min(r.value, max)) - l), height: 18, rx: 9, fill: r.color || T.bar, class: "vz-grow" }, s);
    svg("text", { x: X(Math.min(r.value, max)) + 6, y: y + 18, class: "vz-t vz-val" }, s).textContent = (r.text || fmt(r.value)) + unit;
  });
  if (mark !== undefined) {
    svg("line", { x1: X(mark), x2: X(mark), y1: 2, y2: h - 18, stroke: T.ink, "stroke-dasharray": "3 3" }, s);
    svg("text", { x: X(mark), y: h - 4, "text-anchor": "middle", class: "vz-t" }, s).textContent = markLabel || fmt(mark);
  }
  return s;
}

/* Step-through player: steps [{title, body(html), on(stepIndex)}] */
function stepper(parent, steps, onStep) {
  const box = el("div", { class: "vz-steps" }, parent);
  const nav = el("div", { class: "vz-stepnav" }, box);
  const prev = el("button", { type: "button", "aria-label": "previous step" }, nav, "←");
  const lab = el("span", { class: "vz-stepn", "aria-live": "polite" }, nav);
  const next = el("button", { type: "button", "aria-label": "next step" }, nav, "→");
  const body = el("div", { class: "vz-stepbody" }, box);
  let i = 0;
  const go = (k) => { i = Math.max(0, Math.min(steps.length - 1, k));
    lab.textContent = `step ${i + 1} / ${steps.length} · ${steps[i].title}`;
    body.innerHTML = steps[i].body || ""; onStep && onStep(i);
    prev.disabled = i === 0; next.disabled = i === steps.length - 1;
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([body]).catch(() => {}); };
  prev.onclick = () => go(i - 1); next.onclick = () => go(i + 1);
  go(0);
  return { go, get i() { return i; } };
}

const api = { svg, el, slider, toggle, plot, bars, stepper, scale, fmt, tokens,
  get reduceMotion() { return matchMedia("(prefers-reduced-motion: reduce)").matches; } };

async function boot() {
  try {
    if (!manifest) manifest = await (await fetch(new URL("./figs/index.json", import.meta.url))).json();
  } catch (e) { return; }
  const slug = slugOf(location.pathname);
  if (!manifest.includes(slug)) return;
  let mod;
  try { mod = await import(new URL(`./figs/${slug}.js`, import.meta.url)); }
  catch (e) { console.warn("viz: cannot load figures for", slug, e); return; }
  let n = 0;
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    io.unobserve(e.target);
    const f = e.target._fig;
    try { f.mount(e.target.querySelector(".vz-body"), api); }
    catch (err) { console.warn("viz: figure failed", f.id, err); e.target.querySelector(".vz-body").textContent = "Figure failed to render."; }
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([e.target]).catch(() => {});
  }), { rootMargin: "300px 0px" });
  for (const f of mod.figures || []) {
    const h = document.querySelector(f.anchor);
    if (!h) { console.warn("viz: anchor not found", f.id, f.anchor); continue; }
    if (document.getElementById("vz-" + f.id)) continue;
    n++;
    const card = document.createElement("figure");
    card.className = "vz"; card.id = "vz-" + f.id; card._fig = f;
    card.innerHTML = `<p class="vz-kick">Fig. ${n} · ${h.textContent.replace(/[¶#]\s*$/, "").trim()}</p>` +
      `<h4 class="vz-title"></h4><div class="vz-body"></div><figcaption class="vz-cap"></figcaption>` +
      (f.source ? `<details class="vz-src"><summary>source</summary><code></code></details>` : "");
    card.querySelector(".vz-title").textContent = f.title;
    card.querySelector(".vz-cap").innerHTML = f.caption || "";
    if (f.source) card.querySelector(".vz-src code").textContent = f.source.join(" · ");
    let at = h;
    if (f.place === "after-first-para") { const p = h.nextElementSibling; if (p && p.tagName === "P") at = p; }
    at.after(card);
    io.observe(card);
  }
}

if (window.document$ && window.document$.subscribe) window.document$.subscribe(boot);
else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
