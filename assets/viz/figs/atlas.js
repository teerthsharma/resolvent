/* Proof atlas: every declaration in the Lean development, placed by dependency depth
   (left: foundations, right: results built on them), coloured by file. Click a node to see its
   statement and light up what it uses (upstream) and what uses it (downstream). */
const DATA = new URL("../lean_atlas.json", import.meta.url);

export const figures = [{
  id: "atlas", anchor: "#proof-atlas", title: "Every theorem, and what it stands on",
  caption: "Each dot is a Lean declaration; large dots are theorems and lemmas. Columns are dependency depth. " +
    "Click one: blue lines are what it uses, violet what uses it. Search matches names and statements.",
  source: ["lean/CEQ.lean", "lean/CEQ/*.lean", "scripts/lean_atlas.py"],
  async mount(el, api) {
    const d = await (await fetch(DATA)).json();
    const T = api.tokens(), byName = new Map(d.decls.map((x) => [x.name, x]));
    // depth = 1 + max depth of uses (memoised, cycle-safe)
    const depth = new Map(), seen = new Set();
    const dep = (n) => { if (depth.has(n)) return depth.get(n); if (seen.has(n)) return 0; seen.add(n);
      const v = 1 + Math.max(-1, ...byName.get(n).uses.map(dep)); depth.set(n, v); return v; };
    d.decls.forEach((x) => dep(x.name));
    const files = d.files.map((f) => f.replace("lean/", "").replace(".lean", ""));
    const hue = (f) => `hsl(${(files.indexOf(f.replace("lean/", "").replace(".lean", "")) * 360 / files.length + 205) % 360} 62% 52%)`;

    const nThm = d.counts.theorem + d.counts.lemma;
    const stats = api.el("p", { class: "vz-stats" }, el);
    stats.innerHTML = `<b>${d.counts.theorem}</b> theorems · <b>${d.counts.lemma}</b> lemmas · <b>${d.decls.length - nThm}</b> definitions · <b>${d.decls.reduce((a, x) => a + x.uses.length, 0)}</b> dependencies · <b>${files.length}</b> files`;
    const ctl = api.el("div", { class: "vz-ctls" }, el);
    const q = api.el("input", { type: "search", placeholder: "search a theorem…", "aria-label": "search theorems", class: "vz-search" }, ctl);
    const legend = api.el("div", { class: "vz-legend" }, ctl);
    files.forEach((f) => { const s = api.el("span", {}, legend, f.replace("CEQ/", "")); s.style.setProperty("--c", hue(f)); });

    const cols = Math.max(...depth.values()) + 1, groups = Array.from({ length: cols }, () => []);
    d.decls.forEach((x) => groups[depth.get(x.name)].push(x));
    groups.forEach((g) => g.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line));
    const rows = Math.max(...groups.map((g) => g.length));
    const W = 980, H = Math.max(420, rows * 9 + 40), px = (c) => 30 + c * ((W - 60) / Math.max(1, cols - 1));
    const s = api.svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `dependency graph of ${d.decls.length} Lean declarations` }, el);
    const pos = new Map();
    groups.forEach((g, c) => g.forEach((x, i) => pos.set(x.name, [px(c), 20 + (i + .5) * ((H - 40) / g.length)])));
    const edges = api.svg("g", { opacity: .9 }, s), nodes = api.svg("g", {}, s), hot = api.svg("g", {}, s);
    for (const x of d.decls) for (const u of x.uses) {
      const [x1, y1] = pos.get(x.name), [x2, y2] = pos.get(u);
      api.svg("path", { d: `M${x1} ${y1} C${(x1 + x2) / 2} ${y1} ${(x1 + x2) / 2} ${y2} ${x2} ${y2}`, fill: "none", stroke: T.line2, "stroke-width": .6, opacity: .45 }, edges);
    }
    const dots = new Map();
    for (const x of d.decls) {
      const [cx, cy] = pos.get(x.name), thm = x.kind === "theorem" || x.kind === "lemma";
      const c = api.svg("circle", { cx, cy, r: thm ? 4.2 : 2.6, fill: thm ? hue(x.file) : T.card, stroke: hue(x.file), "stroke-width": 1.4, tabindex: 0, role: "button", "aria-label": x.kind + " " + x.name, style: "cursor:pointer" }, nodes);
      c.addEventListener("click", () => select(x.name));
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(x.name); } });
      dots.set(x.name, c);
    }
    const panel = api.el("div", { class: "vz-panel", "aria-live": "polite" }, el);
    panel.innerHTML = "<p class='vz-hint'>Pick a dot, or search.</p>";

    const cone = (n, key, acc = new Set()) => { for (const u of byName.get(n)[key]) if (!acc.has(u)) { acc.add(u); cone(u, key, acc); } return acc; };
    function select(n) {
      const x = byName.get(n), up = cone(n, "uses"), down = cone(n, "usedBy");
      hot.replaceChildren();
      const line = (a, b, col) => { const [x1, y1] = pos.get(a), [x2, y2] = pos.get(b);
        api.svg("path", { d: `M${x1} ${y1} C${(x1 + x2) / 2} ${y1} ${(x1 + x2) / 2} ${y2} ${x2} ${y2}`, fill: "none", stroke: col, "stroke-width": 1.6 }, hot); };
      [n, ...up].forEach((a) => byName.get(a).uses.forEach((b) => line(a, b, T.bar)));
      [n, ...down].forEach((a) => byName.get(a).usedBy.forEach((b) => line(b, a, T.violet)));
      dots.forEach((c, k) => c.setAttribute("opacity", k === n || up.has(k) || down.has(k) ? 1 : .18));
      const [cx, cy] = pos.get(n);
      api.svg("circle", { cx, cy, r: 9, fill: "none", stroke: T.ink, "stroke-width": 2 }, hot);
      const gh = `https://github.com/teerthsharma/resolvent/blob/master/${x.file}#L${x.line}`;
      const list = (a) => a.length ? a.map((u) => `<button type="button" data-n="${u}">${u}</button>`).join("") : "<i>none</i>";
      panel.innerHTML = `<p class="vz-kind">${x.kind} · <a href="${gh}">${x.file}:${x.line}</a></p><h5>${x.name}</h5>` +
        `<pre>${x.statement.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</pre>` +
        `<p class="vz-rel"><b>uses ${x.uses.length}</b> (${up.size} in its full cone)</p><div class="vz-chips">${list(x.uses)}</div>` +
        `<p class="vz-rel"><b>used by ${x.usedBy.length}</b> (${down.size} downstream)</p><div class="vz-chips">${list(x.usedBy)}</div>`;
      panel.querySelectorAll("button[data-n]").forEach((b) => b.onclick = () => select(b.dataset.n));
    }
    q.addEventListener("input", () => {
      const t = q.value.trim().toLowerCase();
      dots.forEach((c, k) => c.setAttribute("opacity", !t || k.toLowerCase().includes(t) || byName.get(k).statement.toLowerCase().includes(t) ? 1 : .12));
      if (!t) hot.replaceChildren();
    });
    q.addEventListener("keydown", (e) => { if (e.key === "Enter") { const t = q.value.trim().toLowerCase();
      const hit = d.decls.find((x) => x.name.toLowerCase().includes(t)); if (hit) select(hit.name); } });
    const star = d.decls.filter((x) => x.kind === "theorem").sort((a, b) => b.usedBy.length - a.usedBy.length)[0];
    if (star) select(star.name);
  },
}];
