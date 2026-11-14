/* North Star, shown. Values from docs/NORTH_STAR.md (lines in `source`). */
const S = "docs/NORTH_STAR.md";
export const figures = [
  { id: "inout", anchor: "#in-through-the-resolvent-out", title: "Forward only: in-state to out-state",
    caption: "The operator T maps what went in to what came out. No path is drawn, and nothing runs backwards: this is not inverse scattering.",
    source: [S + ":7", S + ":15-22", S + ":63"],
    mount(el, api) {
      const T = api.tokens(), s = api.svg("svg", { viewBox: "0 0 680 200", role: "img", "aria-label": "in-state, T, out-state" }, el);
      const st = (x, t) => { api.svg("circle", { cx: x, cy: 100, r: 36, fill: T.card, stroke: T.ink, "stroke-width": 1.8 }, s);
        api.svg("text", { x, y: 106, "text-anchor": "middle", class: "vz-row" }, s).textContent = t; };
      st(90, "|in⟩"); st(590, "|out⟩");
      api.svg("rect", { x: 230, y: 55, width: 220, height: 90, rx: 14, fill: T.panel2, stroke: T.line2 }, s);
      api.svg("text", { x: 340, y: 96, "text-anchor": "middle", class: "vz-row" }, s).textContent = "T = (I − V G₀)⁻¹ V";
      api.svg("text", { x: 340, y: 120, "text-anchor": "middle", class: "vz-t" }, s).textContent = "causal branch: the +iε margin";
      for (const [a, b] of [[126, 230], [450, 554]]) api.svg("line", { x1: a, y1: 100, x2: b - 6, y2: 100, stroke: T.bar, "stroke-width": 2.5, "marker-end": "" }, s);
      if (!api.reduceMotion) for (let k = 0; k < 3; k++) {
        const c = api.svg("circle", { r: 5, fill: T.violet }, s);
        const a = api.svg("animateMotion", { dur: "3s", begin: `-${k}s`, repeatCount: "indefinite", path: "M126 100 L554 100" }, c);
      }
    } },
  { id: "series", anchor: "#one-series-two-names", title: "The Born series is the hop expansion",
    caption: "Step through the terms. Each hop adds one more factor of V·G₀. Physics calls it the Born series; this project calls it the hop expansion (I − gP)⁻¹ = I + gP + (gP)² + …",
    source: [S + ":22-28"],
    mount(el, api) {
      const T = api.tokens(), s = api.svg("svg", { viewBox: "0 0 680 120", role: "img", "aria-label": "terms of the Born series" }, el);
      const terms = ["V", "VG₀V", "VG₀VG₀V", "VG₀VG₀VG₀V"], boxes = [];
      let x = 20;
      terms.forEach((t, i) => { const w = 40 + t.length * 11;
        const g = api.svg("g", {}, s);
        api.svg("rect", { x, y: 40, width: w, height: 40, rx: 9, fill: T.card, stroke: T.line2 }, g);
        api.svg("text", { x: x + w / 2, y: 65, "text-anchor": "middle", class: "vz-val" }, g).textContent = t;
        boxes.push(g); x += w + 26;
        if (i < terms.length - 1) api.svg("text", { x: x - 17, y: 66, "text-anchor": "middle", class: "vz-row" }, s).textContent = "+"; });
      api.svg("text", { x: x - 6, y: 66, class: "vz-row" }, s).textContent = "+ …";
      const steps = terms.map((t, i) => ({ title: `${i} hop${i === 1 ? "" : "s"}`,
        body: `<p>Term ${i + 1}: <code>${t}</code> — ${i === 0 ? "the direct interaction, no propagation" : `${i} factor${i > 1 ? "s" : ""} of V·G₀: the signal propagates ${i} time${i > 1 ? "s" : ""} before it scatters out`}.</p>` }));
      api.stepper(el, steps, (k) => boxes.forEach((b, i) => b.setAttribute("opacity", i <= k ? 1 : .2)));
    } },
  { id: "converge", anchor: "#it-converges-when-1", title: "Error against the exact T, by number of hops",
    caption: "The two measured instances (seed 0, n = 12, complex128): ρ = 0.2596 falls about six orders of magnitude per ten hops; ρ = 1.6688, with E inside the band, grows. The dashed line is ρ<sup>k</sup> for the slider's ρ: illustrative, not measured.",
    source: [S + ":39-48", S + ":52"],
    mount(el, api) {
      const T = api.tokens(), ctl = api.el("div", { class: "vz-ctls" }, el), box = api.el("div", {}, el);
      const conv = [[5, 1.8e-4], [10, 2.1e-7], [20, 2.8e-13]], div = [[5, 3.5], [10, 25], [20, 4.1e3]];
      const p = api.plot(box, { x: [0, 22], y: [1e-14, 1e5], ylog: true, xlabel: "hops k", ylabel: "‖error‖_F", title: "Born series error" });
      const draw = (r) => p.draw([
        { pts: conv, dots: conv, color: T.proved, label: "ρ = 0.2596 · converges" },
        { pts: div, dots: div, color: T.unmet, label: "ρ = 1.6688 · diverges" },
        { pts: Array.from({ length: 23 }, (_, k) => [k, r ** k]), color: T.bar, dash: "5 5", width: 1.5 }]);
      api.slider(ctl, { label: "illustrative ρ", min: 0.1, max: 1.6, step: 0.01, value: 0.2596, fmt: (v) => v.toFixed(2) + (v < 1 ? " · converges" : " · diverges"), oninput: draw });
      draw(0.2596);
    } },
  { id: "unchanged", anchor: "#what-it-changes", title: "Four verdicts it leaves exactly where they were",
    caption: "The reconciliation gives a vocabulary and a lineage. It adds no measurement.",
    source: [S + ":75-78"],
    mount(el, api) {
      const T = api.tokens(), rows = [
        ["held", T.proved, "Represents order", "0.8620 vs control 0.2860 · 404 params · 5/5 seeds"],
        ["lost", T.unmet, "Does not predict", "resolution 0.001469 of a 0.10117 ceiling (1.45%)"],
        ["open", T.open, "The gate closes, cause unknown", "four proposed mechanisms, four refuted"],
        ["catalogued", T.kill, "Checks that could not fail", "twelve, each for a structural reason"]];
      const g = api.el("div", { class: "vz-verdicts" }, el);
      rows.forEach(([k, c, t, n]) => { const d = api.el("div", {}, g); d.style.setProperty("--c", c);
        d.innerHTML = `<span class="vz-chip">${k}</span><b>${t}</b><span>${n}</span>`; });
    } }];
