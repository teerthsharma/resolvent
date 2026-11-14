/* Status, shown. Every value below is copied from docs/STATUS.md (line numbers in `source`). */
const S = "docs/STATUS.md";
const C = [
  { name: "Understands causality", score: "2.5 of 5", pct: 50, gates: [
    ["An operator whose entries encode path structure, not pairwise affinity", "done", "W_ij = G_ij e^{s_ij} / Z_i^β, exact at the (β, qk) corners"],
    ["A bed that demands order, separated from a commuting control", "done", "operator 0.8620 vs control 0.2860 at 404 params, 5/5 seeds"],
    ["The non-commutative matrix separated from recurrence alone", "half", "84.9% of the gap needs the matrix, 15.1% is recurrence"],
    ["A second, independent bed carrying the same property", "not", "—"],
    ["The mechanism named and tested", "not", "both named candidates failed"]] },
  { name: "Beats anything before it", score: "4 of 9", pct: 44, gates: [
    ["A win exists", "failed", "an ALiBi twin recovers 108.8 / 106.7 / 108.8% of the win: against it, no win"],
    ["At matched parameters, excess reported", "done", "783 excess on all three domains"],
    ["More than one domain", "done", "TinyStories −0.2473, WikiText-103 −0.3285, codeparrot −0.3301"],
    ["Against the strongest baseline lacking the property", "failed", "a forget gate on the twin recovers 100.2 / 100.3 / 110.9%"],
    ["A seed interval on every domain", "not", "5 / 5 / 1 seeds"],
    ["Survives a varied-split protocol", "done", "C_win +0.2470, 95% CI [+0.2271, +0.2668]"],
    ["The mechanism identified", "done", "a data-dependent forget gate: prior art"],
    ["Against a strong published baseline", "failed", "FoX minus family −0.0004 / −0.0008 / −0.0284"],
    ["Matched on compute", "failed", "6.72× the twin's wall clock"]] },
  { name: "Carries the weight of attention or JEPA", score: "1.5 of 5", pct: 30, gates: [
    ["Runs at transformer scale", "not", "~725k params"],
    ["Cost competitive with FlashAttention", "not", "6.72× the twin's clock"],
    ["A prediction capability self-attention lacks", "not", "committor tied by eight bins of piece count"],
    ["The block-summary path proven exact", "done", "8.9e-16 / 3.2e-13 / 2.6e-10"],
    ["Formal backing", "half", "12 Lean files, 6 sorry remaining"]] }];
const COL = (T) => ({ done: T.proved, half: T.kill, failed: T.unmet, not: T.line2 });
const LAB = { done: "done", half: "half", failed: "measured false", not: "not done" };

export const figures = [
  { id: "board", anchor: "#the-gate-board", title: "19 gates, three conditions",
    caption: "Filled green: done. Amber: half. Red: measured false. Hollow: not done. Hover or focus a tile for its evidence.",
    source: [S + ":24-52", S + ":96-106"],
    mount(el, api) {
      const T = api.tokens(), col = COL(T), W = 760, rowH = 92, s = api.svg("svg", { viewBox: `0 0 ${W} ${C.length * rowH + 8}`, role: "img", "aria-label": "gate board" }, el);
      const tip = api.el("p", { class: "vz-tip", "aria-live": "polite" }, el, "Hover a tile.");
      C.forEach((c, r) => {
        const y = 8 + r * rowH;
        api.svg("text", { x: 0, y: y + 14, class: "vz-row" }, s).textContent = `${r + 1}. ${c.name}`;
        api.svg("text", { x: W, y: y + 14, "text-anchor": "end", class: "vz-val" }, s).textContent = `${c.score} · ${c.pct}%`;
        c.gates.forEach(([g, st, ev], i) => {
          const x = i * 82, t = api.svg("g", { tabindex: 0, role: "button", "aria-label": `${g}: ${LAB[st]}. ${ev}`, style: "cursor:default" }, s);
          const rect = api.svg("rect", { x, y: y + 26, width: 72, height: 54, rx: 10, fill: st === "not" ? T.card : col[st],
            stroke: st === "not" ? T.line2 : col[st], "stroke-width": 1.5, "stroke-dasharray": st === "not" ? "4 4" : "" }, t);
          if (st === "half") rect.setAttribute("fill", `url(#half${r}${i})`);
          if (st === "half") { const d = api.svg("defs", {}, s), lg = api.svg("linearGradient", { id: `half${r}${i}` }, d);
            api.svg("stop", { offset: ".5", "stop-color": col.half }, lg); api.svg("stop", { offset: ".5", "stop-color": T.card }, lg); }
          api.svg("text", { x: x + 36, y: y + 58, "text-anchor": "middle", class: "vz-val", fill: st === "not" ? T.faint : "#fff" }, t).textContent = i + 1;
          const show = () => { tip.innerHTML = `<b>${r + 1}.${i + 1} ${g}</b> — ${LAB[st]}<br>${ev}`; rect.setAttribute("stroke", T.ink); };
          const hide = () => rect.setAttribute("stroke", st === "not" ? T.line2 : col[st]);
          t.addEventListener("mouseenter", show); t.addEventListener("focus", show);
          t.addEventListener("mouseleave", hide); t.addEventListener("blur", hide);
        });
      });
    } },
  { id: "weakest", anchor: "#why-30-not-42", title: "A conjunctive goal is its weakest leg",
    caption: "Counting gates across all three conditions gives 8 of 19. But the goal needs all three, so the honest number is the weakest leg: condition 3 at 30%.",
    source: [S + ":8-20"],
    mount(el, api) {
      const T = api.tokens();
      api.bars(el, { max: 100, unit: "%", mark: 30, markLabel: "weakest leg 30%", rows: [
        { label: "gates passed, all three", value: 42, text: "8 of 19 — 42", color: T.bar },
        ...C.map((c, i) => ({ label: `condition ${i + 1}`, value: c.pct, text: `${c.score} — ${c.pct}`, color: c.pct === 30 ? T.unmet : T.line2 }))] });
    } },
  { id: "owned", anchor: "#the-win-and-who-owns-it", title: "The language-model win is recovered by things that already exist",
    caption: "Share of the family's win (C_win) recovered by each zero-parameter or published control, at split seeds 0 / 1 / 2. At 100% the control owns the whole win.",
    source: [S + ":44", S + ":47"],
    mount(el, api) {
      const T = api.tokens(), ctl = api.el("div", { class: "vz-ctls" }, el), box = api.el("div", {}, el);
      const data = { RoPE: [96.3, 99.7, 101.4], ALiBi: [108.8, 106.7, 108.8], "Forget gate (FoX)": [100.2, 100.3, 110.9] };
      const draw = (seed) => { box.replaceChildren(); api.bars(box, { max: 120, unit: "%", mark: 100, markLabel: "the whole win",
        rows: Object.entries(data).map(([k, v]) => ({ label: k, value: v[seed], color: v[seed] >= 100 ? T.unmet : T.kill })) }); };
      api.toggle(ctl, { label: "split seed", value: 0, options: [[0, "seed 0"], [1, "seed 1"], [2, "seed 2"]], onchange: draw });
      draw(0);
    } },
  { id: "cost", anchor: "#what-it-costs", title: "Wall clock per training cell",
    caption: "Same steps and tokens. The family takes 514.8 s; its own twin 76.6 s (6.72×); FoX 131.9 s; a twin grown to 9,976,320 params takes 226 s and beats the family by 0.1672.",
    source: [S + ":51-52"],
    mount(el, api) {
      const T = api.tokens();
      api.bars(el, { max: 540, unit: " s", rows: [
        { label: "the family (f)", value: 514.8, color: T.unmet },
        { label: "twin grown to 9.98M", value: 226, color: T.proved },
        { label: "FoX", value: 131.9, color: T.bar },
        { label: "softmax twin (a)", value: 76.6, color: T.line2 }] });
    } },
  { id: "holds", anchor: "#the-one-result-that-holds", title: "Order, separated from a control that cannot see order",
    caption: "Operator 0.8620 against the commuting control 0.2860 at matched 404 params, 5/5 seeds. The control saturates within 0.031 of its own ceiling, 0.3110: it has used everything it can see.",
    source: [S + ":31"],
    mount(el, api) {
      const T = api.tokens();
      api.bars(el, { max: 1, mark: 0.3110, markLabel: "control's ceiling 0.3110", rows: [
        { label: "operator", value: 0.8620, color: T.proved },
        { label: "commuting control", value: 0.2860, color: T.line2 }] });
    } }];
