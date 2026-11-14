/* The home story: one sticky stage, redrawn by CSS state as each chapter scrolls in.
   Every chapter is a <g data-ch="…"> in one SVG; the active chapter is the stage's data-step,
   and CSS crossfades between them. Numbers are the ones stated in the chapter text. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  function E(tag, a, p, t) { var e = document.createElementNS(NS, tag); for (var k in a) e.setAttribute(k, a[k]); if (t !== undefined) e.textContent = t; if (p) p.appendChild(e); return e; }
  var N = 9, Y = 300, X = function (i) { return 60 + i * 65; }, Q = N - 1;
  var arc = function (a, b, h) { return "M" + X(a) + " " + Y + " Q" + (X(a) + X(b)) / 2 + " " + (Y - h) + " " + X(b) + " " + Y; };

  function build(svg) {
    var ch = function (name) { return E("g", { "data-ch": name, class: "ch" }, svg); };
    var cap = { hop: "the query t₈ reads each earlier token directly", chain: "a relay: each step multiplies", resolvent: "every route, summed in one solve",
      gate: "gate 4 closed: routes to t₀–t₃ read exactly 0", cube: "the family as a cube: β · g · qk", held: "S5 as bare integers · 404 params · 5/5 seeds",
      died: "share of the family's language-model win each control recovers", proofs: "207 theorems and lemmas · 523 dependencies", now: "score only the far band" };

    // shared token row (hop … gate)
    var row = E("g", { class: "row" }, svg);
    // 01 direct arcs, widths = illustrative softmax weights
    var w = [0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.2, 0.35], g1 = ch("hop");
    for (var j = 0; j < Q; j++) E("path", { d: arc(Q, j, 40 + (Q - j) * 26), class: "a-direct", "stroke-width": 1 + w[j] * 14, style: "--d:" + (j * 60) + "ms" }, g1);
    for (j = 0; j < Q; j++) E("rect", { x: X(j) - 12, y: Y + 34, width: 24, height: 2 + w[j] * 180, rx: 3, class: "wbar", style: "--d:" + (j * 50) + "ms" }, g1);
    E("text", { x: 320, y: Y + 118, "text-anchor": "middle", class: "s-dim" }, g1, "one row of proportions · sums to 1 (illustrative)");
    // 02 chain hops with relayed packets
    var g2 = ch("chain");
    for (j = 0; j < Q; j++) E("path", { d: arc(j + 1, j, 34), class: "a-hop", style: "--d:" + ((Q - j) * 90) + "ms" }, g2);
    var relay = "M" + X(Q) + " " + Y; for (j = Q - 1; j >= 0; j--) relay += " Q" + (X(j) + X(j + 1)) / 2 + " " + (Y - 34) + " " + X(j) + " " + Y;
    [0, 1.2, 2.4].forEach(function (b) { var c = E("circle", { r: 5, class: "pkt" }, g2); E("animateMotion", { dur: "3.6s", begin: "-" + b + "s", repeatCount: "indefinite", path: relay }, c); });
    for (j = 0; j < Q; j++) E("text", { x: (X(j) + X(j + 1)) / 2, y: Y - 44, "text-anchor": "middle", class: "s-dim" }, g2, "×m" + (j + 1));
    // 03 all routes
    var g3 = ch("resolvent");
    for (j = 0; j < Q; j++) for (var k = j; k < Q; k++) if ((k - j) % 2 === 0) E("path", { d: arc(k + 1, j, 18 + (k + 1 - j) * 22), class: "a-route", style: "--d:" + ((k - j) * 40) + "ms" }, g3);
    E("text", { x: 320, y: 70, "text-anchor": "middle", class: "s-eq" }, g3, "(I − gP)⁻¹ = I + gP + (gP)² + …");
    // 04 gate
    var g4 = ch("gate");
    for (j = 0; j < Q; j++) E("path", { d: arc(Q, j, 40 + (Q - j) * 26), class: j < 4 ? "a-dead" : "a-live" }, g4);
    E("line", { x1: (X(3) + X(4)) / 2, x2: (X(3) + X(4)) / 2, y1: Y - 30, y2: Y + 30, class: "gatebar" }, g4);
    for (j = 0; j < Q; j++) E("text", { x: X(j), y: Y + 52, "text-anchor": "middle", class: j < 4 ? "s-zero" : "s-val" }, g4, j < 4 ? "0" : "·");
    E("text", { x: (X(3) + X(4)) / 2, y: Y - 40, "text-anchor": "middle", class: "s-kill" }, g4, "m₄ = 0");
    // tokens on top of the row chapters
    for (j = 0; j < N; j++) { E("circle", { cx: X(j), cy: Y, r: j === Q ? 15 : 12, class: j === Q ? "tok q" : "tok" }, row);
      E("text", { x: X(j), y: Y + 5, "text-anchor": "middle", class: "s-tok" }, row, "t" + "₀₁₂₃₄₅₆₇₈"[j]); }
    // 05 cube
    var g5 = ch("cube"), cx = 250, cy = 250, s = 150, o = 70;
    var P = { a: [cx, cy], b: [cx + s, cy], c: [cx + s, cy - s], d: [cx, cy - s], e: [cx + o, cy - o * .7], f: [cx + s + o, cy - o * .7], g: [cx + s + o, cy - s - o * .7], h: [cx + o, cy - s - o * .7] };
    [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["b", "f"], ["c", "g"], ["d", "h"], ["f", "g"], ["g", "h"], ["e", "f"], ["e", "h"], ["a", "e"]].forEach(function (p, i) {
      E("line", { x1: P[p[0]][0], y1: P[p[0]][1], x2: P[p[1]][0], y2: P[p[1]][1], class: "edge", style: "--d:" + (i * 45) + "ms" }, g5); });
    E("line", { x1: P.a[0], y1: P.a[1], x2: P.e[0], y2: P.e[1], class: "edge-hot" }, g5);
    E("circle", { cx: P.c[0], cy: P.c[1], r: 9, class: "pt-soft" }, g5); E("text", { x: P.c[0] + 16, y: P.c[1] + 4, class: "s-lab" }, g5, "softmax attention");
    E("circle", { cx: P.d[0], cy: P.d[1], r: 9, class: "pt-kern" }, g5); E("text", { x: P.d[0] - 16, y: P.d[1] + 4, "text-anchor": "end", class: "s-lab" }, g5, "unnormalized kernel");
    E("text", { x: P.a[0] + 10, y: P.a[1] + 30, class: "s-lab kill" }, g5, "exact path product (an edge)");
    E("text", { x: cx + s / 2, y: cy + 62, "text-anchor": "middle", class: "s-dim" }, g5, "β : total → mean");
    E("text", { x: cx - 30, y: cy - s / 2, "text-anchor": "end", class: "s-dim" }, g5, "qk");
    E("text", { x: P.f[0] + 10, y: P.f[1] + 20, class: "s-dim" }, g5, "g");
    // 06 held, 07 died: bars
    var bars = function (name, rows, max, mark, markLab) {
      var g = ch(name), x0 = 200, W = 380;
      rows.forEach(function (r, i) { var y = 150 + i * 70;
        E("text", { x: x0 - 14, y: y + 22, "text-anchor": "end", class: "s-lab" }, g, r[0]);
        E("rect", { x: x0, y: y, width: W, height: 34, rx: 17, class: "track" }, g);
        var b = E("rect", { x: x0, y: y, width: W * Math.min(r[1], max) / max, height: 34, rx: 17, class: "fill " + r[3], style: "--d:" + (i * 140) + "ms" }, g);
        E("text", { x: x0 + W * Math.min(r[1], max) / max - 12, y: y + 23, "text-anchor": "end", class: "s-in" }, g, r[2]); });
      if (mark !== undefined) { var mx = x0 + W * mark / max;
        E("line", { x1: mx, x2: mx, y1: 130, y2: 150 + rows.length * 70, class: "mark" }, g);
        E("text", { x: mx, y: 122, "text-anchor": "middle", class: "s-dim" }, g, markLab); }
    };
    bars("held", [["operator", .862, "0.8620", "ok"], ["control, blind to order", .286, "0.2860", "dim"]], 1, .311, "control's ceiling 0.3110");
    bars("died", [["ALiBi twin", 108.8, "108.8%", "bad"], ["forget gate (FoX)", 100.2, "100.2%", "bad"], ["RoPE twin", 96.3, "96.3%", "warn"]], 120, 100, "the whole win");
    // 08 proofs constellation, deterministic
    var g8 = ch("proofs"), seed = 7, rnd = function () { seed = seed * 16807 % 2147483647; return seed / 2147483647; }, pts = [];
    for (j = 0; j < 207; j++) { var a = rnd() * 6.283, r = 40 + Math.sqrt(rnd()) * 170; pts.push([320 + Math.cos(a) * r * 1.35, 230 + Math.sin(a) * r * .9]); }
    for (j = 0; j < 150; j++) { var p1 = pts[j], p2 = pts[(j * 37 + 11) % 207];
      if (Math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 120) E("line", { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], class: "link" }, g8); }
    pts.forEach(function (p, i) { E("circle", { cx: p[0], cy: p[1], r: i < 166 ? 3.2 : 2.2, class: i < 166 ? "thm" : "lem", style: "--d:" + (i * 6) + "ms" }, g8); });
    E("text", { x: 320, y: 420, "text-anchor": "middle", class: "s-dim" }, g8, "each dot one Lean declaration (layout illustrative; the atlas has the real graph)");
    // 09 far band
    var g9 = ch("now"), n = 48;
    E("rect", { x: 40, y: 190, width: 560 * .62, height: 150, rx: 14, class: "near" }, g9);
    E("rect", { x: 40 + 560 * .62, y: 190, width: 560 * .38, height: 150, rx: 14, class: "far" }, g9);
    E("text", { x: 40 + 560 * .31, y: 180, "text-anchor": "middle", class: "s-dim" }, g9, "reachable by a softmax stack's best window");
    E("text", { x: 40 + 560 * .81, y: 180, "text-anchor": "middle", class: "s-lab" }, g9, "the far band");
    for (j = 0; j < n; j++) E("circle", { cx: 52 + j * (536 / (n - 1)), cy: 265, r: 4.5, class: j > n * .62 ? "tok far-t" : "tok" }, g9);
    E("path", { d: "M" + (52 + (n - 1) * 536 / (n - 1)) + " 265 Q 330 60 64 265", class: "a-route" }, g9);
    return cap;
  }

  function boot() {
    var root = document.querySelector(".rs.story"); if (!root || root.dataset.ready) return; root.dataset.ready = "1";
    var svg = root.querySelector("[data-stage]"), stage = root.querySelector(".stage"), capEl = root.querySelector("[data-stagecap]");
    var cap = build(svg), steps = root.querySelectorAll(".step");
    var set = function (s) { stage.setAttribute("data-step", s); root.querySelectorAll(".step").forEach(function (x) { x.classList.toggle("on", x.dataset.step === s); }); capEl.textContent = cap[s] || ""; };
    set("hop");
    var io = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) set(e.target.dataset.step); }); }, { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach(function (s) { io.observe(s); });
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(boot);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
