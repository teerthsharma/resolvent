/* resolvent — the figures on the home page are instruments, not pictures.
   Gate demo: click any gate; every path across a closed gate reads exactly 0.
   Cube: the marker tours the three named settings, or jumps to a clicked one.
   Bars: the Hodge shares grow when they first come into view.
   Motion is off under prefers-reduced-motion; every control stays usable. */
(function () {
  "use strict";
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------- the gate demo ---- */
  var fig = document.querySelector('[data-rs="gates"]');
  if (fig) {
    var M_OPEN = 0.9;                       // an open gate passes 0.9 of the path
    var closed = { 5: true };               // m5 starts closed, as the caption says
    var arcs = fig.querySelectorAll("[data-j]");
    var outs = fig.querySelectorAll("[data-out]");
    var ticks = fig.querySelectorAll("[data-k]");
    var hits = fig.querySelectorAll("[data-hit]");
    var state = fig.querySelector("[data-state]");

    var product = function (j) {            // G(8,j) = prod_{k=j+1..8} m_k
      var g = 1;
      for (var k = j + 1; k <= 8; k++) g *= closed[k] ? 0 : M_OPEN;
      return g;
    };

    var render = function () {
      Array.prototype.forEach.call(arcs, function (a) {
        var j = +a.getAttribute("data-j"), g = product(j), live = g > 0;
        a.setAttribute("class", live ? "live" : "dead");
        // Assigning "" to a style property removes it but still leaves a stray
        // style="" attribute behind on a path that never had one. Drop the whole
        // attribute for a dead path instead, so a gate that toggles and returns
        // restores every untouched path byte-identically, not just its class/values.
        if (live) {
          a.style.strokeWidth = (1.1 + 2.2 * g).toFixed(2);
          a.style.opacity = (0.45 + 0.55 * g).toFixed(2);
        } else {
          a.removeAttribute("style");
        }
        var pkt = fig.querySelector('[data-pkt="' + j + '"]');
        if (pkt) pkt.style.opacity = live && !reduce ? 1 : 0;
      });
      Array.prototype.forEach.call(outs, function (o) {
        var g = product(+o.getAttribute("data-out"));
        o.textContent = g === 0 ? "0" : g.toFixed(2).replace(/^0\./, ".");
        o.setAttribute("class", g === 0 ? "out zero" : "out");
      });
      Array.prototype.forEach.call(ticks, function (t) {
        var shut = !!closed[+t.getAttribute("data-k")];
        t.setAttribute("class", shut ? "gatetick closed" : "gatetick");
      });
      Array.prototype.forEach.call(hits, function (h) {
        var k = +h.getAttribute("data-hit"), shut = !!closed[k];
        h.setAttribute("aria-pressed", shut ? "true" : "false");
        h.setAttribute("aria-label", "gate m" + k + ", " + (shut ? "closed" : "open"));
      });
      if (state) {
        var n = 0, first = 0;
        for (var k in closed) if (closed[k]) { n++; if (!first) first = k; }
        state.textContent = n === 0
          ? "every gate open — every path carries weight, none is zero"
          : n === 1
            ? "gate m" + first + " closed — every path across it is exactly 0"
            : n + " gates closed — every path across one of them is exactly 0";
      }
    };

    Array.prototype.forEach.call(hits, function (h) {
      var k = +h.getAttribute("data-hit");
      var toggle = function (e) { e.preventDefault(); closed[k] = !closed[k]; render(); };
      h.addEventListener("click", toggle);
      h.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") toggle(e);
      });
    });
    render();
  }

  /* -------------------------------------------------------- the cube ---- */
  var cube = document.querySelector('[data-rs="cube"]');
  if (cube) {
    var marker = cube.querySelector("[data-marker]");
    var readout = cube.querySelector("[data-read]");
    var stops = cube.querySelectorAll("[data-stop]");
    var at = 0, timer = null;

    var go = function (n) {
      at = n;
      var s = stops[n], colour = "var(" + s.getAttribute("data-colour") + ")";
      if (marker) {
        marker.setAttribute("transform",
          "translate(" + s.getAttribute("data-x") + "," + s.getAttribute("data-y") + ")");
        marker.style.stroke = colour;
      }
      if (readout) {
        readout.textContent = s.getAttribute("data-read");
        readout.style.fill = colour;
      }
      Array.prototype.forEach.call(stops, function (t, k) {
        t.setAttribute("class", k === n ? "stop on" : "stop");
        t.setAttribute("aria-pressed", k === n ? "true" : "false");
      });
    };

    Array.prototype.forEach.call(stops, function (s, n) {
      var pick = function (e) {
        e.preventDefault();
        if (timer) { clearInterval(timer); timer = null; }   // a click ends the tour
        go(n);
      };
      s.addEventListener("click", pick);
      s.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") pick(e);
      });
    });

    go(0);
    if (!reduce && stops.length > 1) {
      timer = setInterval(function () { go((at + 1) % stops.length); }, 4200);
    }
  }

  /* ------------------------------------------------ the Hodge shares ---- */
  var fills = document.querySelectorAll(".rs .bars .fill");
  if (!reduce && fills.length && "IntersectionObserver" in window) {
    Array.prototype.forEach.call(fills, function (b) {
      b.setAttribute("data-w", b.style.width);
      b.style.width = "0";
    });
    // The track is observed, not the fill: a fill at width 0 has no area, and an
    // element with no area never crosses a ratio threshold.
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var fill = e.target.querySelector(".fill");
        if (fill) fill.style.width = fill.getAttribute("data-w");
        io.unobserve(e.target);
      });
    }, { threshold: 0.25 });
    Array.prototype.forEach.call(fills, function (b) {
      if (b.parentNode) io.observe(b.parentNode);
    });
  }
})();


/* ---------- the six teaching figures: row, beta, absorb, wall, fold, blocks ----------
   Each root is <div class="rs fig" data-rs="NAME">. Nothing here assumes the markdown
   pre-builds the SVG or controls: every figure draws its own <svg> contents and its own
   .ctl inputs/outputs into whatever it finds (reusing them if the markup already has
   them), so a bare `<div data-rs="row"></div>` is enough to make it work. SVGs reuse the
   existing .tok/.live/.dead/.edge/.marker/.stop/.hit/.out/.t-mono/.t-dim/.gatetick
   classes; nothing here introduces a new class name or a literal colour. A figure whose
   root is missing from the page is a no-op, never a throw. */
(function () {
  "use strict";
  var SVGNS = "http://www.w3.org/2000/svg";

  // None of the six figures below animate on their own (no interval, no
  // requestAnimationFrame) — they only redraw in response to a slider or a click, so
  // there is no continuous motion for prefers-reduced-motion to disable here. The
  // .marker/.bars transitions a figure's redraw can trigger are the existing global
  // CSS rules the gates and cube figures already use.

  // The three arithmetic constants the wall and fold figures teach. Written once, as
  // literals, so both the figures and the self-check below read the same number.
  // float32 and bfloat16 share an 8-bit exponent but NOT a wall: bf16 has fewer mantissa
  // bits, so its largest finite value is smaller and it overflows first. Measured on this
  // box (torch 2.14.0+cpu, float64 arithmetic over each dtype's finfo):
  //   float32  max = 3.4028234663852886e+38   log = 88.72283905206835
  //   bfloat16 max = 3.3895313892515355e+38   log = 88.71892521235186
  var WALL_F32 = 88.72283905206835;   // exp(x) overflow threshold for float32
  var WALL_BF16 = 88.71892521235186;  // exp(x) overflow threshold for bfloat16 (narrower, not shared)
  var BF16_MAX = 3.3895313892515355e+38; // bfloat16's largest finite value, for comparison-based overflow (Math.fround cannot represent it)
  var WALL_F64 = 709.782712893384;    // exp(x) overflow threshold for float64
  var FOLD_A_CRIT = 0.36787944117144233; // 1/e, the fold point of c = e^{a c}

  var mk = function (tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };

  // Same guarded console.assert the self-check at the bottom of this file uses, so a
  // contract mismatch between markup and an init function is as loud as a math
  // regression — never thrown, never silent. Three rounds have now shipped a figure
  // whose init function authors one value (a viewBox, a min/max/step) while the
  // markup already supplies a different one and quietly wins because it ran first;
  // this is the one place that class of bug gets caught.
  var assertContract = function (cond, msg) {
    try { if (window.console && console.assert) console.assert(cond, "resolvent contract: " + msg); } catch (e) { /* never let a contract check break the page */ }
  };

  // Builds (or reuses) the <figure><svg>, <div class="ctl"> and <p class="cap"> a figure
  // needs inside its root, and clears the svg so each render starts from nothing. Returns
  // null if root is falsy, so every caller can bail with one line.
  var shell = function (root, viewBox, ariaLabel, capText, role) {
    if (!root) return null;
    var fig = root.querySelector("figure");
    if (!fig) { fig = document.createElement("figure"); root.appendChild(fig); }
    var svg = fig.querySelector("svg");
    if (!svg) { svg = document.createElementNS(SVGNS, "svg"); fig.appendChild(svg); }
    var haveViewBox = svg.getAttribute("viewBox");
    if (!haveViewBox) svg.setAttribute("viewBox", viewBox);
    else assertContract(haveViewBox === viewBox,
      "markup viewBox \"" + haveViewBox + "\" overrides authored \"" + viewBox + "\" on " + (root.getAttribute("data-rs") || root.tagName));
    svg.setAttribute("role", role || "img");
    if (!svg.getAttribute("aria-label")) svg.setAttribute("aria-label", ariaLabel);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var ctl = root.querySelector(".ctl");
    if (!ctl) { ctl = document.createElement("div"); ctl.className = "ctl"; root.appendChild(ctl); }

    var cap = root.querySelector(".cap");
    if (!cap) {
      cap = document.createElement("p"); cap.className = "cap"; cap.textContent = capText;
      root.appendChild(cap);
    }
    return { svg: svg, ctl: ctl, cap: cap };
  };

  // Finds an existing control by its data-ctl/data-out name, or builds a <label> (for a
  // range input) / a bare <output> and appends it to .ctl. Keeps each init function down
  // to "describe the control", not "write the same six DOM lines six times".
  var range = function (ctl, name, text, min, max, step, value) {
    var input = ctl.querySelector('input[data-ctl="' + name + '"]');
    if (input) {
      assertContract(input.min === String(min) && input.max === String(max) && input.step === String(step),
        "markup " + name + " min/max/step (" + input.min + "/" + input.max + "/" + input.step +
        ") overrides authored (" + min + "/" + max + "/" + step + ")");
      return input;
    }
    var label = document.createElement("label");
    label.appendChild(document.createTextNode(text + " "));
    input = document.createElement("input");
    input.type = "range"; input.min = min; input.max = max; input.step = step; input.value = value;
    input.setAttribute("data-ctl", name);
    label.appendChild(input);
    ctl.appendChild(label);
    return input;
  };
  var readout = function (ctl, name, text) {
    var out = ctl.querySelector('output[data-out="' + name + '"]');
    if (out) return out;
    var label = document.createElement("label");
    label.appendChild(document.createTextNode(text + " "));
    out = document.createElement("output");
    out.setAttribute("data-out", name);
    label.appendChild(out);
    ctl.appendChild(label);
    return out;
  };

  /* -------------------------------------------- shared, and self-checkable, math ---- */
  var rawWeight = function (dist) { return Math.exp(-0.35 * dist); }; // illustrative recency weight

  // Solves q = R + Q q for a linear absorbing chain 0..n-1 (0 and n-1 absorbing, boundary
  // values fixed) by iterated substitution — the resolvent read done the slow, honest way.
  var solveChain = function (boundaryLo, boundaryHi, n) {
    var q = new Array(n);
    for (var i = 0; i < n; i++) q[i] = i === 0 ? boundaryLo : i === n - 1 ? boundaryHi : 0.5;
    for (var it = 0; it < 200; it++) {
      var next = q.slice();
      for (i = 1; i < n - 1; i++) next[i] = 0.5 * q[i - 1] + 0.5 * q[i + 1];
      q = next;
    }
    return q;
  };

  // Number of real solutions of c = e^{a c}: two below the fold point, one tangency at
  // it, none above. The "a" slider here runs 0.25..0.6 step 0.001 (351 positions); the
  // tolerance below was measured against every one of them, not guessed: the drawn
  // crossing count (counted from the same 200-segment curve the figure renders) steps
  // straight from 2 at a=0.367 to 0 at a=0.368, the two grid points nearest crit=1/e, and
  // the closer of the two (0.368) sits 1.2056e-4 away from crit. Any tolerance at or above
  // that would mislabel 0.368 as a tangency it does not show. 1e-4 stays under it with a
  // safety margin and still catches an exact a === crit (e.g. the datalist tick), so the
  // label never disagrees with the drawing at any reachable slider position.
  var foldSolutionCount = function (a, crit) {
    if (Math.abs(a - crit) < 0.0001) return 1;
    return a < crit ? 2 : 0;
  };

  /* --------------------------------------------------------- 1. what a row is ---- */
  function initRow(root) {
    var s = shell(root, "0 0 640 190",
      "Nine tokens in a row. A slider moves the query position; a bar over each earlier token is its raw attention weight.",
      "Each bar is the row's raw weight on one earlier token, before any normalisation. Illustrative weights, not a measured result.");
    if (!s) return;
    var svg = s.svg, ctl = s.ctl;
    var N = 9, MARGIN = 40, W = 640, TOP = 30, BASE = 150, MAXH = 96;
    var xAt = function (j) { return MARGIN + j * (W - 2 * MARGIN) / (N - 1); };

    var qpos = range(ctl, "qpos", "query position", "0", String(N - 1), "1", String(N - 1));
    var out = readout(ctl, "qpos", "i =");

    var render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var i = +qpos.value;
      for (var j = 0; j < N; j++) {
        var x = xAt(j), future = j > i;
        mk("rect", { x: x - 12, y: TOP, width: 24, height: 24, rx: 4, "class": future ? "dead" : (j === i ? "tok q" : "tok") }, svg);
        mk("text", { x: x, y: TOP + 16, "text-anchor": "middle", "class": "t-mono" }, svg).textContent = String(j);
        if (!future) {
          var h = MAXH * rawWeight(i - j);
          mk("rect", { x: x - 10, y: BASE - h, width: 20, height: h, "class": j === i ? "tok q" : "tok" }, svg);
        }
      }
      mk("line", { x1: MARGIN - 14, y1: BASE, x2: W - MARGIN + 14, y2: BASE, "class": "edge" }, svg);
      out.textContent = String(i);
    };

    qpos.addEventListener("input", render);
    render();
  }

  /* ------------------------------------------ 2. what the normalizer does ---- */
  function initBeta(root) {
    var s = shell(root, "0 0 640 190",
      "The same row, fixed at the last position, with a slider for beta from 0 to 1.",
      "The row is divided by Z_i raised to beta. At beta = 1 it is the usual softmax normalisation and the row sums to 1; at beta = 0 it is not divided at all. Illustrative weights, not a measured result.");
    if (!s) return;
    var svg = s.svg, ctl = s.ctl;
    var N = 9, I = N - 1, MARGIN = 40, W = 640, TOP = 30, BASE = 150, MAXH = 96;
    var xAt = function (j) { return MARGIN + j * (W - 2 * MARGIN) / (N - 1); };
    var raw = [];
    for (var j = 0; j < N; j++) raw.push(rawWeight(I - j));
    var Z = raw.reduce(function (a, b) { return a + b; }, 0);

    var betaCtl = range(ctl, "beta", "beta", "0", "1", "0.01", "1");
    var zOut = readout(ctl, "z", "Z_i^beta =");
    var sumOut = readout(ctl, "rowsum", "row sum =");

    var render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var beta = +betaCtl.value, divisor = Math.pow(Z, beta), sum = 0;
      for (var j = 0; j < N; j++) {
        var w = raw[j] / divisor;
        sum += w;
        var x = xAt(j), h = MAXH * w;
        mk("rect", { x: x - 12, y: TOP, width: 24, height: 24, rx: 4, "class": j === I ? "tok q" : "tok" }, svg);
        mk("text", { x: x, y: TOP + 16, "text-anchor": "middle", "class": "t-mono" }, svg).textContent = String(j);
        mk("rect", { x: x - 10, y: BASE - h, width: 20, height: h, "class": j === I ? "tok q" : "tok" }, svg);
      }
      mk("line", { x1: MARGIN - 14, y1: BASE, x2: W - MARGIN + 14, y2: BASE, "class": "edge" }, svg);
      zOut.textContent = divisor.toFixed(3);
      sumOut.textContent = sum.toFixed(3);
    };

    betaCtl.addEventListener("input", render);
    render();
  }

  /* --------------------------------------------------------- 5. what the read is ---- */
  function initAbsorb(root) {
    var s = shell(root, "0 0 640 160",
      "A six-state chain: A, four transient states, and B. Click a transient state to read the probability of reaching A before B.",
      "Solved by iterating q = R + Q q on this toy six-state chain until it settles — the resolvent read, done by substitution instead of a matrix inverse. Illustrative, not a measured result.",
      "group");
    if (!s) return;
    var svg = s.svg, ctl = s.ctl;
    var LABELS = ["A", "1", "2", "3", "4", "B"], n = LABELS.length;
    var MARGIN = 60, W = 640, Y = 70;
    var xAt = function (k) { return MARGIN + k * (W - 2 * MARGIN) / (n - 1); };
    var pOut = readout(ctl, "pa", "P(reach A first) =");
    var selected = 2;

    var attach = function (hit, k) {
      var pick = function (e) { e.preventDefault(); selected = k; render(); };
      hit.addEventListener("click", pick);
      hit.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") pick(e);
      });
    };

    var render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var k, x;
      for (k = 0; k < n - 1; k++) mk("line", { x1: xAt(k), y1: Y, x2: xAt(k + 1), y2: Y, "class": "edge" }, svg);
      for (k = 0; k < n; k++) {
        x = xAt(k);
        var absorbing = k === 0 || k === n - 1;
        mk("circle", { cx: x, cy: Y, r: 16, "class": absorbing ? "tok q" : "tok" }, svg);
        mk("text", { x: x, y: Y + 5, "text-anchor": "middle", "class": "t-mono" }, svg).textContent = LABELS[k];
        if (!absorbing) {
          var hit = mk("circle", {
            cx: x, cy: Y, r: 22, "class": "hit", tabindex: "0", role: "button",
            "aria-pressed": k === selected ? "true" : "false",
            "aria-label": "start at state " + LABELS[k]
          }, svg);
          attach(hit, k);
        }
      }
      mk("circle", { cx: xAt(selected), cy: Y, r: 24, "class": "marker" }, svg);
      pOut.textContent = solveChain(1, 0, n)[selected].toFixed(3);
    };

    render();
  }

  /* ------------------------------------------------ 6. where the arithmetic ends ---- */
  function initWall(root) {
    var s = shell(root, "0 0 640 170",
      "A number line for x, with exp(x) evaluated live and marked where bfloat16, float32, and float64 overflow.",
      "exp(x) is evaluated at float64. bfloat16 and float32 share an 8-bit exponent but not a wall: bf16 carries fewer mantissa bits, so its largest finite value is smaller and it overflows first. Moving x into the interval between the two ticks is where they disagree — float32 still finite, bf16 already infinite.");
    if (!s) return;
    var svg = s.svg, ctl = s.ctl;
    var XMIN = -20, XMAX = 720, MARGIN = 40, W = 640, Y = 95;
    var px = function (x) { return MARGIN + (x - XMIN) * (W - 2 * MARGIN) / (XMAX - XMIN); };

    var xCtl = range(ctl, "x", "x", String(XMIN), String(XMAX), "0.5", "0");
    var bf16Out = readout(ctl, "bf16", "exp(x) as bfloat16 =");
    var f32Out = readout(ctl, "f32", "exp(x) as float32 =");
    var f64Out = readout(ctl, "f64", "exp(x) as float64 =");
    var fmt = function (v) { return v === Infinity ? "∞ (overflowed)" : v.toExponential(3); };

    var render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var x = +xCtl.value;
      var x64 = Math.exp(x);
      mk("line", { x1: MARGIN, y1: Y, x2: W - MARGIN, y2: Y, "class": "edge" }, svg);
      [[WALL_BF16, "bfloat16"], [WALL_F32, "float32"], [WALL_F64, "float64"]].forEach(function (pair) {
        var wx = px(pair[0]);
        mk("line", { x1: wx, y1: Y - 22, x2: wx, y2: Y + 22, "class": "gatetick closed" }, svg);
        mk("text", { x: wx, y: Y - 28, "text-anchor": "middle", "class": "t-dim" }, svg).textContent = pair[1];
        mk("text", { x: wx, y: Y + 34, "text-anchor": "middle", "class": "t-mono" }, svg).textContent = "x = " + pair[0];
      });
      mk("circle", { cx: px(x), cy: Y, r: 6, "class": "marker" }, svg);
      // bf16 overflow is modeled by comparing the float64 value against bf16's own finite
      // max, not by Math.fround (which rounds to float32 and cannot represent bf16 at all).
      bf16Out.textContent = fmt(x64 > BF16_MAX ? Infinity : x64);
      f32Out.textContent = fmt(Math.fround(x64));
      f64Out.textContent = fmt(x64);
    };

    xCtl.addEventListener("input", render);
    render();
  }

  /* --------------------------------------------------------- 7. when the route closes ---- */
  function initFold(root) {
    var s = shell(root, "0 0 480 320",
      "The line c against the curve exp(a c) for c from 0 to 10, with a slider for a.",
      "c = e^{a c} stops having a solution above a = 1/e; it does not degrade as a rises, it disappears.");
    if (!s) return;
    var svg = s.svg, ctl = s.ctl;
    var CMAX = 10, MARGIN = 36, W = 480, H = 320, PLOTH = H - 2 * MARGIN, PLOTW = W - 2 * MARGIN;
    var px = function (c) { return MARGIN + c * PLOTW / CMAX; };
    var py = function (v) { return H - MARGIN - v * PLOTH / CMAX; };

    var aCtl = range(ctl, "a", "a", "0.25", "0.6", "0.001", "0.3");
    if (!document.getElementById("rs-fold-ticks")) {
      var dl = document.createElement("datalist");
      dl.id = "rs-fold-ticks";
      var opt = document.createElement("option");
      opt.value = String(FOLD_A_CRIT);
      dl.appendChild(opt);
      ctl.appendChild(dl);
    }
    aCtl.setAttribute("list", "rs-fold-ticks");
    var aOut = readout(ctl, "a", "a =");
    var cOut = readout(ctl, "count", "solutions =");

    var render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var a = +aCtl.value;
      mk("line", { x1: MARGIN, y1: H - MARGIN, x2: W - MARGIN, y2: H - MARGIN, "class": "edge" }, svg);
      mk("line", { x1: MARGIN, y1: H - MARGIN, x2: MARGIN, y2: MARGIN, "class": "edge" }, svg);
      mk("line", { x1: px(0), y1: py(0), x2: px(CMAX), y2: py(CMAX), "class": "edge" }, svg);

      var steps = 200, d = "M" + px(0).toFixed(2) + "," + py(1).toFixed(2);
      for (var i = 1; i <= steps; i++) {
        var c = CMAX * i / steps;
        d += " L" + px(c).toFixed(2) + "," + py(Math.exp(a * c)).toFixed(2);
      }
      var solved = foldSolutionCount(a, FOLD_A_CRIT);
      mk("path", { d: d, "class": solved > 0 ? "live" : "dead" }, svg);

      var ex = Math.E, ey = py(ex);
      // 1a: renamed so this never reads like the live "a =" readout six inches away.
      mk("text", { x: px(ex) + 8, y: ey - 2, "class": "t-dim" }, svg).textContent = "fold point a* = 1/e = " + FOLD_A_CRIT;
      // 1b: the marker and "tangent" claim are only true at solved === 1; the other
      // two geometries (two crossings, no crossing) get their own, distinct label.
      if (solved === 1) {
        mk("circle", { cx: px(ex), cy: ey, r: 4, "class": "marker" }, svg);
        mk("text", { x: px(ex) + 8, y: ey - 14, "class": "t-dim" }, svg).textContent = "tangent at c = e";
      } else {
        mk("text", { x: px(ex) + 8, y: ey - 14, "class": "t-dim" }, svg).textContent =
          solved === 2 ? "two crossings below the fold" : "no crossing above the fold";
      }

      aOut.textContent = a.toFixed(3);
      cOut.textContent = solved === 1 ? "1 (tangent)" : String(solved);
    };

    aCtl.addEventListener("input", render);
    render();
  }

  /* --------------------------------------------------------- 8. how it is computed ---- */
  function initBlocks(root) {
    var s = shell(root, "0 0 620 260",
      "Eight tokens cut into four blocks of two, merged left to right into one summary. Closing a gate zeroes every block before it.",
      "Each block carries (m, l, o, carry): the running max, the running sum of exp(logit - m), the running weighted value, and whether it survives the gates to its right. Illustrative logits, not a measured result.",
      "group");
    if (!s) return;
    var svg = s.svg;
    var LOGITS = [0.4, -0.2, 0.9, 0.1, -0.6, 0.3, 0.7, -0.1];
    var VALUES = [1, 2, 3, 4, 5, 6, 7, 8];
    var BLOCKS = 4, PER = 2, BW = 110, GAP = 25, Y0 = 20, BH = 90;
    var xAt = function (b) { return 30 + b * (BW + GAP); };
    var closed = {};

    var localSummary = function (b) {
      var m = -Infinity, k;
      for (k = 0; k < PER; k++) m = Math.max(m, LOGITS[b * PER + k]);
      var l = 0, o = 0;
      for (k = 0; k < PER; k++) {
        var w = Math.exp(LOGITS[b * PER + k] - m);
        l += w; o += w * VALUES[b * PER + k];
      }
      return { m: m, l: l, o: o };
    };
    var merge = function (A, B) {
      var m = Math.max(A.m, B.m);
      var ea = A.m === -Infinity ? 0 : Math.exp(A.m - m);
      var eb = B.m === -Infinity ? 0 : Math.exp(B.m - m);
      return { m: m, l: A.l * ea + B.l * eb, o: A.o * ea + B.o * eb };
    };
    var carryOf = function (b) {
      for (var g = b; g < BLOCKS - 1; g++) if (closed[g]) return 0;
      return 1;
    };
    var isZero = function (n) { return Math.abs(n) < 1e-9; };
    var block = function (label, vals, x, y, w) {
      mk("rect", { x: x, y: y, width: w, height: BH, rx: 6, "class": "tok" }, svg);
      mk("text", { x: x + w / 2, y: y + 16, "text-anchor": "middle", "class": "t-mono" }, svg).textContent = label;
      var rows = [
        ["m", vals.m === -Infinity ? "-∞" : vals.m.toFixed(2), vals.m === -Infinity],
        ["l", vals.l.toFixed(2), isZero(vals.l)],
        ["o", vals.o.toFixed(2), isZero(vals.o)],
        ["carry", String(vals.carry), vals.carry === 0]
      ];
      rows.forEach(function (r, i) {
        mk("text", { x: x + 10, y: y + 34 + i * 14, "class": r[2] ? "out zero" : "out" }, svg).textContent = r[0] + " " + r[1];
      });
    };
    var attachGate = function (hit, g) {
      var toggle = function (e) { e.preventDefault(); closed[g] = !closed[g]; render(); };
      hit.addEventListener("click", toggle);
      hit.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") toggle(e);
      });
    };

    var render = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var running = null, b;
      for (b = 0; b < BLOCKS; b++) {
        var loc = localSummary(b), carry = carryOf(b);
        var gated = carry ? loc : { m: -Infinity, l: 0, o: 0 };
        running = running === null ? gated : merge(running, gated);
        block("B" + (b + 1), { m: loc.m, l: loc.l, o: loc.o, carry: carry }, xAt(b), Y0, BW);
        if (b < BLOCKS - 1) {
          var gx = xAt(b) + BW + GAP / 2;
          mk("line", { x1: xAt(b) + BW, y1: Y0 + BH / 2, x2: xAt(b + 1), y2: Y0 + BH / 2, "class": "edge" }, svg);
          mk("line", { x1: gx, y1: Y0, x2: gx, y2: Y0 + BH, "class": closed[b] ? "gatetick closed" : "gatetick" }, svg);
          var hit = mk("rect", {
            x: gx - 12, y: Y0, width: 24, height: BH, "class": "hit", tabindex: "0", role: "button",
            "aria-pressed": closed[b] ? "true" : "false",
            "aria-label": "gate after block " + (b + 1) + ", " + (closed[b] ? "closed" : "open")
          }, svg);
          attachGate(hit, b);
        }
      }
      var mx = (xAt(0) + xAt(BLOCKS - 1) + BW) / 2 - 70, my = Y0 + BH + 45;
      mk("line", { x1: mx + 70, y1: Y0 + BH, x2: mx + 70, y2: my, "class": "edge" }, svg);
      running.carry = 1;
      block("merged", running, mx, my, 140);
    };

    render();
  }

  /* ---------------------------------------------------------------- dispatch ---- */
  var FIGS = { row: initRow, beta: initBeta, absorb: initAbsorb, wall: initWall, fold: initFold, blocks: initBlocks };
  Array.prototype.forEach.call(document.querySelectorAll("[data-rs]"), function (el) {
    var fn = FIGS[el.getAttribute("data-rs")];
    if (fn) fn(el);
  });

  /* -------------------------------------------------------------- self-check ----
     One runnable smoke test for the non-trivial logic above: the gambler's-ruin
     closed form for the absorbing chain, and the fold point's three regimes. Silent
     unless something regresses. */
  try {
    if (window.console && console.assert) {
      var q = solveChain(1, 0, 6); // A=1, B=0, states 0..5
      for (var i = 1; i <= 4; i++) {
        console.assert(Math.abs(q[i] - (5 - i) / 5) < 1e-6,
          "resolvent self-check: state " + i + " expected " + (5 - i) / 5 + ", got " + q[i]);
      }
      console.assert(foldSolutionCount(0.3, FOLD_A_CRIT) === 2, "resolvent self-check: expected two crossings below 1/e");
      console.assert(foldSolutionCount(0.5, FOLD_A_CRIT) === 0, "resolvent self-check: expected no crossing above 1/e");
      console.assert(foldSolutionCount(FOLD_A_CRIT, FOLD_A_CRIT) === 1, "resolvent self-check: expected tangency at 1/e");
      console.assert(Math.fround(Math.exp(WALL_F32 - 1)) !== Infinity, "resolvent self-check: float32 should not have overflowed yet");
      console.assert(Math.fround(Math.exp(WALL_F32 + 1)) === Infinity, "resolvent self-check: float32 should have overflowed");
      console.assert(Math.exp(WALL_BF16 - 1) <= BF16_MAX, "resolvent self-check: bf16 should not have overflowed yet");
      console.assert(Math.exp(WALL_BF16 + 1) > BF16_MAX, "resolvent self-check: bf16 should have overflowed");
      var xMid = (WALL_BF16 + WALL_F32) / 2;
      console.assert(Math.exp(xMid) > BF16_MAX && Math.fround(Math.exp(xMid)) !== Infinity,
        "resolvent self-check: at the midpoint of the disagreement interval, bf16 should have overflowed and float32 should not");
    }
  } catch (e) { /* never let the self-check break the page */ }
})();
