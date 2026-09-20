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
        a.style.strokeWidth = live ? (1.1 + 2.2 * g).toFixed(2) : "";
        a.style.opacity = live ? (0.45 + 0.55 * g).toFixed(2) : "";
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
