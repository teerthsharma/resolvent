/* The hero scene: a floating ice island carrying one causal head.
   Nine tokens stand on the island; the last one (violet, the query) reads every earlier
   token along an arc, and light runs along each arc. Between neighbours stands a gate:
   click one and it closes, and every arc that crosses it goes dark, which is the same
   rule as the gate figure below it on the page. Decorative for assistive tech (the SVG
   figure carries the accessible version), paused off-screen, one still frame under
   prefers-reduced-motion, and it removes itself if WebGL is unavailable. */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js";

const N = 9;                       // tokens t0..t8; t8 is the query
const X = (i) => -3.3 + i * (6.6 / (N - 1));
const Z = (i) => 0.55 * Math.sin((i / (N - 1)) * Math.PI) - 0.2;

function css(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim() || "#888";
}
function rng(seed) {               // deterministic jitter so the island is the same shape every visit
  return () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
}
function jitter(geo, amount, seed, keepTop) {
  const r = rng(seed), p = geo.attributes.position, seen = new Map();
  for (let i = 0; i < p.count; i++) {
    const key = p.getX(i).toFixed(3) + "," + p.getY(i).toFixed(3) + "," + p.getZ(i).toFixed(3);
    if (!seen.has(key)) seen.set(key, [(r() - .5) * amount, (r() - .5) * amount, (r() - .5) * amount]);
    const d = seen.get(key);
    p.setX(i, p.getX(i) + d[0]); p.setZ(i, p.getZ(i) + d[2]);
    if (!(keepTop && p.getY(i) > 0)) p.setY(i, p.getY(i) + d[1]);
  }
  geo.computeVertexNormals();
  return geo;
}
function glowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d"), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, "rgba(255,255,255,1)"); grd.addColorStop(.35, "rgba(255,255,255,.55)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function boot() {
  const hero = document.querySelector(".rs .hero");
  if (!hero || hero.querySelector(".scene")) return;
  const wrap = document.createElement("div");
  wrap.className = "scene"; wrap.setAttribute("aria-hidden", "true");
  hero.insertBefore(wrap, hero.firstChild);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
  } catch (e) { wrap.remove(); return; }
  if (!renderer.getContext()) { wrap.remove(); return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  wrap.appendChild(renderer.domElement);

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  const world = new THREE.Group(); scene.add(world);

  const hemi = new THREE.HemisphereLight(0xeef5ff, 0x9dbbe0, 1.45); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.position.set(4.5, 9, 5); sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 1, far: 30 });
  sun.shadow.radius = 4; sun.shadow.bias = -0.0015;
  scene.add(sun);

  // the island: a snow slab over an ice keel
  const snow = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .9, flatShading: true, emissive: 0xdfeaf7, emissiveIntensity: .35 });
  const ice = new THREE.MeshStandardMaterial({ color: 0xa9cbee, roughness: .55, metalness: .05, flatShading: true });
  const deep = new THREE.MeshStandardMaterial({ color: 0x7fa9d8, roughness: .5, flatShading: true });
  const slab = new THREE.Mesh(jitter(new THREE.CylinderGeometry(4.55, 4.25, .55, 11, 1), .32, 7, true), snow);
  slab.receiveShadow = true; world.add(slab);
  const rim = new THREE.Mesh(jitter(new THREE.CylinderGeometry(4.3, 3.7, .55, 11, 1), .3, 11), ice);
  rim.position.y = -.5; world.add(rim);
  const keel = new THREE.Mesh(jitter(new THREE.ConeGeometry(3.7, 2.6, 11, 2), .45, 19), deep);
  keel.rotation.x = Math.PI; keel.position.y = -2.05; world.add(keel);
  for (let k = 0; k < 5; k++) {               // a few loose floes around it
    const r = rng(101 + k), s = .35 + r() * .45;
    const floe = new THREE.Mesh(jitter(new THREE.CylinderGeometry(s, s * .8, .18, 7), .12, 31 + k), snow);
    const a = r() * Math.PI * 2, d = 5.0 + r() * .9;
    floe.position.set(Math.cos(a) * d, -.4 - r() * 1.3, Math.sin(a) * d * .6);
    floe.userData.bob = r() * 6.28; world.add(floe);
  }

  // tokens
  const tokMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .45, emissive: 0x2456dc, emissiveIntensity: 0 });
  const tokens = [];
  for (let i = 0; i < N; i++) {
    const m = i === N - 1
      ? new THREE.MeshStandardMaterial({ color: 0xd9731f, roughness: .35, emissive: 0xd9731f, emissiveIntensity: .25 })
      : tokMat.clone();
    const tok = new THREE.Mesh(new THREE.CapsuleGeometry(.2, .32, 6, 16), m);
    tok.position.set(X(i), .6, Z(i)); tok.castShadow = true; world.add(tok);
    const base = new THREE.Mesh(new THREE.CylinderGeometry(.3, .34, .08, 18), ice);
    base.position.set(X(i), .31, Z(i)); base.receiveShadow = true; world.add(base);
    tokens.push(tok);
  }

  // gates between neighbours: gate k sits between t(k-1) and t(k)
  const gates = [];
  for (let k = 1; k < N; k++) {
    const g = new THREE.Mesh(new THREE.BoxGeometry(.07, .5, .46),
      new THREE.MeshStandardMaterial({ color: 0xd9e9f9, roughness: .3, transparent: true, opacity: .85 }));
    g.position.set((X(k - 1) + X(k)) / 2, .56, (Z(k - 1) + Z(k)) / 2);
    g.castShadow = true; g.userData = { k, closed: false, lift: 0 }; world.add(g); gates.push(g);
  }

  // arcs from the query to every earlier token, and the light running along them
  const glow = glowTexture(), arcs = [];
  const q = new THREE.Vector3(X(N - 1), .95, Z(N - 1));
  for (let j = 0; j < N - 1; j++) {
    const end = new THREE.Vector3(X(j), .95, Z(j));
    const mid = q.clone().lerp(end, .5); mid.y += .7 + (N - 1 - j) * .42; mid.z -= .35;
    const curve = new THREE.QuadraticBezierCurve3(q, mid, end);
    const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: .5, depthWrite: false });
    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 72, .018, 6, false), mat);
    world.add(tube);
    const pk = [];
    for (let s = 0; s < 2; s++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending }));
      sp.scale.setScalar(.34); world.add(sp); pk.push(sp);
    }
    arcs.push({ j, curve, mat, pk, dur: 3.1 - j * 0.2, on: 1 });
  }

  function applyScheme() {
    const night = document.body.getAttribute("data-md-color-scheme") === "slate";
    const bar = new THREE.Color(css("--rs-bar")), violet = new THREE.Color(css("--rs-violet"));
    arcs.forEach((a) => { a.mat.color.copy(bar); a.pk.forEach((p) => p.material.color.copy(night ? violet : bar)); });
    hemi.intensity = night ? .6 : 1.45; sun.intensity = night ? 1.0 : 2.0;
    hemi.color.set(night ? 0x8fa9d9 : 0xeef5ff);
    snow.emissiveIntensity = night ? .08 : .35;
    renderer.toneMappingExposure = 1;
  }
  applyScheme();
  new MutationObserver(() => { applyScheme(); if (reduce) draw(0); })
    .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = w / h < 1.15 ? 38 : 30;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(() => { resize(); if (reduce) draw(0); }).observe(wrap);
  resize();

  const mouse = { x: 0, y: 0 }, ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  function pick(ev) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(gates, false)[0];
    return hit ? hit.object : null;
  }
  wrap.addEventListener("pointermove", (ev) => {
    const r = wrap.getBoundingClientRect();
    mouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1; mouse.y = ((ev.clientY - r.top) / r.height) * 2 - 1;
    wrap.style.cursor = pick(ev) ? "pointer" : "";
  });
  wrap.addEventListener("pointerleave", () => { mouse.x = 0; mouse.y = 0; });
  wrap.addEventListener("click", (ev) => {
    const g = pick(ev); if (!g) return;
    g.userData.closed = !g.userData.closed;
    if (reduce) { settle(); draw(0); }
  });

  const amber = new THREE.Color(css("--rs-kill")), iceCol = new THREE.Color(0xd9e9f9);
  function settle() {                     // closed-gate state without animation
    gates.forEach((g) => { g.userData.lift = g.userData.closed ? 1 : 0; });
    arcs.forEach((a) => { a.on = gates.some((g) => g.userData.closed && a.j < g.userData.k) ? 0 : 1; });
  }

  let seen = true, last = 0, t0 = performance.now();
  if ("IntersectionObserver" in window)
    new IntersectionObserver((e) => { seen = e[0].isIntersecting; }).observe(wrap);

  function draw(now) {
    const t = (now - t0) / 1000, dt = Math.min(.05, (now - last) / 1000 || .016); last = now;
    world.position.y = Math.sin(t * .8) * .09;
    world.rotation.y = -.32 + Math.sin(t * .11) * .07;
    camera.position.set(Math.sin(t * .13) * .6 + mouse.x * .8, 5.4 - mouse.y * .4, 16.5);
    camera.lookAt(0, .9, 0);
    world.children.forEach((c) => { if (c.userData.bob !== undefined) c.position.y += Math.sin(t * .9 + c.userData.bob) * .0015; });
    gates.forEach((g) => {
      const d = g.userData, target = d.closed ? 1 : 0;
      d.lift += (target - d.lift) * Math.min(1, dt * 7);
      g.scale.y = 1 + d.lift * .5; g.position.y = .56 + d.lift * .14;
      g.material.color.copy(iceCol).lerp(amber, d.lift);
    });
    arcs.forEach((a) => {
      const target = gates.some((g) => g.userData.closed && a.j < g.userData.k) ? 0 : 1;
      a.on += (target - a.on) * Math.min(1, dt * 5);
      a.mat.opacity = .1 + .45 * a.on;
      a.pk.forEach((p, s) => {
        const u = ((t / a.dur) + s * .5) % 1;
        p.position.copy(a.curve.getPoint(u));
        p.material.opacity = a.on * Math.sin(u * Math.PI);
        if (u > .96) tokens[a.j].material.emissiveIntensity = Math.max(tokens[a.j].material.emissiveIntensity, .5 * a.on);
      });
    });
    tokens.forEach((tk, i) => { if (i < N - 1) tk.material.emissiveIntensity *= .94; });
    renderer.render(scene, camera);
  }
  function loop(now) {
    if (!wrap.isConnected) { renderer.dispose(); return; }
    if (seen && !document.hidden) draw(now);
    requestAnimationFrame(loop);
  }
  if (reduce) { settle(); draw(performance.now()); }
  else requestAnimationFrame(loop);
}

if (window.document$ && typeof window.document$.subscribe === "function") window.document$.subscribe(boot);
else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
