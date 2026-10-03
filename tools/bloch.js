  // ---- Bloch sphere drawing (same view as qsim and Qiskit: +x out of the page to the lower left, +y right, |0> up)
  var NS = 'http://www.w3.org/2000/svg', D2R = Math.PI / 180;
  function blochView(az, el) {
    var a = az * Math.PI / 180, e = el * Math.PI / 180;
    var R = [-Math.sin(a), Math.cos(a), 0];
    var U = [-Math.sin(e) * Math.cos(a), -Math.sin(e) * Math.sin(a), Math.cos(e)];
    var C = [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
    return function (p) {
      return { x: p[0] * R[0] + p[1] * R[1] + p[2] * R[2], y: -(p[0] * U[0] + p[1] * U[1] + p[2] * U[2]), d: p[0] * C[0] + p[1] * C[1] + p[2] * C[2] };
    };
  }
  function sv(svg, tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    svg.appendChild(e);
    return e;
  }
  // cfg: {pts: [{v: [x, y, z], c: colour, l: label, dash: bool}], az, el, labels, font, angles: {theta, phi}}
  function drawBloch(svg, cfg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('viewBox', '-1.55 -1.5 3.1 3.0');
    var P = blochView(cfg.az === undefined ? 25 : cfg.az, cfg.el === undefined ? 20 : cfg.el);
    var f = cfg.font || 0.13, id = 'ah' + Math.random().toString(36).slice(2, 8);
    var defs = sv(svg, 'defs', {});
    sv(svg, 'circle', { cx: 0, cy: 0, r: 1, fill: '#F8F4F6', stroke: '#6A626B', 'stroke-width': 0.012 });
    function ring(fn) {
      var prev = null;
      for (var i = 0; i <= 120; i++) {
        var u = i / 120 * 2 * Math.PI, q = P(fn(u));
        if (prev) sv(svg, 'line', { x1: prev.x, y1: prev.y, x2: q.x, y2: q.y, stroke: '#B9AEB4', 'stroke-width': 0.01, 'stroke-dasharray': prev.d < 0 ? '0.03 0.03' : 'none' });
        prev = q;
      }
    }
    ring(function (u) { return [Math.cos(u), Math.sin(u), 0]; });
    ring(function (u) { return [Math.sin(u), 0, Math.cos(u)]; });
    ring(function (u) { return [0, Math.sin(u), Math.cos(u)]; });
    var axes = [[[0, 0, 1], '|0⟩'], [[0, 0, -1], '|1⟩'], [[1, 0, 0], '|+⟩'], [[-1, 0, 0], '|−⟩'], [[0, 1, 0], '|+i⟩'], [[0, -1, 0], '|−i⟩']];
    axes.forEach(function (ax) {
      var p = P(ax[0]), q = P([1.25 * ax[0][0], 1.25 * ax[0][1], 1.25 * ax[0][2]]);
      sv(svg, 'line', { x1: 0, y1: 0, x2: p.x, y2: p.y, stroke: '#B9AEB4', 'stroke-width': 0.01, 'stroke-dasharray': p.d < -0.01 ? '0.03 0.03' : 'none' });
      if (cfg.labels !== false) sv(svg, 'text', { x: q.x, y: q.y + f * 0.35, 'font-size': f, 'text-anchor': 'middle', fill: '#24313D' }, ax[1]);
    });
    if (cfg.angles) {
      var th = cfg.angles.theta, ph = cfg.angles.phi, k, pts = [], q;
      for (k = 0; k <= 30; k++) { var t = th * k / 30; q = P([0.45 * Math.sin(t) * Math.cos(ph), 0.45 * Math.sin(t) * Math.sin(ph), 0.45 * Math.cos(t)]); pts.push(q.x + ',' + q.y); }
      sv(svg, 'polyline', { points: pts.join(' '), fill: 'none', stroke: '#C9A227', 'stroke-width': 0.02 });
      pts = [];
      for (k = 0; k <= 30; k++) { var w = ph * k / 30; q = P([0.55 * Math.cos(w), 0.55 * Math.sin(w), 0]); pts.push(q.x + ',' + q.y); }
      sv(svg, 'polyline', { points: pts.join(' '), fill: 'none', stroke: '#C9A227', 'stroke-width': 0.02 });
      var foot = P([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), 0]);
      var tip = P([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]);
      sv(svg, 'line', { x1: 0, y1: 0, x2: foot.x, y2: foot.y, stroke: '#C9A227', 'stroke-width': 0.012, 'stroke-dasharray': '0.04 0.03' });
      sv(svg, 'line', { x1: foot.x, y1: foot.y, x2: tip.x, y2: tip.y, stroke: '#C9A227', 'stroke-width': 0.012, 'stroke-dasharray': '0.04 0.03' });
      q = P([0.62 * Math.sin(th / 2) * Math.cos(ph), 0.62 * Math.sin(th / 2) * Math.sin(ph), 0.62 * Math.cos(th / 2)]);
      sv(svg, 'text', { x: q.x + 0.05, y: q.y, 'font-size': f, fill: '#8A6D00', 'font-style': 'italic' }, 'θ');
      q = P([0.7 * Math.cos(ph / 2), 0.7 * Math.sin(ph / 2), 0]);
      sv(svg, 'text', { x: q.x, y: q.y + 0.12, 'font-size': f, fill: '#8A6D00', 'font-style': 'italic' }, 'φ');
    }
    // rotation arcs: {v: start point, n: axis (unit vector), a: angle in degrees (right-hand rule), c, l, axis: true}
    (cfg.rots || []).forEach(function (r, j) {
      var col = r.c || '#C9A227', n = r.n, L = Math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      n = [n[0] / L, n[1] / L, n[2] / L];
      if (r.axis) {
        var a1 = P([1.18 * n[0], 1.18 * n[1], 1.18 * n[2]]), a2 = P([-1.18 * n[0], -1.18 * n[1], -1.18 * n[2]]);
        sv(svg, 'line', { x1: a2.x, y1: a2.y, x2: a1.x, y2: a1.y, stroke: col, 'stroke-width': 0.014, 'stroke-dasharray': '0.05 0.035' });
        if (r.al) sv(svg, 'text', { x: a1.x + 0.03, y: a1.y - 0.03, 'font-size': f * 0.95, fill: col, 'font-style': 'italic' }, r.al);
      }
      var steps = Math.max(8, Math.round(Math.abs(r.a) / 4)), prev = null, pts = [];
      for (var k = 0; k <= steps; k++) {
        var ang = r.a * D2R * k / steps, c = Math.cos(ang), s2 = Math.sin(ang), v = r.v;
        var dot = n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
        var cr = [n[1] * v[2] - n[2] * v[1], n[2] * v[0] - n[0] * v[2], n[0] * v[1] - n[1] * v[0]];
        var w = [v[0] * c + cr[0] * s2 + n[0] * dot * (1 - c), v[1] * c + cr[1] * s2 + n[1] * dot * (1 - c), v[2] * c + cr[2] * s2 + n[2] * dot * (1 - c)];
        pts.push(P(w));
      }
      var mid2 = id + 'r' + j;
      var mk = sv(defs, 'marker', { id: mid2, viewBox: '0 0 10 10', refX: 6, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto' });
      sv(mk, 'path', { d: 'M0,0 L10,5 L0,10 z', fill: col });
      for (k = 1; k < pts.length; k++) {
        var back = pts[k].d < -0.02 && pts[k - 1].d < -0.02;
        var seg = { x1: pts[k - 1].x, y1: pts[k - 1].y, x2: pts[k].x, y2: pts[k].y, stroke: col, 'stroke-width': r.w || 0.026, 'stroke-linecap': 'round', opacity: back ? 0.55 : 1 };
        if (back) seg['stroke-dasharray'] = '0.04 0.03';
        if (k === pts.length - 1) seg['marker-end'] = 'url(#' + mid2 + ')';
        sv(svg, 'line', seg);
      }
      if (r.l) {
        var mpt = pts[Math.floor(pts.length / 2)];
        sv(svg, 'text', { x: mpt.x + (r.dx || 0.05), y: mpt.y + (r.dy || -0.05), 'font-size': f, fill: col, 'font-weight': 600 }, r.l);
      }
    });
    (cfg.pts || []).forEach(function (pt, j) {
      var col = pt.c || '#99004C', mid = id + j;
      var m = sv(defs, 'marker', { id: mid, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' });
      sv(m, 'path', { d: 'M0,0 L10,5 L0,10 z', fill: col });
      var q = P(pt.v), len = Math.sqrt(pt.v[0] * pt.v[0] + pt.v[1] * pt.v[1] + pt.v[2] * pt.v[2]);
      if (pt.l === undefined || pt.arrow !== false) sv(svg, 'line', { x1: 0, y1: 0, x2: q.x, y2: q.y, stroke: col, 'stroke-width': 0.035, 'stroke-dasharray': pt.dash ? '0.06 0.04' : 'none', 'marker-end': len > 0.05 ? 'url(#' + mid + ')' : '' });
      if (pt.dot) sv(svg, 'circle', { cx: q.x, cy: q.y, r: 0.045, fill: col });
      if (pt.l) sv(svg, 'text', { x: q.x + (pt.dx || 0.06), y: q.y + (pt.dy || -0.06), 'font-size': f * 1.05, fill: col, 'font-weight': 600 }, pt.l);
    });
  }
