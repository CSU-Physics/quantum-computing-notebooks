"""Assemble media/level1/m3-slides.html and m3-explore.html from the Module 1 deck's frame,
tools/m3_slides_body.html, tools/m3_explore_src.html and tools/bloch.js."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 3 Slides: Single-Qubit Gates</title>")
head = head.replace("  svg text { font-family:", "  svg.bloch { overflow:visible; }\n  svg text { font-family:")
# keep the 1280 x 720 stage from shrinking inside narrow frames (it is scaled to fit instead)
assert "#stage { width:1280px; height:720px;" in head
head = head.replace("#stage { width:1280px; height:720px;", "#stage { width:1280px; height:720px; flex-shrink:0;")
body = (ROOT / "tools/m3_slides_body.html").read_text()
bar = "\n".join(m1[365:373])
ctrl = "\n".join(m1[393:])
bloch = (ROOT / "tools/bloch.js").read_text()
# Slide 15: the interference fringe. The dots are the lab's Step 6 counts (seed 7, 1,000 shots).
fringe = r'''
  (function () {
    var svg = document.getElementById('fringe');
    var zeros = [1000, 936, 747, 502, 257, 71, 0, 71, 257, 502, 747, 936, 1000];
    var x0 = 70, x1 = 540, y0 = 270, y1 = 30;
    function X(p) { return x0 + (x1 - x0) * p / (2 * Math.PI); }
    function Y(v) { return y0 - (y0 - y1) * v; }
    sv(svg, 'line', { x1: x0, y1: y0, x2: x1, y2: y0, stroke: '#24313D', 'stroke-width': 2 });
    sv(svg, 'line', { x1: x0, y1: y0, x2: x0, y2: y1, stroke: '#24313D', 'stroke-width': 2 });
    [[0, '0'], [0.5, '0.5'], [1, '1']].forEach(function (t) {
      sv(svg, 'text', { x: x0 - 10, y: Y(t[0]) + 6, 'font-size': 18, 'text-anchor': 'end', fill: '#24313D' }, t[1]);
    });
    [[0, '0'], [Math.PI, 'π'], [2 * Math.PI, '2π']].forEach(function (t) {
      sv(svg, 'text', { x: X(t[0]), y: y0 + 26, 'font-size': 19, 'text-anchor': 'middle', fill: '#24313D' }, t[1]);
    });
    sv(svg, 'text', { x: x1, y: y0 + 48, 'font-size': 17, 'text-anchor': 'end', fill: '#6A626B' }, 'phase φ');
    sv(svg, 'text', { x: x0 + 8, y: y1 - 8, 'font-size': 17, fill: '#6A626B' }, 'P(0)');
    var pts = [];
    for (var k = 0; k <= 120; k++) { var p = 2 * Math.PI * k / 120; pts.push(X(p) + ',' + Y(Math.pow(Math.cos(p / 2), 2))); }
    sv(svg, 'polyline', { points: pts.join(' '), fill: 'none', stroke: '#24313D', 'stroke-width': 3 });
    zeros.forEach(function (z, i) { sv(svg, 'circle', { cx: X(2 * Math.PI * i / 12), cy: Y(z / 1000), r: 7, fill: '#99004C' }); });
    sv(svg, 'circle', { cx: 330, cy: 18, r: 7, fill: '#99004C' });
    sv(svg, 'text', { x: 344, y: 24, 'font-size': 16, fill: '#24313D' }, 'measured (1,000 shots)');
  })();
  document.querySelectorAll('svg.bloch').forEach(function (s) { drawBloch(s, JSON.parse(s.getAttribute('data-cfg'))); });

'''
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + bloch + fringe + ctrl
assert "`" not in out and "${" not in out and "—" not in out
(ROOT / "media/level1/m3-slides.html").write_text(out)
print("wrote media/level1/m3-slides.html", len(out))

src = (ROOT / "tools/m3_explore_src.html").read_text()
exp = src.replace("/*BLOCH*/", bloch)
assert "`" not in exp and "${" not in exp and "—" not in exp
(ROOT / "media/level1/m3-explore.html").write_text(exp)
print("wrote media/level1/m3-explore.html", len(exp))
