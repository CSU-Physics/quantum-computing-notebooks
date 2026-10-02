"""Assemble media/level1/m2-slides.html from the Module 1 deck's frame, tools/m2_slides_body.html and tools/bloch.js."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 2 Slides: One Qubit</title>")
head = head.replace("  svg text { font-family:", "  svg.bloch { overflow:visible; }\n  svg text { font-family:")
body = (ROOT / "tools/m2_slides_body.html").read_text()
bar = "\n".join(m1[365:373])
ctrl = "\n".join(m1[393:])
bloch = (ROOT / "tools/bloch.js").read_text()
three = r'''
  // Slide 15: predicted and sampled 0s in three bases
  (function () {
    var svg = document.getElementById('three');
    var data = [['Z', 250, 257], ['X', 717, 718], ['Y', 875, 879]];
    var x0 = 60, yb = 300, top = 80, H = yb - top;
    sv(svg, 'line', { x1: x0, y1: yb, x2: 540, y2: yb, stroke: '#24313D', 'stroke-width': 2 });
    data.forEach(function (d, i) {
      var gx = x0 + 30 + i * 160;
      [[d[1], '#C9A227'], [d[2], '#99004C']].forEach(function (b, j) {
        var h = b[0] / 1000 * H;
        sv(svg, 'rect', { x: gx + j * 58, y: yb - h, width: 52, height: h, fill: b[1] });
        sv(svg, 'text', { x: gx + j * 58 + 26, y: yb - h - 8, 'font-size': 18, 'text-anchor': 'middle', fill: '#24313D' }, String(b[0]));
      });
      sv(svg, 'text', { x: gx + 55, y: yb + 30, 'font-size': 22, 'text-anchor': 'middle', fill: '#24313D' }, d[0] + ' basis');
    });
    sv(svg, 'rect', { x: 70, y: 6, width: 18, height: 18, fill: '#C9A227' });
    sv(svg, 'text', { x: 94, y: 21, 'font-size': 17, fill: '#24313D' }, 'predicted: 1000 × P(0)');
    sv(svg, 'rect', { x: 300, y: 6, width: 18, height: 18, fill: '#99004C' });
    sv(svg, 'text', { x: 324, y: 21, 'font-size': 17, fill: '#24313D' }, 'sampled 0s');
  })();
  document.querySelectorAll('svg.bloch').forEach(function (s) { drawBloch(s, JSON.parse(s.getAttribute('data-cfg'))); });

'''
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + bloch + three + ctrl
assert "`" not in out and "${" not in out and "—" not in out
(ROOT / "media/level1/m2-slides.html").write_text(out)
print("wrote media/level1/m2-slides.html", len(out))

# The Module 2 explorer uses the same Bloch-sphere drawing code.
src = (ROOT / "tools/m2_explore_src.html").read_text()
exp = src.replace("/*BLOCH*/", bloch)
assert "`" not in exp and "${" not in exp and "—" not in exp
(ROOT / "media/level1/m2-explore.html").write_text(exp)
print("wrote media/level1/m2-explore.html", len(exp))
