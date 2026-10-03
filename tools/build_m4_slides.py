"""Assemble media/level1/m4-slides.html and m4-explore.html from the Module 1 deck's frame,
tools/m4_slides_body.html, tools/m4_explore_src.html and tools/bloch.js."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 4 Slides: Two Qubits and Entanglement</title>")
head = head.replace("  svg text { font-family:", "  svg.bloch { overflow:visible; }\n  svg text { font-family:")
# keep the 1280 x 720 stage from shrinking inside narrow frames (it is scaled to fit instead)
assert "#stage { width:1280px; height:720px;" in head
head = head.replace("#stage { width:1280px; height:720px;", "#stage { width:1280px; height:720px; flex-shrink:0;")
body = (ROOT / "tools/m4_slides_body.html").read_text()
bar = "\n".join(m1[365:373])
ctrl = "\n".join(m1[393:])
bloch = (ROOT / "tools/bloch.js").read_text()
fringe = r'''
  document.querySelectorAll('svg.bloch').forEach(function (s) { drawBloch(s, JSON.parse(s.getAttribute('data-cfg'))); });

'''
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + bloch + fringe + ctrl
assert "`" not in out and "${" not in out and "—" not in out
(ROOT / "media/level1/m4-slides.html").write_text(out)
print("wrote media/level1/m4-slides.html", len(out))

src = (ROOT / "tools/m4_explore_src.html").read_text()
exp = src.replace("/*BLOCH*/", bloch)
assert "`" not in exp and "${" not in exp and "—" not in exp
(ROOT / "media/level1/m4-explore.html").write_text(exp)
print("wrote media/level1/m4-explore.html", len(exp))

