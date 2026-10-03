"""Assemble media/level1/m5-slides.html and m5-explore.html from the Module 1 deck's frame,
tools/m5_slides_body.html and tools/m5_explore_src.html."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 5 Slides: Simulators and Real Hardware</title>")
# keep the 1280 x 720 stage from shrinking inside narrow frames (it is scaled to fit instead)
assert "#stage { width:1280px; height:720px;" in head
head = head.replace("#stage { width:1280px; height:720px;", "#stage { width:1280px; height:720px; flex-shrink:0;")
body = (ROOT / "tools/m5_slides_body.html").read_text()
bar = "\n".join(m1[365:373])
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out
(ROOT / "media/level1/m5-slides.html").write_text(out)
print("wrote media/level1/m5-slides.html", len(out))

exp = (ROOT / "tools/m5_explore_src.html").read_text()
assert "`" not in exp and "${" not in exp and "—" not in exp
(ROOT / "media/level1/m5-explore.html").write_text(exp)
print("wrote media/level1/m5-explore.html", len(exp))
