"""Assemble media/level2/m0-slides.html and m0-explore.html (Quantum Computing Intermediate, Module 0)
from the Level 1 deck frame (media/level1/m1-slides.html), tools/l2_m0_slides_body.html,
tools/l2_m0_explore_src.html and tools/bloch.js."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media/level2"
OUT.mkdir(parents=True, exist_ok=True)
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 0 Slides: Dynamic Circuits and Teleportation</title>")
assert "<title>Module 0 Slides" in head
head = head.replace("  svg text { font-family:",
                    "  svg.bloch { overflow:visible; }\n"
                    "  pre.code { font-family: Consolas, \"Courier New\", monospace; font-size:22px; background:var(--tint); "
                    "padding:12px 18px; border-radius:6px; margin:6px 0 14px; line-height:1.3; white-space:pre; }\n"
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                    "  svg text { font-family:")
body = (ROOT / "tools/l2_m0_slides_body.html").read_text()
bar = "\n".join(m1[365:373]).replace("1 / 19", "1 / 18")
ctrl = "\n".join(m1[393:])
bloch = (ROOT / "tools/bloch.js").read_text()
draw = "\n  document.querySelectorAll('svg.bloch').forEach(function (s) { drawBloch(s, JSON.parse(s.getAttribute('data-cfg'))); });\n\n"
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + bloch + draw + ctrl
assert "`" not in out and "${" not in out and "—" not in out
(OUT / "m0-slides.html").write_text(out)
print("wrote media/level2/m0-slides.html", len(out))

exp = (ROOT / "tools/l2_m0_explore_src.html").read_text().replace("/*BLOCH*/", bloch)
assert "`" not in exp and "${" not in exp and "—" not in exp and "/*BLOCH*/" not in exp
(OUT / "m0-explore.html").write_text(exp)
print("wrote media/level2/m0-explore.html", len(exp))
