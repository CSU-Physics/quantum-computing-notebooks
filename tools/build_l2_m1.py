"""Assemble media/level2/m1-slides.html and m1-explore.html (Quantum Computing Intermediate, Module 1)
from the Level 1 deck frame (media/level1/m1-slides.html), tools/l2_m1_slides_body.html and tools/l2_m1_explore_src.html."""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media/level2"
OUT.mkdir(parents=True, exist_ok=True)


def dials(j, n, size, labels=True, per_row=8):
    """A row of phase dials for QFT|j> on n qubits, as inline SVG (phase 0 to the right, anticlockwise)."""
    N = 2 ** n
    gap = size * 1.25
    rows = math.ceil(N / per_row)
    h = rows * (size + (32 if labels else 8))
    w = min(N, per_row) * gap
    parts = [f'<svg width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}" aria-label="Phases of QFT of |{j}&gt; on {n} qubits">']
    r = size / 2 - 4
    for k in range(N):
        cx = (k % per_row) * gap + size / 2
        cy = (k // per_row) * (size + (32 if labels else 8)) + size / 2
        a = 2 * math.pi * ((j * k) % N) / N
        x2, y2 = cx + 0.9 * r * math.cos(a), cy - 0.9 * r * math.sin(a)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#F8F4F6" stroke="#B9AEB4" stroke-width="2"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + r:.1f}" y2="{cy:.1f}" stroke="#E4DCE0" stroke-width="2"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#99004C" stroke-width="{max(3, size / 16):.1f}" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="{max(4, size / 14):.1f}" fill="#99004C"/>')
        if labels:
            deg = round(360 * ((j * k) % N) / N, 1)
            deg_s = f"{deg:g}"
            parts.append(f'<text x="{cx:.1f}" y="{cy + size / 2 + 20:.1f}" font-size="18" text-anchor="middle" fill="#24313D">|{format(k, f"0{n}b")}⟩ {deg_s}°</text>')
    parts.append('</svg>')
    return ''.join(parts)


m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 1 Slides: The QFT and Phase Estimation</title>")
assert "QFT and Phase Estimation" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n  svg text { font-family:")
body = (ROOT / "tools/l2_m1_slides_body.html").read_text()
body = body.replace("/*DIALS_TITLE*/", dials(3, 3, 88, labels=False, per_row=4))
body = body.replace("/*DIALS_3*/", dials(3, 3, 112, labels=True, per_row=8))
bar = "\n".join(m1[365:373]).replace("1 / 19", "1 / 18")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m1-slides.html").write_text(out)
print("wrote media/level2/m1-slides.html", len(out))

exp = (ROOT / "tools/l2_m1_explore_src.html").read_text()
assert "`" not in exp and "${" not in exp and "—" not in exp
(OUT / "m1-explore.html").write_text(exp)
print("wrote media/level2/m1-explore.html", len(exp))
