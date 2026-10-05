"""Assemble media/level2/m3-slides.html and m3-explore.html (Quantum Computing Intermediate, Module 3: Shor's algorithm)
from the Level 1 deck frame (media/level1/m1-slides.html), tools/l2_m3_slides_body.html and tools/l2_m3_explore_src.html."""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media/level2"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, GREY, INK, GOLD, MUTED = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B"


def cycle(values=(1, 7, 4, 13), size=360):
    """The powers of 7 mod 15 as a cycle of four nodes."""
    c, R = size / 2, size / 2 - 60
    p = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" aria-label="The powers of 7 mod 15 repeat: 1, 7, 4, 13, then 1 again">']
    pts = []
    for i, v in enumerate(values):
        a = math.pi / 2 - 2 * math.pi * i / len(values)
        pts.append((c + R * math.cos(a), c - R * math.sin(a)))
    p.append('<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{MAROON}"/></marker></defs>')
    for i in range(len(values)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(values)]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        sx, sy, ex, ey = x1 + dx / L * 40, y1 + dy / L * 40, x2 - dx / L * 44, y2 - dy / L * 44
        p.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{MAROON}" stroke-width="4" marker-end="url(#ar)"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ox, oy = (mx - c) * 0.28, (my - c) * 0.28
        p.append(f'<text x="{mx + ox:.1f}" y="{my + oy + 6:.1f}" font-size="18" text-anchor="middle" fill="{MUTED}">× 7</text>')
    for (x, y), v in zip(pts, values):
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="36" fill="#fff" stroke="{INK}" stroke-width="3"/>')
        p.append(f'<text x="{x:.1f}" y="{y + 10:.1f}" font-size="28" text-anchor="middle" fill="{INK}">{v}</text>')
    p.append(f'<text x="{c}" y="{c + 8}" font-size="20" text-anchor="middle" fill="{MAROON}">r = 4</text>')
    p.append(f'<text x="{c}" y="{size - 6}" font-size="17" text-anchor="middle" fill="{MUTED}">7ᵏ mod 15</text>')
    p.append('</svg>')
    return ''.join(p)


def circuit(w=1120, h=330):
    """Schematic order-finding circuit: counting qubits, controlled U^(2^k), inverse QFT, measurements; register in |1>."""
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Order-finding circuit: H on the counting qubits, controlled U, U squared and U to the fourth on a register in the state 1, the inverse QFT and measurements">',
         '<g font-size="22" fill="#24313D" stroke-width="3">']
    rows = [50, 120, 190]
    labels = ["c0 = |0⟩", "c1 = |0⟩", "c2 = |0⟩"]
    for y, lab in zip(rows, labels):
        p.append(f'<text x="0" y="{y + 7}" font-size="20">{lab}</text><line x1="96" y1="{y}" x2="880" y2="{y}" stroke="{INK}"/>')
    p.append(f'<text x="0" y="287" font-size="20">reg = |1⟩</text>')
    for dy in (-12, -4, 4, 12):
        p.append(f'<line x1="106" y1="{280 + dy}" x2="880" y2="{280 + dy}" stroke="{INK}" stroke-width="1.5"/>')
    p.append(f'<text x="106" y="318" font-size="16" fill="{MUTED}">4 qubits</text>')
    for y in rows:
        p.append(f'<rect x="130" y="{y - 22}" width="44" height="44" rx="5" fill="#fff" stroke="{MAROON}"/><text x="152" y="{y + 8}" text-anchor="middle" fill="{MAROON}">H</text>')
    for x, y, lab in ((270, 50, "U"), (390, 120, "U²"), (510, 190, "U⁴")):
        p.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{INK}"/><line x1="{x}" y1="{y}" x2="{x}" y2="256" stroke="{INK}"/>')
        p.append(f'<rect x="{x - 36}" y="256" width="72" height="48" rx="5" fill="#fff" stroke="{INK}"/><text x="{x}" y="288" text-anchor="middle">{lab}</text>')
    p.append(f'<rect x="590" y="28" width="170" height="184" rx="8" fill="#F8F4F6" stroke="{MAROON}"/><text x="675" y="126" text-anchor="middle" fill="{MAROON}">QFT†</text>')
    for y in rows:
        p.append(f'<rect x="800" y="{y - 22}" width="50" height="44" rx="5" fill="#fff" stroke="{INK}"/><text x="825" y="{y + 8}" text-anchor="middle">M</text>')
    p.append(f'<text x="900" y="126" font-size="20" fill="#2E7D4F">m/2ᵗ ≈ s/r</text>')
    p.append(f'<text x="900" y="156" font-size="18" fill="{MUTED}">s random</text>')
    p.append('</g></svg>')
    return ''.join(p)


def hist(counts, labels, w=470, h=300):
    x0, x1, y0, y1 = 50, w - 10, h - 46, 20
    top = 400
    bw = (x1 - x0) / len(counts)
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Histogram of the four results">']
    for v in (0, 200, 400):
        y = y0 - (y0 - y1) * v / top
        p.append(f'<line x1="{x0 - 4}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{INK if v == 0 else "#E4DCE0"}" stroke-width="{2 if v == 0 else 1}"/>')
        p.append(f'<text x="{x0 - 8}" y="{y + 5:.1f}" font-size="16" text-anchor="end" fill="{INK}">{v}</text>')
    for i, (c, lab) in enumerate(zip(counts, labels)):
        hh = (y0 - y1) * c / top
        x = x0 + i * bw + bw * 0.2
        p.append(f'<rect x="{x:.1f}" y="{y0 - hh:.1f}" width="{bw * 0.6:.1f}" height="{hh:.1f}" fill="{MAROON}"/>')
        p.append(f'<text x="{x + bw * 0.3:.1f}" y="{y0 - hh - 6:.1f}" font-size="16" text-anchor="middle" fill="{INK}">{c}</text>')
        p.append(f'<text x="{x + bw * 0.3:.1f}" y="{y0 + 20}" font-size="17" text-anchor="middle" fill="{INK}">{lab}</text>')
    p.append(f'<text x="{x1}" y="{h - 4}" font-size="15" text-anchor="end" fill="{MUTED}">result m (of 256)</text>')
    p.append('</svg>')
    return ''.join(p)


m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 3 Slides: Shor's Algorithm and Period Finding</title>")
assert "Shor" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n  svg text { font-family:")
body = (ROOT / "tools/l2_m3_slides_body.html").read_text()
body = body.replace("/*CYCLE_TITLE*/", cycle())
body = body.replace("/*CIRCUIT*/", circuit())
body = body.replace("/*HIST7*/", hist([265, 239, 261, 235], ["0", "64", "128", "192"]))
bar = "\n".join(m1[365:373]).replace("1 / 19", "1 / 18")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m3-slides.html").write_text(out)
print("wrote media/level2/m3-slides.html", len(out))

exp = (ROOT / "tools/l2_m3_explore_src.html").read_text()
assert "`" not in exp and "${" not in exp and "—" not in exp
(OUT / "m3-explore.html").write_text(exp)
print("wrote media/level2/m3-explore.html", len(exp))
