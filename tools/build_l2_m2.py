"""Assemble media/level2/m2-slides.html and m2-explore.html (Quantum Computing Intermediate, Module 2: Grover's search)
from the Level 1 deck frame (media/level1/m1-slides.html), tools/l2_m2_slides_body.html and tools/l2_m2_explore_src.html."""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media/level2"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, GREY, INK, GOLD, MUTED = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B"


def bars(amps, n, marked, w=480, h=250, label=None, show_mean=True, font=17):
    """Signed amplitude bars as inline SVG, with a zero line and (optionally) the dashed mean."""
    N = len(amps)
    x0, x1, ymid, sc = 46, w - 8, h / 2 + (8 if label else 0), (h / 2 - 34)
    bw = (x1 - x0) / N
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{label or "amplitudes"}">']
    if label:
        p.append(f'<text x="{x0}" y="18" font-size="{font}" fill="{INK}">{label}</text>')
    for v in (-1, 0, 1):
        y = ymid - sc * v
        p.append(f'<line x1="{x0 - 4}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{INK if v == 0 else "#E4DCE0"}" stroke-width="{2 if v == 0 else 1}"/>')
        p.append(f'<text x="{x0 - 8}" y="{y + 5:.1f}" font-size="{font - 2}" text-anchor="end" fill="{INK}">{v}</text>')
    for k, a in enumerate(amps):
        hh = sc * abs(a)
        y = ymid - hh if a >= 0 else ymid
        p.append(f'<rect x="{x0 + k * bw + bw * 0.14:.1f}" y="{y:.1f}" width="{bw * 0.72:.1f}" height="{max(hh, 0.8):.1f}" fill="{MAROON if k == marked else GREY}"/>')
        p.append(f'<text x="{x0 + (k + 0.5) * bw:.1f}" y="{h - 4}" font-size="{font - 3 if N > 4 else font - 1}" text-anchor="middle" fill="{MAROON if k == marked else INK}">{format(k, f"0{n}b")}</text>')
    if show_mean:
        m = sum(amps) / N
        y = ymid - sc * m
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GOLD}" stroke-width="3" stroke-dasharray="7 5"/>')
    p.append('</svg>')
    return ''.join(p)


def grover_amps(n, marked, steps):
    """Amplitudes after a list of steps: 'o' (oracle) and 'd' (reflection 2m - a)."""
    N = 2 ** n
    a = [1 / math.sqrt(N)] * N
    for s in steps:
        if s == 'o':
            a[marked] = -a[marked]
        else:
            m = sum(a) / N
            a = [2 * m - v for v in a]
    return a


def geometry(theta_deg=20.7048, w=440, h=380):
    """The plane of |w_perp> (right) and |w> (up): the state at angles theta, 3 theta, 5 theta (N = 8)."""
    ox, oy, L = 120, h - 44, 270
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Geometric picture: the state starts at angle theta from |w perp> and turns by 2 theta per iteration">']
    p.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + L + 20}" y2="{oy}" stroke="{INK}" stroke-width="2"/>')
    p.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - L - 20}" stroke="{INK}" stroke-width="2"/>')
    p.append(f'<text x="{ox + L + 20}" y="{oy - 10}" font-size="20" text-anchor="end" fill="{INK}">|w⊥⟩</text>')
    p.append(f'<text x="{ox + 10}" y="{oy - L - 6}" font-size="20" fill="{INK}">|w⟩</text>')
    cols = [GREY, "#C98FAE", MAROON]
    labs = ["start: θ", "t = 1: 3θ", "t = 2: 5θ"]
    for t in range(3):
        ang = math.radians((2 * t + 1) * theta_deg)
        x, y = ox + L * math.cos(ang), oy - L * math.sin(ang)
        p.append(f'<line x1="{ox}" y1="{oy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{cols[t]}" stroke-width="5" stroke-linecap="round"/>')
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{cols[t]}"/>')
        if t < 2:
            p.append(f'<text x="{x + 12:.1f}" y="{y + 6:.1f}" font-size="18" fill="{INK}">{labs[t]}</text>')
        else:
            p.append(f'<text x="{x - 10:.1f}" y="{y - 14:.1f}" font-size="18" text-anchor="middle" fill="{INK}">{labs[t]}</text>')
    r = 70
    a1 = math.radians(theta_deg)
    p.append(f'<path d="M {ox + r} {oy} A {r} {r} 0 0 0 {ox + r * math.cos(a1):.1f} {oy - r * math.sin(a1):.1f}" fill="none" stroke="{GOLD}" stroke-width="3"/>')
    p.append(f'<text x="{ox + r + 8}" y="{oy - 8}" font-size="18" fill="#8A6D00">θ</text>')
    p.append(f'<text x="{w - 6}" y="{h - 6}" font-size="16" text-anchor="end" fill="{MUTED}">N = 8, θ ≈ 20.7°; 5θ ≈ 103.5° passes |w⟩</text>')
    p.append('</svg>')
    return ''.join(p)


def curve(n=3, T=6, w=520, h=300):
    th = math.asin(1 / math.sqrt(2 ** n))
    x0, x1, y0, y1 = 50, w - 16, h - 40, 20
    X = lambda t: x0 + (x1 - x0) * t / T
    Y = lambda v: y0 - (y0 - y1) * v
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Success probability against iterations for N = 8">']
    for v in (0, 0.5, 1):
        p.append(f'<line x1="{x0 - 4}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{INK if v == 0 else "#E4DCE0"}" stroke-width="{2 if v == 0 else 1}"/>')
        p.append(f'<text x="{x0 - 10}" y="{Y(v) + 6:.1f}" font-size="18" text-anchor="end" fill="{INK}">{v:g}</text>')
    pts = []
    for t in range(T + 1):
        v = math.sin((2 * t + 1) * th) ** 2
        pts.append(f'{X(t):.1f},{Y(v):.1f}')
        p.append(f'<text x="{X(t):.1f}" y="{y0 + 24}" font-size="18" text-anchor="middle" fill="{INK}">{t}</text>')
    p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{GREY}" stroke-width="3"/>')
    for t in range(T + 1):
        v = math.sin((2 * t + 1) * th) ** 2
        p.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="{8 if t == 2 else 6}" fill="{MAROON if t == 2 else MUTED}"/>')
        p.append(f'<text x="{X(t) + (12 if t != 6 else -12):.1f}" y="{Y(v) + (22 if v > 0.9 else -10):.1f}" font-size="16" text-anchor="{"start" if t != 6 else "end"}" fill="{INK}">{v:.3f}</text>')
    p.append(f'<text x="{x1}" y="{h - 2}" font-size="17" text-anchor="end" fill="{MUTED}">iterations t</text>')
    p.append('</svg>')
    return ''.join(p)


m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 2 Slides: Grover's Search</title>")
assert "Grover" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                    "  .slide pre { font-family: Consolas, \"Courier New\", monospace; background:var(--tint); border-left:6px solid var(--maroon); padding:12px 18px; margin:6px 0; white-space:pre; }\n"
                    "  svg text { font-family:")
body = (ROOT / "tools/l2_m2_slides_body.html").read_text()
body = body.replace("/*BARS_TITLE*/", bars(grover_amps(3, 5, ['o', 'd', 'o', 'd']), 3, 5, w=420, h=260, show_mean=False))
body = body.replace("/*BARS_ORACLE*/", bars(grover_amps(2, 2, ['o']), 2, 2, w=440, h=270, label="after the oracle for 10", show_mean=False))
body = body.replace("/*BARS_MEAN*/", bars(grover_amps(2, 2, ['o']), 2, 2, w=440, h=270, label="before: gold line = mean 0.25"))
seq = [("start", []), ("oracle", ['o']), ("diffuser", ['o', 'd'])]
row = ''.join(f'<div style="display:inline-block;margin-right:14px">{bars(grover_amps(2, 2, s), 2, 2, w=340, h=250, label=lab, show_mean=(lab != "diffuser"))}</div>' for lab, s in seq)
body = body.replace("/*BARS_N4*/", row)
body = body.replace("/*GEOM*/", geometry())
body = body.replace("/*CURVE8*/", curve())
bar = "\n".join(m1[365:373]).replace("1 / 19", "1 / 18")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m2-slides.html").write_text(out)
print("wrote media/level2/m2-slides.html", len(out))

exp = (ROOT / "tools/l2_m2_explore_src.html").read_text()
assert "`" not in exp and "${" not in exp and "—" not in exp
(OUT / "m2-explore.html").write_text(exp)
print("wrote media/level2/m2-explore.html", len(exp))
