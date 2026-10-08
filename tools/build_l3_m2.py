"""Assemble media/level3/m2-slides.html and m2-explore.html (Quantum Computing Advanced, Module 2: quantum error
correction) from the Level 1 deck frame (media/level1/m1-slides.html), tools/l3_m2_slides_body.html and
tools/l3_m2_explore_src.html. Every number on the slides is computed here with qsim, the reference solutions of
checks/l3_m2_check.py and content/level3/l3_m2_device_run.json, the same way the lab computes them."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.modules["qiskit"] = None
sys.path[:0] = [str(ROOT / "content/level3"), str(ROOT / "checks")]
import qsim  # noqa: E402
import l3_m2_check as ref  # noqa: E402

OUT = ROOT / "media/level3"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, GREY, INK, GOLD, MUTED, OK = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#2E7D4F"

# ---------------------------------------------------------------- the numbers
dev1 = json.loads((ROOT / "content/level3/l3_m1_device_data.json").read_text())
sys.modules["qiskit"] = None


run = json.loads((ROOT / "content/level3/l3_m2_device_run.json").read_text())
MED_CZ, MED_RO = run["device_median_cz_error"], run["device_median_readout_error"]


def failure(p, q=0.0, p_cx=0.0):
    nm = ref.noise_model(p, q)
    if p_cx > 0:
        nm.add_all_qubit_quantum_error(qsim.depolarizing_error(p_cx, 2), ["cx"])
    return 1 - ref.data_fidelity(ref.one_round(0.0), 0.0, nm)


assert abs(failure(0.1) - 0.028) < 1e-12
F0502 = failure(0.05, 0.02)
assert round(F0502, 3) == 0.047


def break_even(p_cx):
    lo, hi = 1e-4, 0.45
    f = lambda p: failure(p, 0, p_cx) - p  # noqa: E731
    for _ in range(40):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


BREAK = break_even(0.01)
assert round(BREAK, 3) == 0.033
F_SMALL = failure(0.0, 0, 0.01)

run = json.loads((ROOT / "content/level3/l3_m2_device_run.json").read_text())
c0 = run["counts_logical_0"]
SHOTS = sum(c0.values())
zeros, events, raw, vote = [0, 0, 0], [0, 0, 0], 0, 0
for key, n in c0.items():
    *syn, data = key.split()
    prev = "00"
    for r, s in enumerate(syn):
        zeros[r] += n * (s == "00")
        events[r] += n * (s != prev)
        prev = s
    bits = [int(b) for b in data]
    raw += n * sum(bits)
    vote += n * (sum(bits) >= 2)
assert zeros[0] == 3942
ONE_ROUND_ONLY = sum(n for k, n in c0.items() if sum(s != "00" for s in k.split()[:3]) == 1 and k.split()[-1] == "000")


# ---------------------------------------------------------------- drawings
def axes(p, w, h, x0, x1, yb, yt, xticks, X, yticks, Y, xlab, ylab):
    for v, lab in yticks:
        p.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="#E4DCE0"/>'
                 f'<text x="{x0 - 8}" y="{Y(v) + 5:.1f}" font-size="14" text-anchor="end">{lab}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    for v in xticks:
        p.append(f'<text x="{X(v):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle">{v:g}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 8}" font-size="15" text-anchor="middle" fill="{MUTED}">{xlab}</text>')
    p.append(f'<text x="16" y="{(yb + yt) / 2}" font-size="15" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 16 {(yb + yt) / 2})">{ylab}</text>')


def poly(X, Y, xs, ys, col, w=3, dash=""):
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}"{d}/>'


def legend(items, x, y):
    out = []
    for i, (col, lab, dash) in enumerate(items):
        yy = y + 22 * i
        d = ' stroke-dasharray="6 4"' if dash else ""
        out.append(f'<line x1="{x}" y1="{yy}" x2="{x + 22}" y2="{yy}" stroke="{col}" stroke-width="3"{d}/>'
                   f'<text x="{x + 30}" y="{yy + 5}" font-size="15">{lab}</text>')
    return "".join(out)


def classical(w=520, h=360):
    x0, x1, yb, yt = 70, w - 20, h - 60, 20
    X = lambda v: x0 + (x1 - x0) * v  # noqa: E731
    Y = lambda v: yb - (yb - yt) * v  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Error probability of one bit, p, and of three bits with a majority vote, 3p squared minus 2p cubed">']
    axes(p, w, h, x0, x1, yb, yt, [0, 0.25, 0.5, 0.75, 1], X, [(v, f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)], Y, "flip probability p", "error probability")
    ps = np.linspace(0, 1, 101)
    p.append(poly(X, Y, ps, ps, INK, 2.5, "7 5"))
    p.append(poly(X, Y, ps, 3 * ps ** 2 - 2 * ps ** 3, MAROON, 3.5))
    p.append(f'<circle cx="{X(0.5):.1f}" cy="{Y(0.5):.1f}" r="6" fill="{GOLD}"/>')
    p.append(legend([(INK, "one bit: p", True), (MAROON, "three bits, majority vote", False)], x0 + 20, yt + 20))
    p.append('</svg>')
    return "".join(p)


def box(x, y, w, h, label, stroke=INK, fill="#fff", fs=18):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + fs / 3}" text-anchor="middle" font-size="{fs}">{label}</text>')


def circuit_svg(w=1130, h=330):
    ys = [40, 95, 150, 215, 270]
    names = ["data 0", "data 1", "data 2", "ancilla 3", "ancilla 4"]
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="One round of the three-qubit code: encoding, syndrome measurement with two ancillas, reset and feedforward correction">',
         f'<g fill="{INK}" stroke-width="2.5">']
    for y, n in zip(ys, names):
        p.append(f'<text x="0" y="{y + 6}" font-size="17">{n}</text><line x1="95" y1="{y}" x2="{w - 10}" y2="{y}" stroke="{INK}"/>')
    p.append(f'<line x1="95" y1="{h - 18}" x2="{w - 10}" y2="{h - 18}" stroke="{GREY}" stroke-width="2"/><line x1="95" y1="{h - 14}" x2="{w - 10}" y2="{h - 14}" stroke="{GREY}" stroke-width="2"/>')
    p.append(f'<text x="0" y="{h - 10}" font-size="15" fill="{MUTED}">bits 0, 1</text>')

    def cnot(x, c, tq):
        return (f'<line x1="{x}" y1="{ys[c]}" x2="{x}" y2="{ys[tq] + (16 if ys[tq] > ys[c] else -16)}" stroke="{INK}"/>'
                f'<circle cx="{x}" cy="{ys[c]}" r="7" fill="{INK}"/>'
                f'<circle cx="{x}" cy="{ys[tq]}" r="16" fill="#fff" stroke="{INK}"/>'
                f'<line x1="{x - 16}" y1="{ys[tq]}" x2="{x + 16}" y2="{ys[tq]}" stroke="{INK}"/><line x1="{x}" y1="{ys[tq] - 16}" x2="{x}" y2="{ys[tq] + 16}" stroke="{INK}"/>')
    p.append(f'<rect x="115" y="12" width="150" height="160" rx="10" fill="none" stroke="{GREY}" stroke-dasharray="7 5"/><text x="190" y="190" text-anchor="middle" font-size="15" fill="{MUTED}">encode</text>')
    p.append(cnot(160, 0, 1) + cnot(225, 0, 2))
    p.append(f'<text x="315" y="{ys[1] + 6}" font-size="16" fill="{GOLD}" text-anchor="middle">noise</text>')
    p.append(f'<rect x="365" y="12" width="420" height="292" rx="10" fill="none" stroke="{GREY}" stroke-dasharray="7 5"/><text x="575" y="324" text-anchor="middle" font-size="15" fill="{MUTED}">syndrome</text>')
    p.append(cnot(410, 0, 3) + cnot(470, 1, 3) + cnot(530, 1, 4) + cnot(590, 2, 4))
    for a, x in ((3, 650), (4, 650)):
        p.append(box(x - 22, ys[a] - 20, 44, 40, "M", MAROON))
        p.append(f'<line x1="{x}" y1="{ys[a] + 20}" x2="{x}" y2="{h - 18}" stroke="{GREY}" stroke-width="2"/>')
        p.append(box(x + 40, ys[a] - 20, 70, 40, "reset", INK, fs=15))
    p.append(f'<rect x="815" y="12" width="300" height="292" rx="10" fill="#F8F4F6" stroke="{MAROON}" stroke-dasharray="7 5"/><text x="965" y="324" text-anchor="middle" font-size="15" fill="{MAROON}">feedforward (if_test)</text>')
    for d, lab in ((0, "X if 10"), (1, "X if 11"), (2, "X if 01")):
        p.append(box(900, ys[d] - 20, 130, 40, lab, MAROON, fs=16))
    p.append('</g></svg>')
    return "".join(p)


def syn_table():
    rows = [("none", "0", "0", "'00'", "nothing"), ("X on qubit 0", "1", "0", "'01'", "X on qubit 0"),
            ("X on qubit 1", "1", "1", "'11'", "X on qubit 1"), ("X on qubit 2", "0", "1", "'10'", "X on qubit 2")]
    s = '<table style="font-size:21px"><tr><th>error</th><th>s₁</th><th>s₂</th><th>counts key</th><th>correction</th></tr>'
    for r in rows:
        s += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return s + "</table>"


def fail_svg(w=540, h=370):
    x0, x1, yb, yt = 70, w - 20, h - 60, 20
    X = lambda v: x0 + (x1 - x0) * v / 0.3  # noqa: E731
    Y = lambda v: yb - (yb - yt) * v / 0.3  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Failure probability of one round against p for syndrome readout errors 0, 0.02 and 0.05, and for no code">']
    axes(p, w, h, x0, x1, yb, yt, [0, 0.1, 0.2, 0.3], X, [(v, f"{v:g}") for v in (0, 0.1, 0.2, 0.3)], Y, "flip probability p", "failure probability")
    ps = np.linspace(0.0, 0.3, 61)
    p.append(poly(X, Y, ps, ps, INK, 2.5, "7 5"))
    for q, col in ((0.0, MAROON), (0.02, GOLD), (0.05, OK)):
        p.append(poly(X, Y, ps, [failure(v, q) for v in ps], col, 3))
    p.append(legend([(INK, "no code", True), (MAROON, "code, q = 0", False), (GOLD, "code, q = 0.02", False), (OK, "code, q = 0.05", False)], x0 + 15, yt + 15))
    p.append('</svg>')
    return "".join(p)


def breakeven_svg(w=540, h=370):
    x0, x1, yb, yt = 70, w - 20, h - 60, 20
    X = lambda v: x0 + (x1 - x0) * v / 0.1  # noqa: E731
    Y = lambda v: yb - (yb - yt) * v / 0.1  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Failure probability with 1 percent CNOT errors against p, crossing the no-code line at the break-even point">']
    axes(p, w, h, x0, x1, yb, yt, [0, 0.025, 0.05, 0.075, 0.1], X, [(v, f"{v:g}") for v in (0, 0.025, 0.05, 0.075, 0.1)], Y, "flip probability p", "failure probability")
    ps = np.linspace(0.0, 0.1, 51)
    p.append(poly(X, Y, ps, ps, INK, 2.5, "7 5"))
    p.append(poly(X, Y, ps, [failure(v, 0, 0.01) for v in ps], MAROON, 3.5))
    p.append(f'<circle cx="{X(BREAK):.1f}" cy="{Y(BREAK):.1f}" r="7" fill="{GOLD}"/>')
    p.append(f'<text x="{X(BREAK) + 12:.1f}" y="{Y(BREAK) + 24:.1f}" font-size="15">break-even p ≈ {BREAK:.3f}</text>')
    p.append(legend([(INK, "no code", True), (MAROON, "code with 1% CNOT errors", False)], x0 + 15, yt + 15))
    p.append('</svg>')
    return "".join(p)


def device_svg(w=540, h=370):
    x0, x1, yb, yt = 80, w - 20, h - 60, 30
    top = 120
    Y = lambda v: yb - (yb - yt) * v / top  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Shots with a nonzero syndrome and detection events in rounds 1, 2 and 3, out of 4,000 shots of logical 0">']
    for v in (0, 30, 60, 90, 120):
        p.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="#E4DCE0"/><text x="{x0 - 8}" y="{Y(v) + 5:.1f}" font-size="14" text-anchor="end">{v}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    bw = 52
    for r in range(3):
        cx = x0 + 70 + r * 140
        nz = SHOTS - zeros[r]
        p.append(f'<rect x="{cx - bw - 3}" y="{Y(nz):.1f}" width="{bw}" height="{yb - Y(nz):.1f}" fill="{GREY}"/>')
        p.append(f'<text x="{cx - bw / 2 - 3}" y="{Y(nz) - 6:.1f}" font-size="14" text-anchor="middle">{nz}</text>')
        p.append(f'<rect x="{cx + 3}" y="{Y(events[r]):.1f}" width="{bw}" height="{yb - Y(events[r]):.1f}" fill="{MAROON}"/>')
        p.append(f'<text x="{cx + bw / 2 + 3}" y="{Y(events[r]) - 6:.1f}" font-size="14" text-anchor="middle">{events[r]}</text>')
        p.append(f'<text x="{cx}" y="{yb + 22}" font-size="15" text-anchor="middle">round {r + 1}</text>')
    p.append(f'<rect x="{x0 + 10}" y="{yt - 18}" width="14" height="14" fill="{GREY}"/><text x="{x0 + 30}" y="{yt - 6}" font-size="14">syndrome not 00</text>')
    p.append(f'<rect x="{x0 + 190}" y="{yt - 18}" width="14" height="14" fill="{MAROON}"/><text x="{x0 + 210}" y="{yt - 6}" font-size="14">detection events</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 10}" font-size="15" text-anchor="middle" fill="{MUTED}">shots out of {SHOTS:,}, logical |0⟩</text>')
    p.append('</svg>')
    return "".join(p)


def surface_svg(w=400, h=400):
    """Rotated distance-3 surface code: 9 data qubits (circles) and 8 measure qubits (squares, X or Z)."""
    s, o = 110, 90
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="A distance-3 surface code: nine data qubits on a grid with X and Z measure qubits between them">']
    # plaquettes
    for i in range(2):
        for j in range(2):
            col = "#F1E3EA" if (i + j) % 2 == 0 else "#E9EEF2"
            p.append(f'<rect x="{o + j * s}" y="{o + i * s}" width="{s}" height="{s}" fill="{col}" stroke="{GREY}"/>')
    for (x, y, col) in ((o + s / 2 + s, o - 18, "#E9EEF2"), (o + s / 2, o + 2 * s + 18, "#E9EEF2"), (o - 18, o + s / 2, "#F1E3EA"), (o + 2 * s + 18, o + s / 2 + s, "#F1E3EA")):
        p.append(f'<circle cx="{x}" cy="{y}" r="34" fill="{col}" stroke="{GREY}"/>')
    meas = [(o + s / 2, o + s / 2, "Z"), (o + 1.5 * s, o + s / 2, "X"), (o + s / 2, o + 1.5 * s, "X"), (o + 1.5 * s, o + 1.5 * s, "Z"),
            (o + 1.5 * s, o - 18, "X"), (o + s / 2, o + 2 * s + 18, "X"), (o - 18, o + s / 2, "Z"), (o + 2 * s + 18, o + 1.5 * s, "Z")]
    for x, y, lab in meas:
        col = MAROON if lab == "Z" else INK
        p.append(f'<rect x="{x - 15}" y="{y - 15}" width="30" height="30" rx="4" fill="{col}"/><text x="{x}" y="{y + 6}" font-size="16" text-anchor="middle" fill="#fff">{lab}</text>')
    for i in range(3):
        for j in range(3):
            p.append(f'<circle cx="{o + j * s}" cy="{o + i * s}" r="16" fill="#fff" stroke="{INK}" stroke-width="3"/>')
    p.append(f'<text x="{w / 2}" y="{h - 8}" font-size="15" text-anchor="middle" fill="{MUTED}">distance 3: 9 data qubits (circles), 8 measure qubits</text>')
    p.append('</svg>')
    return "".join(p)


def title_fig(w=430, h=230):
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Three data qubits with two ancillas measuring their parities">']
    xs, ya, yd = [50, 215, 380], 50, 160
    for a, (i, j) in enumerate(((0, 1), (1, 2))):
        ax = (xs[i] + xs[j]) / 2
        p.append(f'<line x1="{xs[i]}" y1="{yd}" x2="{ax}" y2="{ya}" stroke="{GREY}" stroke-width="3"/><line x1="{xs[j]}" y1="{yd}" x2="{ax}" y2="{ya}" stroke="{GREY}" stroke-width="3"/>')
        p.append(f'<rect x="{ax - 30}" y="{ya - 24}" width="60" height="48" rx="8" fill="{MAROON}"/><text x="{ax}" y="{ya + 7}" font-size="18" text-anchor="middle" fill="#fff">Z{i}Z{j}</text>')
    for k, x in enumerate(xs):
        p.append(f'<circle cx="{x}" cy="{yd}" r="30" fill="#fff" stroke="{INK}" stroke-width="3"/><text x="{x}" y="{yd + 7}" font-size="19" text-anchor="middle">q{k}</text>')
    p.append(f'<text x="{w / 2}" y="{h - 10}" font-size="17" text-anchor="middle" fill="{MUTED}">α|000⟩ + β|111⟩</text>')
    p.append('</svg>')
    return "".join(p)


m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 2 Slides: Quantum Error Correction</title>")
assert "Quantum Error Correction" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                    "  .slide table td, .slide table th { padding:5px 12px; text-align:left; }\n  svg text { font-family:")
body = (ROOT / "tools/l3_m2_slides_body.html").read_text()
pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
rep = {"/*TITLE_FIG*/": title_fig(), "/*CLASSICAL*/": classical(), "/*CIRCUIT*/": circuit_svg(), "/*SYN_TABLE*/": syn_table(),
       "/*FAIL*/": fail_svg(), "/*BREAKEVEN*/": breakeven_svg(), "/*DEVICE*/": device_svg(), "/*SURFACE*/": surface_svg(),
       "/*MED_CZ*/": f"{MED_CZ * 1e4:.0f}", "/*MED_RO*/": f"{MED_RO * 1e4:.0f}", "/*PL01*/": "0.028",
       "/*F0502*/": f"{F0502:.3f}", "/*BREAK*/": f"{BREAK:.3f}", "/*F_SMALL*/": pct(F_SMALL),
       "/*Q_RAW*/": pct(raw / (3 * SHOTS)), "/*Q_VOTE*/": (f"once in {SHOTS:,} shots" if vote == 1 else f"{vote} times in {SHOTS:,}"), "/*QSIM_V*/": qsim.__version__}
for k, v in rep.items():
    assert k in body, k
    body = body.replace(k, v)
n_slides = body.count('<section class="slide')
bar = "\n".join(m1[365:373]).replace("1 / 19", f"1 / {n_slides}")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m2-slides.html").write_text(out)
print("wrote media/level3/m2-slides.html", len(out), "bytes,", n_slides, "slides")
print(f"median CZ {MED_CZ:.5f}, RO {MED_RO:.5f}; F(0.05, 0.02) {F0502:.4f}; break-even {BREAK:.4f}; F at p=0 with 1% CX {F_SMALL:.4f}; "
      f"zeros {zeros}, events {events}, raw {raw / (3 * SHOTS):.4f}, vote {vote / SHOTS:.4f}, one-round-only {ONE_ROUND_ONLY}")

exp = (ROOT / "tools/l3_m2_explore_src.html").read_text()
assert "`" not in exp and "${" not in exp and "—" not in exp
(OUT / "m2-explore.html").write_text(exp)
print("wrote media/level3/m2-explore.html", len(exp))
