"""Build the Level 3 Module 5 slides and explorer: media/level3/m5-slides.html and media/level3/m5-explore.html
(two tabs: error budget across platforms; one chip, many qubits). Every number on the slides is computed here from
content/level3/l3_m5_data.json with the lab's reference functions, and the values the lab prints are asserted."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media" / "level3"
sys.path[:0] = [str(ROOT / "content" / "level3")]
import l3_m5_check as C  # noqa: E402

MAROON, GREY, INK, GOLD, MUTED, LINE = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#E4DCE0"
FAM = {"superconducting": MAROON, "trapped": INK, "neutral": "#C08A00", "silicon": "#2E7D32", "photons": "#6A5ACD"}

DATA = json.loads((ROOT / "content/level3/l3_m5_data.json").read_text())
PL = {p["name"]: p for p in DATA["platforms"]}
BOS = DATA["devices"]["ibm_boston"]


def fam(p):
    return next(v for k, v in FAM.items() if p["platform"].startswith(k))


# ---------------------------------------------------------------- the lab's numbers
def succ(name, n1, n2, nm):
    p = PL[name]
    return C.reference_circuit_success(p["e1"], p["e2"], p["e_ro"], n1, n2, nm)


GHZ = {n: succ(n, 1, 19, 20) for n in ("IBM Heron r3 (ibm_boston)", "IBM Nighthawk r1 (ibm_miami)", "Google Willow", "Quantinuum Helios", "IonQ Aria 1")}
LAY = {n: succ(n, 1000, 500, 50) for n in ("IBM Heron r3 (ibm_boston)", "IBM Nighthawk r1 (ibm_miami)", "Google Willow", "Quantinuum Helios")}
assert [round(v, 3) for v in GHZ.values()] == [0.881, 0.659, 0.804, 0.976, 0.857], GHZ
assert [round(v, 3) for v in LAY.values()] == [0.355, 0.077, 0.092, 0.641], LAY
assert C.reference_max_two_qubit_gates(0.0033) == 209 and C.reference_max_two_qubit_gates(0.00079) == 877
assert C.personal_value(500, 100, 20) == 739
E2 = C.usable(BOS["e2"])
S_E2 = C.reference_summarize(E2, False)
S_T1 = C.reference_summarize(BOS["qubit_data"]["t1_us"], True)
assert (round(S_E2["median"], 6), round(S_T1["median"], 1)) == (0.001271, 292.4)
B = PL["IBM Heron r3 (ibm_boston)"]
A = PL["IonQ Aria 1"]
R_B, R_A = B["t2_us"] / B["t2q_us"], A["t2_us"] / A["t2q_us"]
assert (round(R_B), round(R_A)) == (5191, 1667)
assert round(0.998 ** 300, 2) == 0.55 and round(0.996 ** 300, 2) == 0.30 and round(0.999 ** 300, 2) == 0.74


# ---------------------------------------------------------------- figures
def svg(w, h, label, parts):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{label}">' + "".join(parts) + "</svg>"


def compare_fig(w=600, h=380, labels=True):
    x0, x1, yb, yt = (70, w - 32, h - 50, 20)
    lx0, lx1, ly0, ly1 = 0, 4, -4, -1.5
    X = lambda q: x0 + (x1 - x0) * (math.log10(q) - lx0) / (lx1 - lx0)  # noqa: E731
    Y = lambda e: yb - (yb - yt) * (math.log10(e) - ly0) / (ly1 - ly0)  # noqa: E731
    p = []
    for e, lab in ((1e-4, "0.0001"), (1e-3, "0.001"), (1e-2, "0.01")):
        p.append(f'<line x1="{x0}" y1="{Y(e):.1f}" x2="{x1}" y2="{Y(e):.1f}" stroke="{LINE}"/>'
                 f'<text x="{x0 - 8}" y="{Y(e) + 5:.1f}" font-size="14" text-anchor="end">{lab}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    for q in (1, 10, 100, 1000, 10000):
        p.append(f'<text x="{X(q):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle">{q:,}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 8}" font-size="15" text-anchor="middle" fill="{MUTED}">qubits (log scale)</text>')
    p.append(f'<text x="16" y="{(yb + yt) / 2}" font-size="15" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 16 {(yb + yt) / 2})">two-qubit error</text>')
    short = {"IBM Heron r3 (ibm_boston)": "ibm_boston", "IBM Nighthawk r1 (ibm_miami)": "ibm_miami", "Google Willow": "Willow",
             "Quantinuum Helios": "Helios", "IonQ Aria 1": "IonQ Aria", "Harvard / MIT / QuEra array": "Harvard/QuEra atoms",
             "Diraq / imec silicon unit cells": "silicon (bound)"}
    off = {"ibm_boston": (8, 16), "ibm_miami": (8, -8), "Willow": (-10, 18), "Helios": (8, 5), "IonQ Aria": (-8, -8),
           "Harvard/QuEra atoms": (8, -10), "silicon (bound)": (10, 5)}
    for name, pr in PL.items():
        if not (pr["qubits"] and pr["e2"]):
            continue
        c = fam(pr)
        p.append(f'<circle cx="{X(pr["qubits"]):.1f}" cy="{Y(pr["e2"]):.1f}" r="7" fill="{c}"/>')
        if labels:
            s = short[name]
            dx, dy = off[s]
            p.append(f'<text x="{X(pr["qubits"]) + dx:.1f}" y="{Y(pr["e2"]) + dy:.1f}" font-size="14" '
                     f'text-anchor="{"end" if dx < 0 else "start"}" fill="{c}">{s}</text>')
    if labels:
        items = [("superconducting", MAROON), ("trapped ions", INK), ("neutral atoms", FAM["neutral"]), ("silicon spins", FAM["silicon"])]
        for i, (lab, col) in enumerate(items):
            lx, ly = x1 - 150, Y(10 ** -3.2)
            p.append(f'<circle cx="{lx}" cy="{ly + 20 * i:.1f}" r="6" fill="{col}"/><text x="{lx + 12}" y="{ly + 5 + 20 * i:.1f}" font-size="14">{lab}</text>')
    return svg(w, h, "Published two-qubit errors against qubit count for the platforms", p)


def budget_fig(w=600, h=320):
    x0, x1 = 150, w - 60
    rows = [("ibm_boston", "IBM Heron r3 (ibm_boston)"), ("ibm_miami", "IBM Nighthawk r1 (ibm_miami)"), ("Willow", "Google Willow"),
            ("Helios", "Quantinuum Helios"), ("IonQ Aria", "IonQ Aria 1")]
    p, top, bh = [], 34, 22
    p.append(f'<rect x="{x0}" y="6" width="14" height="12" fill="{GREY}"/><text x="{x0 + 20}" y="17" font-size="14">GHZ, 20 qubits</text>'
             f'<rect x="{x0 + 160}" y="6" width="14" height="12" fill="{MAROON}"/><text x="{x0 + 180}" y="17" font-size="14">20 layers, 50 qubits</text>')
    for i, (lab, name) in enumerate(rows):
        y = top + i * 54
        p.append(f'<text x="{x0 - 10}" y="{y + 26}" font-size="15" text-anchor="end">{lab}</text>')
        g = GHZ[name]
        p.append(f'<rect x="{x0}" y="{y}" width="{(x1 - x0) * g:.1f}" height="{bh}" fill="{GREY}"/>'
                 f'<text x="{x0 + (x1 - x0) * g + 6:.1f}" y="{y + 16}" font-size="14">{g:.3f}</text>')
        if name in LAY:
            v = LAY[name]
            p.append(f'<rect x="{x0}" y="{y + bh + 2}" width="{(x1 - x0) * v:.1f}" height="{bh}" fill="{MAROON}"/>'
                     f'<text x="{x0 + (x1 - x0) * v + 6:.1f}" y="{y + bh + 18}" font-size="14">{v:.3f}</text>')
        else:
            p.append(f'<text x="{x0 + 6}" y="{y + bh + 18}" font-size="13" fill="{MUTED}">only 25 qubits</text>')
    p.append(f'<line x1="{x0}" y1="{top - 4}" x2="{x0}" y2="{top + 5 * 54 - 6}" stroke="{INK}" stroke-width="2"/>')
    return svg(w, h, "Probability of an error-free run for two circuits on five devices", p)


def spread_fig(w=600, h=330):
    x0, x1, yb, yt = 60, w - 20, h - 50, 30
    lo, hi, nb = -3.5, -0.5, 30
    lv = np.log10(E2)
    counts, _ = np.histogram(lv, bins=nb, range=(lo, hi))
    cmax = counts.max()
    X = lambda u: x0 + (x1 - x0) * (u - lo) / (hi - lo)  # noqa: E731
    Y = lambda n: yb - (yb - yt) * n / cmax  # noqa: E731
    p = [f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>']
    for k, n in enumerate(counts):
        if n:
            p.append(f'<rect x="{X(lo + (hi - lo) * k / nb) + 1:.1f}" y="{Y(n):.1f}" width="{(x1 - x0) / nb - 2:.1f}" height="{yb - Y(n):.1f}" fill="{GREY}"/>')
    for t in (-3, -2, -1):
        p.append(f'<text x="{X(t):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle">{10.0 ** t:g}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 8}" font-size="15" text-anchor="middle" fill="{MUTED}">CZ error of each pair (log scale)</text>')
    p.append(f'<text x="16" y="{(yb + yt) / 2}" font-size="15" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 16 {(yb + yt) / 2})">pairs</text>')
    for v, col, dash, lab, i in ((S_E2["median"], INK, "6 4", "median", 0), (S_E2["best"], GOLD, "", "best", 1), (S_E2["worst"], MAROON, "", "worst", 2)):
        x = X(math.log10(v))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{x:.1f}" y1="{yt - 6}" x2="{x:.1f}" y2="{yb}" stroke="{col}" stroke-width="3"{d}/>'
                 f'<text x="{x - 6 if i else x + 4:.1f}" y="{yt + 8 + 16 * i}" font-size="14" fill="{col}" text-anchor="{"end" if i else "start"}">{lab}</text>')
    return svg(w, h, "Histogram of the CZ errors of the pairs of ibm_boston", p)


def speed_fig(w=600, h=300):
    rows = [("ibm_boston", B), ("ibm_miami", PL["IBM Nighthawk r1 (ibm_miami)"]), ("IonQ Aria", A)]
    x0, x1 = 120, w - 70
    rmax = 6000
    p = [f'<text x="{x0}" y="18" font-size="15" fill="{MUTED}">two-qubit gate times within T2</text>']
    for i, (lab, r) in enumerate(rows):
        y = 34 + i * 48
        v = r["t2_us"] / r["t2q_us"]
        p.append(f'<text x="{x0 - 10}" y="{y + 18}" font-size="15" text-anchor="end">{lab}</text>'
                 f'<rect x="{x0}" y="{y}" width="{(x1 - x0) * v / rmax:.1f}" height="26" fill="{MAROON if lab != "IonQ Aria" else INK}"/>'
                 f'<text x="{x0 + (x1 - x0) * v / rmax + 6:.1f}" y="{y + 18}" font-size="14">{v:,.0f}</text>')
    p.append(f'<text x="{x0}" y="198" font-size="15" fill="{MUTED}">time for 1,000 layers of two-qubit gates</text>')
    for i, (lab, r) in enumerate(rows):
        y = 210 + i * 28
        t = 1000 * r["t2q_us"]
        txt = f"{t:,.0f} µs" if t < 1000 else f"{t / 1e6:.1f} s"
        p.append(f'<text x="{x0 - 10}" y="{y + 14}" font-size="15" text-anchor="end">{lab}</text><text x="{x0}" y="{y + 14}" font-size="15">{txt}</text>')
    return svg(w, h, "Gate times within the coherence time, and time for 1,000 layers", p)


# ---------------------------------------------------------------- the deck
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
TITLE = "Module 5 Slides: Quantum Hardware and Careers"
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>", f"<title>{TITLE}</title>")
assert TITLE in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                    "  .slide table td, .slide table th { padding:5px 12px; text-align:left; }\n  svg text { font-family:")
rep = {"/*TITLE_FIG*/": compare_fig(430, 300, labels=False), "/*COMPARE_FIG*/": compare_fig(), "/*BUDGET_FIG*/": budget_fig(),
       "/*SPREAD_FIG*/": spread_fig(), "/*SPEED_FIG*/": speed_fig(),
       "/*BOSTON_T2Q*/": f"{B['t2q_us'] * 1000:.0f}", "/*BOSTON_E2*/": f"{S_E2['median']:.4f}", "/*BOSTON_T2*/": f"{B['t2_us']:.0f}",
       "/*HELIOS_L*/": f"{LAY['Quantinuum Helios']:.2f}", "/*BOSTON_L*/": f"{LAY['IBM Heron r3 (ibm_boston)']:.2f}",
       "/*WILLOW_L*/": f"{LAY['Google Willow']:.2f}", "/*BOSTON_NPAIR*/": str(len(E2)), "/*BOSTON_E2B*/": f"{S_E2['best']:.5f}",
       "/*BOSTON_E2W*/": f"{S_E2['worst']:.2f}", "/*BOSTON_T1W*/": f"{S_T1['worst']:.1f}", "/*BOSTON_T1B*/": f"{S_T1['best']:.0f}",
       "/*BOSTON_T1*/": f"{S_T1['median']:.0f}", "/*ARIA_R*/": f"{R_A:,.0f}", "/*BOSTON_R*/": f"{R_B:,.0f}",
       "/*BOSTON_1000*/": f"{1000 * B['t2q_us']:.0f}"}
body = (ROOT / "tools/l3_m5_slides_body.html").read_text()
for k, v in rep.items():
    assert k in body, k
    body = body.replace(k, v)
n_slides = body.count('<section class="slide')
bar = "\n".join(m1[365:373]).replace("1 / 19", f"1 / {n_slides}")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m5-slides.html").write_text(out)
print("wrote media/level3/m5-slides.html", len(out), "bytes,", n_slides, "slides")

src = (ROOT / "tools/l3_m5_explore_src.html").read_text()
exp = src.replace("/*DATA*/", json.dumps(DATA, separators=(",", ":")))
assert "`" not in exp and "${" not in exp and "—" not in exp and "/*DATA*/" not in exp
(OUT / "m5-explore.html").write_text(exp)
print("wrote media/level3/m5-explore.html", len(exp))
print("GHZ", {k: round(v, 3) for k, v in GHZ.items()}, "LAY", {k: round(v, 3) for k, v in LAY.items()})
print("boston E2", S_E2, "T1", S_T1, "ratios", R_B, R_A)
