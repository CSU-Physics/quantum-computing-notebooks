"""Assemble media/level2/m6-slides.html and m6-explore.html (Quantum Computing Intermediate, Module 6: the mini-project)
from the Level 1 deck frame (media/level1/m1-slides.html), tools/l2_m6_slides_body.html and tools/l2_m6_explore_src.html.

Every number is computed here with the exact noisy simulator of the project check cells (checks/l2_m6_common.py), which
the tests compare with qsim and with Qiskit Aer; the one sampled run and the transpiler numbers come from qsim and from
the course's saved runs, the same way the notebooks get them."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.modules["qiskit"] = None                       # use qsim, as the browser notebooks do
sys.path[:0] = [str(ROOT / "content/level1"), str(ROOT / "checks")]
import qsim  # noqa: E402
import l2_m6_grover_check as G  # noqa: E402
import l2_m6_qpe_check as Q  # noqa: E402
import l2_m6_qaoa_check as A  # noqa: E402

OUT = ROOT / "media/level2"
MAROON, GREY, INK, GOLD, MUTED, OK = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#2E7D4F"
S = 4000

# ---------------------------------------------------------------- the numbers
g_ideal = G.expected_success("101", 2, 0, 0)
g_exp = G.expected_success("101", 2, 0.01, 0.02)
g1 = G.expected_success("101", 1, 0.01, 0.02)
dist = G.exact_probabilities(G.reference_grover_ops("101", 2), 3, [0, 1, 2], 0.01, 0.02)


def course_noise_model(p2, ro):
    nm = qsim.NoiseModel()
    nm.add_all_qubit_quantum_error(qsim.depolarizing_error(p2 / 10, 1), G.ONE_QUBIT_GATES)
    nm.add_all_qubit_quantum_error(qsim.depolarizing_error(p2, 2), G.TWO_QUBIT_GATES)
    nm.add_all_qubit_readout_error(qsim.ReadoutError([[1 - ro, ro], [2 * ro, 1 - 2 * ro]]))
    return nm


run = qsim.AerSimulator(noise_model=course_noise_model(0.01, 0.02)).run(G.reference_grover("101", 2), shots=S, seed_simulator=1).result().get_counts()
f_run = run.get("101", 0) / S
sig = math.sqrt(g_exp * (1 - g_exp) / S)
gsd = math.sqrt(g1 * (1 - g1) / S + g_exp * (1 - g_exp) / S)

P2_GRID = [round(0.001 * i, 3) for i in range(31)]
grover_curves = {it: [G.expected_success("101", it, p, 0.02) for p in P2_GRID] for it in range(6)}
cross = next(p for p, a, b in zip(P2_GRID, grover_curves[1], grover_curves[2]) if a > b)
x1, x2 = grover_curves[1][15], grover_curves[2][15]
xs = 9 * (x1 * (1 - x1) + x2 * (1 - x2)) / (x2 - x1) ** 2

QPE_MS = [3, 4, 5, 6, 7]
qpe_err = {m: [Q.expected_error(m, p, 0.02) for p in P2_GRID] for m in QPE_MS}
qpe_ideal = {m: Q.expected_error(m, 0, 0) for m in QPE_MS}

P2_WIDE = [round(0.001 * i, 3) for i in range(51)]
qaoa_cut = {L: [A.expected_cut(L, p, 0.02) for p in P2_WIDE] for L in (1, 2, 3)}
qaoa_pmax = {L: [A.expected_max_cut_probability(L, p, 0.02) for p in P2_WIDE] for L in (1, 2, 3)}
qcross = next(p for p, a, b in zip(P2_WIDE, qaoa_cut[2], qaoa_cut[3]) if a > b)
qa2 = A.expected_cut(2, 0.01, 0.02)
q_sem = A.cut_sigma(2, 0.01, 0.02)

saved = json.loads((ROOT / "content/level2/m6_transpiler_runs.json").read_text())
lvl0 = [r["success"] / S for r in saved["runs"] if r["n"] == 4 and r["level"] == 0]
ts0 = float(np.std(lvl0, ddof=1))
tsh = math.sqrt(np.mean(lvl0) * (1 - np.mean(lvl0)) / S)

assert round(g_exp, 4) == 0.687 and round(g_ideal, 4) == 0.9453 and cross == 0.018 and qcross == 0.032
assert round(ts0, 4) == 0.011 and abs(xs - 14365) < 1


# ---------------------------------------------------------------- drawings
def bars_svg(w=560, h=330, title=False):
    x0, x1, yb, yt = 50, w - 15, h - 50, 30
    labels = [format(i, "03b") for i in range(8)]
    ideal = G.exact_probabilities(G.reference_grover_ops("101", 2), 3, [0, 1, 2], 0, 0)
    bw = (x1 - x0) / 8
    Y = lambda v: yb - (yb - yt) * v
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Grover for 101 with 2 iterations: '
         + ('ideal and noisy probabilities' if title else 'the noise model\'s expected probabilities and one run of 4,000 shots') + '">']
    for v in (0.25, 0.5, 0.75, 1.0):
        p.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="#E4DCE0"/>'
                 f'<text x="{x0 - 6}" y="{Y(v) + 5:.1f}" font-size="14" text-anchor="end" fill="{INK}">{v:.2f}</text>')
    for i, s in enumerate(labels):
        k = int(s, 2)
        a = ideal[k] if title else dist[k]
        b = dist[k] if title else run.get(s, 0) / S
        xa = x0 + i * bw + bw * 0.12
        p.append(f'<rect x="{xa:.1f}" y="{Y(a):.1f}" width="{bw * 0.36:.1f}" height="{yb - Y(a):.1f}" fill="{GREY if title else MAROON}"/>')
        p.append(f'<rect x="{xa + bw * 0.38:.1f}" y="{Y(b):.1f}" width="{bw * 0.36:.1f}" height="{yb - Y(b):.1f}" fill="{MAROON if title else GOLD}"/>')
        p.append(f'<text x="{x0 + (i + 0.5) * bw:.1f}" y="{yb + 20}" font-size="14" text-anchor="middle" fill="{INK}">{s}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    la, lb = ("ideal", "with noise") if title else ("noise model, expected", "4,000 shots")
    ca, cb = (GREY, MAROON) if title else (MAROON, GOLD)
    p.append(f'<rect x="{x0 + 10}" y="6" width="14" height="14" fill="{ca}"/><text x="{x0 + 30}" y="18" font-size="15">{la}</text>')
    p.append(f'<rect x="{x0 + 220}" y="6" width="14" height="14" fill="{cb}"/><text x="{x0 + 240}" y="18" font-size="15">{lb}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 12}" font-size="15" text-anchor="middle" fill="{MUTED}">result (marked: 101)</text>')
    p.append('</svg>')
    return "".join(p)


def cross_svg(w=620, h=380):
    x0, x1, yb, yt = 60, w - 20, h - 60, 20
    lo, hi = 0.40, 0.90
    X = lambda p: x0 + (x1 - x0) * p / 0.030
    Y = lambda v: yb - (yb - yt) * (v - lo) / (hi - lo)
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Expected success of Grover with 1 and 2 iterations against P2; the curves cross at P2 = {cross}">']
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        p.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="#E4DCE0"/>'
                 f'<text x="{x0 - 8}" y="{Y(v) + 5:.1f}" font-size="14" text-anchor="end">{v:.1f}</text>')
    for p2 in (0, 0.01, 0.02, 0.03):
        p.append(f'<text x="{X(p2):.1f}" y="{yb + 22}" font-size="14" text-anchor="middle">{p2:.2f}</text>')
    for it, col, lab in ((1, GREY, "1 iteration"), (2, MAROON, "2 iterations")):
        pts = " ".join(f"{X(q):.1f},{Y(v):.1f}" for q, v in zip(P2_GRID, grover_curves[it]))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="4"/>')
        yl = Y(grover_curves[it][-1])
        p.append(f'<text x="{X(0.030) - 4:.1f}" y="{yl + (-20 if it == 1 else 22):.1f}" font-size="16" text-anchor="end" fill="{col if col != GREY else INK}">{lab}</text>')
    p.append(f'<line x1="{X(cross):.1f}" y1="{yt}" x2="{X(cross):.1f}" y2="{yb}" stroke="{INK}" stroke-dasharray="6 5"/>')
    p.append(f'<text x="{X(cross) + 6:.1f}" y="{yt + 16}" font-size="15">P2 = {cross}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 14}" font-size="15" text-anchor="middle" fill="{MUTED}">P2, two-qubit depolarizing error</text>')
    p.append('</svg>')
    return "".join(p)


SIGMA_TABLE = ('<table style="font-size:22px"><tr><th>shots S</th>' + "".join(f"<th>{s:,}</th>" for s in (100, 1000, 4000, 16000))
               + "</tr><tr><td>σ for p = 0.687</td>" + "".join(f"<td>{math.sqrt(0.687 * 0.313 / s):.4f}</td>" for s in (100, 1000, 4000, 16000))
               + "</tr></table>")
rows = []
for p2 in (0, 0.01, 0.02, 0.03):
    i = P2_GRID.index(p2)
    errs = {m: (qpe_err[m][i] if p2 else qpe_ideal[m]) for m in QPE_MS}
    best = min(errs, key=errs.get)
    rows.append(f"<tr><td>{p2:.2f}</td>" + "".join(
        f'<td{" style=\"font-weight:700;color:#99004C\"" if m == best else ""}>{errs[m]:.3f}</td>' for m in QPE_MS) + "</tr>")
QPE_TABLE = ('<table style="font-size:19px"><tr><th>P2</th>' + "".join(f"<th>m={m}</th>" for m in QPE_MS) + "</tr>" + "".join(rows) + "</table>")

m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 6 Slides: Mini-Project, an Experiment with Noise</title>")
assert "Module 6" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n  svg text { font-family:")
body = (ROOT / "tools/l2_m6_slides_body.html").read_text()
rep = {"/*FIG_TITLE*/": bars_svg(420, 300, title=True), "/*FIG_BARS*/": bars_svg(), "/*FIG_CROSS*/": cross_svg(),
       "/*SIGMA_TABLE*/": SIGMA_TABLE, "/*QPE_TABLE*/": QPE_TABLE,
       "/*G_EXP*/": f"{g_exp:.4f}", "/*G_IDEAL*/": f"{g_ideal:.4f}", "/*G_RUN*/": f"{f_run:.4f}",
       "/*Z_MODEL*/": f"{(f_run - g_exp) / sig:+.1f}", "/*Z_IDEAL*/": f"{(f_run - g_ideal) / sig:+.0f}",
       "/*G1*/": f"{g1:.4f}", "/*G2*/": f"{g_exp:.4f}", "/*GD*/": f"{g_exp - g1:.4f}", "/*GSD*/": f"{gsd:.4f}",
       "/*GZ*/": f"{(g_exp - g1) / gsd:.0f}", "/*X1*/": f"{x1:.4f}", "/*X2*/": f"{x2:.4f}", "/*XD*/": f"{x2 - x1:.4f}",
       "/*XS*/": f"{round(xs, -3):,.0f}", "/*CROSS*/": f"{cross:.3f}", "/*QA2*/": f"{qa2:.3f}",
       "/*QS*/": f"{q_sem * math.sqrt(S):.2f}", "/*QSEM*/": f"{q_sem:.3f}", "/*TS0*/": f"{ts0:.3f}", "/*TSH*/": f"{tsh:.3f}",
       "/*QG3*/": f"{A.expected_cut(3, 0, 0) - A.expected_cut(2, 0, 0):.3f}",
       "/*QG3N*/": f"{A.expected_cut(3, 0.01, 0.02) - qa2:.3f}", "/*QCROSS*/": f"{qcross:.3f}", "/*QSIM*/": qsim.__version__}
for k_, v in rep.items():
    assert k_ in body, k_
    body = body.replace(k_, v)
bar = "\n".join(m1[365:373]).replace("1 / 19", "1 / 18")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m6-slides.html").write_text(out)
print("wrote media/level2/m6-slides.html", len(out))
print(f"Grover: ideal {g_ideal:.4f}, expected {g_exp:.4f}, run {f_run:.4f}, 1 it {g1:.4f}, crossover {cross}, shots {xs:.0f}; "
      f"QAOA crossover {qcross}, SEM {q_sem:.4f}; transpiler level-0 spread {ts0:.4f}, shot {tsh:.4f}")

# ---------------------------------------------------------------- explorer data (exact values, no sampling)
data = {"p2": P2_GRID, "p2wide": P2_WIDE,
        "grover": {str(it): [round(v, 5) for v in grover_curves[it]] for it in range(6)},
        "qpe": {str(m): [round(v, 5) for v in qpe_err[m]] for m in QPE_MS},
        "qaoa": {str(L): [round(v, 5) for v in qaoa_cut[L]] for L in (1, 2, 3)}}
src = ROOT / "tools/l2_m6_explore_src.html"
if src.exists():
    exp = src.read_text().replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    assert "`" not in exp and "${" not in exp and "—" not in exp and "/*DATA*/" not in exp
    (OUT / "m6-explore.html").write_text(exp)
    print("wrote media/level2/m6-explore.html", len(exp))
