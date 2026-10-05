"""Assemble media/level2/m5-slides.html and m5-explore.html (Quantum Computing Intermediate, Module 5: variational
algorithms, VQE and QAOA) from the Level 1 deck frame (media/level1/m1-slides.html), tools/l2_m5_slides_body.html and
tools/l2_m5_explore_src.html. Every number on the slides is computed here with qsim (content/level1/qsim.py) and the
reference solutions of checks/l2_m5_check.py, the same way the lab computes them."""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.modules["qiskit"] = None                       # use qsim, as the browser lab does
sys.path[:0] = [str(ROOT / "content/level1"), str(ROOT / "checks")]
import qsim  # noqa: E402
from qsim import SparsePauliOp, StatevectorEstimator, Statevector, QuantumCircuit  # noqa: E402
import l2_m5_check as ref  # noqa: E402

OUT = ROOT / "media/level2"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, GREY, INK, GOLD, MUTED, OK, PURPLE = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#2E7D4F", "#6E2C4F"

# ---------------------------------------------------------------- the numbers
H2 = SparsePauliOp.from_list([("II", -1.052373245772859), ("IZ", 0.39793742484318045), ("ZI", -0.39793742484318045),
                              ("ZZ", -0.01128010425623538), ("XX", 0.18093119978423156)])
NUC = 0.7199689944
E_EXACT = float(np.linalg.eigvalsh(H2.to_matrix())[0])
est = StatevectorEstimator()
hf = QuantumCircuit(2); hf.x(0)
labels = [lab for lab, _ in H2.to_list()]
terms = est.run([(hf, [SparsePauliOp(l) for l in labels])]).result()[0].data.evs
E_HF = float(est.run([(hf, H2)]).result()[0].data.evs)


def vqe(layers):
    qc = ref.reference_ansatz(layers)
    hist = []

    def cost(p):
        e = float(est.run([(qc, H2, p)]).result()[0].data.evs)
        hist.append(e)
        return e
    r = minimize(cost, np.zeros(qc.num_parameters), method="COBYLA", options={"maxiter": 300})
    return r, hist


r1, h1 = vqe(1)
r2, h2 = vqe(2)
EDGES, N = ref.COURSE_EDGES, 4
HC = ref.reference_maxcut_hamiltonian(EDGES, N)


def exp_cut(x):
    p = len(x) // 2
    qc = ref.reference_qaoa_circuit(list(x[:p]), list(x[p:]), EDGES, N)
    return (len(EDGES) - float(est.run([(qc, HC)]).result()[0].data.evs)) / 2


cost = lambda x: -exp_cut(x)  # noqa: E731
ra = minimize(cost, [0.5, 0.5], method="COBYLA", options={"maxiter": 300})
gammas, betas = np.linspace(0, np.pi, 33), np.linspace(0, np.pi / 2, 17)
grid = np.array([[exp_cut([g, b]) for g in gammas] for b in betas])
k = np.unravel_index(np.argmax(np.round(grid, 9)), grid.shape)
rc = minimize(cost, [gammas[k[1]], betas[k[0]]], method="COBYLA", options={"maxiter": 300})
rd = minimize(cost, [0.5] * 4, method="COBYLA", options={"maxiter": 500})


def p_opt(x):
    p = len(x) // 2
    pr = Statevector(ref.reference_qaoa_circuit(list(x[:p]), list(x[p:]), EDGES, N)).probabilities_dict()
    return pr.get("0101", 0) + pr.get("1010", 0)


idx = np.arange(16)
cuts = sum(((idx >> i) & 1) ^ ((idx >> j) & 1) for i, j in EDGES)
MAXCUT = int(cuts.max())
assert MAXCUT == 4 and abs(cuts.mean() - 2.5) < 1e-12
assert round(r1.fun, 6) == -1.836968 and round(r2.fun, 6) == -1.857275 and round(E_EXACT + NUC, 6) == -1.137306
assert round(-ra.fun, 3) == 3.086 and round(-rc.fun, 2) == 3.24 and round(-rd.fun, 3) == 3.856


# ---------------------------------------------------------------- drawings
def graph_svg(w=380, h=330, colors=None, cut_edges=None, labels=True, aria="The course graph: a square of four nodes with one diagonal"):
    pos = {0: (80, 70), 1: (300, 70), 2: (300, 260), 3: (80, 260)}
    sx, sy = w / 380, h / 330
    P = lambda q: (pos[q][0] * sx, pos[q][1] * sy)
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{aria}">']
    for i, j in EDGES:
        (x1, y1), (x2, y2) = P(i), P(j)
        cut = cut_edges is not None and (i, j) in cut_edges
        p.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{GOLD if cut else INK}" '
                 f'stroke-width="{6 if cut else 3}"{" stroke-dasharray=\"10 6\"" if cut else ""}/>')
    for q in range(4):
        x, y = P(q)
        fill = (colors or {}).get(q, "#fff")
        txt = "#fff" if fill not in ("#fff", "#ffffff") else INK
        p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{30 * min(sx, sy):.0f}" fill="{fill}" stroke="{INK}" stroke-width="3"/>')
        if labels:
            p.append(f'<text x="{x:.0f}" y="{y + 9 * min(sx, sy):.0f}" font-size="{26 * min(sx, sy):.0f}" text-anchor="middle" fill="{txt}">{q}</text>')
    p.append('</svg>')
    return "".join(p)


CUT_COLORS = {0: MAROON, 2: MAROON, 1: GREY, 3: GREY}
CUT_EDGES = {(0, 1), (1, 2), (2, 3), (3, 0)}


def loop_svg(w=1130, h=330):
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="The hybrid loop: the quantum computer prepares the ansatz state and estimates the energy; the classical optimizer proposes new angles">',
         f'<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MAROON}"/></marker></defs>',
         '<g font-size="22" fill="#24313D">']
    p.append(f'<rect x="20" y="40" width="440" height="230" rx="14" fill="#F8F4F6" stroke="{MAROON}" stroke-width="3"/>')
    p.append(f'<text x="240" y="80" text-anchor="middle" font-size="24" fill="{MAROON}">Quantum computer</text>')
    p.append('<text x="240" y="130" text-anchor="middle">prepare |ψ(θ)⟩ with the ansatz</text>')
    p.append('<text x="240" y="170" text-anchor="middle">estimate E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩</text>')
    p.append(f'<text x="240" y="215" text-anchor="middle" font-size="18" fill="{MUTED}">the Estimator primitive</text>')
    p.append(f'<rect x="670" y="40" width="440" height="230" rx="14" fill="#fff" stroke="{INK}" stroke-width="3"/>')
    p.append(f'<text x="890" y="80" text-anchor="middle" font-size="24" fill="{INK}">Classical computer</text>')
    p.append('<text x="890" y="130" text-anchor="middle">optimizer (COBYLA)</text>')
    p.append('<text x="890" y="170" text-anchor="middle">choose new angles θ</text>')
    p.append(f'<text x="890" y="215" text-anchor="middle" font-size="18" fill="{MUTED}">stop when E stops falling</text>')
    p.append(f'<path d="M 460 110 C 540 80, 600 80, 666 110" fill="none" stroke="{MAROON}" stroke-width="4" marker-end="url(#ar)"/>')
    p.append('<text x="565" y="70" text-anchor="middle" font-size="20">E(θ)</text>')
    p.append(f'<path d="M 670 210 C 600 240, 540 240, 464 210" fill="none" stroke="{MAROON}" stroke-width="4" marker-end="url(#ar)"/>')
    p.append('<text x="565" y="265" text-anchor="middle" font-size="20">new θ</text>')
    p.append('</g></svg>')
    return "".join(p)


def box(x, y, w, h, label, fill="#fff", stroke=INK, fs=19):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + fs / 3}" text-anchor="middle" font-size="{fs}">{label}</text>')


def ansatz_svg(w=1130, h=210):
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="The ansatz with two layers: RY on each qubit, a CNOT from qubit 0 to qubit 1, RY on each qubit">',
         f'<g fill="{INK}" stroke-width="3">']
    for y, lab in ((60, "q₀ = |0⟩"), (150, "q₁ = |0⟩")):
        p.append(f'<text x="0" y="{y + 7}" font-size="21">{lab}</text><line x1="100" y1="{y}" x2="{w - 20}" y2="{y}" stroke="{INK}"/>')
    p.append(f'<rect x="130" y="15" width="300" height="185" rx="10" fill="none" stroke="{GREY}" stroke-dasharray="8 6"/>')
    p.append(f'<text x="280" y="12" text-anchor="middle" font-size="17" fill="{MUTED}">layer 1</text>')
    p.append(box(170, 35, 120, 50, "RY(θ₀)", stroke=MAROON))
    p.append(box(170, 125, 120, 50, "RY(θ₁)", stroke=MAROON))
    p.append(f'<rect x="460" y="15" width="380" height="185" rx="10" fill="none" stroke="{GREY}" stroke-dasharray="8 6"/>')
    p.append(f'<text x="650" y="12" text-anchor="middle" font-size="17" fill="{MUTED}">layer 2</text>')
    p.append(f'<circle cx="520" cy="60" r="9" fill="{INK}"/><line x1="520" y1="60" x2="520" y2="168" stroke="{INK}"/>')
    p.append(f'<circle cx="520" cy="150" r="20" fill="#fff" stroke="{INK}"/><line x1="500" y1="150" x2="540" y2="150" stroke="{INK}"/>')
    p.append(box(620, 35, 120, 50, "RY(θ₂)", stroke=MAROON))
    p.append(box(620, 125, 120, 50, "RY(θ₃)", stroke=MAROON))
    p.append(f'<text x="960" y="110" font-size="20" fill="{MUTED}">… more layers</text>')
    p.append('</g></svg>')
    return "".join(p)


def qaoa_svg(w=1130, h=300):
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="QAOA with p layers: H on every qubit, then cost and mixer steps for each layer, then measurement">',
         f'<g fill="{INK}" stroke-width="3">']
    ys = [45, 105, 165, 225]
    for q, y in enumerate(ys):
        p.append(f'<text x="0" y="{y + 7}" font-size="20">q{q}</text><line x1="40" y1="{y}" x2="{w - 10}" y2="{y}" stroke="{INK}"/>')
        p.append(box(60, y - 20, 44, 40, "H", stroke=MAROON))
        p.append(box(990, y - 20, 50, 40, "M"))
    for x0, lab in ((130, "layer 1 (γ₁, β₁)"), (560, "layer p (γₚ, βₚ)")):
        p.append(box(x0, 20, 190, 230, "", fill="#F8F4F6", stroke=MAROON))
        p.append(f'<text x="{x0 + 95}" y="130" text-anchor="middle" font-size="21" fill="{MAROON}">cost step</text>')
        p.append(f'<text x="{x0 + 95}" y="160" text-anchor="middle" font-size="15" fill="{MUTED}">rzz(2γ) on each edge</text>')
        p.append(box(x0 + 205, 20, 175, 230, "", fill="#fff", stroke=INK))
        p.append(f'<text x="{x0 + 292}" y="130" text-anchor="middle" font-size="21">mixer step</text>')
        p.append(f'<text x="{x0 + 292}" y="160" text-anchor="middle" font-size="15" fill="{MUTED}">rx(2β) on each qubit</text>')
        p.append(f'<text x="{x0 + 190}" y="285" text-anchor="middle" font-size="17" fill="{MUTED}">{lab}</text>')
    p.append(f'<text x="535" y="140" text-anchor="middle" font-size="26">…</text>')
    p.append('</g></svg>')
    return "".join(p)


def converge_svg(w=560, h=360):
    x0, x1, yb, yt = 70, w - 20, h - 60, 20
    lo, hi = E_EXACT - 0.02, E_EXACT + 0.25
    n = max(len(h1), len(h2))
    X = lambda i: x0 + (x1 - x0) * i / (n - 1)
    Y = lambda e: yb - (yb - yt) * (min(max(e, lo), hi) - lo) / (hi - lo)
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Energy against cost function evaluation for one and two layers">']
    for e in (-1.85, -1.80, -1.75, -1.70, -1.65):
        p.append(f'<line x1="{x0}" y1="{Y(e):.1f}" x2="{x1}" y2="{Y(e):.1f}" stroke="#E4DCE0"/>'
                 f'<text x="{x0 - 8}" y="{Y(e) + 5:.1f}" font-size="14" text-anchor="end" fill="{INK}">{e:.2f}</text>')
    p.append(f'<line x1="{x0}" y1="{Y(E_EXACT):.1f}" x2="{x1}" y2="{Y(E_EXACT):.1f}" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>')
    p.append(f'<text x="{x1}" y="{Y(E_EXACT) + 20:.1f}" font-size="14" text-anchor="end" fill="{INK}">exact</text>')
    for hist, col in ((h1, GREY), (h2, MAROON)):
        pts = " ".join(f"{X(i):.1f},{Y(e):.1f}" for i, e in enumerate(hist))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3"/>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    for i in range(0, n, 20):
        p.append(f'<text x="{X(i):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle" fill="{INK}">{i}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 18}" font-size="15" text-anchor="middle" fill="{MUTED}">cost function evaluation</text>')
    p.append(f'<rect x="{x0 + 260}" y="{yt + 10}" width="14" height="14" fill="{GREY}"/><text x="{x0 + 280}" y="{yt + 22}" font-size="15">1 layer</text>')
    p.append(f'<rect x="{x0 + 360}" y="{yt + 10}" width="14" height="14" fill="{MAROON}"/><text x="{x0 + 380}" y="{yt + 22}" font-size="15">2 layers</text>')
    p.append('</svg>')
    return "".join(p)


def viridis(t):
    stops = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return "#%02x%02x%02x" % tuple(int(ca[i] + f * (cb[i] - ca[i])) for i in range(3))
    return "#fde725"


def landscape_svg(w=560, h=380):
    x0, x1, yb, yt = 60, w - 20, h - 60, 15
    lo, hi = grid.min(), grid.max()
    cw, chh = (x1 - x0) / len(gammas), (yb - yt) / len(betas)
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Expected cut over gamma and beta for one QAOA layer, with the two COBYLA results marked">']
    for bi in range(len(betas)):
        for gi in range(len(gammas)):
            t = (grid[bi, gi] - lo) / (hi - lo)
            p.append(f'<rect x="{x0 + gi * cw:.1f}" y="{yb - (bi + 1) * chh:.1f}" width="{cw + 0.6:.1f}" height="{chh + 0.6:.1f}" fill="{viridis(t)}"/>')
    G = lambda g: x0 + (x1 - x0) * (g % np.pi) / np.pi
    B = lambda b: yb - (yb - yt) * (b % (np.pi / 2)) / (np.pi / 2)
    p.append(f'<text x="{G(ra.x[0]):.1f}" y="{B(ra.x[1]) + 8:.1f}" font-size="26" text-anchor="middle" fill="#fff">×</text>')
    p.append(f'<text x="{G(rc.x[0]):.1f}" y="{B(rc.x[1]) + 9:.1f}" font-size="28" text-anchor="middle" fill="#ff3b3b">★</text>')
    for g, lab in ((0, "0"), (np.pi / 2, "π/2"), (np.pi, "π")):
        p.append(f'<text x="{x0 + (x1 - x0) * g / np.pi:.1f}" y="{yb + 20}" font-size="15" text-anchor="middle">{lab}</text>')
    for b, lab in ((0, "0"), (np.pi / 4, "π/4"), (np.pi / 2, "π/2")):
        p.append(f'<text x="{x0 - 8}" y="{yb - (yb - yt) * b / (np.pi / 2) + 5:.1f}" font-size="15" text-anchor="end">{lab}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 18}" font-size="16" text-anchor="middle" fill="{MUTED}">γ (cost angle)</text>')
    p.append(f'<text x="18" y="{(yb + yt) / 2}" font-size="16" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 18 {(yb + yt) / 2})">β (mixer angle)</text>')
    p.append(f'<text x="{x1}" y="{h - 2}" font-size="13" text-anchor="end" fill="{MUTED}">dark: {lo:.2f} · bright: {hi:.2f} · ×: from (0.5, 0.5) · ★: from the grid</text>')
    p.append('</svg>')
    return "".join(p)


TD = 'style="padding:6px 12px"'
term_rows = "".join(f"<tr><td>{l}</td><td>{c.real:+.4f}</td><td>{ev:+.0f}</td></tr>" for (l, c), ev in zip(H2.to_list(), terms))
TERMS = f'<table style="font-size:21px"><tr><th>Pauli</th><th>coefficient</th><th>⟨01|P|01⟩</th></tr>{term_rows}</table>'
VQE_TABLE = ('<table style="font-size:21px"><tr><th>ansatz</th><th>energy (hartree)</th><th>error</th></tr>'
             f'<tr><td>1 layer (no CNOT)</td><td>{r1.fun:.6f}</td><td>{1000 * (r1.fun - E_EXACT):.1f} mHa</td></tr>'
             f'<tr><td>2 layers (1 CNOT)</td><td>{r2.fun:.6f}</td><td>{abs(1000 * (r2.fun - E_EXACT)):.3f} mHa</td></tr>'
             f'<tr><td>exact</td><td>{E_EXACT:.6f}</td><td></td></tr></table>'
             f'<p class="small" style="margin-top:8px">Total energy of H₂ with 2 layers: {r2.fun + NUC:.6f} hartree.</p>')

m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 5 Slides: Variational Algorithms, VQE and QAOA</title>")
assert "Module 5" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n  svg text { font-family:")
body = (ROOT / "tools/l2_m5_slides_body.html").read_text()
rep = {"/*GRAPH_TITLE*/": graph_svg(380, 330, CUT_COLORS, CUT_EDGES), "/*LOOP*/": loop_svg(), "/*ANSATZ*/": ansatz_svg(),
       "/*CONVERGE*/": converge_svg(), "/*QAOA_CIRCUIT*/": qaoa_svg(), "/*LANDSCAPE*/": landscape_svg(),
       "/*GRAPH_CUT*/": graph_svg(330, 290, CUT_COLORS, CUT_EDGES, aria="A maximum cut of the course graph: nodes 0 and 2 in one group, 1 and 3 in the other; four edges cut"),
       "/*TERMS*/": TERMS, "/*VQE_TABLE*/": VQE_TABLE, "/*E_EXACT*/": f"{E_EXACT:.6f}", "/*E_TOTAL*/": f"{E_EXACT + NUC:.6f}",
       "/*E_HF*/": f"{E_HF:.6f}", "/*ERR_HF*/": f"{1000 * (E_HF - E_EXACT):.1f}", "/*MAXCUT*/": str(MAXCUT),
       "/*QA*/": f"{-ra.fun:.3f}", "/*QC*/": f"{-rc.fun:.3f}", "/*Q2*/": f"{-rd.fun:.3f}",
       "/*P1*/": f"{100 * p_opt(rc.x):.0f}%", "/*P2*/": f"{100 * p_opt(rd.x):.0f}%"}
for k_, v in rep.items():
    assert k_ in body, k_
    body = body.replace(k_, v)
bar = "\n".join(m1[365:373]).replace("1 / 19", "1 / 18")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m5-slides.html").write_text(out)
print("wrote media/level2/m5-slides.html", len(out))
print(f"E_exact {E_EXACT:.6f}, HF {E_HF:.6f}, 1 layer {r1.fun:.6f} ({len(h1)} evals), 2 layers {r2.fun:.6f} ({len(h2)} evals); "
      f"QAOA a {-ra.fun:.4f}, grid {grid.max():.4f}, c {-rc.fun:.4f} P {p_opt(rc.x):.3f}, p2 {-rd.fun:.4f} P {p_opt(rd.x):.3f}")

src = ROOT / "tools/l2_m5_explore_src.html"
if src.exists():
    exp = src.read_text()
    assert "`" not in exp and "${" not in exp and "—" not in exp
    (OUT / "m5-explore.html").write_text(exp)
    print("wrote media/level2/m5-explore.html", len(exp))
