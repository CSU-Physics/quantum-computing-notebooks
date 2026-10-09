"""Assemble media/level3/m3-slides.html and m3-explore.html (Quantum Computing Advanced, Module 3: error suppression
and mitigation) from the Level 1 deck frame (media/level1/m1-slides.html), tools/l3_m3_slides_body.html and
tools/l3_m3_explore_src.html. Every number on the slides is computed here with qsim, the reference solutions of
checks/l3_m3_check.py and content/level3/l3_m3_device_run.json, the same way the lab computes them."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.modules["qiskit"] = None
sys.path[:0] = [str(ROOT / "content/level3"), str(ROOT / "checks")]
import qsim  # noqa: E402
import l3_m3_check as ref  # noqa: E402

OUT = ROOT / "media/level3"
MAROON, GREY, INK, GOLD, MUTED, OK = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#2E7D4F"

# ---------------------------------------------------------------- the numbers (as in the lab)
D, PRO, PCX = 0.05, 0.02, 0.02
nm = ref.noise_model(D, PRO, PCX)
P = ref.pipeline(D, PRO, PCX)
RAW, DD, DD_RO, LIN, QUAD = P["raw"], P["dd"], P["dd_ro"], P["linear"], P["quadratic"]
assert (round(RAW, 3), round(DD, 3), round(DD_RO, 3), round(LIN, 3)) == (0.409, 0.753, 0.853, 0.869)
TARGET = ref.parity_expectation(ref.reference_mitigate_readout(
    ref.exact_probabilities(ref.experiment(True), ref.noise_model(D, PRO, 0.0), PRO),
    ref.calibration_matrix(ref.noise_model(D, PRO, 0.0), PRO)))


def nm_parts(delta, p_ro, p_cx, relax=True):
    m = qsim.NoiseModel()
    idle = qsim.thermal_relaxation_error(ref.T1, ref.T2, ref.T_ID) if relax else None
    if delta:
        coh = qsim.coherent_unitary_error(ref.rz_matrix(delta))
        idle = coh if idle is None else idle.compose(coh)
    if idle is not None:
        m.add_all_qubit_quantum_error(idle, ["id"])
    m.add_all_qubit_quantum_error(qsim.depolarizing_error(ref.P1, 1), ["h", "x"])
    if p_cx:
        m.add_all_qubit_quantum_error(qsim.depolarizing_error(p_cx, 2), ["cx"])
    if p_ro:
        m.add_all_qubit_readout_error(qsim.ReadoutError(ref.readout_matrix(p_ro)))
    return m


def xx(qc, m, pro):
    return ref.parity_expectation(ref.exact_probabilities(qc, m, pro))


qc0 = ref.experiment(False)
PARTS = [("dephasing and relaxation", xx(qc0, nm_parts(0, 0, 0), 0)), ("detuning", xx(qc0, nm_parts(D, 0, 0, False), 0)),
         ("readout error", xx(qc0, nm_parts(0, PRO, 0, False), PRO)), ("CNOT error", xx(qc0, nm_parts(0, 0, PCX, False), 0)),
         ("all four", RAW)]
assert abs(PARTS[4][1] - xx(qc0, nm_parts(D, PRO, PCX), PRO)) < 1e-12
CALM = nm_parts(0, PRO, PCX)
CALM_RAW, CALM_DD = xx(ref.experiment(False), CALM, PRO), xx(ref.experiment(True), CALM, PRO)
A = ref.calibration_matrix(nm, PRO)

# Step 6 of the lab, with the same seed
rng = np.random.default_rng(2026)
SH, REP = 4000, 300
base = ref.experiment(True)
exact = {s: ref.exact_probabilities(ref.reference_fold_cx(base, s), nm, PRO) for s in ref.SCALES}
est = {"DD": [], "DD + readout": [], "DD + readout + ZNE": []}
for _ in range(REP):
    A_s = np.column_stack([rng.multinomial(SH, A[:, j] / A[:, j].sum()) / SH for j in range(4)])
    samp = {}
    for s in ref.SCALES:
        n = rng.multinomial(SH, [exact[s][k] for k in ref.KEYS])
        samp[s] = {k: n[i] / SH for i, k in enumerate(ref.KEYS)}
    est["DD"].append(ref.parity_expectation(samp[1]))
    pts = [ref.parity_expectation(ref.reference_mitigate_readout(samp[s], A_s)) for s in ref.SCALES]
    est["DD + readout"].append(pts[0])
    est["DD + readout + ZNE"].append(ref.reference_extrapolate_zero(ref.SCALES, pts, 1))
STD = {k: float(np.std(v)) for k, v in est.items()}
AMP = STD["DD + readout + ZNE"] / STD["DD"]

run = json.loads((ROOT / "content/level3/l3_m3_device_run.json").read_text())
shots = run["shots"]
pr = {n: {k: c / shots for k, c in cnt.items()} for n, cnt in run["counts"].items()}
A_dev = np.array([[pr[f"cal_{j}"][i] for j in ref.KEYS] for i in ref.KEYS])
DEV = {"raw": ref.parity_expectation(pr["raw"]), "DD": ref.parity_expectation(pr["dd"])}
dpts = [ref.parity_expectation(ref.reference_mitigate_readout(pr[n], A_dev)) for n in ("dd", "dd_fold3", "dd_fold5")]
DEV["DD + readout"] = dpts[0]
DEV["DD + readout + ZNE"] = ref.reference_extrapolate_zero(ref.SCALES, dpts, 1)
t2 = [run["qubit_properties"][str(q)]["t2_us"] for q in run["physical_qubits"]]
DEPH = float(np.exp(-run["idle_us"] / t2[0] - run["idle_us"] / t2[1]))
assert round(DEV["DD + readout"], 3) == 0.900


# ---------------------------------------------------------------- drawings
def axes(p, w, h, x0, x1, yb, yt, xticks, X, yticks, Y, xlab, ylab):
    for v, lab in yticks:
        p.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="#E4DCE0"/>'
                 f'<text x="{x0 - 8}" y="{Y(v) + 5:.1f}" font-size="14" text-anchor="end">{lab}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    for v in xticks:
        p.append(f'<text x="{X(v):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle">{v:g}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 8}" font-size="15" text-anchor="middle" fill="{MUTED}">{xlab}</text>')
    if ylab:
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


def hbars(rows, w=600, h=330, top=1.0, fmt="{:.3f}", label_w=230, cols=None):
    x0, x1 = label_w, w - 70
    X = lambda v: x0 + (x1 - x0) * v / top  # noqa: E731
    bh, gap = 34, 18
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Bar chart">']
    for i, (lab, v) in enumerate(rows):
        y = 20 + i * (bh + gap)
        col = (cols or [GREY] * len(rows))[i]
        p.append(f'<text x="{x0 - 10}" y="{y + bh / 2 + 6}" font-size="17" text-anchor="end">{lab}</text>'
                 f'<rect x="{x0}" y="{y}" width="{X(v) - x0:.1f}" height="{bh}" fill="{col}"/>'
                 f'<text x="{X(v) + 8:.1f}" y="{y + bh / 2 + 6}" font-size="16">{fmt.format(v)}</text>')
    yb = 20 + len(rows) * (bh + gap) - gap + 8
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    return p, X, yb


def noise_bars():
    p, X, yb = hbars([(n, v) for n, v in PARTS], cols=[GREY, MAROON, GREY, GREY, INK], h=300)
    p.append(f'<line x1="{X(1)}" y1="10" x2="{X(1)}" y2="{yb}" stroke="{GOLD}" stroke-width="2" stroke-dasharray="6 4"/>'
             f'<text x="{X(1) - 4}" y="{yb + 20}" font-size="14" text-anchor="end" fill="{MUTED}">ideal ⟨XX⟩ = 1</text>')
    p.append('</svg>')
    return "".join(p).replace('aria-label="Bar chart"', 'aria-label="XX expectation with each kind of noise alone and all together"')


def std_bars():
    p, X, yb = hbars([(k, v) for k, v in STD.items()], top=0.02, fmt="{:.4f}", cols=[GREY, MAROON, INK], h=220)
    p.append(f'<text x="{(230 + 530) / 2}" y="{yb + 24}" font-size="15" text-anchor="middle" fill="{MUTED}">standard deviation of ⟨XX⟩, 4,000 shots per circuit</text>')
    p.append('</svg>')
    return "".join(p).replace('aria-label="Bar chart"', 'aria-label="Standard deviation of three estimates of XX"')


def device_bars():
    p, X, yb = hbars([(k, v) for k, v in DEV.items()], w=580, cols=[GREY, GREY, MAROON, INK], h=260, label_w=220)
    se = 1 / np.sqrt(shots)
    for i, (k, v) in enumerate(DEV.items()):
        y = 20 + i * 52 + 17
        p.append(f'<line x1="{X(v - se):.1f}" y1="{y}" x2="{X(v + se):.1f}" y2="{y}" stroke="{GOLD}" stroke-width="3"/>')
    p.append(f'<line x1="{X(DEPH):.1f}" y1="10" x2="{X(DEPH):.1f}" y2="{yb}" stroke="{OK}" stroke-width="2" stroke-dasharray="6 4"/>'
             f'<text x="{X(DEPH) - 4:.1f}" y="{yb + 20}" font-size="14" text-anchor="end" fill="{OK}">dephasing limit {DEPH:.3f}</text>')
    p.append('</svg>')
    return "".join(p).replace('aria-label="Bar chart"', 'aria-label="XX on FakePittsburgh: raw, with DD, with readout mitigation, with ZNE"')


def gate(x, y, lab, col=INK, w=46):
    return (f'<rect x="{x - w / 2}" y="{y - 20}" width="{w}" height="40" rx="5" fill="#fff" stroke="{col}" stroke-width="2.5"/>'
            f'<text x="{x}" y="{y + 7}" font-size="18" text-anchor="middle" fill="{col}">{lab}</text>')


def cnot(x, y0, y1):
    return (f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1 + 16}" stroke="{INK}" stroke-width="2.5"/><circle cx="{x}" cy="{y0}" r="7" fill="{INK}"/>'
            f'<circle cx="{x}" cy="{y1}" r="16" fill="#fff" stroke="{INK}" stroke-width="2.5"/>'
            f'<line x1="{x - 16}" y1="{y1}" x2="{x + 16}" y2="{y1}" stroke="{INK}" stroke-width="2.5"/><line x1="{x}" y1="{y1 - 16}" x2="{x}" y2="{y1 + 16}" stroke="{INK}" stroke-width="2.5"/>')


def circuit_svg(w=1100, h=170, dd=False):
    y0, y1 = 45, 120
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="The experiment: H, CNOT, a 10 microsecond wait, H on both qubits and measurement">']
    for y, n in ((y0, "qubit 0"), (y1, "qubit 1")):
        p.append(f'<text x="0" y="{y + 6}" font-size="17">{n}</text><line x1="85" y1="{y}" x2="{w - 10}" y2="{y}" stroke="{INK}" stroke-width="2.5"/>')
    p.append(gate(130, y0, "H") + cnot(200, y0, y1))
    p.append(f'<rect x="250" y="15" width="560" height="135" rx="10" fill="#F8F4F6" stroke="{GREY}" stroke-dasharray="7 5"/>'
             f'<text x="530" y="{h - 2}" font-size="15" text-anchor="middle" fill="{MUTED}">wait 10 µs (10 id steps): detuning, dephasing</text>')
    for k in range(10):
        for y in (y0, y1):
            p.append(gate(285 + 54 * k, y, "id", GREY, 40).replace('font-size="18"', 'font-size="15"'))
    for y in (y0, y1):
        p.append(gate(870, y, "H") + gate(950, y, "M", MAROON))
    p.append(f'<text x="1020" y="{(y0 + y1) / 2 + 6}" font-size="18">⟨XX⟩</text></svg>')
    return "".join(p)


def echo_fig(w=1100, h=150):
    y = 70
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Spin echo: wait half, X, wait half, X">',
         f'<line x1="20" y1="{y}" x2="{w - 20}" y2="{y}" stroke="{INK}" stroke-width="2.5"/>']
    p.append(f'<rect x="60" y="{y - 40}" width="380" height="80" rx="10" fill="#F8F4F6" stroke="{GREY}" stroke-dasharray="7 5"/>'
             f'<text x="250" y="{y + 6}" font-size="18" text-anchor="middle">wait: turn by +φ/2</text>')
    p.append(gate(490, y, "X", MAROON))
    p.append(f'<rect x="540" y="{y - 40}" width="380" height="80" rx="10" fill="#F8F4F6" stroke="{GREY}" stroke-dasharray="7 5"/>'
             f'<text x="730" y="{y + 6}" font-size="18" text-anchor="middle">wait: the same rotation now undoes it</text>')
    p.append(gate(970, y, "X", MAROON))
    p.append(f'<text x="550" y="{h - 8}" font-size="15" text-anchor="middle" fill="{MUTED}">on both qubits; net effect of a steady Z rotation: none</text></svg>')
    return "".join(p)


def fold_fig(w=1100, h=120):
    y0, y1 = 35, 95
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="One CNOT and three CNOTs in a row do the same thing">']
    for xs, lab, xa in (((150,), "scale 1", 80), ((480, 560, 640), "scale 3", 400)):
        for y in (y0, y1):
            p.append(f'<line x1="{xa}" y1="{y}" x2="{xa + 310}" y2="{y}" stroke="{INK}" stroke-width="2.5"/>')
        for x in xs:
            p.append(cnot(x, y0, y1))
        p.append(f'<text x="{xa - 10}" y="{(y0 + y1) / 2 + 6}" font-size="17" fill="{MUTED}" text-anchor="end">{lab}</text>')
    p.append(f'<text x="740" y="{(y0 + y1) / 2 + 6}" font-size="20">the same gate, about 3 × the noise</text></svg>')
    return "".join(p)


def dd_curves(w=560, h=370):
    x0, x1, yb, yt = 70, w - 20, h - 60, 20
    X = lambda v: x0 + (x1 - x0) * v / 0.2  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (v + 1) / 2  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="XX against detuning, raw and with dynamical decoupling">']
    axes(p, w, h, x0, x1, yb, yt, [0, 0.05, 0.1, 0.15, 0.2], X, [(v, f"{v:g}") for v in (-1, -0.5, 0, 0.5, 1)], Y, "detuning per step (rad)", "⟨XX⟩")
    ds = np.linspace(0, 0.2, 41)
    raw = [xx(ref.experiment(False), ref.noise_model(d, PRO, PCX), PRO) for d in ds]
    dd = [xx(ref.experiment(True), ref.noise_model(d, PRO, PCX), PRO) for d in ds]
    p.append(poly(X, Y, ds, raw, INK, 3, "7 5") + poly(X, Y, ds, dd, MAROON, 3.5))
    p.append(f'<line x1="{X(D)}" y1="{yt}" x2="{X(D)}" y2="{yb}" stroke="{GOLD}" stroke-width="2"/>')
    p.append(legend([(INK, "raw", True), (MAROON, "dynamical decoupling", False)], x0 + 15, yb - 50))
    p.append('</svg>')
    return "".join(p)


def zne_plot(w=560, h=370):
    x0, x1, yb, yt = 70, w - 20, h - 60, 20
    lo, hi = 0.75, 0.9
    X = lambda v: x0 + (x1 - x0) * v / 5.5  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (v - lo) / (hi - lo)  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="XX at noise scales 1, 3 and 5 with linear and quadratic fits extended to zero">']
    axes(p, w, h, x0, x1, yb, yt, [0, 1, 2, 3, 4, 5], X, [(v, f"{v:.2f}") for v in (0.75, 0.8, 0.85, 0.9)], Y, "noise scale (number of CNOTs)", "⟨XX⟩")
    xs = np.linspace(0, 5.5, 60)
    pts = P["zne_points"]
    p.append(poly(X, Y, xs, np.polyval(np.polyfit(ref.SCALES, pts, 1), xs), MAROON, 2.5, "7 5"))
    p.append(poly(X, Y, xs, np.polyval(np.polyfit(ref.SCALES, pts, 2), xs), GOLD, 2.5, "3 4"))
    for s, v in zip(ref.SCALES, pts):
        p.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="7" fill="{INK}"/>')
    p.append(f'<path d="M {X(0) - 9:.1f} {Y(TARGET):.1f} l 9 -9 l 9 9 l -9 9 z" fill="{OK}"/>')
    p.append(legend([(INK, "measured", False), (MAROON, "linear fit", True), (GOLD, "quadratic fit", True), (OK, "no CNOT error", False)], x0 + 230, yt + 20))
    p.append('</svg>')
    return "".join(p)


def a_table():
    s = '<table style="font-size:20px"><tr><th>measured ↓ / prepared →</th>' + "".join(f"<th>{k}</th>" for k in ref.KEYS) + "</tr>"
    for i, k in enumerate(ref.KEYS):
        s += f"<tr><th>{k}</th>" + "".join(f"<td>{A[i, j]:.4f}</td>" for j in range(4)) + "</tr>"
    return s + "</table>"


def title_fig(w=430, h=240):
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Measured values rising towards the ideal value as each tool is added">']
    vals = [("raw", RAW), ("DD", DD), ("+ readout", DD_RO), ("+ ZNE", LIN)]
    yb = 200
    for i, (lab, v) in enumerate(vals):
        x = 40 + i * 95
        hgt = 170 * v
        p.append(f'<rect x="{x}" y="{yb - hgt:.1f}" width="60" height="{hgt:.1f}" fill="{MAROON if i else GREY}"/>'
                 f'<text x="{x + 30}" y="{yb + 22}" font-size="15" text-anchor="middle">{lab}</text>'
                 f'<text x="{x + 30}" y="{yb - hgt - 8:.1f}" font-size="15" text-anchor="middle">{v:.2f}</text>')
    p.append(f'<line x1="25" y1="{yb - 170}" x2="{w - 10}" y2="{yb - 170}" stroke="{GOLD}" stroke-width="2.5" stroke-dasharray="6 4"/>'
             f'<text x="{w - 12}" y="{yb - 178}" font-size="15" text-anchor="end" fill="{MUTED}">ideal 1</text></svg>')
    return "".join(p)


m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 3 Slides: Error Suppression and Mitigation</title>")
assert "Error Suppression" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                    "  .slide table td, .slide table th { padding:5px 12px; text-align:left; }\n  svg text { font-family:")
body = (ROOT / "tools/l3_m3_slides_body.html").read_text()
f3 = lambda v: f"{v:.3f}"  # noqa: E731
rep = {"/*TITLE_FIG*/": title_fig(), "/*CIRCUIT*/": circuit_svg(), "/*NOISE_BARS*/": noise_bars(), "/*ECHO_FIG*/": echo_fig(),
       "/*DD_CURVES*/": dd_curves(), "/*A_TABLE*/": a_table(), "/*FOLD_FIG*/": fold_fig(), "/*ZNE_PLOT*/": zne_plot(),
       "/*STD_BARS*/": std_bars(), "/*DEVICE*/": device_bars(),
       "/*RAW*/": f3(RAW), "/*DD*/": f3(DD), "/*DD_RO*/": f3(DD_RO), "/*LIN*/": f3(LIN), "/*QUAD*/": f3(QUAD),
       "/*TARGET*/": f3(TARGET), "/*CALM_RAW*/": f3(CALM_RAW), "/*CALM_DD*/": f3(CALM_DD),
       "/*ZNE_POINTS*/": ", ".join(f3(v) for v in P["zne_points"]), "/*AMP*/": f"{AMP:.1f}", "/*AMP2*/": f"{AMP ** 2:.1f}",
       "/*SE*/": f3(1 / np.sqrt(shots)), "/*CZ*/": f"{100 * run['cz_error']:.1f}", "/*DEPH*/": f3(DEPH), "/*QSIM_V*/": qsim.__version__}
for k, v in rep.items():
    assert k in body, k
    body = body.replace(k, v)
n_slides = body.count('<section class="slide')
bar = "\n".join(m1[365:373]).replace("1 / 19", f"1 / {n_slides}")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m3-slides.html").write_text(out)
print("wrote media/level3/m3-slides.html", len(out), "bytes,", n_slides, "slides")
print("parts", [(n, round(v, 4)) for n, v in PARTS], "calm", round(CALM_RAW, 4), round(CALM_DD, 4), "target", round(TARGET, 4))
print("std", {k: round(v, 4) for k, v in STD.items()}, "amp", round(AMP, 3), "device", {k: round(v, 4) for k, v in DEV.items()}, "deph", round(DEPH, 4))

exp = (ROOT / "tools/l3_m3_explore_src.html").read_text()
assert "`" not in exp and "${" not in exp and "—" not in exp
(OUT / "m3-explore.html").write_text(exp)
print("wrote media/level3/m3-explore.html", len(exp))
