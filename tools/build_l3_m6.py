"""Build the Level 3 Module 6 slides and explorer: media/level3/m6-slides.html and media/level3/m6-explore.html
(two tabs: the shot budget of the mitigation study; combining uncertainties). Every number is computed here from the
capstone check modules (exact density matrices, course noise model P2 = 0.01, READOUT = 0.02)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media" / "level3"
sys.path[:0] = [str(ROOT / "content" / "level3")]
import l3_m6_kernel_check as KC  # noqa: E402
import l3_m6_mit_check as MC  # noqa: E402
import l3_m6_rep_check as RC  # noqa: E402
import l3_m6_spin_check as SC  # noqa: E402
from l3_m6_common import exact_probabilities  # noqa: E402

MAROON, GREY, INK, GOLD, MUTED, LINE, OK = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#E4DCE0", "#2E7D4F"
P2, RO, SHOTS = 0.01, 0.02, 4000
LIN, QUAD = MC.reference_zne_coefficients([1, 3, 5], 1), MC.reference_zne_coefficients([1, 3, 5], 2)


# ---------------------------------------------------------------- numbers
def exact_sigma(n, scale, mitigate, shots=SHOTS):
    p = exact_probabilities(MC.measured(MC.reference_fold_global(MC.reference_ghz_circuit(n), scale)), P2, RO)
    keys = [format(i, f"0{n}b") for i in range(2 ** n)]
    w = []
    for k in keys:
        unit = {x: (1.0 if x == k else 0.0) for x in keys}
        w.append(MC.reference_parity_expectation(MC.reference_mitigate_readout(unit, RO) if mitigate else unit))
    pv = np.array([p.get(k, 0.0) for k in keys])
    E = float(np.dot(w, pv))
    return math.sqrt((float(np.dot(np.square(w), pv)) - E * E) / shots)


BUDGET = {}
for n in range(2, 6):
    st = MC.study(n, P2, RO)
    s_raw = exact_sigma(n, 1, False)
    s_ro = {s: exact_sigma(n, s, True) for s in (1, 3, 5)}
    s_lin = math.sqrt(sum((c * s_ro[s]) ** 2 for c, s in zip(LIN, (1, 3, 5))))
    s_quad = math.sqrt(sum((c * s_ro[s]) ** 2 for c, s in zip(QUAD, (1, 3, 5))))
    BUDGET[n] = {"bias": [1 - st["raw"][1], 1 - st["ro"][1], 1 - st["zne_lin"], 1 - st["zne_quad"]],
                 "var": [s_raw ** 2 * SHOTS, s_ro[1] ** 2 * SHOTS, s_lin ** 2 * 3 * SHOTS, s_quad ** 2 * 3 * SHOTS],
                 "value": [st["raw"][1], st["ro"][1], st["zne_lin"], st["zne_quad"]]}
B3 = BUDGET[3]
cross1 = (B3["var"][2] - B3["var"][1]) / (B3["bias"][1] ** 2 - B3["bias"][2] ** 2)
cross2 = (B3["var"][3] - B3["var"][2]) / (B3["bias"][2] ** 2 - B3["bias"][3] ** 2)
assert 3000 < cross1 < 3400 and 1.8e6 < cross2 < 2.1e6, (cross1, cross2)

enc = {r: RC.logical_error(RC.record_distribution(RC.reference_memory_circuit(r), P2, RO, P2), r) for r in range(1, 6)}
bare = {r: RC.record_distribution(RC.bare_circuit(r), P2, RO, P2).get("1", 0.0) for r in range(1, 6)}
assert round(enc[3], 4) == 0.0152 and round(bare[3], 4) == 0.034

STEPS = [2, 4, 6, 8, 10, 12]
SPIN = {s: SC.study(s, P2, RO) for s in STEPS}
EXACT_M = SC.exact_magnetization(3, 0.7, 2.0)
best_raw = min(STEPS, key=lambda s: abs(SPIN[s]["raw"][1] - EXACT_M))
best_mit = min(STEPS, key=lambda s: abs(SPIN[s]["mitigated"] - EXACT_M))
assert (best_raw, best_mit) == (6, 6)

Xtr, ytr, Xte, yte = KC.capstone_data(str(ROOT / "content" / "level3" / "l3_m4a_data.json"))
Ke = KC.reference_training_kernel(Xtr, lambda a, b: KC.exact_entry(a, b, 0, 0))
Kn = KC.reference_training_kernel(Xtr, lambda a, b: KC.exact_entry(a, b, P2, RO))
iu = np.triu_indices(8, 1)
ka, kb = np.polyfit(Ke[iu], Kn[iu], 1)
kdiag = float(np.mean([KC.exact_entry(x, x, P2, RO) for x in Xtr]))


def r2(x):
    e = int(math.floor(math.log10(x))) - 1
    return int(round(x, -e))


# ---------------------------------------------------------------- figures
def svg(w, h, label, parts):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{label}">' + "".join(parts) + "</svg>"


def axes(p, x0, x1, yb, yt, xlab, ylab, w, h):
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>'
             f'<line x1="{x0}" y1="{yb}" x2="{x0}" y2="{yt}" stroke="{INK}" stroke-width="2"/>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 6}" font-size="20" text-anchor="middle" fill="{MUTED}">{xlab}</text>')
    p.append(f'<text x="18" y="{(yb + yt) / 2}" font-size="20" text-anchor="middle" fill="{MUTED}" '
             f'transform="rotate(-90 18 {(yb + yt) / 2})">{ylab}</text>')


def legend(p, x, y, items):
    for i, (col, lab, dash) in enumerate(items):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{x}" y1="{y + 26 * i}" x2="{x + 30}" y2="{y + 26 * i}" stroke="{col}" stroke-width="4"{d}/>'
                 f'<text x="{x + 38}" y="{y + 26 * i + 7}" font-size="20">{lab}</text>')


def rep_fig(w=560, h=380):
    x0, x1, yb, yt = 80, w - 20, h - 56, 20
    ymax = 0.05
    X = lambda r: x0 + (x1 - x0) * (r - 0.5) / 5  # noqa: E731
    Y = lambda v: yb - (yb - yt) * v / ymax  # noqa: E731
    p = []
    axes(p, x0, x1, yb, yt, "syndrome rounds", "logical error", w, h)
    for v in (0, 0.01, 0.02, 0.03, 0.04, 0.05):
        p.append(f'<text x="{x0 - 8}" y="{Y(v) + 6:.1f}" font-size="18" text-anchor="end">{v:.2f}</text>'
                 f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
    for r in range(1, 6):
        p.append(f'<text x="{X(r):.1f}" y="{yb + 24}" font-size="18" text-anchor="middle">{r}</text>')
        p.append(f'<rect x="{X(r) - 30:.1f}" y="{Y(bare[r]):.1f}" width="28" height="{yb - Y(bare[r]):.1f}" fill="{GREY}"/>'
                 f'<rect x="{X(r) + 2:.1f}" y="{Y(enc[r]):.1f}" width="28" height="{yb - Y(enc[r]):.1f}" fill="{MAROON}"/>')
    p.append(f'<rect x="{x0 + 20}" y="{yt + 4}" width="22" height="16" fill="{GREY}"/><text x="{x0 + 50}" y="{yt + 18}" font-size="20">one qubit</text>'
             f'<rect x="{x0 + 20}" y="{yt + 32}" width="22" height="16" fill="{MAROON}"/><text x="{x0 + 50}" y="{yt + 46}" font-size="20">repetition code</text>')
    return svg(w, h, "Logical error of the repetition code and of one qubit for 1 to 5 rounds", p)


def mit_fig(w=560, h=380):
    x0, x1, yb, yt = 80, w - 20, h - 70, 20
    lo, hi = 0.765, 1.04
    labels = [("raw", ""), ("readout", ""), ("readout +", "ZNE linear"), ("readout +", "ZNE quadratic")]
    sig = [math.sqrt(v / SHOTS) if i < 2 else math.sqrt(v / (3 * SHOTS)) for i, v in enumerate(B3["var"])]
    X = lambda i: x0 + (x1 - x0) * (i + 0.5) / 4  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (v - lo) / (hi - lo)  # noqa: E731
    p = []
    axes(p, x0, x1, yb, yt, "", "⟨X X X⟩", w, h)
    for v in (0.8, 0.85, 0.9, 0.95, 1.0):
        p.append(f'<text x="{x0 - 8}" y="{Y(v) + 6:.1f}" font-size="18" text-anchor="end">{v:.2f}</text>'
                 f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
    p.append(f'<line x1="{x0}" y1="{Y(1):.1f}" x2="{x1}" y2="{Y(1):.1f}" stroke="{OK}" stroke-width="2" stroke-dasharray="7 5"/>'
             f'<text x="{x0 + 8}" y="{Y(1) - 8:.1f}" font-size="18" fill="{OK}">ideal 1</text>')
    for i, ((l1, l2), v, s) in enumerate(zip(labels, B3["value"], sig)):
        x = X(i)
        p.append(f'<line x1="{x:.1f}" y1="{Y(v - s):.1f}" x2="{x:.1f}" y2="{Y(v + s):.1f}" stroke="{INK}" stroke-width="3"/>'
                 f'<line x1="{x - 9:.1f}" y1="{Y(v - s):.1f}" x2="{x + 9:.1f}" y2="{Y(v - s):.1f}" stroke="{INK}" stroke-width="3"/>'
                 f'<line x1="{x - 9:.1f}" y1="{Y(v + s):.1f}" x2="{x + 9:.1f}" y2="{Y(v + s):.1f}" stroke="{INK}" stroke-width="3"/>'
                 f'<circle cx="{x:.1f}" cy="{Y(v):.1f}" r="8" fill="{MAROON}"/>'
                 f'<text x="{x:.1f}" y="{yb + 22}" font-size="17" text-anchor="middle">{l1}</text>'
                 f'<text x="{x:.1f}" y="{yb + 42}" font-size="17" text-anchor="middle">{l2}</text>'
                 f'<text x="{x:.1f}" y="{Y(v - s) + 24:.1f}" font-size="18" text-anchor="middle">{v:.3f}</text>')
    return svg(w, h, "GHZ parity: raw, readout-mitigated and extrapolated values with error bars", p)


def rms_fig(w=580, h=380):
    x0, x1, yb, yt = 80, w - 20, h - 56, 20
    lx0, lx1, ly0, ly1 = 2, 7, -3, 0
    X = lambda b: x0 + (x1 - x0) * (math.log10(b) - lx0) / (lx1 - lx0)  # noqa: E731
    Y = lambda r: yb - (yb - yt) * (math.log10(r) - ly0) / (ly1 - ly0)  # noqa: E731
    p = []
    axes(p, x0, x1, yb, yt, "total shots", "RMS error", w, h)
    for e in range(2, 8):
        p.append(f'<text x="{X(10 ** e):.1f}" y="{yb + 24}" font-size="18" text-anchor="middle">10{"⁰¹²³⁴⁵⁶⁷"[e]}</text>')
    for v in (0.001, 0.01, 0.1, 1):
        p.append(f'<text x="{x0 - 8}" y="{Y(v) + 6:.1f}" font-size="18" text-anchor="end">{v:g}</text>'
                 f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
    cols = [GREY, INK, MAROON, GOLD]
    names = ["raw", "readout", "readout + ZNE linear", "readout + ZNE quadratic"]
    bs = np.logspace(2, 7, 120)
    for k in range(4):
        pts = " ".join(f"{X(b):.1f},{Y(math.sqrt(B3['bias'][k] ** 2 + B3['var'][k] / b)):.1f}" for b in bs)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k]}" stroke-width="4"/>')
    legend(p, x0 + 16, yb - 96, [(cols[k], names[k], "") for k in range(4)])
    return svg(w, h, "RMS error of four methods against the total number of shots", p)


def spin_fig(w=560, h=380):
    x0, x1, yb, yt = 80, w - 20, h - 56, 20
    lo, hi = 0.2, 0.42
    X = lambda s: x0 + (x1 - x0) * (s - 1) / 12  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (v - lo) / (hi - lo)  # noqa: E731
    p = []
    axes(p, x0, x1, yb, yt, "Trotter steps", "M(T = 2)", w, h)
    for v in (0.2, 0.25, 0.3, 0.35, 0.4):
        p.append(f'<text x="{x0 - 8}" y="{Y(v) + 6:.1f}" font-size="18" text-anchor="end">{v:.2f}</text>'
                 f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
    for s in STEPS:
        p.append(f'<text x="{X(s):.1f}" y="{yb + 24}" font-size="18" text-anchor="middle">{s}</text>')
    p.append(f'<line x1="{x0}" y1="{Y(EXACT_M):.1f}" x2="{x1}" y2="{Y(EXACT_M):.1f}" stroke="{OK}" stroke-width="2" stroke-dasharray="7 5"/>')
    for key, col, dash in (("ideal", INK, "6 4"), ("raw", GREY, ""), ("mitigated", MAROON, "")):
        vals = [SPIN[s][key] if key != "raw" else SPIN[s]["raw"][1] for s in STEPS]
        pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(STEPS, vals))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="4"{d}/>'
                 + "".join(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="5" fill="{col}"/>' for s, v in zip(STEPS, vals)))
    legend(p, x0 + 190, yb - 96, [(OK, "exact", "7 5"), (INK, "Trotter, no noise", "6 4"), (MAROON, "mitigated", ""), (GREY, "raw", "")])
    return svg(w, h, "Magnetization at T = 2 against the number of Trotter steps", p)


def ker_fig(w=520, h=380):
    x0, x1, yb, yt = 80, w - 20, h - 56, 20
    X = lambda v: x0 + (x1 - x0) * v  # noqa: E731
    Y = lambda v: yb - (yb - yt) * v  # noqa: E731
    p = []
    axes(p, x0, x1, yb, yt, "exact kernel entry", "noisy entry", w, h)
    for v in (0, 0.25, 0.5, 0.75, 1):
        p.append(f'<text x="{x0 - 8}" y="{Y(v) + 6:.1f}" font-size="18" text-anchor="end">{v:g}</text>'
                 f'<text x="{X(v):.1f}" y="{yb + 24}" font-size="18" text-anchor="middle">{v:g}</text>')
    p.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="{LINE}" stroke-width="2"/>'
             f'<line x1="{X(0)}" y1="{Y(kb):.1f}" x2="{X(1)}" y2="{Y(ka + kb):.1f}" stroke="{INK}" stroke-width="2" stroke-dasharray="7 5"/>')
    for a, b in zip(Ke[iu], Kn[iu]):
        p.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="6" fill="{MAROON}"/>')
    p.append(f'<circle cx="{X(1):.1f}" cy="{Y(kdiag):.1f}" r="8" fill="{GOLD}"/>'
             f'<circle cx="{X(0.5):.1f}" cy="{Y(0.08):.1f}" r="8" fill="{GOLD}"/><text x="{X(0.5) + 14:.1f}" y="{Y(0.08) + 6:.1f}" font-size="18">K(x, x), the diagonal</text>'
             f'<text x="{X(0.04):.1f}" y="{Y(0.93):.1f}" font-size="19">{ka:.3f} · exact + {kb:.3f}</text>')
    return svg(w, h, "Noisy against exact kernel entries of the 28 training pairs", p)


def title_fig(w=460, h=320):
    return rep_fig(w, h)


# ---------------------------------------------------------------- the deck
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
TITLE = "Module 6 Slides: Capstone"
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>", f"<title>{TITLE}</title>")
assert TITLE in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                    "  .slide table td, .slide table th { padding:6px 12px; text-align:left; }\n  svg text { font-family:")
rep = {"/*TITLE_FIG*/": title_fig(), "/*REP_FIG*/": rep_fig(), "/*MIT_FIG*/": mit_fig(), "/*RMS_FIG*/": rms_fig(),
       "/*SPIN_FIG*/": spin_fig(), "/*KER_FIG*/": ker_fig(),
       "/*REP_E3*/": f"{enc[3]:.4f}", "/*REP_B3*/": f"{bare[3]:.4f}", "/*REP_CROSS*/": "0.037",
       "/*MIT_LINRAW*/": f"{MC.study(3, P2, RO)['zne_lin_raw']:.3f}", "/*CROSS1*/": f"{r2(cross1):,}", "/*CROSS2*/": f"{r2(cross2):,}",
       "/*SPIN_BEST_RAW*/": str(best_raw), "/*SPIN_EXACT*/": f"{EXACT_M:.3f}", "/*SPIN_BEST*/": str(best_mit),
       "/*KER_DIAG*/": f"{kdiag:.2f}", "/*KER_A*/": f"{ka:.2f}", "/*KER_B*/": f"{kb:.2f}", "/*KER_ACC20*/": "0.90"}
body = (ROOT / "tools/l3_m6_slides_body.html").read_text()
for k, v in rep.items():
    assert k in body, k
    body = body.replace(k, v)
n_slides = body.count('<section class="slide')
bar = "\n".join(m1[365:373]).replace("1 / 19", f"1 / {n_slides}")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m6-slides.html").write_text(out)
print("wrote media/level3/m6-slides.html", len(out), "bytes,", n_slides, "slides")

src = (ROOT / "tools/l3_m6_explore_src.html").read_text()
data = {str(n): {k: [round(x, 6) for x in v] for k, v in b.items()} for n, b in BUDGET.items()}
exp = src.replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
assert "`" not in exp and "${" not in exp and "—" not in exp and "/*DATA*/" not in exp
(OUT / "m6-explore.html").write_text(exp)
print("wrote media/level3/m6-explore.html", len(exp))
print("crossings", round(cross1), round(cross2), "kernel", round(ka, 3), round(kb, 3), round(kdiag, 3))
