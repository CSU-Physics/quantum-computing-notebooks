"""Build the Level 3 Module 4 slides and explorer:
media/level3/m4a-slides.html (Track A, quantum-kernel classifier), media/level3/m4b-slides.html (Track B, Trotter
simulation) and media/level3/m4-explore.html (two tabs: what a kernel sees; Trotter steps and noise).
Every number on the slides is computed here with the labs' own reference functions and data, and the values the
labs print are asserted. Needs scikit-learn (as the Track A lab); Qiskit is not used."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media" / "level3"
sys.modules["qiskit"] = None
sys.path[:0] = [str(ROOT / "content" / "level3")]
import qsim  # noqa: E402
import l3_m4a_check as A  # noqa: E402
import l3_m4b_check as B  # noqa: E402
from qsim import NoiseModel, Statevector, depolarizing_error, simulate_density_matrix  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

MAROON, GREY, INK, GOLD, MUTED, OK, LINE = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#2E7D4F", "#E4DCE0"

# ---------------------------------------------------------------- Track A numbers (as in the lab)
DATA = json.loads((ROOT / "content/level3/l3_m4a_data.json").read_text())
CV = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
C_GRID = [0.1, 1, 10, 100]


def load(name):
    d = DATA[name]
    return np.array(d["X_train"]), np.array(d["y_train"]), np.array(d["X_test"]), np.array(d["y_test"])


ACC = {}
for name in ("moons", "adhoc"):
    Xtr, ytr, Xte, yte = load(name)
    q = GridSearchCV(SVC(kernel="precomputed"), {"C": C_GRID}, cv=CV).fit(A.reference_kernel_matrix(Xtr, Xtr), ytr)
    c = GridSearchCV(SVC(kernel="rbf"), {"C": C_GRID, "gamma": [0.1, 0.3, 1, 3, 10]}, cv=CV).fit(Xtr, ytr)
    ACC[name] = (q.score(A.reference_kernel_matrix(Xte, Xtr), yte), c.score(Xte, yte))
assert [round(v, 2) for v in ACC["moons"] + ACC["adhoc"]] == [0.70, 1.00, 1.00, 0.80], ACC
assert A.personal_value(1.25, 0.40, 2.10) == 604

rng = np.random.default_rng(7)
CONC = []
for n in range(1, 9):
    vals = np.array([abs(np.vdot(Statevector(A.reference_feature_map(rng.uniform(0, 2 * np.pi, n))).data,
                                 Statevector(A.reference_feature_map(rng.uniform(0, 2 * np.pi, n))).data)) ** 2 for _ in range(50)])
    CONC.append((n, vals.mean(), vals.std(), np.sqrt(vals.mean() * (1 - vals.mean()) / 1000)))
assert round(CONC[5][1], 4) == 0.0165 and round(CONC[7][1], 4) == 0.0046

# ---------------------------------------------------------------- Track B numbers (as in the lab)
N, J, H0, T2 = 4, 1.0, 0.5, 2.0
Hop = B.reference_ising_hamiltonian(N, J, H0)


def mag_state(psi):
    return float(np.real(np.conj(psi) @ B.magnetization_op(N).to_matrix() @ psi))


M_EX = mag_state(B.exact_state(Hop, T2))
assert round(M_EX, 3) == 0.670
psi_ex = B.exact_state(Hop, T2)
STEPS = [4, 8, 16, 32]
INF1 = [1 - abs(np.vdot(psi_ex, Statevector(B.trotter_circuit(N, J, H0, T2, n)).data)) ** 2 for n in STEPS]
INF2 = [1 - abs(np.vdot(psi_ex, Statevector(B.trotter_circuit(N, J, H0, T2, n, B.reference_trotter_step2)).data)) ** 2 for n in STEPS]
SL1, SL2 = np.polyfit(np.log(STEPS), np.log(INF1), 1)[0], np.polyfit(np.log(STEPS), np.log(INF2), 1)[0]
assert (round(INF1[1], 3), round(SL1, 2), round(SL2, 2)) == (0.037, -2.09, -4.05)
P2 = 0.005
nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(P2, 2), ["rzz"])
NS = list(range(1, 17))
NOISY = {}
for name, step in (("first", B.reference_trotter_step), ("second", B.reference_trotter_step2)):
    vals = [float(np.real(np.trace(simulate_density_matrix(B.trotter_circuit(N, J, H0, T2, n, step), nm).data
                                    @ B.magnetization_op(N).to_matrix()))) for n in NS]
    NOISY[name] = np.abs(np.array(vals) - M_EX)
BEST = {k: (NS[int(np.argmin(v))], float(v.min())) for k, v in NOISY.items()}
assert (BEST["first"][0], round(BEST["first"][1], 4), BEST["second"][0], round(BEST["second"][1], 4)) == (5, 0.0429, 6, 0.0081), BEST
assert B.personal_value(0.50, 2.00, 8) == 665


def exact_curve(h, ts):
    Hh = B.reference_ising_hamiltonian(N, J, h)
    E, V = np.linalg.eigh(Hh.to_matrix())
    out = []
    for t in ts:
        psi = V @ (np.exp(-1j * E * t) * V.conj().T[:, 0])
        out.append(mag_state(psi))
    return out


# ---------------------------------------------------------------- drawing helpers
def axes(p, w, h, x0, x1, yb, yt, xticks, X, yticks, Y, xlab, ylab):
    for v, lab in yticks:
        p.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>'
                 f'<text x="{x0 - 8}" y="{Y(v) + 5:.1f}" font-size="14" text-anchor="end">{lab}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    for v, lab in xticks:
        p.append(f'<text x="{X(v):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle">{lab}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 6}" font-size="15" text-anchor="middle" fill="{MUTED}">{xlab}</text>')
    if ylab:
        p.append(f'<text x="16" y="{(yb + yt) / 2}" font-size="15" text-anchor="middle" fill="{MUTED}" '
                 f'transform="rotate(-90 16 {(yb + yt) / 2})">{ylab}</text>')


def poly(X, Y, xs, ys, col, w=3, dash=""):
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}"{d}/>'


def dots(X, Y, xs, ys, col, r=5, square=False):
    if square:
        return "".join(f'<rect x="{X(a) - r:.1f}" y="{Y(b) - r:.1f}" width="{2 * r}" height="{2 * r}" fill="{col}"/>' for a, b in zip(xs, ys))
    return "".join(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="{r}" fill="{col}"/>' for a, b in zip(xs, ys))


def legend(items, x, y):
    out = []
    for i, (col, lab, dash) in enumerate(items):
        yy = y + 22 * i
        d = ' stroke-dasharray="6 4"' if dash else ""
        out.append(f'<line x1="{x}" y1="{yy}" x2="{x + 22}" y2="{yy}" stroke="{col}" stroke-width="3"{d}/>'
                   f'<text x="{x + 30}" y="{yy + 5}" font-size="15">{lab}</text>')
    return "".join(out)


def svg(w, h, label, parts):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{label}">' + "".join(parts) + "</svg>"


def heat(v):
    stops = [(36, 49, 61), (153, 0, 76), (201, 162, 39), (255, 248, 230)]
    x = min(max(v, 0), 1) * 3
    i = min(2, int(x))
    f = x - i
    return "rgb(" + ",".join(str(round(a + (b - a) * f)) for a, b in zip(stops[i], stops[i + 1])) + ")"


def gate(x, y, lab, w=46, col=INK, fill="#FFFFFF", fs=15):
    return (f'<rect x="{x - w / 2}" y="{y - 18}" width="{w}" height="36" rx="4" fill="{fill}" stroke="{col}" stroke-width="2"/>'
            f'<text x="{x}" y="{y + 5}" font-size="{fs}" text-anchor="middle" fill="{col}">{lab}</text>')


def cnot(x, yc, yt):
    return (f'<line x1="{x}" y1="{yc}" x2="{x}" y2="{yt + 13}" stroke="{INK}" stroke-width="2"/>'
            f'<circle cx="{x}" cy="{yc}" r="6" fill="{INK}"/><circle cx="{x}" cy="{yt}" r="13" fill="none" stroke="{INK}" stroke-width="2"/>'
            f'<line x1="{x - 13}" y1="{yt}" x2="{x + 13}" y2="{yt}" stroke="{INK}" stroke-width="2"/>')


# ---------------------------------------------------------------- Track A figures
def acc_bars(w=560, h=340, title=True):
    p = []
    x0, x1, yb, yt = 60, w - 20, h - 80, 30
    Y = lambda v: yb - (yb - yt) * v / 1.1  # noqa: E731
    axes(p, w, h, x0, x1, yb, yt, [], None, [(0, "0"), (0.5, "0.5"), (1, "1.0")], Y, "", "test accuracy")
    groups = [("moons", ACC["moons"]), ("adhoc", ACC["adhoc"])]
    gw = (x1 - x0) / 2
    for g, (name, (q, c)) in enumerate(groups):
        for k, (v, col) in enumerate(((q, MAROON), (c, INK))):
            bx = x0 + g * gw + gw * 0.2 + k * gw * 0.32
            p.append(f'<rect x="{bx:.1f}" y="{Y(v):.1f}" width="{gw * 0.28:.1f}" height="{yb - Y(v):.1f}" fill="{col}"/>'
                     f'<text x="{bx + gw * 0.14:.1f}" y="{Y(v) - 7:.1f}" font-size="16" text-anchor="middle">{v:.2f}</text>')
        p.append(f'<text x="{x0 + g * gw + gw / 2:.1f}" y="{yb + 24}" font-size="17" text-anchor="middle">{name}</text>')
    p.append(f'<rect x="{x0 + 10}" y="{h - 34}" width="14" height="14" fill="{MAROON}"/><text x="{x0 + 30}" y="{h - 22}" font-size="15">quantum ZZ kernel</text>'
             f'<rect x="{x0 + 200}" y="{h - 34}" width="14" height="14" fill="{INK}"/><text x="{x0 + 220}" y="{h - 22}" font-size="15">classical RBF kernel</text>')
    return svg(w, h, "Test accuracy of the two kernels on the two data sets", p)


def fmap_fig(w=1100, h=170):
    p = [f'<line x1="70" y1="55" x2="{w - 40}" y2="55" stroke="{INK}" stroke-width="2"/>',
         f'<line x1="70" y1="125" x2="{w - 40}" y2="125" stroke="{INK}" stroke-width="2"/>',
         '<text x="30" y="61" font-size="18">q₀</text><text x="30" y="131" font-size="18">q₁</text>',
         f'<rect x="100" y="20" width="{w - 200}" height="140" rx="10" fill="none" stroke="{GOLD}" stroke-width="2" stroke-dasharray="7 5"/>',
         f'<text x="{w - 110}" y="15" font-size="16" fill="{GOLD}" text-anchor="end">repeated twice (reps = 2)</text>']
    p += [gate(150, 55, "H"), gate(150, 125, "H"), gate(260, 55, "P(2x₀)", 100), gate(260, 125, "P(2x₁)", 100),
          cnot(390, 55, 125), gate(590, 125, "P(2(π − x₀)(π − x₁))", 250, MAROON), cnot(790, 55, 125)]
    p.append(f'<text x="{(390 + 790) / 2}" y="35" font-size="15" text-anchor="middle" fill="{MAROON}">the pair term: entangles the qubits</text>')
    return svg(w, h, "The ZZ feature map for two features", p)


def data_fig(w=1100, h=330):
    p = []
    for k, name in enumerate(("moons", "adhoc")):
        Xtr, ytr, Xte, yte = load(name)
        top = np.pi if name == "moons" else 2 * np.pi
        ox, sz = 90 + k * 560, 250
        X = lambda v, ox=ox, top=top: ox + sz * v / top  # noqa: E731
        Y = lambda v, top=top: 30 + sz - sz * v / top  # noqa: E731
        p.append(f'<rect x="{ox}" y="30" width="{sz}" height="{sz}" fill="#FFFFFF" stroke="{INK}"/>'
                 f'<text x="{ox + sz / 2}" y="20" font-size="18" text-anchor="middle">{name}</text>'
                 f'<text x="{ox + sz / 2}" y="{sz + 52}" font-size="13" text-anchor="middle" fill="{MUTED}">feature 1 (0 to {"π" if top < 4 else "2π"})</text>')
        for XX, yy, filled in ((Xtr, ytr, True), (Xte, yte, False)):
            for (a, b), c in zip(XX, yy):
                col = MAROON if c == 1 else INK
                if filled:
                    p.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="5" fill="{col}"/>')
                else:
                    p.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="7" fill="none" stroke="{col}" stroke-width="2"/>')
    p.append(f'<text x="{w - 20}" y="{h - 8}" font-size="14" text-anchor="end" fill="{MUTED}">filled: 40 training points · rings: 20 test points · maroon: class 1</text>')
    return svg(w, h, "The moons and adhoc data sets", p)


def kmat_fig(w=1100, h=330):
    p = []
    for k, name in enumerate(("moons", "adhoc")):
        Xtr, ytr, _, _ = load(name)
        idx = np.concatenate([np.flatnonzero(ytr == 0)[:10], np.flatnonzero(ytr == 1)[:10]])
        K = A.reference_kernel_matrix(Xtr[idx], Xtr[idx])
        ox, cell = 120 + k * 520, 12.5
        p.append(f'<text x="{ox + 125}" y="22" font-size="18" text-anchor="middle">{name}</text>')
        for i in range(20):
            for j in range(20):
                p.append(f'<rect x="{ox + j * cell:.1f}" y="{35 + i * cell:.1f}" width="{cell + 0.3:.1f}" height="{cell + 0.3:.1f}" fill="{heat(K[i, j])}"/>')
        p.append(f'<rect x="{ox}" y="35" width="250" height="250" fill="none" stroke="{INK}"/>'
                 f'<line x1="{ox + 125}" y1="35" x2="{ox + 125}" y2="285" stroke="#FFFFFF" stroke-width="2"/>'
                 f'<line x1="{ox}" y1="160" x2="{ox + 250}" y2="160" stroke="#FFFFFF" stroke-width="2"/>'
                 f'<text x="{ox - 8}" y="100" font-size="13" text-anchor="end">class 0</text>'
                 f'<text x="{ox - 8}" y="225" font-size="13" text-anchor="end">class 1</text>')
    for i, v in enumerate(np.linspace(0, 1, 11)):
        p.append(f'<rect x="{w - 150 + i * 11}" y="300" width="11" height="14" fill="{heat(v)}"/>')
    p.append(f'<text x="{w - 158}" y="312" font-size="13" text-anchor="end">K = 0</text><text x="{w - 26}" y="312" font-size="13">1</text>')
    p.append(f'<text x="120" y="318" font-size="13" fill="{MUTED}">10 training points of each class, class 0 first</text>')
    return svg(w, h, "Kernel matrices of ten points of each class", p)


def overlap_fig(w=1000, h=150):
    p = [f'<line x1="70" y1="50" x2="{w - 204}" y2="50" stroke="{INK}" stroke-width="2"/>',
         f'<line x1="70" y1="110" x2="{w - 204}" y2="110" stroke="{INK}" stroke-width="2"/>',
         '<text x="10" y="56" font-size="18">|0⟩</text><text x="10" y="116" font-size="18">|0⟩</text>',
         f'<rect x="130" y="25" width="250" height="110" rx="6" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>',
         '<text x="255" y="87" font-size="20" text-anchor="middle">feature map of x</text>',
         f'<rect x="440" y="25" width="280" height="110" rx="6" fill="#FFFFFF" stroke="{MAROON}" stroke-width="2"/>',
         f'<text x="580" y="87" font-size="20" text-anchor="middle" fill="{MAROON}">feature map of x′, inverted</text>']
    for y in (50, 110):
        p.append(f'<rect x="{w - 250}" y="{y - 17}" width="46" height="34" rx="4" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>'
                 f'<path d="M {w - 242} {y + 8} Q {w - 227} {y - 12} {w - 212} {y + 8}" fill="none" stroke="{INK}" stroke-width="2"/>')
    p.append(f'<text x="{w - 190}" y="86" font-size="18">P(00) = K(x, x′)</text>')
    return svg(w, h, "The overlap circuit used to estimate a kernel value", p)


def conc_fig(w=560, h=360):
    p = []
    x0, x1, yb, yt = 70, w - 20, h - 50, 20
    lo, hi = -3.5, 0
    X = lambda n: x0 + (x1 - x0) * (n - 1) / 7  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (np.log10(max(v, 10 ** lo)) - lo) / (hi - lo)  # noqa: E731
    axes(p, w, h, x0, x1, yb, yt, [(n, str(n)) for n in range(1, 9)], X,
         [(1, "1"), (0.1, "0.1"), (0.01, "0.01"), (0.001, "0.001")], Y, "qubits", "")
    ns = [r[0] for r in CONC]
    p.append(poly(X, Y, ns, [r[1] for r in CONC], MAROON) + dots(X, Y, ns, [r[1] for r in CONC], MAROON))
    p.append(poly(X, Y, ns, [r[2] for r in CONC], INK) + dots(X, Y, ns, [r[2] for r in CONC], INK, 5, True))
    p.append(poly(X, Y, ns, [r[3] for r in CONC], GOLD, 3, "7 5"))
    p.append(legend([(MAROON, "mean kernel value", False), (INK, "spread between pairs", False), (GOLD, "1,000-shot error", True)], x0 + 205, 32))
    return svg(w, h, "Kernel values and their spread fall with the number of qubits", p)


# ---------------------------------------------------------------- Track B figures
def chain_fig(w=900, h=170):
    p = []
    xs = [150 + 190 * i for i in range(4)]
    for i in range(3):
        p.append(f'<line x1="{xs[i]}" y1="95" x2="{xs[i + 1]}" y2="95" stroke="{MAROON}" stroke-width="5"/>'
                 f'<text x="{(xs[i] + xs[i + 1]) / 2}" y="83" font-size="17" text-anchor="middle" fill="{MAROON}">−J ZZ</text>')
    for i, x in enumerate(xs):
        p.append(f'<circle cx="{x}" cy="95" r="30" fill="#FFFFFF" stroke="{INK}" stroke-width="3"/>'
                 f'<line x1="{x}" y1="112" x2="{x}" y2="74" stroke="{INK}" stroke-width="4"/>'
                 f'<path d="M {x - 8} 82 L {x} 70 L {x + 8} 82" fill="none" stroke="{INK}" stroke-width="4"/>'
                 f'<text x="{x}" y="150" font-size="17" text-anchor="middle">spin {i}</text>'
                 f'<line x1="{x + 36}" y1="40" x2="{x + 66}" y2="40" stroke="{GOLD}" stroke-width="4"/>'
                 f'<path d="M {x + 58} 33 L {x + 68} 40 L {x + 58} 47" fill="none" stroke="{GOLD}" stroke-width="4"/>')
    p.append(f'<text x="{w - 10}" y="30" font-size="16" text-anchor="end" fill="{GOLD}">field −h X on each spin</text>')
    return svg(w, h, "Four spins in a chain with Ising couplings and a transverse field", p)


def exact_fig(w=560, h=330):
    ts = np.linspace(0, 3, 61)
    p = []
    x0, x1, yb, yt = 60, w - 20, h - 50, 20
    X = lambda t: x0 + (x1 - x0) * t / 3  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (v + 0.2) / 1.2  # noqa: E731
    axes(p, w, h, x0, x1, yb, yt, [(t, str(t)) for t in range(4)], X, [(0, "0"), (0.5, "0.5"), (1, "1")], Y, "time t (units of 1/J)", "magnetization M")
    p.append(poly(X, Y, ts, exact_curve(0.5, ts), MAROON) + poly(X, Y, ts, exact_curve(1.0, ts), INK))
    p.append(f'<circle cx="{X(2):.1f}" cy="{Y(M_EX):.1f}" r="6" fill="{GOLD}"/><text x="{X(2) + 10:.1f}" y="{Y(M_EX) + 22:.1f}" font-size="15">{M_EX:.3f} at t = 2</text>')
    p.append(legend([(MAROON, "h = 0.5", False), (INK, "h = 1.0", False)], x1 - 130, 150))
    return svg(w, h, "Exact magnetization against time", p)


def step_fig(w=1000, h=250):
    ys = [40 + 55 * q for q in range(4)]
    p = [f'<line x1="60" y1="{y}" x2="{w - 30}" y2="{y}" stroke="{INK}" stroke-width="2"/><text x="20" y="{y + 6}" font-size="17">q{q}</text>'
         for q, y in enumerate(ys)]
    for k in range(3):
        x = 150 + 150 * k
        p.append(f'<rect x="{x - 55}" y="{ys[k] - 18}" width="110" height="{ys[k + 1] - ys[k] + 36}" rx="5" fill="#FFFFFF" stroke="{MAROON}" stroke-width="2"/>'
                 f'<text x="{x}" y="{(ys[k] + ys[k + 1]) / 2 + 6}" font-size="15" text-anchor="middle" fill="{MAROON}">RZZ(−2J dt)</text>')
    for q in range(4):
        p.append(gate(720, ys[q], "RX(−2h dt)", 120))
    p.append(f'<text x="{w - 30}" y="{h - 8}" font-size="14" text-anchor="end" fill="{MUTED}">one first-order step: exp(+iJ dt Σ ZZ), then exp(+ih dt Σ X); repeat n = t/dt times</text>')
    return svg(w, h, "One first-order Trotter step for four spins", p)


def trot_fig(w=560, h=330):
    ts = np.linspace(0, 3, 61)
    p = []
    x0, x1, yb, yt = 60, w - 20, h - 50, 20
    X = lambda t: x0 + (x1 - x0) * t / 3  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (v - 0.35) / 0.7  # noqa: E731
    axes(p, w, h, x0, x1, yb, yt, [(t, str(t)) for t in range(4)], X, [(0.5, "0.5"), (0.75, "0.75"), (1, "1")], Y, "time t", "M (h = 0.5)")
    p.append(poly(X, Y, ts, exact_curve(0.5, ts), MAROON))
    for dt, sq, col in ((0.5, False, INK), (0.2, True, GOLD)):
        tt = np.arange(0, 3.0001, dt)
        mm = [B.magnetization(B.trotter_circuit(N, J, 0.5, t, max(1, round(t / dt))), N) for t in tt]
        p.append(dots(X, Y, tt, mm, col, 5, sq))
    lx, ly = x0 + 250, yb - 62
    p.append(legend([(MAROON, "exact", False)], lx, ly) +
             f'<circle cx="{lx + 11}" cy="{ly + 22}" r="5" fill="{INK}"/><text x="{lx + 30}" y="{ly + 27}" font-size="15">Trotter, dt = 0.5</text>'
             f'<rect x="{lx + 6}" y="{ly + 39}" width="10" height="10" fill="{GOLD}"/><text x="{lx + 30}" y="{ly + 49}" font-size="15">Trotter, dt = 0.2</text>')
    return svg(w, h, "Trotter results against the exact magnetization", p)


def scaling_fig(w=560, h=330):
    p = []
    x0, x1, yb, yt = 70, w - 20, h - 50, 20
    lo, hi = -6, 0
    X = lambda n: x0 + (x1 - x0) * (np.log2(n) - 2) / 3  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (np.log10(v) - lo) / (hi - lo)  # noqa: E731
    axes(p, w, h, x0, x1, yb, yt, [(n, str(n)) for n in STEPS], X, [(10.0 ** k, f"1e{k}" if k else "1") for k in range(-6, 1, 2)], Y,
         "Trotter steps to t = 2", "infidelity")
    p.append(poly(X, Y, STEPS, INF1, INK) + dots(X, Y, STEPS, INF1, INK))
    p.append(poly(X, Y, STEPS, INF2, MAROON) + dots(X, Y, STEPS, INF2, MAROON, 5, True))
    p.append(legend([(INK, f"first order, slope {SL1:.2f}", False), (MAROON, f"second order, slope {SL2:.2f}", False)], x0 + 20, yb - 60))
    return svg(w, h, "Infidelity against the number of Trotter steps", p)


def noise_fig(w=560, h=330):
    p = []
    x0, x1, yb, yt = 70, w - 20, h - 50, 20
    lo, hi = -2.5, 0
    X = lambda n: x0 + (x1 - x0) * (n - 1) / 15  # noqa: E731
    Y = lambda v: yb - (yb - yt) * (np.log10(max(v, 10 ** lo)) - lo) / (hi - lo)  # noqa: E731
    axes(p, w, h, x0, x1, yb, yt, [(n, str(n)) for n in (1, 4, 8, 12, 16)], X, [(1, "1"), (0.1, "0.1"), (0.01, "0.01")], Y,
         "Trotter steps to t = 2", "|M − M_exact|")
    p.append(poly(X, Y, NS, NOISY["first"], INK) + dots(X, Y, NS, NOISY["first"], INK, 4))
    p.append(poly(X, Y, NS, NOISY["second"], MAROON) + dots(X, Y, NS, NOISY["second"], MAROON, 4, True))
    for k, col in (("first", INK), ("second", MAROON)):
        n, e = BEST[k]
        p.append(f'<circle cx="{X(n):.1f}" cy="{Y(e):.1f}" r="10" fill="none" stroke="{GOLD}" stroke-width="3"/>')
    p.append(legend([(INK, "first order", False), (MAROON, "second order", False)], x1 - 160, 40))
    return svg(w, h, "Error with gate noise against the number of Trotter steps", p)


def title_b(w=430, h=250):
    ts = np.linspace(0, 3, 61)
    p = []
    X = lambda t: 20 + 390 * t / 3  # noqa: E731
    Y = lambda v: 220 - 190 * (v - 0.5) / 0.55  # noqa: E731
    p.append(poly(X, Y, ts, exact_curve(0.5, ts), MAROON, 4))
    tt = np.arange(0, 3.0001, 0.25)
    p.append(dots(X, Y, tt, [B.magnetization(B.trotter_circuit(N, J, 0.5, t, max(1, round(t / 0.25))), N) for t in tt], INK, 5))
    p.append(f'<text x="410" y="244" font-size="14" text-anchor="end" fill="{MUTED}">magnetization: exact (line), Trotter (dots)</text>')
    return svg(w, h, "Exact and Trotter magnetization", p)


# ---------------------------------------------------------------- assemble the decks
m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
f2, f3, f4 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}")


def deck(body_file, title, out_name, rep):
    head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>", f"<title>{title}</title>")
    assert title in head
    head = head.replace("  svg text { font-family:",
                        "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                        "  .slide table td, .slide table th { padding:5px 12px; text-align:left; }\n  svg text { font-family:")
    body = (ROOT / "tools" / body_file).read_text()
    for k, v in rep.items():
        assert k in body, k
        body = body.replace(k, v)
    n_slides = body.count('<section class="slide')
    bar = "\n".join(m1[365:373]).replace("1 / 19", f"1 / {n_slides}")
    ctrl = "\n".join(m1[393:])
    out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
    assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body, out_name
    (OUT / out_name).write_text(out)
    print("wrote media/level3/" + out_name, len(out), "bytes,", n_slides, "slides")


QV = qsim.__version__
deck("l3_m4a_slides_body.html", "Module 4 Track A Slides: A Quantum-Kernel Classifier", "m4a-slides.html", {
    "/*TITLE_FIG*/": acc_bars(430, 300), "/*FMAP_FIG*/": fmap_fig(), "/*DATA_FIG*/": data_fig(), "/*KMAT_FIG*/": kmat_fig(),
    "/*ACC_FIG*/": acc_bars(), "/*OVERLAP_FIG*/": overlap_fig(), "/*CONC_FIG*/": conc_fig(),
    "/*MQ*/": f2(ACC["moons"][0]), "/*MC*/": f2(ACC["moons"][1]), "/*AQ*/": f2(ACC["adhoc"][0]), "/*AC*/": f2(ACC["adhoc"][1]),
    "/*C6*/": f4(CONC[5][1]), "/*C8*/": f4(CONC[7][1]), "/*S8*/": f4(CONC[7][2]), "/*E8*/": f4(CONC[7][3]), "/*QSIM_V*/": QV})
deck("l3_m4b_slides_body.html", "Module 4 Track B Slides: Simulating a Spin Chain", "m4b-slides.html", {
    "/*TITLE_FIG*/": title_b(), "/*CHAIN_FIG*/": chain_fig(), "/*EXACT_FIG*/": exact_fig(), "/*STEP_FIG*/": step_fig(),
    "/*TROT_FIG*/": trot_fig(), "/*SCALING_FIG*/": scaling_fig(), "/*NOISE_FIG*/": noise_fig(),
    "/*MEX*/": f3(M_EX), "/*INF1_8*/": f3(INF1[1]), "/*INF2_8*/": f"{INF2[1]:.4f}", "/*RATIO8*/": f"{INF1[1] / INF2[1]:.0f}",
    "/*SL1*/": f2(SL1), "/*SL2*/": f2(SL2), "/*B1N*/": str(BEST["first"][0]), "/*B1E*/": f3(BEST["first"][1]),
    "/*B2N*/": str(BEST["second"][0]), "/*B2E*/": f3(BEST["second"][1]), "/*QSIM_V*/": QV})

src = (ROOT / "tools/l3_m4_explore_src.html").read_text()
core = (ROOT / "tools/l3_m4_explore_core.js").read_text()
data_js = json.dumps({k: {kk: DATA[k][kk] for kk in ("X_train", "y_train", "X_test", "y_test")} for k in ("moons", "adhoc")},
                     separators=(",", ":"))
exp = src.replace("/*CORE*/", core).replace("/*DATA*/", data_js)
assert "`" not in exp and "${" not in exp and "—" not in exp and "/*CORE*/" not in exp and "/*DATA*/" not in exp
(OUT / "m4-explore.html").write_text(exp)
print("wrote media/level3/m4-explore.html", len(exp))
print("ACC", ACC, "CONC6/8", CONC[5][1], CONC[7][1], "M_EX", M_EX, "INF1", INF1, "INF2", INF2, "BEST", BEST)
