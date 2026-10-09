"""Tests for l3_m6_mit_check.py. Run from the repository root: python3 checks/test_l3_m6_mit_check.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content", "level3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsim import QuantumCircuit  # noqa: E402
import l3_m6_mit_check as c  # noqa: E402
from l3_m6_common import confusion  # noqa: E402
from test_l3_m6_stats import run  # noqa: E402

G, PE, MR, FG, ZC, BS = (c.reference_ghz_circuit, c.reference_parity_expectation, c.reference_mitigate_readout,
                         c.reference_fold_global, c.reference_zne_coefficients, c.reference_bootstrap_sigma)


def ghz_star(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    for i in range(n):
        qc.h(i)
    return qc


def ghz_no_h(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def ghz_measured(n):
    qc = QuantumCircuit(n, n)
    qc.compose(G(n), inplace=True)
    for i in range(n):
        qc.measure(i, i)
    return qc


def parity_alt(probs):
    tot = 0.0
    s = 0.0
    for k, v in probs.items():
        tot += v
        s += v if sum(map(int, k)) % 2 == 0 else -v
    return s / tot


def mitigate_full(probs, readout):
    n = len(next(iter(probs)))
    Cfull = np.array([[1.0]])
    for _ in range(n):
        Cfull = np.kron(Cfull, confusion(readout).T)
    m = np.array([probs.get(format(i, f"0{n}b"), 0.0) for i in range(2 ** n)], dtype=float)
    t = np.linalg.solve(Cfull, m / m.sum())
    return {format(i, f"0{n}b"): float(x) for i, x in enumerate(t)}


def mitigate_variant(probs, readout, mat):
    n = len(next(iter(probs)))
    Cfull = np.array([[1.0]])
    for _ in range(n):
        Cfull = np.kron(Cfull, mat)
    m = np.array([probs.get(format(i, f"0{n}b"), 0.0) for i in range(2 ** n)], dtype=float)
    t = np.linalg.solve(Cfull, m / m.sum())
    return {format(i, f"0{n}b"): float(x) for i, x in enumerate(t)}


def fold_loop(qc, scale):
    out = QuantumCircuit(qc.num_qubits)
    out.compose(qc, inplace=True)
    k = 0
    while k < (scale - 1) // 2:
        out.compose(qc.inverse(), inplace=True)
        out.compose(qc, inplace=True)
        k += 1
    return out


def fold_in_place(qc, scale):
    inv = qc.inverse()
    base = qc.copy()
    for _ in range((scale - 1) // 2):
        qc.compose(inv, inplace=True)
        qc.compose(base, inplace=True)
    return qc


def fold_wrong_order(qc, scale):
    out = qc.copy()
    for _ in range((scale - 1) // 2):
        out.compose(qc, inplace=True)
        out.compose(qc.inverse(), inplace=True)
    return out


def coeff_lstsq(scales, degree):
    V = np.array([[s ** k for k in range(degree + 1)] for s in scales], dtype=float)
    out = []
    for i in range(len(scales)):
        e = np.zeros(len(scales))
        e[i] = 1
        out.append(np.linalg.lstsq(V, e, rcond=None)[0][0])
    return out


def boot_alt(counts, estimator, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(counts)
    shots = sum(counts.values())
    p = np.array([counts[k] for k in keys]) / shots
    vals = [estimator(dict(zip(keys, rng.multinomial(shots, p)))) for _ in range(n_boot)]
    return float(np.std(vals, ddof=1))


def boot_shots_list(counts, estimator, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    shots = [k for k, v in counts.items() for _ in range(v)]
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(shots), size=len(shots), replace=True)
        d = {}
        for i in pick:
            d[shots[i]] = d.get(shots[i], 0) + 1
        vals.append(estimator(d))
    return float(np.std(vals))


GOOD = [(G, PE, MR, FG, ZC, BS), (G, parity_alt, mitigate_full, fold_loop, coeff_lstsq, boot_alt),
        (G, PE, MR, FG, lambda s, d: list(np.polyfit(s, np.eye(len(s)), d)[-1]), boot_shots_list),
        (ghz_star, PE, MR, FG, ZC, BS)]   # a star of CX gates from qubit 0 gives the same results with this noise model
BAD = {
    "no final H": (ghz_no_h, PE, MR, FG, ZC, BS),
    "ghz with measurements": (ghz_measured, PE, MR, FG, ZC, BS),
    "parity not normalized": (G, lambda p: sum((-1) ** k.count("1") * v for k, v in p.items()), MR, FG, ZC, BS),
    "parity counts zeros": (G, lambda p: sum((-1) ** k.count("0") * v for k, v in p.items()) / sum(p.values()), MR, FG, ZC, BS),
    "fraction of all-zero": (G, lambda p: p.get("0" * len(next(iter(p))), 0) / sum(p.values()), MR, FG, ZC, BS),
    "symmetric readout": (G, PE, lambda p, r: mitigate_variant(p, r, np.array([[1 - r, r], [r, 1 - r]])), FG, ZC, BS),
    "confusion not transposed": (G, PE, lambda p, r: mitigate_variant(p, r, confusion(r)), FG, ZC, BS),
    "no mitigation": (G, PE, lambda p, r: {k: v / sum(p.values()) for k, v in p.items()}, FG, ZC, BS),
    "fold in place": (G, PE, MR, fold_in_place, ZC, BS),
    "fold wrong order": (G, PE, MR, fold_wrong_order, ZC, BS),
    "fold scale times": (G, PE, MR, lambda qc, s: fold_loop(qc, 2 * s - 1), ZC, BS),
    "polynomial coefficients": (G, PE, MR, FG, lambda s, d: list(np.polyfit(s, [1.0] * len(s), d)), BS),
    "last row of pinv": (G, PE, MR, FG, lambda s, d: list(np.linalg.pinv(np.vander(s, d + 1, increasing=True))[-1]), BS),
    "decreasing vander": (G, PE, MR, FG, lambda s, d: list(np.linalg.pinv(np.vander(np.array(s, float), d + 1))[0]), BS),
    "bootstrap one shot": (G, PE, MR, FG, ZC, lambda c_, e, n_boot=200, seed=0: float(np.std(
        [e(dict(zip(list(c_), np.random.default_rng(seed + i).multinomial(1, np.array(list(c_.values())) / sum(c_.values())))))
         for i in range(n_boot)], ddof=1))),
    "bootstrap no resampling": (G, PE, MR, FG, ZC, lambda c_, e, n_boot=200, seed=0: float(np.std([e(c_) for _ in range(n_boot)]))),
    "bootstrap unseeded": (G, PE, MR, FG, ZC, lambda c_, e, n_boot=200, seed=0: boot_alt(c_, e, n_boot, None)),
}

if __name__ == "__main__":
    want = c.personal_value(3, 0.01, 0.02)
    assert want == 1874, want
    wrong = run("mit", c.check_l3_m6_mit, GOOD, BAD, (3, 0.01, 0.02), want,
                [(1, 0.01, 0.02), (6, 0.01, 0.02), (3, 0.04, 0.02), (3, 0.01, 0.001), (2.5, 0.01, 0.02), ("x", 1, 1)])
    sys.exit(1 if wrong else 0)
