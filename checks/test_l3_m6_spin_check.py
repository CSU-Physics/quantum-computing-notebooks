"""Tests for l3_m6_spin_check.py. Run from the repository root: python3 checks/test_l3_m6_spin_check.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content", "level3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsim import QuantumCircuit  # noqa: E402
import l3_m6_spin_check as c  # noqa: E402
from test_l3_m6_stats import run  # noqa: E402

TC, MG, CR, FR, EL = (c.reference_trotter_circuit, c.reference_magnetization, c.reference_correct_readout_z,
                      c.reference_fold_rzz, c.reference_extrapolate_linear)


def trotter(N, h, T, steps, zz=-2, xx=-2, order="zx", skip_last_rx=False):
    qc = QuantumCircuit(N)
    dt = T / steps
    for s in range(steps):
        parts = {"z": lambda: [qc.rzz(zz * dt, i, i + 1) for i in range(N - 1)],
                 "x": lambda: [qc.rx(xx * h * dt, i) for i in range(N)]}
        for p in order:
            if p == "x" and skip_last_rx and s == steps - 1:
                continue
            parts[p]()
    return qc


def mag_alt(probs):
    n = len(next(iter(probs)))
    tot = sum(probs.values())
    z = [sum(v * (1 if k[n - 1 - i] == "0" else -1) for k, v in probs.items()) / tot for i in range(n)]
    return float(np.mean(z))


def fold_alt(qc, scale):
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.name == "rzz":
            th = inst.params[0]
            q = inst.qubits
            for k in range(scale):
                out.rzz(th if k % 2 == 0 else -th, q[0], q[1])
        else:
            out.data.append(inst)
    return out


def fold_all(qc, scale):
    out = qc.copy_empty_like()
    for inst in qc.data:
        th = float(inst.params[0])
        a = inst.qubits
        g = out.rzz if inst.name == "rzz" else (lambda t, q: out.rx(t, q))
        g(th, *a)
        for _ in range((scale - 1) // 2):
            g(-th, *a)
            g(th, *a)
    return out


def fold_global_like(qc, scale):
    out = qc.copy()
    for _ in range((scale - 1) // 2):
        out.compose(qc.inverse(), inplace=True)
        out.compose(qc, inplace=True)
    return out


def fold_in_place(qc, scale):
    new = []
    for inst in qc.data:
        new.append(inst)
        if inst.name == "rzz":
            for _ in range((scale - 1) // 2):
                qc.rzz(-float(inst.params[0]), *inst.qubits)
                new.append(qc.data.pop())
                qc.rzz(float(inst.params[0]), *inst.qubits)
                new.append(qc.data.pop())
    qc.data = new
    return qc


def fold_at_end(qc, scale):
    out = qc.copy()
    for inst in qc.data:
        if inst.name == "rzz":
            for _ in range((scale - 1) // 2):
                out.rzz(-float(inst.params[0]), *inst.qubits)
                out.rzz(float(inst.params[0]), *inst.qubits)
    return out


def ext_formula(s, v):
    s, v = np.asarray(s, float), np.asarray(v, float)
    b = np.sum((s - s.mean()) * (v - v.mean())) / np.sum((s - s.mean()) ** 2)
    return float(v.mean() - b * s.mean())


GOOD = [(TC, MG, CR, FR, EL), (lambda N, h, T, st: trotter(N, h, T, st), mag_alt, lambda z, r: (z - r) / (1 - 3 * r), fold_alt,
                               ext_formula),
        (TC, MG, lambda z, r: (z - (2 * r - r)) / (1 - r - 2 * r), FR, lambda s, v: float(np.polyval(np.polyfit(s, v, 1), 0)))]
BAD = {
    "signs flipped": (lambda N, h, T, st: trotter(N, h, T, st, zz=2, xx=2), MG, CR, FR, EL),
    "half angles": (lambda N, h, T, st: trotter(N, h, T, st, zz=-1, xx=-1), MG, CR, FR, EL),
    "rx before rzz": (lambda N, h, T, st: trotter(N, h, T, st, order="xz"), MG, CR, FR, EL),
    "missing last rx": (lambda N, h, T, st: trotter(N, h, T, st, skip_last_rx=True), MG, CR, FR, EL),
    "dt = T": (lambda N, h, T, st: trotter(N, h, T * st, st), MG, CR, FR, EL),
    "parity not magnetization": (TC, lambda p: sum((-1) ** k.count("1") * v for k, v in p.items()) / sum(p.values()), CR, FR, EL),
    "magnetization not averaged": (TC, lambda p: MG(p) * len(next(iter(p))), CR, FR, EL),
    "magnetization not normalized": (TC, lambda p: MG(p) * sum(p.values()), CR, FR, EL),
    "readout no offset": (TC, MG, lambda z, r: z / (1 - 3 * r), FR, EL),
    "readout offset sign": (TC, MG, lambda z, r: (z + r) / (1 - 3 * r), FR, EL),
    "readout symmetric": (TC, MG, lambda z, r: z / (1 - 2 * r), FR, EL),
    "fold every gate": (TC, MG, CR, fold_all, EL),
    "global folding": (TC, MG, CR, fold_global_like, EL),
    "fold in place": (TC, MG, CR, fold_in_place, EL),
    "folds at the end": (TC, MG, CR, fold_at_end, EL),
    "extrapolate value at scale 1": (TC, MG, CR, FR, lambda s, v: v[0]),
    "extrapolate slope": (TC, MG, CR, FR, lambda s, v: float(np.polyfit(s, v, 1)[0])),
    "extrapolate through two points": (TC, MG, CR, FR, lambda s, v: v[0] - s[0] * (v[1] - v[0]) / (s[1] - s[0])),
}

if __name__ == "__main__":
    want = c.personal_value(8, 0.01, 0.02)
    assert want == 3618, want
    wrong = run("spin", c.check_l3_m6_spin, GOOD, BAD, (8, 0.01, 0.02), want,
                [(1, 0.01, 0.02), (13, 0.01, 0.02), (8, 0.04, 0.02), (8, 0.01, 0.001), (7.5, 0.01, 0.02), ("x", 1, 1)])
    sys.exit(1 if wrong else 0)
