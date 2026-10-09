"""Tests of checks/l3_m4a_check.py: the reference feature map against Qiskit's zz_feature_map, the kernel against
Qiskit statevectors, the shot estimate against Qiskit Aer sampling of the same overlap circuit, and the check's
verdicts on correct and wrong versions of the learner's functions. Run with Qiskit 2.5.2 and Aer 0.17.2:
    cd checks && python test_l3_m4a_check.py"""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "content" / "level3"), str(Path(__file__).resolve().parent)]
import l3_m4a_check as c  # noqa: E402
from qsim import QuantumCircuit, Statevector  # noqa: E402

from qiskit import transpile  # noqa: E402
from qiskit.circuit.library import zz_feature_map, unitary_overlap  # noqa: E402
from qiskit.quantum_info import Statevector as QSV  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

rng = np.random.default_rng(5)
worst = 0.0
for n in (2, 3, 4):
    for reps in (1, 2):
        for _ in range(4):
            x = rng.uniform(0, 2 * np.pi, n)
            q = QSV(zz_feature_map(n, reps=reps).assign_parameters(x)).data
            s = Statevector(c.reference_feature_map(x, reps)).data
            worst = max(worst, np.abs(q - s).max())
print(f"feature map vs Qiskit zz_feature_map (2 to 4 qubits, reps 1 and 2): max amplitude difference {worst:.1e}")
assert worst < 1e-12

XA, XB = rng.uniform(0, 2 * np.pi, (5, 2)), rng.uniform(0, 2 * np.pi, (4, 2))
fm = zz_feature_map(2, reps=2)
Kq = np.array([[abs(np.vdot(QSV(fm.assign_parameters(a)).data, QSV(fm.assign_parameters(b)).data)) ** 2 for b in XB] for a in XA])
d = np.abs(Kq - c.reference_kernel_matrix(XA, XB)).max()
print(f"kernel matrix vs Qiskit statevectors: max difference {d:.1e}")
assert d < 1e-12

x1, x2 = XA[0], XB[0]
ov = unitary_overlap(fm.assign_parameters(x1), fm.assign_parameters(x2))
ov.measure_all()
sim = AerSimulator(seed_simulator=11)
shots = 400_000
counts = sim.run(transpile(ov, sim), shots=shots).result().get_counts()
aer = counts.get("00", 0) / shots
mine = np.mean([c.reference_kernel_entry_shots(x1, x2, shots, np.random.default_rng(s)) for s in range(5)])
exact = c.reference_kernel_matrix([x1], [x2])[0, 0]
print(f"one kernel entry: exact {exact:.5f}, Aer {aer:.5f} ({shots} shots), qsim estimate {mine:.5f}")
assert abs(aer - exact) < 4 * np.sqrt(exact * (1 - exact) / shots) + 1e-9


# ---------------------------------------------------------------- learner-style versions
def fm_good(x, reps=2):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
            qc.p(2 * x[i], i)
        for i in range(n - 1):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def km_good(XA, XB):
    K = np.zeros((len(XA), len(XB)))
    for a, xa in enumerate(XA):
        for b, xb in enumerate(XB):
            K[a, b] = abs(Statevector(fm_good(xa)).inner(Statevector(fm_good(xb)))) ** 2
    return K


def km_good2(XA, XB):
    A = [Statevector(fm_good(x)).data for x in XA]
    B = [Statevector(fm_good(x)).data for x in XB]
    return np.array([[abs(np.vdot(a, b)) ** 2 for b in B] for a in A])


def sh_good(x1, x2, shots, rng):
    qc = fm_good(x1).compose(fm_good(x2).inverse())
    p = abs(Statevector(qc).data[0]) ** 2
    return rng.binomial(shots, p) / shots


def sh_good2(x1, x2, shots, rng):
    p = km_good2([x1], [x2])[0, 0]
    return int((rng.random(shots) < p).sum()) / shots


def fm_reps1(x, reps=1):
    return fm_good(x, reps)


def fm_noH(x, reps=2):
    return _drop(fm_good(x, reps), "h")


def _drop(qc, name):
    new = qc.copy_empty_like()
    for inst in qc.data:
        if inst.name != name:
            new.append(inst)
    return new


def fm_half(x, reps=2):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
            qc.p(x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def fm_wrongq(x, reps=2):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
            qc.p(2 * x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), i)
                qc.cx(i, j)
    return qc


def fm_linear(x, reps=2):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
            qc.p(2 * x[i], i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.p(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), i + 1)
            qc.cx(i, i + 1)
    return qc


def fm_noreps(x):
    return fm_good(x)


def km_abs(XA, XB):
    return np.sqrt(km_good2(XA, XB))


def km_T(XA, XB):
    return km_good2(XB, XA)


def km_inner(XA, XB):
    A = [Statevector(fm_good(x)).data for x in XA]
    B = [Statevector(fm_good(x)).data for x in XB]
    return np.array([[np.real(np.vdot(a, b)) for b in B] for a in A])


def sh_exact(x1, x2, shots, rng):
    return km_good2([x1], [x2])[0, 0]


def sh_flip(x1, x2, shots, rng):
    return 1 - sh_good(x1, x2, shots, rng)


def sh_sqrt(x1, x2, shots, rng):
    qc = fm_good(x1).compose(fm_good(x2).inverse())
    return rng.binomial(shots, abs(Statevector(qc).data[0])) / shots


def sh_global(x1, x2, shots, rng):
    p = km_good2([x1], [x2])[0, 0]
    return np.random.default_rng(1).binomial(shots, p) / shots


def sh_noncompose(x1, x2, shots, rng):
    qc = fm_good(x1).compose(fm_good(x2))
    return rng.binomial(shots, abs(Statevector(qc).data[0]) ** 2) / shots


def raise_(*a, **k):
    raise NotImplementedError


GOOD = {"reference": (c.reference_feature_map, c.reference_kernel_matrix, c.reference_kernel_entry_shots),
        "loops": (fm_good, km_good, sh_good), "vdot": (fm_good, km_good2, sh_good2)}
BAD = {"reps=1 default": (fm_reps1, km_good2, sh_good), "no H": (fm_noH, km_good2, sh_good),
       "P(x) not P(2x)": (fm_half, km_good2, sh_good), "pair phase on wrong qubit": (fm_wrongq, km_good2, sh_good),
       "linear entanglement": (fm_linear, km_good2, sh_good), "no reps argument": (fm_noreps, km_good2, sh_good),
       "|overlap| not squared": (fm_good, km_abs, sh_good), "transposed": (fm_good, km_T, sh_good),
       "real part of overlap": (fm_good, km_inner, sh_good), "exact, no shots": (fm_good, km_good2, sh_exact),
       "1 - p": (fm_good, km_good2, sh_flip), "sqrt probability": (fm_good, km_good2, sh_sqrt),
       "ignores rng": (fm_good, km_good2, sh_global), "no inverse": (fm_good, km_good2, sh_noncompose),
       "feature_map not written": (raise_, km_good2, sh_good), "kernel_matrix not written": (fm_good, raise_, sh_good)}

wrong = 0
for name, fns in GOOD.items():
    ok, msgs, val = c.check_l3_module4a(*fns, "1.25", "0.40", "2.10")
    print(f"GOOD {name:28s} -> {ok} {val}")
    wrong += (not ok) or val != c.personal_value(1.25, 0.40, 2.10)
for name, fns in BAD.items():
    ok, msgs, val = c.check_l3_module4a(*fns, "1.25", "0.40", "2.10")
    print(f"BAD  {name:28s} -> {ok} | {msgs[-1][:150]}")
    wrong += ok
for args in [("0.05", "1", "1"), ("1.234", "1", "1"), ("x", "1", "1")]:
    ok, msgs, _ = c.check_l3_module4a(*GOOD["reference"], *args)
    wrong += ok
print("wrong verdicts:", wrong)
assert wrong == 0
