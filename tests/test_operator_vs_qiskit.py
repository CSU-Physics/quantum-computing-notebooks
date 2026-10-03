"""Check qsim.Operator against qiskit.quantum_info.Operator on random 1- to 3-qubit circuits.

Run with: PYTHONPATH=content/level1 python tests/test_operator_vs_qiskit.py (needs qiskit).
"""
import numpy as np
import qsim
import qiskit
from qiskit.quantum_info import Operator as QOperator

rng = np.random.default_rng(3)
GATES = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "p", "cx", "cz", "swap"]
worst = 0.0
agree = 0
for trial in range(300):
    n = int(rng.integers(1, 4))
    a, b = qsim.QuantumCircuit(n), qiskit.QuantumCircuit(n)
    for _ in range(10):
        g = str(rng.choice(GATES))
        q = int(rng.integers(0, n))
        if g in ("cx", "cz", "swap"):
            if n == 1:
                continue
            t = int((q + 1 + rng.integers(0, n - 1)) % n)
            getattr(a, g)(q, t); getattr(b, g)(q, t)
        elif g in ("rx", "ry", "rz", "p"):
            th = float(rng.uniform(-6.28, 6.28))
            getattr(a, g)(th, q); getattr(b, g)(th, q)
        else:
            getattr(a, g)(q); getattr(b, g)(q)
    mine, ref = qsim.Operator(a), QOperator(b)
    worst = max(worst, float(np.max(np.abs(mine.data - ref.data))))
    # equality and equivalence must agree with Qiskit's
    other_q = QOperator(b).compose(QOperator.from_label("Z" * n)) if trial % 3 == 0 else ref
    other_m = qsim.Operator(other_q.data)
    agree += (mine == other_m) == (ref == other_q)
    agree += mine.equiv(other_m) == ref.equiv(other_q)
# global phase: rz versus p
ra, pa = qsim.QuantumCircuit(1), qsim.QuantumCircuit(1)
ra.rz(0.7, 0); pa.p(0.7, 0)
assert not (qsim.Operator(ra) == qsim.Operator(pa)) and qsim.Operator(ra).equiv(qsim.Operator(pa))
# from_label, adjoint, compose order
for lab in ["X", "Y", "Z", "H", "S", "T", "HZ", "XYI"]:
    assert np.allclose(qsim.Operator.from_label(lab).data, QOperator.from_label(lab).data), lab
A, B = qsim.Operator.from_label("H"), qsim.Operator.from_label("S")
QA, QB = QOperator.from_label("H"), QOperator.from_label("S")
assert np.allclose(A.compose(B).data, QA.compose(QB).data)
assert np.allclose(A.dot(B).data, QA.dot(QB).data)
assert np.allclose(B.adjoint().data, QB.adjoint().data)
assert A.is_unitary() and not qsim.Operator([[1, 1], [0, 1]]).is_unitary()
print(f"300 random circuits: max |difference| = {worst:.2e}; == and equiv agree with Qiskit in {agree} of 600 cases")
assert worst < 1e-12 and agree == 600
print("OK")
