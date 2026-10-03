"""qsim's QuantumCircuit.compose against Qiskit's, for the Module 6 lab.

300 random pairs of circuits are composed with random qubit and classical-bit maps, at the end
or at the front, and the composed circuits are compared: the unitary (Operator) for circuits
without measurements, and the exact result distribution for circuits with final measurements.

Run: PYTHONPATH=content/level1 python tests/test_compose_vs_qiskit.py
"""
import numpy as np
from qiskit import QuantumCircuit as QQC
from qiskit.quantum_info import Operator as QOp
from qiskit.quantum_info import Statevector as QSV

import qsim

rng = np.random.default_rng(66)
ONE = ["h", "x", "s", "t", "ry", "rz"]
TWO = ["cx", "cz"]


def rand_ops(n, depth):
    ops = []
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
            ops.append((str(rng.choice(TWO)), [a, b], None))
        else:
            g = str(rng.choice(ONE))
            ops.append((g, [int(rng.integers(n))], float(rng.uniform(-3, 3)) if g in ("ry", "rz") else None))
    return ops


def build(cls, n, ops, nc=0):
    qc = cls(n, nc) if nc else cls(n)
    for g, qs, p in ops:
        getattr(qc, g)(*([p] if p is not None else []), *qs)
    return qc


worst, cases = 0.0, 0
for trial in range(300):
    n = int(rng.integers(2, 5))
    m = int(rng.integers(1, n + 1))
    base, other = rand_ops(n, int(rng.integers(1, 8))), rand_ops(m, int(rng.integers(1, 8)))
    qmap = [int(x) for x in rng.choice(n, size=m, replace=False)] if rng.random() < 0.7 else None
    if qmap is None and m != n:
        qmap = list(range(m))
    front = bool(rng.random() < 0.3)
    q = build(qsim.QuantumCircuit, n, base).compose(build(qsim.QuantumCircuit, m, other), qubits=qmap, front=front)
    a = build(QQC, n, base).compose(build(QQC, m, other), qubits=qmap, front=front)
    d = float(np.max(np.abs(qsim.Operator(q).data - QOp(a).data)))
    worst = max(worst, d)
    cases += 1
print(f"Operator of composed circuits, {cases} random cases: max |difference| = {worst:.1e}")
assert worst < 1e-12

# with classical bits: measured circuits composed onto a larger circuit with a clbit map
worst = 0.0
for trial in range(100):
    n = int(rng.integers(2, 5))
    m = int(rng.integers(1, n + 1))
    base, other = rand_ops(n, 5), rand_ops(m, 5)
    qmap = [int(x) for x in rng.choice(n, size=m, replace=False)]
    cmap = [int(x) for x in rng.choice(n, size=m, replace=False)]
    oq, oa = build(qsim.QuantumCircuit, m, other, m), build(QQC, m, other, m)
    oq.measure(list(range(m)), list(range(m)))
    oa.measure(list(range(m)), list(range(m)))
    q = build(qsim.QuantumCircuit, n, base, n).compose(oq, qubits=qmap, clbits=cmap)
    a = build(QQC, n, base, n).compose(oa, qubits=qmap, clbits=cmap)
    # exact distribution of the classical register: from the state before the measurements
    sq = q.copy(); sq.remove_final_measurements()
    sa = a.copy(); sa.remove_final_measurements()
    pq, pa = qsim.Statevector(sq).probabilities(), QSV(sa).probabilities()
    assert [(i.qubits[0], i.clbits[0]) for i in q.data if i.name == "measure"] == \
           [(a.find_bit(i.qubits[0]).index, a.find_bit(i.clbits[0]).index) for i in a.data if i.operation.name == "measure"]
    worst = max(worst, float(np.max(np.abs(pq - pa))))
print(f"composed circuits with measurements and clbit maps, 100 cases: same measure targets; max |p difference| = {worst:.1e}")
assert worst < 1e-12

# in-place and errors
c = qsim.QuantumCircuit(2)
assert c.compose(qsim.QuantumCircuit(1), inplace=True) is None
try:
    qsim.QuantumCircuit(1).compose(qsim.QuantumCircuit(2))
    raise SystemExit("expected an error")
except qsim.CircuitError as e:
    print("bigger circuit:", e)
print("ALL PASSED")
