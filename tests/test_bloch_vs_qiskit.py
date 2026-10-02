"""Check qsim's Bloch vectors against Qiskit's partial trace, on random 1- to 3-qubit circuits.

Run with: PYTHONPATH=content/level1 python tests/test_bloch_vs_qiskit.py (needs qiskit).
"""
import numpy as np
import qsim
import qiskit
from qiskit.quantum_info import Statevector as QStatevector, partial_trace

rng = np.random.default_rng(2)
P = (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]))
worst = 0.0
for trial in range(200):
    n = int(rng.integers(1, 4))
    a, b = qsim.QuantumCircuit(n), qiskit.QuantumCircuit(n)
    for _ in range(8):
        g = rng.choice(["h", "s", "sdg", "x", "ry", "rz", "p", "cx"])
        q = int(rng.integers(0, n))
        if g == "cx" and n > 1:
            t = int((q + 1 + rng.integers(0, n - 1)) % n)
            a.cx(q, t); b.cx(q, t)
        elif g in ("ry", "rz", "p"):
            th = float(rng.uniform(0, 6.28))
            getattr(a, g)(th, q); getattr(b, g)(th, q)
        elif g != "cx":
            getattr(a, g)(q); getattr(b, g)(q)
    mine = qsim.bloch_vectors(qsim.Statevector(a))
    psi = QStatevector(b)
    for q in range(n):
        red = partial_trace(psi, [j for j in range(n) if j != q]).data if n > 1 else psi.to_operator().data
        ref = [float(np.real(np.trace(red @ m))) for m in P]
        worst = max(worst, float(np.max(np.abs(np.array(mine[q]) - ref))))
    # one qubit: plot_bloch_vector's formula agrees too
    if n == 1:
        sv = qsim.Statevector(a).data
        worst = max(worst, float(np.max(np.abs(np.array(qsim._bloch_of(sv)) - np.array(mine[0])))))
print(f"Bloch vectors on 200 random circuits: max difference from Qiskit {worst:.2e}")
assert worst < 1e-12
