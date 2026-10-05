"""qsim 1.8.0: cswap against Qiskit 2.5.2 (operators of random circuits, and inverses) and order-finding counts against Aer 0.17.2."""
import random, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "content/level1"))
import qsim
from qiskit import QuantumCircuit as QQC, transpile as qtranspile
from qiskit.quantum_info import Operator as QOp
from qiskit_aer import AerSimulator as QAer

rng = random.Random(2027)
worst = 0.0
for trial in range(300):
    n = rng.randint(3, 6)
    a, b = qsim.QuantumCircuit(n), QQC(n)
    for _ in range(rng.randint(3, 14)):
        g = rng.choice(["h", "x", "t", "ry", "cswap", "cswap", "cx", "ccz"])
        if g in ("h", "x", "t"):
            q = rng.randrange(n); getattr(a, g)(q); getattr(b, g)(q)
        elif g == "ry":
            q, th = rng.randrange(n), rng.uniform(0, 6.3); a.ry(th, q); b.ry(th, q)
        elif g == "cx":
            c, t = rng.sample(range(n), 2); a.cx(c, t); b.cx(c, t)
        else:
            c, t1, t2 = rng.sample(range(n), 3); getattr(a, g)(c, t1, t2); getattr(b, g)(c, t1, t2)
    worst = max(worst, np.abs(qsim.Operator(a).data - QOp(b).data).max(),
                np.abs(qsim.Operator(a.inverse()).data - QOp(b.inverse()).data).max())
print("300 random circuits with cswap: largest operator difference from Qiskit", f"{worst:.1e}")
assert worst < 1e-10


def c_amod15(cls, a, power):
    """Controlled multiplication by a^power mod 15 on qubits [control, 4 targets] (course reference)."""
    U = cls(5)
    for _ in range(power):
        if a in (2, 13):
            U.cswap(0, 3, 4); U.cswap(0, 2, 3); U.cswap(0, 1, 2)
        if a in (7, 8):
            U.cswap(0, 1, 2); U.cswap(0, 2, 3); U.cswap(0, 3, 4)
        if a in (4, 11):
            U.cswap(0, 2, 4); U.cswap(0, 1, 3)
        if a in (7, 11, 13):
            for q in range(1, 5):
                U.cx(0, q)
    return U


def order_finding(cls, a, t):
    qc = cls(t + 4, t)
    qc.x(t)
    for k in range(t):
        qc.h(k)
    for k in range(t):
        qc.compose(c_amod15(cls, a, 2 ** k), qubits=[k] + list(range(t, t + 4)), inplace=True)
    inv = cls(t)
    for i in range(t // 2):
        inv.swap(i, t - 1 - i)
    for j in range(t):
        for k in range(j):
            inv.cp(-np.pi / 2 ** (j - k), k, j)
        inv.h(j)
    qc.compose(inv, qubits=list(range(t)), inplace=True)
    qc.measure(list(range(t)), list(range(t)))
    return qc

worst = 0.0
for a in (2, 4, 7, 8, 11, 13):
    shots = 100000
    ca = qsim.AerSimulator().run(order_finding(qsim.QuantumCircuit, a, 4), shots=shots, seed_simulator=5).result().get_counts()
    qb = order_finding(QQC, a, 4)
    cb = QAer().run(qtranspile(qb, QAer()), shots=shots, seed_simulator=5).result().get_counts()
    keys = set(ca) | set(cb)
    tvd = 0.5 * sum(abs(ca.get(k, 0) - cb.get(k, 0)) for k in keys) / shots
    worst = max(worst, tvd)
    print("a =", a, "results", sorted(ca), "TVD", round(tvd, 4))
assert worst < 0.01
print("order finding for N = 15 against Aer: worst total variation distance", round(worst, 4))
