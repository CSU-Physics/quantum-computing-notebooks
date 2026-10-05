"""qsim 1.7.0: ccz and mcx against Qiskit 2.5.2 (operators of random circuits) and Aer 0.17.2 (counts)."""
import random, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "content/level1"))
import qsim
from qiskit import QuantumCircuit as QQC
from qiskit.quantum_info import Operator as QOp

rng = random.Random(2026)
worst = 0.0
for trial in range(300):
    n = rng.randint(2, 6)
    a, b = qsim.QuantumCircuit(n), QQC(n)
    for _ in range(rng.randint(3, 14)):
        g = rng.choice(["h", "x", "t", "ry", "ccz", "mcx", "cz", "cx"])
        if g in ("h", "x", "t"):
            q = rng.randrange(n); getattr(a, g)(q); getattr(b, g)(q)
        elif g == "ry":
            q, th = rng.randrange(n), rng.uniform(0, 6.3); a.ry(th, q); b.ry(th, q)
        elif g in ("cz", "cx"):
            c, t = rng.sample(range(n), 2); getattr(a, g)(c, t); getattr(b, g)(c, t)
        elif g == "ccz":
            if n < 3: continue
            c1, c2, t = rng.sample(range(n), 3); a.ccz(c1, c2, t); b.ccz(c1, c2, t)
        else:
            k = rng.randint(1, n - 1)
            qs = rng.sample(range(n), k + 1); a.mcx(qs[:-1], qs[-1]); b.mcx(qs[:-1], qs[-1])
    d = np.abs(qsim.Operator(a).data - QOp(b).data).max()
    worst = max(worst, d)
    # inverse
    d2 = np.abs(qsim.Operator(a.inverse()).data - QOp(b.inverse()).data).max()
    worst = max(worst, d2)
print("300 random circuits with ccz and mcx: largest operator difference from Qiskit", f"{worst:.1e}")
assert worst < 1e-10

# Grover circuits: counts against Aer
from qiskit_aer import AerSimulator as QAer
from qiskit import transpile as qtranspile

def grover(cls, marked, its):
    n = len(marked); qc = cls(n, n)
    qc.h(list(range(n)))
    for _ in range(its):
        zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
        if zeros: qc.x(zeros)
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        if zeros: qc.x(zeros)
        qc.h(list(range(n))); qc.x(list(range(n)))
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(list(range(n))); qc.h(list(range(n)))
    qc.measure(list(range(n)), list(range(n)))
    return qc

worst = 0
for marked, its in [("11", 1), ("101", 2), ("0110", 3), ("10011", 4), ("010", 3)]:
    shots = 100000
    qa = qsim.AerSimulator().run(grover(qsim.QuantumCircuit, marked, its), shots=shots, seed_simulator=7).result().get_counts()
    bq = grover(QQC, marked, its)
    qb = QAer().run(qtranspile(bq, QAer()), shots=shots, seed_simulator=7).result().get_counts()
    keys = set(qa) | set(qb)
    tvd = 0.5 * sum(abs(qa.get(k, 0) - qb.get(k, 0)) for k in keys) / shots
    worst = max(worst, tvd)
    print(marked, its, "P(marked) qsim", qa.get(marked, 0) / shots, "Aer", qb.get(marked, 0) / shots, "TVD", round(tvd, 4))
assert worst < 0.01
print("Grover counts against Aer: worst total variation distance", round(worst, 4))
