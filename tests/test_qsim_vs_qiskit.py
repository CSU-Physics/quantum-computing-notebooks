"""Compare qsim with Qiskit 2.5.2 / Aer 0.17.2 on random circuits."""
import math, random, sys
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "content", "level1"))
import qsim
import qiskit
from qiskit.quantum_info import Statevector as QSV, DensityMatrix as QDM, Operator
from qiskit_aer import AerSimulator as QAer

GATES1 = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"]
PAR1 = ["rx", "ry", "rz", "p"]
GATES2 = ["cx", "cy", "cz", "ch", "swap"]
PAR2 = ["cp", "crz"]

def build(n, steps, rng, allow_nonunitary=False, nc=0):
    a = qsim.QuantumCircuit(n, nc) if nc else qsim.QuantumCircuit(n)
    b = qiskit.QuantumCircuit(n, nc) if nc else qiskit.QuantumCircuit(n)
    for _ in range(steps):
        kind = rng.random()
        if kind < 0.35:
            g = rng.choice(GATES1); q = rng.randrange(n)
            getattr(a, g)(q); getattr(b, g)(q)
        elif kind < 0.55:
            g = rng.choice(PAR1); q = rng.randrange(n); t = rng.uniform(-3, 3)
            getattr(a, g)(t, q); getattr(b, g)(t, q)
        elif kind < 0.8 and n >= 2:
            g = rng.choice(GATES2); x, y = rng.sample(range(n), 2)
            getattr(a, g)(x, y); getattr(b, g)(x, y)
        elif kind < 0.9 and n >= 2:
            g = rng.choice(PAR2); x, y = rng.sample(range(n), 2); t = rng.uniform(-3, 3)
            getattr(a, g)(t, x, y); getattr(b, g)(t, x, y)
        elif n >= 3:
            x, y, z = rng.sample(range(n), 3)
            a.ccx(x, y, z); b.ccx(x, y, z)
        if allow_nonunitary and rng.random() < 0.12:
            q = rng.randrange(n)
            if rng.random() < 0.5 and nc:
                c = rng.randrange(nc); a.measure(q, c); b.measure(q, c)
            else:
                a.reset(q); b.reset(q)
    return a, b

rng = random.Random(7)
worst_sv = 0
for trial in range(300):
    n = rng.randint(1, 5)
    a, b = build(n, rng.randint(1, 25), rng)
    d = np.max(np.abs(qsim.Statevector(a).data - QSV(b).data))
    worst_sv = max(worst_sv, d)
print("statevector max abs difference over 300 random circuits:", f"{worst_sv:.2e}")

# Exact reference for mid-circuit measurement and reset: Qiskit quantum_info with Kraus channels.
# (Aer's save_density_matrix averages over shots, so it is not an exact reference.)
from qiskit.quantum_info import Kraus
_P0 = np.array([[1, 0], [0, 0]]); _P1 = np.array([[0, 0], [0, 1]]); _R1 = np.array([[0, 1], [0, 0]])


def exact_rho(b):
    rho = QDM.from_label("0" * b.num_qubits)
    for ci in b.data:
        name = ci.operation.name
        qs = [b.find_bit(q).index for q in ci.qubits]
        if name == "measure":
            rho = rho.evolve(Kraus([_P0, _P1]), qs)
        elif name == "reset":
            rho = rho.evolve(Kraus([_P0, _R1]), qs)
        elif name != "barrier":
            rho = rho.evolve(Operator(ci.operation), qs)
    return np.asarray(rho)


worst_dm = 0
for trial in range(200):
    n = rng.randint(1, 4)
    a, b = build(n, rng.randint(2, 20), rng, allow_nonunitary=True, nc=n)
    worst_dm = max(worst_dm, np.max(np.abs(qsim.simulate_density_matrix(a).data - exact_rho(b))))
print("density matrix (with mid-circuit measure/reset) max abs difference over 200 circuits:", f"{worst_dm:.2e}")

# counts: distribution agreement (total variation distance) and key format
worst_tv = 0
for trial in range(60):
    n = rng.randint(1, 4)
    nc = n
    a, b = build(n, rng.randint(2, 15), rng, allow_nonunitary=(trial % 2 == 0), nc=nc)
    if trial % 3 == 0:
        a.measure_all(); b.measure_all()
    else:
        a.measure(range(n), range(n)); b.measure(range(n), range(n))
    shots = 20000
    ca = qsim.AerSimulator(seed_simulator=1).run(a, shots=shots).result().get_counts()
    sim = QAer(seed_simulator=1)
    cb = sim.run(qiskit.transpile(b, sim), shots=shots).result().get_counts()
    keys = set(ca) | set(cb)
    assert all(len(k) == len(next(iter(cb))) for k in ca), (ca, cb)
    tv = 0.5 * sum(abs(ca.get(k, 0) - cb.get(k, 0)) / shots for k in keys)
    worst_tv = max(worst_tv, tv)
print("counts total variation distance, worst of 60 circuits at 20,000 shots:", f"{worst_tv:.3f}")

# Pauli expectation values with Qiskit label order
q = qsim.QuantumCircuit(2); q.ry(0.7, 0); q.cx(0, 1); q.rx(0.3, 1)
k = qiskit.QuantumCircuit(2); k.ry(0.7, 0); k.cx(0, 1); k.rx(0.3, 1)
from qiskit.quantum_info import SparsePauliOp
for lab in ["ZI", "IZ", "XY", "YX", "ZZ"]:
    e1 = qsim.Statevector(q).expectation_value(lab); e2 = QSV(k).expectation_value(SparsePauliOp(lab))
    assert abs(e1 - e2) < 1e-10, (lab, e1, e2)
print("Pauli expectation values match Qiskit's label order")

# error message for the planted bug
try:
    qsim.QuantumCircuit(1, 1).measure(0, 1)
except qsim.CircuitError as e:
    print("planted-bug message:", e)
try:
    qiskit.QuantumCircuit(1, 1).measure(0, 1)
except Exception as e:
    print("Qiskit's message:   ", e)
print(qsim.QuantumCircuit(3, 3).h(0).cx(0, 1).cx(1, 2).measure([0, 1, 2], [0, 1, 2]).draw())
