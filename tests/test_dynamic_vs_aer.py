"""qsim 1.5.0 dynamic circuits (if_test) against Qiskit 2.5.2 and Qiskit Aer 0.17.2.

1. Random dynamic circuits (mid-circuit measurements, resets, if_test with and without else,
   nested blocks): qsim's exact distribution of the recorded bits (from its branch simulator)
   against Aer's counts from 100,000 shots. Pass: total variation distance below 0.015
   (shot noise alone gives about 0.003 to 0.006 here).
2. The same circuits: qsim's own sampled counts (shot by shot, 4,000 shots) against qsim's exact
   distribution. Pass: total variation distance below 0.05.
3. Teleportation of 200 random states: the final density matrix of the receiving qubit equals the
   input state (fidelity 1 to 1e-12), with and without noise the identity check is exact.
4. A noise model with readout errors: qsim's exact distribution against Aer (200,000 shots).
Run: PYTHONPATH=content/level1 python tests/test_dynamic_vs_aer.py
"""
import sys
import numpy as np
import qsim
from qiskit import QuantumCircuit as QQC
from qiskit_aer import AerSimulator as QAer
from qiskit_aer.noise import NoiseModel as QNM, depolarizing_error as qdep, ReadoutError as QRO

rng = np.random.default_rng(2026)
GATES1 = ["h", "x", "s", "t", "sdg"]
fails = 0


def add(dst, src):
    for ci in src.data:
        dst.append(ci.operation, [dst.qubits[src.find_bit(q).index] for q in ci.qubits],
                   [dst.clbits[src.find_bit(c).index] for c in ci.clbits])


def build_pair(n, nc, steps):
    a, b = qsim.QuantumCircuit(n, nc), QQC(n, nc)

    def emit(ca, cb, depth):
        for _ in range(steps):
            r = rng.random()
            if r < 0.35:
                g = rng.choice(GATES1); q = int(rng.integers(n))
                getattr(ca, g)(q); getattr(cb, g)(q)
            elif r < 0.5:
                th = float(rng.uniform(0, 3)); q = int(rng.integers(n))
                ca.ry(th, q); cb.ry(th, q)
            elif r < 0.65 and n > 1:
                c, t = rng.choice(n, 2, replace=False); c, t = int(c), int(t)
                ca.cx(c, t); cb.cx(c, t)
            elif r < 0.8:
                q = int(rng.integers(n)); c = int(rng.integers(nc))
                ca.measure(q, c); cb.measure(q, c)
            elif r < 0.85:
                q = int(rng.integers(n)); ca.reset(q); cb.reset(q)
            elif depth < 2:
                c = int(rng.integers(nc)); v = int(rng.integers(2))
                with_else = rng.random() < 0.4
                if with_else:
                    sub_a_true, sub_b_true = qsim.QuantumCircuit(n, nc), QQC(n, nc)
                    sub_a_false, sub_b_false = qsim.QuantumCircuit(n, nc), QQC(n, nc)
                    emit(sub_a_true, sub_b_true, depth + 1)
                    emit(sub_a_false, sub_b_false, depth + 1)
                    with ca.if_test((c, v)) as ea:
                        for i in sub_a_true.data: ca.data.append(i)
                    with ea:
                        for i in sub_a_false.data: ca.data.append(i)
                    with cb.if_test((cb.clbits[c], v)) as eb:
                        add(cb, sub_b_true)
                    with eb:
                        add(cb, sub_b_false)
                else:
                    sub_a, sub_b = qsim.QuantumCircuit(n, nc), QQC(n, nc)
                    emit(sub_a, sub_b, depth + 1)
                    with ca.if_test((c, v)):
                        for i in sub_a.data: ca.data.append(i)
                    with cb.if_test((cb.clbits[c], v)):
                        add(cb, sub_b)
    emit(a, b, 0)
    for q in range(n):
        if q < nc:
            a.measure(q, q); b.measure(q, q)
    return a, b


def exact_dist(qc, nm=None):
    out = {}
    for r, bits in qsim._branch_states(qc, nm):
        w = float(np.real(np.trace(r)))
        key = qsim._format_key(list(bits), qc._cregs)
        out[key] = out.get(key, 0) + w
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def tv(p, counts):
    shots = sum(counts.values())
    keys = set(p) | set(counts)
    return 0.5 * sum(abs(p.get(k, 0) - counts.get(k, 0) / shots) for k in keys)


aer = QAer(seed_simulator=11)
worst1 = worst2 = 0
n_if = 0
for case in range(40):
    n = int(rng.integers(2, 4)); nc = n
    a, b = build_pair(n, nc, 7)
    n_if += sum(1 for i in a.data if i.name == "if_else")
    p = exact_dist(a)
    ca = aer.run(b, shots=100000).result().get_counts()
    d1 = tv(p, ca); worst1 = max(worst1, d1)
    cq = qsim.AerSimulator(seed_simulator=case).run(a, shots=4000).result().get_counts()
    d2 = tv(p, cq); worst2 = max(worst2, d2)
    if d1 > 0.015 or d2 > 0.05:
        fails += 1
        print("FAIL case", case, round(d1, 4), round(d2, 4)); print(a.draw())
print(f"1-2. 40 random dynamic circuits ({n_if} if_test blocks): worst TV vs Aer {worst1:.4f}, "
      f"qsim sampled vs exact {worst2:.4f}")

# 3. teleportation
worstf = 0
for k in range(200):
    th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
    qc = qsim.QuantumCircuit(3, 2)
    qc.ry(th, 0); qc.rz(ph, 0)
    qc.h(1); qc.cx(1, 2); qc.cx(0, 1); qc.h(0)
    qc.measure(0, 0); qc.measure(1, 1)
    with qc.if_test((1, 1)):
        qc.x(2)
    with qc.if_test((0, 1)):
        qc.z(2)
    rho = np.array(qsim.simulate_density_matrix(qc)).reshape(2, 4, 2, 4)
    r2 = np.einsum("aibi->ab", rho)
    ref = qsim.QuantumCircuit(1); ref.ry(th, 0); ref.rz(ph, 0)
    f = qsim.state_fidelity(qsim.Statevector(ref), qsim.DensityMatrix(r2))
    worstf = max(worstf, abs(1 - f))
if worstf > 1e-12:
    fails += 1
print(f"3. teleportation of 200 random states: max |1 - fidelity| = {worstf:.1e}")

# 4. noise + readout errors
nm_q = qsim.NoiseModel(); nm_a = QNM()
nm_q.add_all_qubit_quantum_error(qsim.depolarizing_error(0.04, 2), ["cx"])
nm_a.add_all_qubit_quantum_error(qdep(0.04, 2), ["cx"])
nm_q.add_all_qubit_quantum_error(qsim.depolarizing_error(0.01, 1), ["h", "x", "z"])
nm_a.add_all_qubit_quantum_error(qdep(0.01, 1), ["h", "x", "z"])
ro = [[0.97, 0.03], [0.06, 0.94]]
nm_q.add_all_qubit_readout_error(qsim.ReadoutError(ro)); nm_a.add_all_qubit_readout_error(QRO(ro))
worst4 = 0
for k in range(20):
    th = float(rng.uniform(0, np.pi))
    a, b = qsim.QuantumCircuit(3, 3), QQC(3, 3)
    for c in (a, b):
        c.ry(th, 0); c.h(1); c.cx(1, 2); c.cx(0, 1); c.h(0); c.measure(0, 0); c.measure(1, 1)
    with a.if_test((1, 1)):
        a.x(2)
    with a.if_test((0, 1)):
        a.z(2)
    with b.if_test((b.clbits[1], 1)):
        b.x(2)
    with b.if_test((b.clbits[0], 1)):
        b.z(2)
    a.measure(2, 2); b.measure(2, 2)
    p = exact_dist(a, nm_q)
    cnt = QAer(seed_simulator=k, noise_model=nm_a).run(b, shots=200000).result().get_counts()
    worst4 = max(worst4, tv(p, cnt))
if worst4 > 0.012:
    fails += 1
print(f"4. noisy teleportation with readout errors, 20 cases: worst TV vs Aer {worst4:.4f}")
print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
