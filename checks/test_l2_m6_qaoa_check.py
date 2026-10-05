"""Tests for checks/l2_m6_qaoa_check.py, with real Qiskit and with qsim (as in the browser).
Run: PYTHONPATH=content/level2:checks python checks/test_l2_m6_qaoa_check.py          (Qiskit if installed)
     PYTHONPATH=content/level2:checks python checks/test_l2_m6_qaoa_check.py qsim     (force qsim)"""
import math
import sys

QSIM = len(sys.argv) > 1 and sys.argv[1] == "qsim"
if QSIM:
    sys.modules["qiskit"] = None
import numpy as np
import l2_m6_qaoa_check as chk
from l2_m6_qaoa_check import check_l2_m6_qaoa, QuantumCircuit, PROJECT_EDGES, REFERENCE_ANGLES

print("checker uses", QuantumCircuit.__module__)
if QSIM:
    import qsim as nmod
    from qsim import AerSimulator
else:
    import qiskit_aer.noise as nmod
    from qiskit_aer import AerSimulator


def course_noise_model(p2=0.01, readout=0.02):
    nm = nmod.NoiseModel()
    nm.add_all_qubit_quantum_error(nmod.depolarizing_error(p2 / 10, 1), chk.ONE_QUBIT_GATES)
    nm.add_all_qubit_quantum_error(nmod.depolarizing_error(p2, 2), chk.TWO_QUBIT_GATES)
    nm.add_all_qubit_readout_error(nmod.ReadoutError([[1 - readout, readout], [2 * readout, 1 - 2 * readout]]))
    return nm


def sig(p, shots): return math.sqrt(p * (1 - p) / shots)
def w3(k, shots, p): return abs(k / shots - p) <= 3 * sig(p, shots)
def dsig(f1, f2, shots): return math.sqrt(sig(f1, shots) ** 2 + sig(f2, shots) ** 2)
def dsig_bad(f1, f2, shots): return sig(f1, shots) + sig(f2, shots)


def measured(qc):
    c = qc.copy()
    c.measure_all()
    return c


def q_ok(gammas, betas, edges, n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.rzz(2 * g, i, j)
        qc.rx(2 * b, range(n))
    return qc


def q_ok_barriers(gammas, betas, edges, n):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for k in range(len(gammas)):
        qc.barrier()
        for i, j in edges:
            qc.rzz(2 * gammas[k], j, i)            # rzz is symmetric
        qc.barrier()
        for q in range(n):
            qc.rx(2 * betas[k], q)
    return qc


def q_cx(gammas, betas, edges, n):                # rzz as cx, rz, cx: same state, twice the two-qubit gates
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.cx(i, j); qc.rz(2 * g, j); qc.cx(i, j)
        qc.rx(2 * b, range(n))
    return qc


def q_half(gammas, betas, edges, n): return q_ok([g / 2 for g in gammas], [b / 2 for b in betas], edges, n)
def q_measured(gammas, betas, edges, n): return measured(q_ok(gammas, betas, edges, n))
def q_first_layer(gammas, betas, edges, n): return q_ok(gammas[:1], betas[:1], edges, n)
def q_todo(gammas, betas, edges, n): raise NotImplementedError


def c_ok(counts, edges):
    total = 0
    for key, v in counts.items():
        total += v * sum(key[-1 - i] != key[-1 - j] for i, j in edges)
    return total / sum(counts.values())


def c_ok_int(counts, edges):
    return sum(v * sum(((int(k, 2) >> i) & 1) != ((int(k, 2) >> j) & 1) for i, j in edges) for k, v in counts.items()) / sum(counts.values())


def c_left(counts, edges):
    return sum(v * sum(k[i] != k[j] for i, j in edges) for k, v in counts.items()) / sum(counts.values())


def c_unweighted(counts, edges):
    return float(np.mean([sum(k[-1 - i] != k[-1 - j] for i, j in edges) for k in counts]))


def c_total(counts, edges):
    return sum(v * sum(k[-1 - i] != k[-1 - j] for i, j in edges) for k, v in counts.items())


def c_todo(counts, edges): raise NotImplementedError


def counts_for(layers, p2, ro, seed, circuit=q_ok, shots=4000, noise=True):
    g, b = REFERENCE_ANGLES[layers]
    sim = AerSimulator(noise_model=course_noise_model(p2, ro)) if noise else AerSimulator()
    return sim.run(measured(circuit(g, b, PROJECT_EDGES, 5)), shots=shots, seed_simulator=seed).result().get_counts()


results = []


def expect(label, args, want):
    passed, msgs, value = check_l2_m6_qaoa(*args)
    ok = passed == want
    results.append(ok)
    print(("OK   " if ok else "FAIL ") + f"{label}: passed={passed} value={value}")
    if not ok or not passed:
        print("      " + msgs[-1][:230])


for L, p2, ro in [(1, 0.010, 0.020), (2, 0.005, 0.005), (3, 0.030, 0.030), (2, 0.021, 0.015)]:
    c = counts_for(L, p2, ro, 21)
    print(f"\n--- LAYERS {L}, P2 {p2}, READOUT {ro}: expected cut {chk.expected_cut(L, p2, ro):.6f} "
          f"(sigma {chk.cut_sigma(L, p2, ro):.5f}), P(max cut) {chk.expected_max_cut_probability(L, p2, ro):.4f}")
    expect("all correct", (q_ok, c_ok, sig, w3, dsig, L, p2, ro, c), True)
    expect("barriers, integer bits", (q_ok_barriers, c_ok_int, sig, w3, dsig, L, p2, ro, counts_for(L, p2, ro, 12, q_ok_barriers)), True)
    for name, q, cf in [("rzz as cx rz cx", q_cx, c_ok), ("half angles", q_half, c_ok), ("measured", q_measured, c_ok),
                        ("first layer only", q_first_layer, c_ok), ("qaoa todo", q_todo, c_ok),
                        ("cut from the left", q_ok, c_left), ("cut unweighted", q_ok, c_unweighted),
                        ("cut total", q_ok, c_total), ("cut todo", q_ok, c_todo)]:
        expect(name, (q, cf, sig, w3, dsig, L, p2, ro, c), False)
    expect("statistics wrong", (q_ok, c_ok, sig, w3, dsig_bad, L, p2, ro, c), False)
    other = 1 if L != 1 else 3
    expect("counts for other layers", (q_ok, c_ok, sig, w3, dsig, L, p2, ro, counts_for(other, p2, ro, 11)), False)
    e_ideal = chk.expected_cut(L, 0, 0)
    if abs(e_ideal - chk.expected_cut(L, p2, ro)) > 5 * chk.cut_sigma(L, p2, ro):
        expect("counts without noise", (q_ok, c_ok, sig, w3, dsig, L, p2, ro, counts_for(L, p2, ro, 11, noise=False)), False)
    expect("LAYERS 4", (q_ok, c_ok, sig, w3, dsig, 4, p2, ro, c), False)

fails = sum(not check_l2_m6_qaoa(q_ok, c_ok, sig, w3, dsig, 2, 0.01, 0.02, counts_for(2, 0.01, 0.02, 700 + s))[0] for s in range(200))
print(f"\ncorrect runs failing the 3-sigma counts test: {fails} of 200")
print(f"{sum(results)} of {len(results)} verdicts as expected")
sys.exit(0 if all(results) else 1)
