"""Tests for checks/l2_m6_qpe_check.py, with real Qiskit and with qsim (as in the browser).
Run: PYTHONPATH=content/level2:checks python checks/test_l2_m6_qpe_check.py          (Qiskit if installed)
     PYTHONPATH=content/level2:checks python checks/test_l2_m6_qpe_check.py qsim     (force qsim)"""
import math
import sys

QSIM = len(sys.argv) > 1 and sys.argv[1] == "qsim"
if QSIM:
    sys.modules["qiskit"] = None
import numpy as np
import l2_m6_qpe_check as chk
from l2_m6_qpe_check import check_l2_m6_qpe, QuantumCircuit

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
def w3_bad(k, shots, p): return abs(k / shots - p) <= 2 * sig(p, shots)


def inverse_qft(qc, qubits):
    for name, qs, params in chk.reference_iqft_ops(list(qubits)):
        getattr(qc, name)(*params, *qs)


def qft_forward(qc, qubits):
    qubits = list(qubits)
    n = len(qubits)
    for j in reversed(range(n)):
        qc.h(qubits[j])
        for k in reversed(range(j)):
            qc.cp(math.pi / 2 ** (j - k), qubits[k], qubits[j])
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])


def q_ok(m):
    qc = QuantumCircuit(m + 1, m)
    qc.x(m)
    qc.h(range(m))
    for k in range(m):
        qc.cp(2 * math.pi * (1 / 3) * 2 ** k, k, m)
    inverse_qft(qc, range(m))
    qc.measure(range(m), range(m))
    return qc


def q_ok_repeated(m):                   # U^(2^k) as 2^k repeated cp gates, and barriers
    qc = QuantumCircuit(m + 1, m)
    qc.x(m)
    for k in range(m):
        qc.h(k)
    qc.barrier()
    for k in range(m):
        for _ in range(2 ** k):
            qc.cp(2 * math.pi / 3, k, m)
    qc.barrier()
    inverse_qft(qc, list(range(m)))
    for k in range(m):
        qc.measure(k, k)
    return qc


def q_ok_barriers(m):                   # one gate at a time, with barriers
    qc = QuantumCircuit(m + 1, m)
    qc.x(m)
    for k in range(m):
        qc.h(k)
    qc.barrier()
    for k in range(m):
        qc.cp(2 * math.pi * 2 ** k / 3, k, m)
    qc.barrier()
    inverse_qft(qc, list(range(m)))
    for k in range(m):
        qc.measure(k, k)
    return qc


def q_forward_qft(m):
    qc = QuantumCircuit(m + 1, m)
    qc.x(m); qc.h(range(m))
    for k in range(m):
        qc.cp(2 * math.pi / 3 * 2 ** k, k, m)
    qft_forward(qc, range(m))
    qc.measure(range(m), range(m))
    return qc


def q_no_x(m):
    qc = QuantumCircuit(m + 1, m)
    qc.h(range(m))
    for k in range(m):
        qc.cp(2 * math.pi / 3 * 2 ** k, k, m)
    inverse_qft(qc, range(m))
    qc.measure(range(m), range(m))
    return qc


def q_reversed_powers(m):
    qc = QuantumCircuit(m + 1, m)
    qc.x(m); qc.h(range(m))
    for k in range(m):
        qc.cp(2 * math.pi / 3 * 2 ** (m - 1 - k), k, m)
    inverse_qft(qc, range(m))
    qc.measure(range(m), range(m))
    return qc


def q_measure_target(m):
    qc = QuantumCircuit(m + 1, m + 1)
    qc.x(m); qc.h(range(m))
    for k in range(m):
        qc.cp(2 * math.pi / 3 * 2 ** k, k, m)
    inverse_qft(qc, range(m))
    qc.measure(range(m + 1), range(m + 1))
    return qc


def q_ccx(m):
    qc = q_ok(m)
    qc2 = QuantumCircuit(m + 1, m)
    qc2.ccx(0, 1, m); qc2.ccx(0, 1, m)
    return qc2.compose(qc)


def q_todo(m): raise NotImplementedError


def e_ok(counts, m):
    shots = sum(counts.values())
    total = 0.0
    for key, c in counts.items():
        d = abs(int(key, 2) / 2 ** m - 1 / 3)
        total += c * min(d, 1 - d)
    return total / shots


def e_ok_np(counts, m):
    y = np.array([int(k, 2) for k in counts]); c = np.array(list(counts.values()))
    d = np.abs(y / 2 ** m - 1 / 3)
    return float(np.sum(c * np.minimum(d, 1 - d)) / c.sum())


def e_plain(counts, m):
    return sum(c * abs(int(k, 2) / 2 ** m - 1 / 3) for k, c in counts.items()) / sum(counts.values())


def e_reversed(counts, m):
    return e_ok({k[::-1]: v for k, v in counts.items()}, m)


def e_unweighted(counts, m):
    return float(np.mean([min(abs(int(k, 2) / 2 ** m - 1 / 3), 1 - abs(int(k, 2) / 2 ** m - 1 / 3)) for k in counts]))


def e_todo(counts, m): raise NotImplementedError


def counts_for(m, p2, ro, seed, circuit=q_ok, shots=4000):
    return AerSimulator(noise_model=course_noise_model(p2, ro)).run(circuit(m), shots=shots, seed_simulator=seed).result().get_counts()


results = []


def expect(label, args, want):
    passed, msgs, value = check_l2_m6_qpe(*args)
    ok = passed == want
    results.append(ok)
    print(("OK   " if ok else "FAIL ") + f"{label}: passed={passed} value={value}")
    if not ok or not passed:
        print("      " + msgs[-1][:230])


for m, p2, ro in [(4, 0.010, 0.020), (3, 0.005, 0.005), (6, 0.030, 0.030), (5, 0.017, 0.010)]:
    c = counts_for(m, p2, ro, 11)
    print(f"\n--- COUNT {m}, P2 {p2}, READOUT {ro}: P(best) {chk.expected_best(m, p2, ro):.6f}, error {chk.expected_error(m, p2, ro):.5f}")
    expect("all correct", (q_ok, e_ok, sig, w3, dsig, m, p2, ro, c), True)
    expect("barriers, numpy error", (q_ok_barriers, e_ok_np, sig, w3, dsig, m, p2, ro, counts_for(m, p2, ro, 12, q_ok_barriers)), True)
    expect("repeated cp (more gates)", (q_ok_repeated, e_ok, sig, w3, dsig, m, p2, ro, c), False)
    for name, q, e in [("forward QFT", q_forward_qft, e_ok), ("no X on target", q_no_x, e_ok),
                       ("reversed powers", q_reversed_powers, e_ok), ("target measured", q_measure_target, e_ok),
                       ("ccx", q_ccx, e_ok), ("qpe todo", q_todo, e_ok), ("error not circular", q_ok, e_plain),
                       ("error keys reversed", q_ok, e_reversed), ("error unweighted", q_ok, e_unweighted),
                       ("error todo", q_ok, e_todo)]:
        expect(name, (q, e, sig, w3, dsig, m, p2, ro, c), False)
    expect("statistics wrong", (q_ok, e_ok, sig, w3_bad, dsig, m, p2, ro, c), False)
    expect("counts for another COUNT", (q_ok, e_ok, sig, w3, dsig, m, p2, ro, counts_for(3 if m != 3 else 4, p2, ro, 11)), False)
    pb, pi = chk.expected_best(m, p2, ro), chk.expected_best(m, 0, 0)
    if abs(pi - pb) > 5 * sig(pb, 4000):            # only then can 4,000 shots tell the ideal run from the noisy one
        expect("counts without noise", (q_ok, e_ok, sig, w3, dsig, m, p2, ro,
                                         AerSimulator().run(q_ok(m), shots=4000, seed_simulator=3).result().get_counts()), False)
    expect("COUNT 7", (q_ok, e_ok, sig, w3, dsig, 7, p2, ro, c), False)

fails = sum(not check_l2_m6_qpe(q_ok, e_ok, sig, w3, dsig, 4, 0.01, 0.02, counts_for(4, 0.01, 0.02, 500 + s))[0] for s in range(200))
print(f"\ncorrect runs failing the 3-sigma counts test: {fails} of 200")
print(f"{sum(results)} of {len(results)} verdicts as expected")
sys.exit(0 if all(results) else 1)
