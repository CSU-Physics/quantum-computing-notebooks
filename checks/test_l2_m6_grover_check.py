"""Tests for checks/l2_m6_grover_check.py, with real Qiskit and with qsim (as in the browser).
Run: PYTHONPATH=content/level2:checks python checks/test_l2_m6_grover_check.py          (Qiskit if installed)
     PYTHONPATH=content/level2:checks python checks/test_l2_m6_grover_check.py qsim     (force qsim)"""
import math
import sys

QSIM = len(sys.argv) > 1 and sys.argv[1] == "qsim"
if QSIM:
    sys.modules["qiskit"] = None                    # make "import qiskit" fail, as in the browser
import numpy as np
import l2_m6_grover_check as chk
from l2_m6_grover_check import check_l2_m6_grover, QuantumCircuit

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


# ---------------------------------------------------------------- statistics functions
def sig_ok(p, shots): return math.sqrt(p * (1 - p) / shots)
def sig_ok_np(p, shots): return float(np.sqrt(p * (1 - p)) / np.sqrt(shots))
def sig_count(p, shots): return math.sqrt(p * (1 - p) * shots)
def sig_nosqrt(p, shots): return p * (1 - p) / shots
def sig_todo(p, shots): raise NotImplementedError


def w3_ok(k, shots, p): return abs(k / shots - p) <= 3 * sig_ok(p, shots)
def w3_ok_counts(k, shots, p): return bool(abs(k - shots * p) <= 3 * math.sqrt(shots * p * (1 - p)))
def w3_onesided(k, shots, p): return (p - k / shots) <= 3 * sig_ok(p, shots)
def w3_2sigma(k, shots, p): return abs(k / shots - p) <= 2 * sig_ok(p, shots)
def w3_number(k, shots, p): return abs(k / shots - p) / sig_ok(p, shots)


def d_ok(f1, f2, shots): return math.sqrt(sig_ok(f1, shots) ** 2 + sig_ok(f2, shots) ** 2)
def d_linear(f1, f2, shots): return sig_ok(f1, shots) + sig_ok(f2, shots)
def d_minus(f1, f2, shots): return abs(sig_ok(f1, shots) - sig_ok(f2, shots))


# ---------------------------------------------------------------- Grover circuits
def mcz(qc, qubits):
    for name, qs, params in chk.reference_mcz_ops(list(qubits)):
        getattr(qc, name)(*params, *qs)


def g_ok(marked, iterations):
    n = len(marked)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
        if zeros:
            qc.x(zeros)
        mcz(qc, range(n))
        if zeros:
            qc.x(zeros)
        qc.h(range(n)); qc.x(range(n)); mcz(qc, range(n)); qc.x(range(n)); qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def g_ok_loops(marked, iterations):        # one gate at a time, with barriers
    n = len(marked)
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.h(q)
    for _ in range(iterations):
        qc.barrier()
        for q in range(n):
            if marked[::-1][q] == "0":
                qc.x(q)
        mcz(qc, list(range(n)))
        for q in range(n):
            if marked[::-1][q] == "0":
                qc.x(q)
        for q in range(n):
            qc.h(q); qc.x(q)
        mcz(qc, list(range(n)))
        for q in range(n):
            qc.x(q); qc.h(q)
    for q in range(n):
        qc.measure(q, q)
    return qc


def g_mcx(marked, iterations):             # the Module 2 mcz with mcx: a 3-qubit gate without noise
    n = len(marked)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    def mz(qs):
        if len(qs) == 2:
            qc.cz(qs[0], qs[1])
        else:
            qc.h(qs[-1]); qc.mcx(qs[:-1], qs[-1]); qc.h(qs[-1])
    for _ in range(iterations):
        zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
        if zeros:
            qc.x(zeros)
        mz(list(range(n)))
        if zeros:
            qc.x(zeros)
        qc.h(range(n)); qc.x(range(n)); mz(list(range(n))); qc.x(range(n)); qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def g_reversed(marked, iterations): return g_ok(marked[::-1], iterations)
def g_one_less(marked, iterations): return g_ok(marked, max(iterations - 1, 0)) if iterations else g_ok(marked, 0)


def g_no_diffuser(marked, iterations):
    n = len(marked)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
        if zeros:
            qc.x(zeros)
        mcz(qc, range(n))
        if zeros:
            qc.x(zeros)
    qc.measure(range(n), range(n))
    return qc


def g_no_measure(marked, iterations):
    qc = g_ok(marked, iterations)
    qc.remove_final_measurements()
    return qc


def g_measure_all(marked, iterations):      # measure_all adds a second register: bits are fine but registers differ
    n = len(marked)
    qc = g_ok(marked, iterations)
    qc.remove_final_measurements()
    qc.measure_all()
    return qc


def g_measure_reversed(marked, iterations):
    n = len(marked)
    qc = g_ok(marked, iterations)
    out = QuantumCircuit(n, n)
    for inst in qc.data:
        op = getattr(inst, "operation", inst)
        if op.name != "measure":
            out.append(op, inst.qubits) if hasattr(inst, "operation") else out.data.append(inst)
    for q in range(n):
        out.measure(q, n - 1 - q)
    return out


def g_todo(marked, iterations): raise NotImplementedError


def counts_for(marked_int, p2, ro, seed, circuit=g_ok, shots=4000):
    w = format(marked_int, "03b")
    return AerSimulator(noise_model=course_noise_model(p2, ro)).run(circuit(w, 2), shots=shots, seed_simulator=seed).result().get_counts()


PARAMS = [(5, 0.010, 0.020), (0, 0.005, 0.005), (7, 0.030, 0.030)]
results = []


def expect(label, fn_args, want_pass):
    passed, msgs, value = check_l2_m6_grover(*fn_args)
    ok = passed == want_pass
    results.append(ok)
    print(("OK   " if ok else "FAIL ") + f"{label}: passed={passed} value={value}")
    if not ok or not passed:
        print("      " + msgs[-1][:230])
    return value


for m, p2, ro in PARAMS:
    c = counts_for(m, p2, ro, 11)
    print(f"\n--- MARKED {m}, P2 {p2}, READOUT {ro}: expected {chk.expected_success(format(m, '03b'), 2, p2, ro):.6f}")
    v = expect("all correct", (g_ok, sig_ok, w3_ok, d_ok, m, p2, ro, c), True)
    expect("loops and barriers, numpy sigma, counts form", (g_ok_loops, sig_ok_np, w3_ok_counts, d_ok, m, p2, ro,
                                                            counts_for(m, p2, ro, 12, g_ok_loops)), True)
    for name, fns in [("sigma of the count", (g_ok, sig_count, w3_ok, d_ok)), ("no square root", (g_ok, sig_nosqrt, w3_ok, d_ok)),
                      ("sigma todo", (g_ok, sig_todo, w3_ok, d_ok)), ("one-sided", (g_ok, sig_ok, w3_onesided, d_ok)),
                      ("2 sigma", (g_ok, sig_ok, w3_2sigma, d_ok)), ("returns a number", (g_ok, sig_ok, w3_number, d_ok)),
                      ("difference linear", (g_ok, sig_ok, w3_ok, d_linear)), ("difference minus", (g_ok, sig_ok, w3_ok, d_minus)),
                      ("mcx", (g_mcx, sig_ok, w3_ok, d_ok)), ("reversed string", (g_reversed, sig_ok, w3_ok, d_ok)),
                      ("one iteration less", (g_one_less, sig_ok, w3_ok, d_ok)), ("no diffuser", (g_no_diffuser, sig_ok, w3_ok, d_ok)),
                      ("no measurements", (g_no_measure, sig_ok, w3_ok, d_ok)),
                      ("measurements reversed", (g_measure_reversed, sig_ok, w3_ok, d_ok)),
                      ("grover todo", (g_todo, sig_ok, w3_ok, d_ok))]:
        expect(name, (*fns, m, p2, ro, c), False)
    # counts problems
    expect("counts with 1000 shots", (g_ok, sig_ok, w3_ok, d_ok, m, p2, ro, counts_for(m, p2, ro, 3, shots=1000)), False)
    expect("counts from the ideal circuit", (g_ok, sig_ok, w3_ok, d_ok, m, p2, ro,
                                             AerSimulator().run(g_ok(format(m, '03b'), 2), shots=4000, seed_simulator=5).result().get_counts()), False)
    expect("counts for another MARKED", (g_ok, sig_ok, w3_ok, d_ok, m, p2, ro, counts_for((m + 1) % 8, p2, ro, 11)), False)
    expect("counts empty", (g_ok, sig_ok, w3_ok, d_ok, m, p2, ro, {}), False)
    expect("MARKED out of range", (g_ok, sig_ok, w3_ok, d_ok, 9, p2, ro, c), False)
    expect("P2 not from Canvas", (g_ok, sig_ok, w3_ok, d_ok, m, 0.0105, ro, c), False)
    expect("READOUT 0", (g_ok, sig_ok, w3_ok, d_ok, m, p2, 0.0, c), False)

# the measure_all version: keys carry no space with one register? report what happens
p, msgs, v = check_l2_m6_grover(g_measure_all, sig_ok, w3_ok, d_ok, 5, 0.01, 0.02, counts_for(5, 0.01, 0.02, 11, g_measure_all))
print("\nmeasure_all version:", p, msgs[-1][:200])

# how often does a correct run fail the 3-sigma rule?
fails = 0
for s in range(300):
    c = counts_for(5, 0.01, 0.02, 1000 + s)
    fails += not check_l2_m6_grover(g_ok, sig_ok, w3_ok, d_ok, 5, 0.01, 0.02, c)[0]
print(f"correct runs failing the 3-sigma counts test: {fails} of 300")
print(f"\n{sum(results)} of {len(results)} verdicts as expected")
sys.exit(0 if all(results) else 1)
