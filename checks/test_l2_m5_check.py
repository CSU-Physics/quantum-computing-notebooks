"""Tests for checks/l2_m5_check.py, with real Qiskit and with qsim (as in the browser).
Run: PYTHONPATH=content/level2:checks python checks/test_l2_m5_check.py          (Qiskit if installed)
     PYTHONPATH=content/level2:checks python checks/test_l2_m5_check.py qsim     (force qsim)"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "qsim":
    sys.modules["qiskit"] = None                    # make "import qiskit" fail, as in the browser
import numpy as np
import l2_m5_check as chk
from l2_m5_check import check_l2_module5, QuantumCircuit, ParameterVector, SparsePauliOp

print("checker uses", QuantumCircuit.__module__)


# ---------------------------------------------------------------- correct versions
def ansatz_ok(layers):
    th = ParameterVector("θ", 2 * layers)
    qc = QuantumCircuit(2)
    qc.ry(th[0], 0); qc.ry(th[1], 1)
    for k in range(1, layers):
        qc.cx(0, 1); qc.ry(th[2 * k], 0); qc.ry(th[2 * k + 1], 1)
    return qc


def ansatz_ok_x(layers):            # another vector name, and a barrier between layers
    x = ParameterVector("x", 2 * layers)
    qc = QuantumCircuit(2)
    qc.ry(x[0], 0); qc.ry(x[1], 1)
    for k in range(1, layers):
        qc.barrier(); qc.cx(0, 1); qc.ry(x[2 * k], 0); qc.ry(x[2 * k + 1], 1)
    return qc


def ham_ok(edges, n):
    return SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0) for i, j in edges], num_qubits=n)


def ham_ok_labels(edges, n):        # building the labels by hand, the edges in another order
    labels = []
    for i, j in reversed(edges):
        lab = ["I"] * n
        lab[n - 1 - i] = "Z"; lab[n - 1 - j] = "Z"
        labels.append("".join(lab))
    return SparsePauliOp(labels)


def qaoa_ok(gammas, betas, edges, n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.rzz(2 * g, i, j)
        qc.rx(2 * b, range(n))
    return qc


def qaoa_ok_cx(gammas, betas, edges, n):   # rzz written as cx, rz, cx
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.cx(i, j); qc.rz(2 * g, j); qc.cx(i, j)
        for q in range(n):
            qc.rx(2 * b, q)
    return qc


# ---------------------------------------------------------------- wrong versions
def an_cx_reversed(layers):
    th = ParameterVector("θ", 2 * layers); qc = QuantumCircuit(2); qc.ry(th[0], 0); qc.ry(th[1], 1)
    for k in range(1, layers):
        qc.cx(1, 0); qc.ry(th[2 * k], 0); qc.ry(th[2 * k + 1], 1)
    return qc


def an_rx(layers):
    th = ParameterVector("θ", 2 * layers); qc = QuantumCircuit(2); qc.rx(th[0], 0); qc.rx(th[1], 1)
    for k in range(1, layers):
        qc.cx(0, 1); qc.rx(th[2 * k], 0); qc.rx(th[2 * k + 1], 1)
    return qc


def an_one_param_per_layer(layers):
    th = ParameterVector("θ", layers); qc = QuantumCircuit(2); qc.ry(th[0], 0); qc.ry(th[0], 1)
    for k in range(1, layers):
        qc.cx(0, 1); qc.ry(th[k], 0); qc.ry(th[k], 1)
    return qc


def an_no_cx(layers):
    th = ParameterVector("θ", 2 * layers); qc = QuantumCircuit(2); qc.ry(th[0], 0); qc.ry(th[1], 1)
    for k in range(1, layers):
        qc.ry(th[2 * k], 0); qc.ry(th[2 * k + 1], 1)
    return qc


def an_qubits_swapped(layers):     # θ[0] on qubit 1 and θ[1] on qubit 0
    th = ParameterVector("θ", 2 * layers); qc = QuantumCircuit(2); qc.ry(th[0], 1); qc.ry(th[1], 0)
    for k in range(1, layers):
        qc.cx(0, 1); qc.ry(th[2 * k], 1); qc.ry(th[2 * k + 1], 0)
    return qc


def ansatz_ok_cx_first(layers):     # a cx before layer 1 acts on |00> and changes nothing, so this is also right
    th = ParameterVector("θ", 2 * layers); qc = QuantumCircuit(2)
    for k in range(layers):
        qc.cx(0, 1); qc.ry(th[2 * k], 0); qc.ry(th[2 * k + 1], 1)
    return qc


def an_measured(layers):
    qc = ansatz_ok(layers); qc.measure_all(); return qc


def an_todo(layers):
    raise NotImplementedError


def h_cut_form(edges, n):          # the cut-counting operator (I - ZZ)/2 summed over edges
    return (SparsePauliOp.from_sparse_list([("ZZ", [i, j], -0.5) for i, j in edges], num_qubits=n)
            + SparsePauliOp("I" * n, [0.5 * len(edges)]))


def h_reversed_index(edges, n):     # qubit 0 put on the left
    labels = []
    for i, j in edges:
        lab = ["I"] * n; lab[i] = "Z"; lab[j] = "Z"; labels.append("".join(lab))
    return SparsePauliOp(labels)


def h_negative(edges, n):
    return SparsePauliOp.from_sparse_list([("ZZ", [i, j], -1.0) for i, j in edges], num_qubits=n)


def h_xx(edges, n):
    return SparsePauliOp.from_sparse_list([("XX", [i, j], 1.0) for i, j in edges], num_qubits=n)


def h_todo(edges, n):
    raise NotImplementedError


def q_half(gammas, betas, edges, n):
    qc = QuantumCircuit(n); qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.rzz(g, i, j)
        qc.rx(b, range(n))
    return qc


def q_half_beta(gammas, betas, edges, n):
    qc = QuantumCircuit(n); qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.rzz(2 * g, i, j)
        qc.rx(b, range(n))
    return qc


def q_no_h(gammas, betas, edges, n):
    qc = QuantumCircuit(n)
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.rzz(2 * g, i, j)
        qc.rx(2 * b, range(n))
    return qc


def q_only_first_layer(gammas, betas, edges, n):
    return qaoa_ok(gammas[:1], betas[:1], edges, n)


def q_mixer_first(gammas, betas, edges, n):
    qc = QuantumCircuit(n); qc.h(range(n))
    for g, b in zip(gammas, betas):
        qc.rx(2 * b, range(n))
        for i, j in edges:
            qc.rzz(2 * g, i, j)
    return qc


def q_measured(gammas, betas, edges, n):
    qc = qaoa_ok(gammas, betas, edges, n); qc.measure_all(); return qc


def q_todo(gammas, betas, edges, n):
    raise NotImplementedError


good_a, good_h, good_q = [ansatz_ok, ansatz_ok_x, ansatz_ok_cx_first], [ham_ok, ham_ok_labels], [qaoa_ok, qaoa_ok_cx]
bad_a = [an_cx_reversed, an_rx, an_one_param_per_layer, an_no_cx, an_qubits_swapped, an_measured, an_todo]
bad_h = [h_cut_form, h_reversed_index, h_negative, h_xx, h_todo]
bad_q = [q_half, q_half_beta, q_no_h, q_only_first_layer, q_mixer_first, q_measured, q_todo]
PARAMS = [(0.37, 1.21), (1.5, 0.1), (0.1, 0.88)]

wrong, values = 0, []
for g, b in PARAMS:
    for a in good_a:
        for h in good_h:
            for q in good_q:
                ok, msgs, v = check_l2_module5(a, h, q, g, b)
                if not ok:
                    wrong += 1; print("WRONG (should pass):", a.__name__, h.__name__, q.__name__, msgs)
    values.append(check_l2_module5(ansatz_ok, ham_ok, qaoa_ok, g, b)[2])
    for f in bad_a:
        ok, msgs, _ = check_l2_module5(f, ham_ok, qaoa_ok, g, b)
        if ok: wrong += 1; print("WRONG (should fail):", f.__name__)
        elif (g, b) == PARAMS[0]: print("  ", f.__name__, "->", [m for m in msgs if "passed" not in m][0][:150])
    for f in bad_h:
        ok, msgs, _ = check_l2_module5(ansatz_ok, f, qaoa_ok, g, b)
        if ok: wrong += 1; print("WRONG (should fail):", f.__name__)
        elif (g, b) == PARAMS[0]: print("  ", f.__name__, "->", [m for m in msgs if "passed" not in m][0][:150])
    for f in bad_q:
        ok, msgs, _ = check_l2_module5(ansatz_ok, ham_ok, f, g, b)
        if ok: wrong += 1; print("WRONG (should fail):", f.__name__)
        elif (g, b) == PARAMS[0]: print("  ", f.__name__, "->", [m for m in msgs if "passed" not in m][0][:150])
for bad in [(0.05, 0.5), (0.5, 1.6), (0.333, 0.5), ("x", 0.5), (None, 0.5)]:
    if check_l2_module5(ansatz_ok, ham_ok, qaoa_ok, *bad)[0]:
        wrong += 1; print("WRONG: out-of-range parameters accepted", bad)
n_good = len(good_a) * len(good_h) * len(good_q)
print(f"{n_good} correct combinations and {len(bad_a) + len(bad_h) + len(bad_q)} wrong versions at {len(PARAMS)} "
      f"parameter sets: {wrong} wrong verdicts")
print("values:", values, "expected:", [chk.personal_value(g, b) for g, b in PARAMS])
sys.exit(1 if wrong else 0)
