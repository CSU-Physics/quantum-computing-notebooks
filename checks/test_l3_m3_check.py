"""Tests for checks/l3_m3_check.py: correct and wrong learner functions, and the noise against Qiskit Aer.

Run from the repository root:  python3 checks/test_l3_m3_check.py
Needs qiskit and qiskit-aer for the last part (skipped if they are missing).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "content", "level3"), HERE]
import l3_m3_check as c  # noqa: E402

R = (c.reference_idle_block, c.reference_mitigate_readout, c.reference_fold_cx, c.reference_extrapolate_zero)


# ---------------------------------------------------------------- correct variants
def idle_barrier(qc, n, dd=False):
    for k in range(n):
        if dd and k == n // 2:
            qc.x(1)
            qc.x(0)
        qc.id(0)
        qc.id(1)
        qc.barrier(0, 1)
    if dd:
        qc.x(0)
        qc.x(1)


def idle_by_qubit(qc, n, dd=False):
    for q in (0, 1):
        for k in range(n):
            if dd and k == n // 2:
                qc.x(q)
            qc.id(q)
        if dd:
            qc.x(q)


def mitigate_inv(probs, A):
    v = np.array([probs[k] for k in ("00", "01", "10", "11")])
    t = np.linalg.inv(A) @ v
    return dict(zip(("00", "01", "10", "11"), t))


def fold_loop(qc, scale):
    new = qc.copy_empty_like()
    for inst in qc.data:
        if inst.name == "cx":
            a, b = inst.qubits
            for _ in range(scale):
                new.cx(a, b)
        else:
            new.append(inst)
    return new


def extrap_lstsq(scales, values, degree=1):
    X = np.vander(np.asarray(scales, float), degree + 1)
    return float(np.linalg.lstsq(X, np.asarray(values, float), rcond=None)[0][-1])


def idle_x_first(qc, n, dd=False):
    """X before the idle period and halfway: also a valid echo."""
    if dd:
        qc.x(0)
        qc.x(1)
    for k in range(n):
        if dd and k == n // 2:
            qc.x(0)
            qc.x(1)
        qc.id(0)
        qc.id(1)


GOOD = [R, (idle_barrier,) + R[1:], (idle_by_qubit, mitigate_inv, fold_loop, extrap_lstsq), (idle_x_first,) + R[1:]]


# ---------------------------------------------------------------- wrong variants
def idle_no_dd(qc, n, dd=False):
    for _ in range(n):
        qc.id(0)
        qc.id(1)


def idle_one_x(qc, n, dd=False):
    for k in range(n):
        if dd and k == n // 2:
            qc.x(0)
            qc.x(1)
        qc.id(0)
        qc.id(1)


def idle_x_start(qc, n, dd=False):
    for k in range(n):
        if dd and k in (0, n // 2):
            qc.x(0)
            qc.x(1)
        qc.id(0)
        qc.id(1)
        if dd and k == n - 1:
            qc.x(0)
def idle_off_center(qc, n, dd=False):
    for k in range(n):
        if dd and k == 1:
            qc.x(0)
            qc.x(1)
        qc.id(0)
        qc.id(1)
    if dd:
        qc.x(0)
        qc.x(1)


def idle_too_many(qc, n, dd=False):
    for _ in range(n + 1):
        qc.id(0)
        qc.id(1)


def idle_h(qc, n, dd=False):
    for k in range(n):
        if dd and k == n // 2:
            qc.y(0)
            qc.y(1)
        qc.id(0)
        qc.id(1)
    if dd:
        qc.y(0)
        qc.y(1)


def mit_multiply(probs, A):
    v = np.array([probs[k] for k in ("00", "01", "10", "11")])
    return dict(zip(("00", "01", "10", "11"), A @ v))


def mit_transpose(probs, A):
    v = np.array([probs[k] for k in ("00", "01", "10", "11")])
    return dict(zip(("00", "01", "10", "11"), np.linalg.solve(A.T, v)))


def mit_order(probs, A):
    v = np.array([probs[k] for k in ("00", "10", "01", "11")])
    return dict(zip(("00", "10", "01", "11"), np.linalg.solve(A, v)))


def mit_identity(probs, A):
    return dict(probs)


def fold_inplace(qc, scale):
    data = list(qc.data)
    qc.data = []
    for inst in data:
        for _ in range(scale if inst.name == "cx" else 1):
            qc.data.append(inst)
    return qc


def fold_even(qc, scale):
    return c.reference_fold_cx(qc, scale + 1)


def fold_end(qc, scale):
    new = qc.copy_empty_like()
    for inst in qc.data:
        new.append(inst)
    a = [i for i in qc.data if i.name == "cx"][0]
    for _ in range(scale - 1):
        new.cx(*a.qubits)
    return new


def ex_at_one(scales, values, degree=1):
    return float(values[0])


def ex_first_coef(scales, values, degree=1):
    return float(np.polyfit(scales, values, degree)[0])


def ex_no_degree(scales, values, degree=1):
    return float(np.polyfit(scales, values, 1)[-1])


BAD = [
    ((idle_no_dd,) + R[1:], "two on each qubit"),
    ((idle_one_x,) + R[1:], "two on each qubit"),
    ((idle_x_start,) + R[1:], "two on each qubit"),
    ((idle_off_center,) + R[1:], "do not cancel"),
    ((idle_too_many,) + R[1:], "id gates on qubit"),
    ((idle_h,) + R[1:], "only id gates"),
    ((R[0], mit_multiply) + R[2:], "multiplies by A"),
    ((R[0], mit_transpose) + R[2:], "transposed"),
    ((R[0], mit_order) + R[2:], "swapped"),
    ((R[0], mit_identity) + R[2:], "unchanged"),
    (R[:2] + (fold_inplace, R[3]), "changed the circuit"),
    (R[:2] + (fold_even, R[3]), "CNOTs in a row"),
    (R[:2] + (fold_end, R[3]), "order"),
    (R[:3] + (ex_at_one,), "value at scale 1"),
    (R[:3] + (ex_first_coef,), "highest power first"),
    (R[:3] + (ex_no_degree,), "ignores degree"),
]


def main():
    wrong = 0
    for fns in GOOD:
        ok, msgs, val = c.check_l3_module3(*fns, "0.05", "0.02", "0.02")
        if not ok or val != 460:
            wrong += 1
            print("GOOD variant failed:", [f.__name__ for f in fns], msgs[-1:])
    for fns, hint in BAD:
        ok, msgs, val = c.check_l3_module3(*fns, "0.05", "0.02", "0.02")
        if ok or hint not in msgs[-1]:
            wrong += 1
            print("BAD variant not caught as expected:", [f.__name__ for f in fns], ok, msgs[-1:])
    print(f"{len(GOOD)} correct and {len(BAD)} wrong variants: {wrong} wrong verdicts")

    # exact probabilities against qsim's own noisy distribution (readout in the noise model)
    import qsim
    worst = 0.0
    for args in [(0.07, 0.03, 0.04), (0.13, 0.05, 0.07)]:
        nm = c.noise_model(*args)
        for s in c.SCALES:
            qc = c.reference_fold_cx(c.experiment(True), s)
            p = c.exact_probabilities(qc, nm, args[1])
            d, _ = qsim._noisy_distribution(qc, nm)
            worst = max(worst, max(abs(p[k] - d[i]) for i, k in enumerate(c.KEYS)))
    print(f"exact probabilities against qsim's noisy distribution: worst difference {worst:.1e}")

    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer import noise as an
    except ImportError:
        print("qiskit / qiskit-aer not installed: Aer comparison skipped")
        return wrong
    def aer_nm(delta, p_ro, p_cx):
        nm = an.NoiseModel()
        idle = an.thermal_relaxation_error(c.T1, c.T2, c.T_ID).compose(an.coherent_unitary_error(c.rz_matrix(delta)))
        nm.add_all_qubit_quantum_error(idle, ["id"])
        nm.add_all_qubit_quantum_error(an.depolarizing_error(c.P1, 1), ["h", "x"])
        nm.add_all_qubit_quantum_error(an.depolarizing_error(p_cx, 2), ["cx"])
        nm.add_all_qubit_readout_error(an.ReadoutError(c.readout_matrix(p_ro)))
        return nm

    def to_qiskit(qc):
        q = QuantumCircuit(2, 2)
        for inst in qc.data:
            if inst.name == "measure":
                q.measure(inst.qubits[0], inst.clbits[0])
            elif inst.name == "barrier":
                q.barrier(*inst.qubits)
            else:
                getattr(q, inst.name)(*inst.qubits)
        return q
    args = (0.05, 0.02, 0.02)
    nm = aer_nm(*args)
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    worst = 0.0
    for s in c.SCALES:
        qc = c.reference_fold_cx(c.experiment(True), s)
        q = to_qiskit(qc).remove_final_measurements(inplace=False)
        q.save_density_matrix()
        rho = np.asarray(sim.run(q).result().data()["density_matrix"])
        p = np.real(np.diag(rho))
        Rm = c.readout_matrix(args[1])
        p = np.kron(Rm, Rm).T @ p
        exact = c.exact_probabilities(qc, c.noise_model(*args), args[1])
        worst = max(worst, max(abs(p[i] - exact[k]) for i, k in enumerate(c.KEYS)))
    print(f"Aer density matrix with the same readout matrices against qsim exact: worst difference {worst:.1e}")
    shots = 2000000
    vals = []
    for s in c.SCALES:
        qc = c.reference_fold_cx(c.experiment(True), s)
        counts = sim.run(to_qiskit(qc), shots=shots, seed_simulator=11 + s).result().get_counts()
        p = {k: counts.get(k, 0) / shots for k in c.KEYS}
        exact = c.exact_probabilities(qc, c.noise_model(*args), args[1])
        vals.append((c.parity_expectation(p), c.parity_expectation(exact)))
    raw = sim.run(to_qiskit(c.experiment(False)), shots=shots, seed_simulator=5).result().get_counts()
    e_raw = c.parity_expectation({k: raw.get(k, 0) / shots for k in c.KEYS})
    print("Aer sampling (2,000,000 shots) against qsim exact, <XX>:",
          f"raw {e_raw:.4f} vs {c.pipeline(*args)['raw']:.4f};",
          "; ".join(f"scale {s} (DD) {a:.4f} vs {b:.4f}" for s, (a, b) in zip(c.SCALES, vals)))
    return wrong


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
