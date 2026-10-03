"""Check cell logic for the Module 5 lab (shot noise, readout errors, noise models, mitigation), on qsim.

check_module5(shot_noise, two_qubit_readout, mitigate, DEP_PCT, RO_PCT, SEED)
returns (passed, messages, verification_value).
The learner writes the three functions. The verification value is the number of shots, out of 1,000,
in which the two qubits give DIFFERENT results, for the Bell circuit H on qubit 0, CNOT 0 -> 1, measure
both, on qsim's AerSimulator with seed SEED and a noise model with a depolarizing error of DEP_PCT %
on the CNOT and a readout error of RO_PCT % on each qubit. It is printed only when every test passes,
and it depends on the seeded draw, so it cannot be worked out by hand.
"""
import math

import numpy as np
from qsim import AerSimulator, NoiseModel, QuantumCircuit, ReadoutError, depolarizing_error

LABELS = ["00", "01", "10", "11"]


def _rand_single(rng):
    e0, e1 = rng.uniform(0.0, 0.12, size=2)
    return np.array([[1 - e0, e1], [e0, 1 - e1]])


def personal_noise_model(dep_pct, ro_pct):
    lam, e = float(dep_pct) / 100, float(ro_pct) / 100
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]))
    return nm


def personal_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def personal_value(dep_pct, ro_pct, seed):
    sim = AerSimulator(seed_simulator=int(seed), noise_model=personal_noise_model(dep_pct, ro_pct))
    counts = sim.run(personal_circuit(), shots=1000).result().get_counts()
    return int(counts.get("01", 0) + counts.get("10", 0))


def predicted_different(dep_pct, ro_pct):
    lam, e = float(dep_pct) / 100, float(ro_pct) / 100
    flip_one = 2 * e * (1 - e)
    return (lam / 2) * (1 - flip_one) + (1 - lam / 2) * flip_one


def check_module5(shot_noise, two_qubit_readout, mitigate, DEP_PCT, RO_PCT, SEED):
    msgs = []

    # Test 1: shot_noise(p, shots) = sqrt(p (1 - p) / shots), the standard deviation of a measured fraction.
    for p, n in [(0.5, 1000), (0.1, 400), (0.97, 1000), (0.0, 1000), (1.0, 50), (0.25, 10000), (0.5, 100)]:
        want = math.sqrt(p * (1 - p) / n)
        try:
            got = float(shot_noise(p, n))
        except Exception as exc:  # noqa: BLE001
            return False, [f"shot_noise({p}, {n}) stopped with an error: {exc}"], None
        if not abs(got - want) <= 1e-9 + 1e-9 * want:
            hint = "Use np.sqrt(p * (1 - p) / shots): the spread of a fraction shrinks as 1 / sqrt(shots)."
            if abs(got - want ** 2) < 1e-12 and want > 0:
                hint = "You returned the variance. Take the square root. " + hint
            elif abs(got - math.sqrt(p * (1 - p) * n)) < 1e-9 and want > 0:
                hint = "That is the spread of the count, not of the fraction. Divide by shots, not multiply. " + hint
            return False, [f"shot_noise({p}, {n}) gave {got:.6f}; it should be {want:.6f}. {hint}"], None
    msgs.append("Test 1, shot_noise(): passed")

    # Test 2: two_qubit_readout(A0, A1) = np.kron(A1, A0), in Qiskit's order.
    rng = np.random.default_rng(51)
    for _ in range(6):
        a0, a1 = _rand_single(rng), _rand_single(rng)
        want = np.kron(a1, a0)
        try:
            got = np.asarray(two_qubit_readout(a0.copy(), a1.copy()), dtype=float)
        except Exception as exc:  # noqa: BLE001
            return False, [f"two_qubit_readout() stopped with an error: {exc}"], None
        if got.shape != (4, 4):
            return False, ["two_qubit_readout() must return a 4 x 4 matrix (rows and columns in the order 00, 01, 10, 11)."], None
        if not np.allclose(got, want, atol=1e-12):
            hint = "Use np.kron(A1, A0): qubit 1 is the left bit of each label, so its matrix goes first (as in Module 4)."
            if np.allclose(got, np.kron(a0, a1), atol=1e-12):
                hint = "The qubits are the wrong way round. " + hint
            elif np.allclose(got, want.T, atol=1e-12):
                hint = "You have the transpose. Columns are the true results and rows the recorded ones. " + hint
            return False, [f"two_qubit_readout() gave the wrong matrix. {hint}"], None
    msgs.append("Test 2, two_qubit_readout(): passed, in Qiskit's order")

    # Test 3: mitigate(counts, A) solves A x = measured probabilities and keeps every quasi-probability.
    cases = []
    for k in range(6):
        a = np.kron(_rand_single(rng), _rand_single(rng))
        raw = rng.integers(0, 400, size=4)
        if k == 0:
            raw = np.array([480, 0, 25, 495])          # one result never seen
        if k == 1:
            raw = np.array([500, 1, 0, 499])           # mitigation gives a small negative value
            a = np.kron(np.array([[0.95, 0.04], [0.05, 0.96]]), np.array([[0.93, 0.06], [0.07, 0.94]]))
        counts = {lab: int(c) for lab, c in zip(LABELS, raw) if c > 0}
        cases.append((counts, a))
    for counts, a in cases:
        total = sum(counts.values())
        p = np.array([counts.get(lab, 0) / total for lab in LABELS])
        want = np.linalg.solve(a, p)
        try:
            got = mitigate(dict(counts), a.copy())
        except Exception as exc:  # noqa: BLE001
            return False, [f"mitigate({counts}, A) stopped with an error: {exc}. Use counts.get(label, 0) for results that never appeared."], None
        if not isinstance(got, dict) or set(got) != set(LABELS):
            return False, ["mitigate() must return a dictionary with the four keys '00', '01', '10' and '11'."], None
        g = np.array([float(got[lab]) for lab in LABELS])
        if not np.allclose(g, want, atol=1e-9):
            hint = ("Turn the counts into probabilities in the order 00, 01, 10, 11, then solve A x = p with "
                    "np.linalg.solve(A, p).")
            if np.allclose(g, a @ p, atol=1e-9):
                hint = "You multiplied by A, which adds readout errors. Mitigation undoes them: solve A x = p. " + hint
            elif np.allclose(g, want * total, atol=1e-6):
                hint = "Divide the counts by the number of shots first. " + hint
            elif np.allclose(g, np.clip(want, 0, None) / np.clip(want, 0, None).sum(), atol=1e-9) and want.min() < 0:
                hint = ("Do not clip or rescale the result: mitigated values are quasi-probabilities and can be slightly "
                        "negative. Return them as they are. " + hint)
            return False, [f"mitigate({counts}, A) gave {np.round(g, 4).tolist()}; it should be {np.round(want, 4).tolist()}. {hint}"], None
    msgs.append("Test 3, mitigate(): passed on 6 sets of counts")

    if DEP_PCT is None or RO_PCT is None or SEED is None:
        return False, msgs + ["Enter DEP_PCT, RO_PCT and SEED from Canvas in Step 9, then run this cell again."], None
    value = personal_value(DEP_PCT, RO_PCT, SEED)
    pred = predicted_different(DEP_PCT, RO_PCT)
    msgs.append(f"Your noisy Bell circuit is predicted to give different results with probability {pred:.3f}: "
                f"about {1000 * pred:.0f} of 1,000 shots. Your shots: 3-sigma band "
                f"{1000 * max(pred - 3 * math.sqrt(pred * (1 - pred) / 1000), 0):.0f} to "
                f"{1000 * (pred + 3 * math.sqrt(pred * (1 - pred) / 1000)):.0f}.")
    return True, msgs, value
