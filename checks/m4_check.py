"""Check cell logic for the Module 4 lab (two qubits, CNOT, Bell states, correlations), on qsim.

check_module4(product_state, bell_circuit, correlation, THETA_DEG, PHI_DEG, SEED)
returns (passed, messages, verification_value).
The learner writes the three functions. The verification value is the number of shots, out of 1,000,
in which the two qubits give the same result, for the learner's personal circuit
ry(THETA) on qubit 0, cx(0, 1), rz(PHI) on qubit 1, H on both, measure both, on qsim's AerSimulator
with seed SEED. It is printed only when every test passes, and it depends on the seeded draw, so it
cannot be worked out by hand.
"""
import math

import numpy as np
from qsim import AerSimulator, QuantumCircuit, Statevector

S2 = 1 / math.sqrt(2)
BELL = {
    "phi+": np.array([S2, 0, 0, S2], dtype=complex),
    "phi-": np.array([S2, 0, 0, -S2], dtype=complex),
    "psi+": np.array([0, S2, S2, 0], dtype=complex),
    "psi-": np.array([0, S2, -S2, 0], dtype=complex),
}
BELL_HINT = {
    "phi+": "(|00> + |11>)/sqrt(2): H on qubit 0, then cx(0, 1).",
    "phi-": "(|00> - |11>)/sqrt(2): start with X on qubit 0 (or add Z after H), then H on qubit 0 and cx(0, 1).",
    "psi+": "(|01> + |10>)/sqrt(2): start with X on qubit 1, then H on qubit 0 and cx(0, 1).",
    "psi-": "(|01> - |10>)/sqrt(2): start with X on both qubits, then H on qubit 0 and cx(0, 1).",
}


def _num(z):
    z = complex(z)
    re, im = round(z.real, 4) + 0.0, round(z.imag, 4) + 0.0
    if abs(im) < 1e-9:
        return f"{re:g}"
    if abs(re) < 1e-9:
        return f"{im:g}i"
    return f"{re:g}{'+' if im > 0 else '-'}{abs(im):g}i"


def _fmt(v):
    return "[" + ", ".join(_num(x) for x in np.asarray(v, dtype=complex).reshape(-1)) + "]"


def personal_circuit(theta_deg, phi_deg):
    qc = QuantumCircuit(2, 2)
    qc.ry(math.radians(theta_deg), 0)
    qc.cx(0, 1)
    qc.rz(math.radians(phi_deg), 1)
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc


def personal_value(theta_deg, phi_deg, seed):
    counts = AerSimulator(seed_simulator=int(seed)).run(personal_circuit(theta_deg, phi_deg), shots=1000).result().get_counts()
    return int(counts.get("00", 0) + counts.get("11", 0))


def check_module4(product_state, bell_circuit, correlation, THETA_DEG, PHI_DEG, SEED):
    msgs = []

    # Test 1: product_state(q0, q1) is the two-qubit state with qubit 0 in q0 and qubit 1 in q1, in Qiskit's order.
    rng = np.random.default_rng(41)
    cases = [([1, 0], [0, 1]), ([0, 1], [1, 0]), ([0.6, 0.8], [S2, S2]), ([S2, 1j * S2], [1, 0])]
    for _ in range(3):
        a = rng.normal(size=2) + 1j * rng.normal(size=2)
        b = rng.normal(size=2) + 1j * rng.normal(size=2)
        cases.append((list(a / np.linalg.norm(a)), list(b / np.linalg.norm(b))))
    for q0, q1 in cases:
        q0a, q1a = np.array(q0, dtype=complex), np.array(q1, dtype=complex)
        try:
            got = np.asarray(product_state(q0a.copy(), q1a.copy()), dtype=complex).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return False, [f"product_state() stopped with an error: {exc}"], None
        want = np.kron(q1a, q0a)
        if got.shape != (4,):
            return False, ["product_state() must return 4 amplitudes, in the order |00>, |01>, |10>, |11>."], None
        if not np.allclose(got, want, atol=1e-9):
            hint = "Use np.kron(q1, q0): in Qiskit's order qubit 0 is the rightmost bit, so qubit 1's state goes first in the product."
            if np.allclose(got, np.kron(q0a, q1a), atol=1e-9):
                hint = "The qubits are the wrong way round. " + hint
            return False, [f"product_state(q0={_fmt(q0a)}, q1={_fmt(q1a)}) gave {_fmt(got)}; it should be {_fmt(want)}. {hint}"], None
    msgs.append("Test 1, product_state(): passed, in Qiskit's bit order")

    # Test 2: bell_circuit(name) makes each Bell state (up to a global phase), with 2 qubits and no measurements.
    for name, target in BELL.items():
        try:
            qc = bell_circuit(name)
        except Exception as exc:  # noqa: BLE001
            return False, [f"bell_circuit('{name}') stopped with an error: {exc}"], None
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != 2:
            return False, [f"bell_circuit('{name}') must return a QuantumCircuit with 2 qubits."], None
        if any(i.name in ("measure", "reset") for i in qc.data):
            return False, [f"bell_circuit('{name}') must not measure: return the circuit that makes the state. "
                           "The measurements are added later."], None
        state = Statevector(qc).data
        fid = abs(np.vdot(target, state)) ** 2
        if fid < 0.999:
            return False, [f"bell_circuit('{name}') makes {_fmt(state)}, which is not the Bell state {name}. "
                           f"It should be {BELL_HINT[name]}"], None
    msgs.append("Test 2, bell_circuit(): passed for phi+, phi-, psi+ and psi-")

    # Test 3: correlation(counts) = (number of equal results - number of different results) / total.
    ccases = [({"00": 500, "11": 500}, 1.0), ({"01": 300, "10": 700}, -1.0),
              ({"00": 250, "01": 250, "10": 250, "11": 250}, 0.0),
              ({"00": 400, "01": 100, "10": 100, "11": 400}, 0.6),
              ({"11": 10}, 1.0), ({"00": 30, "10": 90}, -0.5), ({"01": 7, "11": 21, "00": 2}, 16 / 30)]
    for counts, want in ccases:
        try:
            got = float(correlation(dict(counts)))
        except Exception as exc:  # noqa: BLE001
            return False, [f"correlation({counts}) stopped with an error: {exc}. Use counts.get(key, 0) for results that never happened."], None
        if abs(got - want) > 1e-9:
            return False, [f"correlation({counts}) gave {got:.4f}; it should be {want:.4f}. Add the counts of '00' and '11' "
                           "(the same result), subtract the counts of '01' and '10' (different results), and divide by "
                           "the total number of shots."], None
    msgs.append("Test 3, correlation(): passed on 7 sets of counts")

    # The personal value.
    if THETA_DEG is None or PHI_DEG is None or SEED is None:
        return False, msgs + ["Enter THETA_DEG, PHI_DEG and SEED from Canvas in Step 8, then run this cell again."], None
    value = personal_value(THETA_DEG, PHI_DEG, SEED)
    predicted = (1 + math.sin(math.radians(float(THETA_DEG))) * math.cos(math.radians(float(PHI_DEG)))) / 2
    msgs.append(f"Your circuit is predicted to give the same result on both qubits with probability {predicted:.3f}: "
                f"about {1000 * predicted:.0f} of 1,000 shots.")
    return True, msgs, value
