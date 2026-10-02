"""Check cell logic for the Module 2 lab (Dirac notation, the Bloch sphere, measurement bases), on qsim.

check_module2(state_from_angles, bloch_vector, measure_in_basis, THETA_DEG, PHI_DEG, SEED)
returns (passed, messages, verification_value).
The learner writes the three functions. The verification value is the number of 0 results in 1,000
shots of the learner's personal state cos(t/2)|0> + e^(i p) sin(t/2)|1>, measured in the Y basis
(S-dagger, then H, then measure) on qsim's AerSimulator with seed SEED. It is printed only when every
test passes, and it depends on the seeded draw, so it cannot be worked out by hand.
"""
import math

import numpy as np
from qsim import AerSimulator, QuantumCircuit, Statevector


def _close(x, y, tol=1e-9):
    x, y = np.asarray(x, dtype=complex), np.asarray(y, dtype=complex)
    return x.shape == y.shape and bool(np.allclose(x, y, atol=tol))


def _ref_state(t, p):
    return np.array([math.cos(t / 2), np.exp(1j * p) * math.sin(t / 2)])


def _ref_bloch(v):
    a, b = complex(v[0]), complex(v[1])
    return np.array([2 * (a.conjugate() * b).real, 2 * (a.conjugate() * b).imag, abs(a) ** 2 - abs(b) ** 2])


def _prep(t, p):
    qc = QuantumCircuit(1, 1)
    qc.ry(t, 0)
    qc.p(p, 0)
    return qc


def _p0_of(qc):
    """Probability of reading 0 from a 1-qubit circuit that ends with one measurement of qubit 0."""
    body = qc.copy()
    body.remove_final_measurements()
    return float(Statevector(body).probabilities()[0])


def personal_value(theta_deg, phi_deg, seed):
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    qc = _prep(t, p)
    qc.sdg(0)
    qc.h(0)
    qc.measure(0, 0)
    counts = AerSimulator(seed_simulator=int(seed)).run(qc, shots=1000).result().get_counts()
    return int(counts.get("0", 0))


def check_module2(state_from_angles, bloch_vector, measure_in_basis, THETA_DEG, PHI_DEG, SEED):
    msgs = []

    # Test 1: state_from_angles(theta, phi) = [cos(theta/2), e^(i phi) sin(theta/2)], with angles in radians.
    for t, p in [(0.0, 0.0), (math.pi, 1.0), (math.pi / 2, math.pi / 2), (2.1, 4.0), (0.7, -1.2)]:
        try:
            got = np.asarray(state_from_angles(t, p), dtype=complex).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return False, [f"state_from_angles() stopped with an error: {exc}"], None
        want = _ref_state(t, p)
        if not _close(got, want, 1e-9):
            return False, [f"state_from_angles({t:.3f}, {p:.3f}) gave {np.round(got, 4)}; it should be {np.round(want, 4)}. "
                           "Use [np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], with the angles in radians."], None
    msgs.append("Test 1, state_from_angles(): passed")

    # Test 2: bloch_vector(state) = (x, y, z), real numbers, unchanged by a global phase.
    cases = [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1j / math.sqrt(2)],
             [0.6, 0.8j], list(np.exp(1j * 2.3) * _ref_state(1.1, 2.7))]
    for v in cases:
        arr = np.array(v, dtype=complex)
        try:
            got = np.asarray(bloch_vector(arr.copy()))
        except Exception as exc:  # noqa: BLE001
            return False, [f"bloch_vector() stopped with an error: {exc}"], None
        want = _ref_bloch(arr)
        if got.shape != (3,) or np.iscomplexobj(got) and np.abs(np.imag(got)).max() > 1e-9:
            return False, ["bloch_vector() must return three real numbers (x, y, z). "
                           "Use the real and imaginary parts: x = 2*Re(conj(a)*b), y = 2*Im(conj(a)*b), z = |a|^2 - |b|^2."], None
        if not _close(np.real(got), want, 1e-9):
            return False, [f"bloch_vector({np.round(arr, 4)}) gave {np.round(np.real(got), 4)}; it should be {np.round(want, 4)}."], None
    msgs.append("Test 2, bloch_vector(): passed (including a state with a global phase)")

    # Test 3: measure_in_basis(qc, basis) returns a new circuit that measures in the Z, X or Y basis.
    for t, p in [(0.9, 0.4), (2.2, 3.9), (math.pi / 2, math.pi / 2), (1.6, 5.5)]:
        r = _ref_bloch(_ref_state(t, p))
        for basis, comp in (("Z", 2), ("X", 0), ("Y", 1)):
            qc = _prep(t, p)
            before = [(i.name, i.qubits, i.params) for i in qc.data]
            try:
                out = measure_in_basis(qc, basis)
            except Exception as exc:  # noqa: BLE001
                return False, [f"measure_in_basis(qc, '{basis}') stopped with an error: {exc}"], None
            if not isinstance(out, QuantumCircuit):
                return False, ["measure_in_basis() must return a QuantumCircuit."], None
            if [(i.name, i.qubits, i.params) for i in qc.data] != before:
                return False, ["measure_in_basis() changed the circuit it was given. Start with new = qc.copy() "
                               "and add the gates and the measurement to new."], None
            if sum(1 for i in out.data if i.name == "measure") != 1 or out.data[-1].name != "measure":
                return False, [f"The circuit from measure_in_basis(qc, '{basis}') must end with one measurement: new.measure(0, 0)."], None
            want = (1 + r[comp]) / 2
            got = _p0_of(out)
            if abs(got - want) > 1e-9:
                hint = {"Z": "For Z, add nothing before the measurement.",
                        "X": "For X, add new.h(0) before the measurement.",
                        "Y": "For Y, add new.sdg(0) and then new.h(0) before the measurement (S-dagger first, then H)."}[basis]
                return False, [f"In the {basis} basis your circuit gives P(0) = {got:.3f}; it should be {want:.3f}. {hint}"], None
    msgs.append("Test 3, measure_in_basis(): passed for Z, X and Y")

    # The personal value.
    if THETA_DEG is None or PHI_DEG is None or SEED is None:
        return False, msgs + ["Enter THETA_DEG, PHI_DEG and SEED from Canvas in Step 8, then run this cell again."], None
    t, p = math.radians(float(THETA_DEG)), math.radians(float(PHI_DEG))
    mine = np.asarray(state_from_angles(t, p), dtype=complex)
    y = float(np.real(np.asarray(bloch_vector(mine)))[1])
    value = personal_value(THETA_DEG, PHI_DEG, SEED)
    msgs.append(f"Your state has Bloch vector y = {y:.3f}, so P(+i) = {(1 + y) / 2:.3f}: "
                f"about {1000 * (1 + y) / 2:.0f} zeros are expected in 1,000 shots in the Y basis.")
    return True, msgs, value
