"""Check cell logic for the Module 3 lab (single-qubit gates as matrices, rotations, sequences), on qsim.

check_module3(gate_matrix, rotation, sequence_matrix, THETA_DEG, PHI_DEG, SEED)
returns (passed, messages, verification_value).
The learner writes the three functions. The verification value is the number of 0 results in 1,000
shots of the learner's personal circuit ry(THETA), rx(PHI), measure, on qsim's AerSimulator with seed
SEED. It is printed only when every test passes, and it depends on the seeded draw, so it cannot be
worked out by hand.
"""
import math

import numpy as np
from qsim import AerSimulator, Operator, QuantumCircuit

_HINTS = {
    "X": "X swaps the two amplitudes: [[0, 1], [1, 0]].",
    "Y": "Y is [[0, -1j], [1j, 0]]: note the -1j in the top row.",
    "Z": "Z flips the sign of the |1> amplitude: [[1, 0], [0, -1]].",
    "H": "H is [[1, 1], [1, -1]] divided by np.sqrt(2).",
    "S": "S is [[1, 0], [0, 1j]].",
    "T": "T is [[1, 0], [0, np.exp(1j * np.pi / 4)]].",
}


def _matrix(x):
    return np.asarray(x, dtype=complex)


def _num(z):
    z = complex(z)
    re, im = round(z.real, 4) + 0.0, round(z.imag, 4) + 0.0
    if abs(im) < 1e-9:
        return f"{re:g}"
    if abs(re) < 1e-9:
        return "i" if im == 1 else "-i" if im == -1 else f"{im:g}i"
    return f"{re:g}{'+' if im > 0 else '-'}{abs(im):g}i"


def _fmt(m):
    m = np.asarray(m, dtype=complex)
    if m.ndim != 2:
        return str(np.round(m, 4).tolist())
    return "[" + ", ".join("[" + ", ".join(_num(x) for x in row) + "]" for row in m) + "]"


def _one_gate(name, theta=None):
    qc = QuantumCircuit(1)
    if theta is None:
        getattr(qc, name.lower())(0)
    else:
        getattr(qc, name.lower())(theta, 0)
    return Operator(qc).data


def personal_value(theta_deg, phi_deg, seed):
    qc = QuantumCircuit(1, 1)
    qc.ry(math.radians(theta_deg), 0)
    qc.rx(math.radians(phi_deg), 0)
    qc.measure(0, 0)
    counts = AerSimulator(seed_simulator=int(seed)).run(qc, shots=1000).result().get_counts()
    return int(counts.get("0", 0))


def check_module3(gate_matrix, rotation, sequence_matrix, THETA_DEG, PHI_DEG, SEED):
    msgs = []

    # Test 1: gate_matrix(name) for the six fixed gates, compared with Operator of a one-gate circuit.
    for name in ["X", "Y", "Z", "H", "S", "T"]:
        try:
            got = _matrix(gate_matrix(name))
        except Exception as exc:  # noqa: BLE001
            return False, [f"gate_matrix('{name}') stopped with an error: {exc}"], None
        want = _one_gate(name)
        if got.shape != (2, 2):
            return False, [f"gate_matrix('{name}') must return a 2 x 2 matrix, for example np.array([[0, 1], [1, 0]])."], None
        if not np.allclose(got, want, atol=1e-9):
            extra = ""
            if Operator(got).equiv(want):
                extra = " It differs from the right matrix only by a global phase, but here the matrix must match exactly."
            return False, [f"gate_matrix('{name}') gave {_fmt(got)}; Operator gives "
                           f"{_fmt(want)}. {_HINTS[name]}{extra}"], None
    msgs.append("Test 1, gate_matrix(): passed for X, Y, Z, H, S and T")

    # Test 2: rotation(axis, theta) = cos(theta/2) I - i sin(theta/2) sigma_axis, as Qiskit's rx, ry, rz.
    for axis in ["X", "Y", "Z"]:
        for theta in [1.234, 0.0, math.pi, math.pi / 2, -2.5, 4.0]:
            try:
                got = _matrix(rotation(axis, theta))
            except Exception as exc:  # noqa: BLE001
                return False, [f"rotation('{axis}', {theta:.3f}) stopped with an error: {exc}"], None
            want = _one_gate("r" + axis.lower(), theta)
            if got.shape != (2, 2) or not np.allclose(got, want, atol=1e-9):
                hint = ("Use the half angle: R(theta) = cos(theta/2) I - 1j sin(theta/2) P, where P is the Pauli "
                        "matrix for the axis.")
                if got.shape == (2, 2) and Operator(got).equiv(want):
                    hint = "Your matrix equals the right one only up to a global phase, but here it must match exactly."
                    if axis == "Z":
                        hint += (" For Z, use [[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], not the phase gate "
                                 "[[1, 0], [0, np.exp(1j*theta)]].")
                return False, [f"rotation('{axis}', {theta:.3f}) gave {_fmt(got)}; Operator of "
                               f"qc.r{axis.lower()}({theta:.3f}, 0) gives {_fmt(want)}. {hint}"], None
    msgs.append("Test 2, rotation(): passed for X, Y and Z at six angles")

    # Test 3: sequence_matrix(list of 2 x 2 matrices in time order) = the matrix of the whole circuit.
    rng = np.random.default_rng(31)
    named = ["x", "y", "z", "h", "s", "sdg", "t", "tdg"]
    cases = [[], ["h"], ["h", "s"], ["s", "h"], ["h", "t", "s", "h"]]
    for _ in range(6):
        cases.append([str(g) for g in rng.choice(named + ["rx", "ry", "rz"], size=int(rng.integers(2, 7)))])
    for case in cases:
        qc = QuantumCircuit(1)
        mats = []
        for g in case:
            if g in ("rx", "ry", "rz"):
                th = float(rng.uniform(-3, 3))
                getattr(qc, g)(th, 0)
                mats.append(_one_gate(g, th))
            else:
                getattr(qc, g)(0)
                mats.append(_one_gate(g))
        before = [m.copy() for m in mats]
        try:
            got = _matrix(sequence_matrix(mats))
        except Exception as exc:  # noqa: BLE001
            return False, [f"sequence_matrix() stopped with an error: {exc}"], None
        if any(not np.array_equal(a, b) for a, b in zip(mats, before)) or len(mats) != len(before):
            return False, ["sequence_matrix() changed the list it was given. Build a new matrix instead."], None
        want = Operator(qc).data
        label = " then ".join(g.upper() for g in case) or "no gates"
        if got.shape != (2, 2) or not np.allclose(got, want, atol=1e-9):
            if not case:
                return False, ["sequence_matrix([]) should be the 2 x 2 identity matrix: with no gates, nothing changes."], None
            rev = np.eye(2, dtype=complex)
            for m in mats:
                rev = rev @ m
            hint = ("The gates are in time order, so the first gate's matrix must end up on the right: "
                    "start with U = identity and, for each gate G, set U = G @ U.")
            if got.shape == (2, 2) and np.allclose(got, rev, atol=1e-9):
                hint = "You multiplied in the wrong order. " + hint
            return False, [f"For the circuit {label}, sequence_matrix() gave {_fmt(got)}; "
                           f"Operator(qc) gives {_fmt(want)}. {hint}"], None
    msgs.append("Test 3, sequence_matrix(): passed on 11 circuits, including H then S and S then H")

    # The personal value.
    if THETA_DEG is None or PHI_DEG is None or SEED is None:
        return False, msgs + ["Enter THETA_DEG, PHI_DEG and SEED from Canvas in Step 8, then run this cell again."], None
    t, p = math.radians(float(THETA_DEG)), math.radians(float(PHI_DEG))
    u = _matrix(sequence_matrix([_matrix(rotation("Y", t)), _matrix(rotation("X", p))]))
    p0 = float(abs(u[0, 0]) ** 2)
    value = personal_value(THETA_DEG, PHI_DEG, SEED)
    msgs.append(f"Your circuit ry({THETA_DEG} deg), rx({PHI_DEG} deg) gives P(0) = {p0:.3f}: "
                f"about {1000 * p0:.0f} zeros are expected in 1,000 shots.")
    return True, msgs, value
