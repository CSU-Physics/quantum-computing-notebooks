"""Shared parts of the Level 3 Module 6 capstone checks.

Every capstone notebook imports this file (served next to the notebook) together with its project's check file.
It holds:
- the capstone noise model: depolarizing error P2 after two-qubit gates and P2 / 10 after one-qubit gates, an idle
  error IDLE on `id` gates (used by the repetition-code capstone), and readout errors (a 0 is recorded as 1 with
  probability READOUT, a 1 as 0 with probability 2 * READOUT);
- exact_probabilities(qc, p2, readout, idle): the noise model's exact probability of every recorded result of a
  circuit whose measurements all come at the end (density matrix, then the readout confusion of each bit);
- the tests of the three statistics functions every capstone asks for:
  fraction_sigma(p, shots), expectation_sigma(E, shots) and combined_sigma(coefficients, sigmas).
"""
import math

import numpy as np

import qsim
from qsim import NoiseModel, ReadoutError, depolarizing_error, simulate_density_matrix

ONE_QUBIT_GATES = ["h", "x", "y", "z", "rx", "ry", "rz", "p", "s", "sdg", "t", "tdg", "sx"]
TWO_QUBIT_GATES = ["cx", "cz", "cp", "rzz", "swap"]
SHOTS = 4000
P2_COURSE, READOUT_COURSE = 0.01, 0.02


def confusion(readout):
    """Rows: the true bit; columns: the recorded bit."""
    return np.array([[1 - readout, readout], [2 * readout, 1 - 2 * readout]])


def capstone_noise_model(p2=P2_COURSE, readout=READOUT_COURSE, idle=0.0):
    nm = NoiseModel()
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ONE_QUBIT_GATES)
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), TWO_QUBIT_GATES)
    if idle > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(idle, 1), ["id"])
    if readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError(confusion(readout).tolist()))
    return nm


def apply_readout(probs, readout):
    """probs: array over m recorded bits, index bit j = classical bit j. Each bit is flipped by the confusion."""
    probs = np.asarray(probs, dtype=float)
    m = int(round(math.log2(len(probs))))
    c = confusion(readout)
    for j in range(m):
        t = probs.reshape(2 ** (m - 1 - j), 2, 2 ** j)
        probs = np.einsum("aib,ij->ajb", t, c).reshape(-1)
    return probs


def exact_probabilities(qc, p2=P2_COURSE, readout=READOUT_COURSE, idle=0.0):
    """The noise model's exact probability of each recorded result, {"bits": p} in Qiskit's key order.
    qc must end with its measurements (each classical bit receives one qubit); no shots are taken."""
    measured = {inst.clbits[0]: inst.qubits[0] for inst in qc.data if inst.name == "measure"}
    body = qc.copy()
    body.remove_final_measurements()
    nm = capstone_noise_model(p2, 0.0, idle)
    rho = simulate_density_matrix(body, nm if not nm.is_ideal() else None)
    m = len(measured)
    probs = rho.probabilities(qargs=[measured[j] for j in range(m)])
    probs = apply_readout(probs, readout)
    return {format(i, f"0{m}b"): float(p) for i, p in enumerate(probs) if p > 1e-15}


# ---------------------------------------------------------------- reference statistics functions
def reference_fraction_sigma(p, shots):
    return math.sqrt(p * (1 - p) / shots)


def reference_expectation_sigma(E, shots):
    return math.sqrt(max(0.0, 1 - E * E) / shots)


def reference_combined_sigma(coefficients, sigmas):
    return math.sqrt(sum((c * s) ** 2 for c, s in zip(coefficients, sigmas)))


def _try(fn, name, *args, **kw):
    try:
        return fn(*args, **kw), None
    except NotImplementedError:
        return None, f"{name}() is not written yet."
    except Exception as e:  # noqa: BLE001
        return None, f"{name}() raised {type(e).__name__}: {e}"


def _num(x):
    if isinstance(x, (bool, str)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def test_statistics(fraction_sigma, expectation_sigma, combined_sigma):
    """Returns (passed, messages)."""
    for p, s in ((0.5, 4000), (0.0613, 4000), (0.9, 1000), (0.0, 100), (0.25, 37)):
        got, err = _try(fraction_sigma, "fraction_sigma", p, s)
        if err:
            return False, [err]
        g, want = _num(got), reference_fraction_sigma(p, s)
        if g is None or abs(g - want) > 1e-9:
            hint = ""
            if g is not None and abs(g - p * (1 - p) / s) < 1e-12:
                hint = " That is the variance; take its square root."
            elif g is not None and abs(g - math.sqrt(p * (1 - p)) / s) < 1e-12:
                hint = " Divide by the number of shots inside the square root: sqrt(p (1 - p) / shots)."
            elif g is not None and abs(g - math.sqrt((1 - p * p) / s)) < 1e-12:
                hint = " That is the formula for an expectation value of +-1 results; a fraction uses p (1 - p)."
            return False, [f"fraction_sigma({p}, {s}) gave {got!r}; expected {want:.6g}.{hint}"]
    for E, s in ((0.0, 4000), (0.8, 4000), (-0.5, 1000), (1.0, 4000), (0.37, 250)):
        got, err = _try(expectation_sigma, "expectation_sigma", E, s)
        if err:
            return False, [err]
        g, want = _num(got), reference_expectation_sigma(E, s)
        if g is None or abs(g - want) > 1e-9:
            hint = ""
            p = (1 + E) / 2
            if g is not None and abs(g - math.sqrt(p * (1 - p) / s)) < 1e-12:
                hint = (" That is the sigma of the fraction of +1 results. The expectation value is E = 2 p - 1, so its sigma "
                        "is twice as large: sqrt((1 - E^2) / shots).")
            elif g is not None and abs(g - (1 - E * E) / s) < 1e-12:
                hint = " That is the variance; take its square root."
            elif g is not None and abs(g - math.sqrt((1 - E) / s)) < 1e-12:
                hint = " Square E: sqrt((1 - E^2) / shots)."
            return False, [f"expectation_sigma({E}, {s}) gave {got!r}; expected {want:.6g}.{hint}"]
    cases = (([1, -1], [0.01, 0.02]), ([1.875, -1.25, 0.375], [0.012, 0.015, 0.02]), ([2.0], [0.01]),
             ([0.5, 0.5], [0.03, 0.04]), ([1, 1, 1, 1], [0.01, 0.01, 0.01, 0.01]))
    for c, s in cases:
        got, err = _try(combined_sigma, "combined_sigma", list(c), list(s))
        if err:
            return False, [err]
        g, want = _num(got), reference_combined_sigma(c, s)
        if g is None or abs(g - want) > 1e-9:
            hint = ""
            if g is not None and abs(g - sum(abs(a) * b for a, b in zip(c, s))) < 1e-12:
                hint = " Independent uncertainties add in quadrature: square, add, then take the square root."
            elif g is not None and abs(g - math.sqrt(sum(a * b * b for a, b in zip(c, s)))) < 1e-12:
                hint = " Square the coefficients too: sqrt(sum (c_i sigma_i)^2)."
            elif g is not None and abs(g - math.sqrt(sum(b * b for b in s))) < 1e-12:
                hint = " Multiply each sigma by its coefficient before squaring."
            return False, [f"combined_sigma({list(c)}, {list(s)}) gave {got!r}; expected {want:.6g}.{hint}"]
    return True, ["fraction_sigma(), expectation_sigma() and combined_sigma(): passed."]


def check_personal_numbers(p2, readout):
    try:
        p2, readout = float(p2), float(readout)
    except (TypeError, ValueError):
        return None
    if not (0.005 <= p2 <= 0.030 and 0.005 <= readout <= 0.030):
        return None
    if abs(round(p2, 3) - p2) > 1e-12 or abs(round(readout, 3) - readout) > 1e-12:
        return None
    return p2, readout
