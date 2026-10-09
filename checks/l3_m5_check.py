"""Check cell logic for the Level 3 Module 5 lab (comparing quantum hardware from published data).

check_l3_module5(circuit_success, max_two_qubit_gates, summarize, N1, N2, NM)
returns (passed, messages, verification_value).

The learner writes three functions:
- circuit_success(e1, e2, e_ro, n1, n2, nm): the estimated probability that a circuit with n1 single-qubit gates,
  n2 two-qubit gates and nm measurements runs without any error, if every operation fails independently with the
  given error rate: (1 - e1)**n1 * (1 - e2)**n2 * (1 - e_ro)**nm.
- max_two_qubit_gates(e2, target=0.5): the largest whole number n of two-qubit gates with (1 - e2)**n >= target.
- summarize(values, higher_is_better): a dict with the median, the best and the worst of a list of numbers; "best" is
  the largest value when higher_is_better is True (for example T1) and the smallest when it is False (error rates).

The verification value is round(1000 * P), where P is the reference circuit_success for N1 single-qubit gates, N2
two-qubit gates and NM measurements with the median error rates of ibm_boston (calibration of 17 April 2026) from
l3_m5_data.json, counting only the qubits and pairs IBM reports as usable (error below 1).
"""
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- reference solutions
def reference_circuit_success(e1, e2, e_ro, n1, n2, nm):
    return (1 - e1) ** n1 * (1 - e2) ** n2 * (1 - e_ro) ** nm


def reference_max_two_qubit_gates(e2, target=0.5):
    n = int(math.floor(math.log(target) / math.log(1 - e2)))
    while (1 - e2) ** (n + 1) >= target:      # guard against rounding in the logarithms
        n += 1
    while n > 0 and (1 - e2) ** n < target:
        n -= 1
    return n


def reference_summarize(values, higher_is_better):
    v = [float(x) for x in values]
    return {"median": float(np.median(v)), "best": max(v) if higher_is_better else min(v),
            "worst": min(v) if higher_is_better else max(v)}


def usable(values):
    """Error rates of 1 mark a qubit or pair IBM reports as unusable; leave them out."""
    return [x for x in values if x < 1]


def boston_medians(path=None):
    d = json.load(open(path or os.path.join(_HERE, "l3_m5_data.json")))["devices"]["ibm_boston"]
    return (float(np.median(usable(d["qubit_data"]["e1"]))), float(np.median(usable(d["e2"]))),
            float(np.median(usable(d["qubit_data"]["e_ro"]))))


def personal_value(n1, n2, nm):
    e1, e2, er = boston_medians()
    return int(round(1000 * reference_circuit_success(e1, e2, er, int(n1), int(n2), int(nm))))


# ---------------------------------------------------------------- helpers
def _try(fn, name, *args, **kw):
    try:
        return fn(*args, **kw), None
    except NotImplementedError:
        return None, f"{name}() is not written yet."
    except Exception as e:  # noqa: BLE001
        return None, f"{name}() raised {type(e).__name__}: {e}"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- tests
_CASES = [(0.0002, 0.003, 0.01, 100, 50, 10), (0.001, 0.01, 0.02, 0, 0, 0), (0.00005, 0.0008, 0.0005, 400, 300, 98),
          (0.0, 0.0, 0.0, 10, 10, 10), (0.003, 0.02, 0.05, 37, 11, 3), (0.0001, 0.0015, 0.005, 1000, 0, 50)]


def _test_success(circuit_success):
    for c in _CASES:
        got, err = _try(circuit_success, "circuit_success", *c)
        if err:
            return False, [err]
        g = None if isinstance(got, (bool, str)) else _num(got)
        want = reference_circuit_success(*c)
        if g is None:
            return False, [f"circuit_success{c} returned {got!r}; it should return a number between 0 and 1."]
        if abs(g - want) > 1e-9:
            e1, e2, er, n1, n2, nm = c
            hint = ""
            if abs(g - (1 - e1 * n1 - e2 * n2 - er * nm)) < 1e-9:
                hint = " This is 1 minus the sum of the errors, the first-order approximation; multiply the success probabilities instead."
            elif abs(g - (e1 ** n1 * e2 ** n2 * er ** nm)) < 1e-9:
                hint = " Raise the success probability (1 - e) to a power, not the error e."
            elif abs(g - (1 - e1) * n1 * (1 - e2) * n2 * (1 - er) * nm) < 1e-6:
                hint = " Use ** (a power), not * (a product), for the number of operations."
            elif abs(g - (1 - e1) ** n1 * (1 - e2) ** n2) < 1e-9:
                hint = " The measurements are missing: multiply by (1 - e_ro) ** nm."
            return False, [f"circuit_success{c} gave {g:.6g}; expected {want:.6g}.{hint}"]
    return True, ["circuit_success(): passed (six circuits, including an empty one and error-free operations)."]


_GATES = [(0.0033, 0.5), (0.001271, 0.5), (0.00079, 0.5), (0.01, 0.5), (0.004, 0.9), (0.002, 0.99), (0.5, 0.25),
          (0.1, 0.9), (0.03, 0.01)]


def _test_gates(max_two_qubit_gates):
    for e2, target in _GATES:
        got, err = _try(max_two_qubit_gates, "max_two_qubit_gates", e2, target)
        if err:
            return False, [err]
        want = reference_max_two_qubit_gates(e2, target)
        if isinstance(got, bool) or not isinstance(got, (int, np.integer)):
            g = _num(got)
            if g is not None and abs(g - want) < 1 and g != int(g):
                return False, [f"max_two_qubit_gates({e2}, {target}) returned {got}; return a whole number (int), rounded down."]
            if g is None or g != int(g):
                return False, [f"max_two_qubit_gates({e2}, {target}) returned {got!r}; it should return a whole number."]
            got = int(g)
        if got != want:
            hint = ""
            if got == want + 1:
                hint = (" One gate too many: with that many gates the success falls below the target. Round down, "
                        "not up or to the nearest whole number.")
            elif got == want - 1 and (1 - e2) ** want >= target:
                hint = " One gate too few: (1 - e2) ** n may equal the target exactly or be just above it."
            elif abs(got - math.log(target) / math.log(e2)) < 1.5:
                hint = " Use the success probability 1 - e2 inside the logarithm, not e2."
            elif abs(got - math.log(target) / -e2) < 1.5 and abs(want - math.log(target) / -e2) >= 1:
                hint = " ln(1 - e2) is only roughly -e2; use the exact logarithm, math.log(1 - e2)."
            return False, [f"max_two_qubit_gates({e2}, target={target}) gave {got}; expected {want}.{hint}"]
    got, err = _try(max_two_qubit_gates, "max_two_qubit_gates", 0.0033)
    if err:
        return False, [err + " Give target the default value 0.5."]
    if got != reference_max_two_qubit_gates(0.0033):
        return False, ["max_two_qubit_gates(0.0033) without a target should use target = 0.5."]
    return True, ["max_two_qubit_gates(): passed (nine error rates and targets, rounding down, the default target 0.5)."]


_LISTS = [([3.0, 1.0, 2.0], True), ([3.0, 1.0, 2.0], False), ([0.004, 0.001, 0.003, 0.002], False),
          ([250.5, 310.0, 120.2, 400.1, 299.9, 15.0], True), ([7.0], True), ([0.02, 0.5, 0.0101, 0.0099, 0.3], False)]


def _test_summarize(summarize):
    for vals, hib in _LISTS:
        got, err = _try(summarize, "summarize", list(vals), hib)
        if err:
            return False, [err]
        if not isinstance(got, dict) or not {"median", "best", "worst"} <= set(got):
            return False, [f"summarize() returned {got!r}; return a dict with the keys 'median', 'best' and 'worst'."]
        want = reference_summarize(vals, hib)
        for k in ("median", "best", "worst"):
            g = _num(got[k])
            if g is None or abs(g - want[k]) > 1e-12:
                hint = ""
                if k == "median" and g is not None and abs(g - float(np.mean(vals))) < 1e-12:
                    hint = " That is the mean; the median is the middle value (np.median)."
                elif k in ("best", "worst") and g is not None and abs(g - want["worst" if k == "best" else "best"]) < 1e-12:
                    hint = (" Best and worst are swapped: for error rates (higher_is_better=False) the best value is the "
                            "smallest; for T1 or T2 (higher_is_better=True) it is the largest.")
                elif k == "median" and len(vals) % 2 == 0:
                    hint = " With an even number of values the median is the average of the two middle values."
                return False, [f"summarize({vals}, higher_is_better={hib}) gave {k} = {got[k]!r}; expected {want[k]}.{hint}"]
    return True, ["summarize(): passed (odd and even lengths, higher and lower is better)."]


# ---------------------------------------------------------------- the check
def check_l3_module5(circuit_success, max_two_qubit_gates, summarize, N1, N2, NM):
    try:
        n = [float(N1), float(N2), float(NM)]
    except (TypeError, ValueError):
        return False, ["N1, N2 and NM must be the numbers shown in your Canvas lab check."], None
    if not (all(v == int(v) for v in n) and 100 <= n[0] <= 1000 and 20 <= n[1] <= 300 and 5 <= n[2] <= 50):
        return False, ["Copy your numbers again from the lab check: N1 (100 to 1,000), N2 (20 to 300) and NM (5 to 50), "
                       "each a whole number."], None
    msgs = []
    for test in (lambda: _test_success(circuit_success), lambda: _test_gates(max_two_qubit_gates),
                 lambda: _test_summarize(summarize)):
        good, m = test()
        msgs += m
        if not good:
            return False, msgs, None
    return True, msgs, personal_value(*[int(v) for v in n])
