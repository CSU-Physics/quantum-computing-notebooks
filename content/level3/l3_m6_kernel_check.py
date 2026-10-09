"""Check cell logic for the Level 3 Module 6 capstone "Quantum-kernel classifier within hardware limits".

check_l3_m6_kernel(fraction_sigma, expectation_sigma, combined_sigma, overlap_circuit, kernel_from_counts,
                   training_kernel, test_kernel, PAIR, P2, READOUT) returns (passed, messages, verification_value).

The classifier of Module 4 Track A with the hardware limits of a real run: 8 training points, 4 test points, 1,000
shots per kernel entry, two features, the ZZ feature map with 2 repetitions (feature_map is prepared in the notebook,
the reference of Module 4).
- overlap_circuit(x1, x2): feature_map(x1), then feature_map(x2).inverse(), then a measurement of every qubit
  (qubit i into classical bit i); the fraction of shots with every bit 0 estimates K(x1, x2);
- kernel_from_counts(counts): that fraction (counts may also be probabilities);
- training_kernel(X, estimate): the len(X) x len(X) matrix with 1 on the diagonal and estimate(X[i], X[j]) for i < j,
  copied to [j][i] (one circuit per pair: 28 for 8 points);
- test_kernel(X_test, X_train, estimate): the len(X_test) x len(X_train) matrix of estimate(X_test[a], X_train[b]).

The verification value is round(10000 * K) where K is the exact noisy kernel entry of training pair PAIR (1 to 28:
pairs (0, 1), (0, 2), ..., (6, 7) in that order) under the capstone noise model with the learner's P2 and READOUT:
the noise model's exact probability of reading 00 from the reference overlap circuit.
"""
import itertools

import numpy as np

from qsim import QuantumCircuit

from l3_m6_common import _num, _try, exact_probabilities, test_statistics

PAIRS = list(itertools.combinations(range(8), 2))


# ---------------------------------------------------------------- the data and the feature map (prepared)
def feature_map(x, reps=2):
    """The ZZ feature map of Module 4 (Qiskit's zz_feature_map with full entanglement)."""
    x = [float(v) for v in x]
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.p(2 * x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def capstone_data(path=None):
    """8 training points (4 of each class) and 4 test points (2 of each) from the Module 4 'adhoc' data set."""
    import json
    import os
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "l3_m4a_data.json")
        if not os.path.exists(path):                     # the course repository keeps the data in content/level3
            path = os.path.join(here, "..", "content", "level3", "l3_m4a_data.json")
    d = json.load(open(path))["adhoc"]

    def pick(X, y, k):
        X, y = np.array(X), np.array(y)
        idx = [int(i) for c in sorted(set(y.tolist())) for i in np.where(y == c)[0][:k]]
        return X[idx], y[idx]
    Xtr, ytr = pick(d["X_train"], d["y_train"], 4)
    Xte, yte = pick(d["X_test"], d["y_test"], 2)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------- reference solutions
def reference_overlap_circuit(x1, x2):
    n = len(x1)
    qc = QuantumCircuit(n, n)
    qc.compose(feature_map(x1), inplace=True)
    qc.compose(feature_map(x2).inverse(), inplace=True)
    for i in range(n):
        qc.measure(i, i)
    return qc


def reference_kernel_from_counts(counts):
    total = sum(counts.values())
    n = len(next(iter(counts)))
    return float(counts.get("0" * n, 0) / total)


def reference_training_kernel(X, estimate):
    m = len(X)
    K = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            K[i, j] = K[j, i] = estimate(X[i], X[j])
    return K


def reference_test_kernel(X_test, X_train, estimate):
    return np.array([[estimate(a, b) for b in X_train] for a in X_test], dtype=float)


# ---------------------------------------------------------------- exact values
def exact_entry(x1, x2, p2, readout, overlap=reference_overlap_circuit):
    return float(exact_probabilities(overlap(x1, x2), p2, readout).get("0" * len(x1), 0.0))


def personal_value(pair, p2, readout, data=None):
    Xtr = (data or capstone_data())[0]
    i, j = PAIRS[int(pair) - 1]
    return int(round(10000 * exact_entry(Xtr[i], Xtr[j], float(p2), float(readout))))


# ---------------------------------------------------------------- tests
def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


_PTS = [(0.4, 2.1), (1.3, 0.2), (2.9, 1.7), (0.0, 3.1)]


def _test_overlap(overlap_circuit):
    for a, b in ((0, 1), (2, 3), (1, 1)):
        x1, x2 = _PTS[a], _PTS[b]
        got, err = _try(overlap_circuit, "overlap_circuit", x1, x2)
        if err:
            return False, [err]
        if not isinstance(got, QuantumCircuit) or got.num_qubits != 2 or got.num_clbits != 2:
            return False, ["overlap_circuit() should return a QuantumCircuit with 2 qubits and 2 classical bits."]
        if not any(i.name == "measure" for i in got.data):
            return False, ["overlap_circuit() needs a measurement of both qubits at the end (qubit i into classical bit i)."]
        ref = reference_overlap_circuit(x1, x2)
        ideal_g, ideal_r = exact_probabilities(got, 0, 0), exact_probabilities(ref, 0, 0)
        if _tvd(ideal_g, ideal_r) > 1e-9:
            hint = ""
            if a == b and abs(ideal_g.get("00", 0) - 1) > 1e-6:
                hint = " For x1 = x2 the circuit should give 00 every time: use feature_map(x2).inverse() after feature_map(x1)."
            elif abs(ideal_g.get("00", 0) - ideal_r.get("00", 0)) < 1e-9:
                hint = " The 00 probability is right but other results differ: measure qubit i into classical bit i."
            return False, [f"overlap_circuit({x1}, {x2}) does not give the expected results.{hint}"]
        if _tvd(exact_probabilities(got, 0.02, 0.01), exact_probabilities(ref, 0.02, 0.01)) > 1e-9:
            return False, [f"overlap_circuit({x1}, {x2}) is right without noise but not with it: build it from feature_map(x1) and "
                           "feature_map(x2).inverse() only, with no extra gates."]
    return True, ["overlap_circuit(): passed (three pairs, with and without noise)."]


def _test_counts(kernel_from_counts):
    for counts, want in (({"00": 700, "01": 100, "10": 150, "11": 50}, 0.7), ({"11": 3, "10": 1}, 0.0),
                         ({"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}, 0.25), ({"000": 10, "001": 30}, 0.25)):
        got, err = _try(kernel_from_counts, "kernel_from_counts", dict(counts))
        if err:
            return False, [err]
        g = _num(got)
        if g is None or abs(g - want) > 1e-12:
            hint = ""
            if g is not None and abs(g - counts.get("0" * len(next(iter(counts))), 0)) < 1e-12 and want > 0:
                hint = " Divide by the total number of shots (or the total probability)."
            return False, [f"kernel_from_counts({counts}) gave {got!r}; expected {want}.{hint}"]
    return True, ["kernel_from_counts(): passed."]


def _test_matrices(training_kernel, test_kernel):
    X = [np.array(p) for p in _PTS]
    calls = []

    def est(a, b):
        calls.append((tuple(a), tuple(b)))
        return round(0.1 + 0.2 * a[0] + 0.05 * b[1], 6)
    got, err = _try(training_kernel, "training_kernel", X, est)
    if err:
        return False, [err]
    want = reference_training_kernel(X, lambda a, b: round(0.1 + 0.2 * a[0] + 0.05 * b[1], 6))
    try:
        g = np.array(got, dtype=float)
    except Exception:  # noqa: BLE001
        g = None
    if g is None or g.shape != want.shape or np.max(np.abs(g - want)) > 1e-12:
        hint = ""
        if g is not None and g.shape == want.shape and not np.allclose(np.diag(g), 1):
            hint = " Put 1 on the diagonal: K(x, x) = 1 needs no circuit."
        elif g is not None and g.shape == want.shape and not np.allclose(g, g.T):
            hint = " Copy each estimate to [j][i]: one circuit per pair, and the matrix must be symmetric."
        elif len(calls) != 6:
            hint = (f" It called estimate {len(calls)} times for 4 points; use one call per pair i < j (6 for 4 points), "
                    "estimate(X[i], X[j]), and copy that value to [j][i].")
        return False, [f"training_kernel() does not give the expected matrix.{hint}"]
    if len(calls) != 6:
        return False, [f"training_kernel() called estimate {len(calls)} times for 4 points; it needs one call per pair i < j "
                       "(6 for 4 points, 28 for 8): every call is a circuit on the hardware."]
    calls.clear()
    got, err = _try(test_kernel, "test_kernel", X[:2], X, est)
    if err:
        return False, [err]
    want = reference_test_kernel(X[:2], X, lambda a, b: round(0.1 + 0.2 * a[0] + 0.05 * b[1], 6))
    try:
        g = np.array(got, dtype=float)
    except Exception:  # noqa: BLE001
        g = None
    if g is None or g.shape != want.shape or np.max(np.abs(g - want)) > 1e-12:
        hint = ""
        if g is not None and g.shape == want.T.shape:
            hint = " The shape is transposed: one row per test point, one column per training point."
        elif g is not None and g.shape == want.shape and np.allclose(
                g, reference_test_kernel(X[:2], X, lambda a, b: round(0.1 + 0.2 * b[0] + 0.05 * a[1], 6))):
            hint = " The arguments are swapped: call estimate(X_test[a], X_train[b]), the test point first."
        return False, [f"test_kernel() does not give the expected matrix.{hint}"]
    return True, ["training_kernel() and test_kernel(): passed."]


def check_l3_m6_kernel(fraction_sigma, expectation_sigma, combined_sigma, overlap_circuit, kernel_from_counts,
                       training_kernel, test_kernel, PAIR, P2, READOUT):
    try:
        pair, p2, ro = int(PAIR), float(P2), float(READOUT)
        ok = float(PAIR) == pair and 1 <= pair <= 28 and 0.005 <= p2 <= 0.030 and 0.005 <= ro <= 0.030
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return False, ["Copy your numbers again from your capstone quiz: PAIR (1 to 28), P2 and READOUT (0.005 to 0.030)."], None
    msgs = []
    for test in (lambda: test_statistics(fraction_sigma, expectation_sigma, combined_sigma),
                 lambda: _test_overlap(overlap_circuit), lambda: _test_counts(kernel_from_counts),
                 lambda: _test_matrices(training_kernel, test_kernel)):
        good, m = test()
        msgs += m
        if not good:
            return False, msgs, None
    return True, msgs, personal_value(pair, p2, ro)
