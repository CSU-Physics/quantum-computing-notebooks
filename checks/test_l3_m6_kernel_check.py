"""Tests for l3_m6_kernel_check.py. Run from the repository root: python3 checks/test_l3_m6_kernel_check.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content", "level3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsim import QuantumCircuit  # noqa: E402
import l3_m6_kernel_check as c  # noqa: E402
from test_l3_m6_stats import run  # noqa: E402

OC, KC, TK, TE = c.reference_overlap_circuit, c.reference_kernel_from_counts, c.reference_training_kernel, c.reference_test_kernel
fm = c.feature_map


def overlap(x1, x2, first=None, second=None, meas=((0, 0), (1, 1)), extra=False):
    qc = QuantumCircuit(2, 2)
    qc.compose(first or fm(x1), inplace=True)
    qc.compose(second or fm(x2).inverse(), inplace=True)
    if extra:
        qc.barrier()
        qc.id(0)
        qc.x(0)
        qc.x(0)
    for q, b in meas:
        qc.measure(q, b)
    return qc


def overlap_alt(x1, x2):
    qc = QuantumCircuit(2, 2)
    qc.compose(fm(x1), inplace=True)
    qc.compose(fm(x2).inverse(), inplace=True)
    for i in range(2):
        qc.measure(i, i)
    return qc


def kernel_alt(counts):
    zero = [k for k in counts if set(k) == {"0"}]
    return sum(counts[k] for k in zero) / sum(counts.values())


def training_loop(X, estimate):
    m = len(X)
    K = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(m):
            if i < j:
                K[i][j] = estimate(X[i], X[j])
                K[j][i] = K[i][j]
    return K


def training_all(X, estimate):
    m = len(X)
    return np.array([[1.0 if i == j else estimate(X[i], X[j]) for j in range(m)] for i in range(m)])


def training_average(X, estimate):
    m = len(X)
    K = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            K[i, j] = K[j, i] = 0.5 * (estimate(X[i], X[j]) + estimate(X[j], X[i]))
    return K


def training_lower(X, estimate):
    m = len(X)
    K = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            K[i, j] = estimate(X[i], X[j])
    return K


def training_diag_estimated(X, estimate):
    m = len(X)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            K[i, j] = K[j, i] = estimate(X[i], X[j])
    return K


GOOD = [(OC, KC, TK, TE), (overlap_alt, kernel_alt, training_loop, lambda Xt, Xr, e: [[e(a, b) for b in Xr] for a in Xt]),
        (lambda a, b: overlap(a, b, extra=False), KC, TK, lambda Xt, Xr, e: np.array([[e(Xt[i], Xr[j]) for j in range(len(Xr))]
                                                                                      for i in range(len(Xt))]))]
BAD = {
    "no inverse": (lambda a, b: overlap(a, b, second=fm(b)), KC, TK, TE),
    "inverse of the first": (lambda a, b: overlap(a, b, first=fm(a).inverse(), second=fm(b)), KC, TK, TE),
    "no measurement": (lambda a, b: overlap(a, b, meas=()), KC, TK, TE),
    "measurements crossed": (lambda a, b: overlap(a, b, meas=((0, 1), (1, 0))), KC, TK, TE),
    "extra gates": (lambda a, b: overlap(a, b, extra=True), KC, TK, TE),
    "one repetition": (lambda a, b: overlap(a, b, first=fm(a, 1), second=fm(b, 1).inverse()), KC, TK, TE),
    "count not fraction": (OC, lambda cn: cn.get("0" * len(next(iter(cn))), 0), TK, TE),
    "fraction of 11": (OC, lambda cn: cn.get("1" * len(next(iter(cn))), 0) / sum(cn.values()), TK, TE),
    "fixed key 00": (OC, lambda cn: cn.get("00", 0) / sum(cn.values()), TK, TE),
    "every entry estimated": (OC, KC, training_all, TE),
    "averages both orders": (OC, KC, training_average, TE),
    "not mirrored": (OC, KC, training_lower, TE),
    "diagonal estimated": (OC, KC, training_diag_estimated, TE),
    "test kernel transposed": (OC, KC, TK, lambda Xt, Xr, e: np.array([[e(a, b) for a in Xt] for b in Xr])),
    "test arguments swapped": (OC, KC, TK, lambda Xt, Xr, e: np.array([[e(b, a) for b in Xr] for a in Xt])),
    "test kernel not written": (OC, KC, TK, lambda Xt, Xr, e: (_ for _ in ()).throw(NotImplementedError())),
}

if __name__ == "__main__":
    want = c.personal_value(1, 0.01, 0.02)
    assert want == 5583, want
    wrong = run("kernel", c.check_l3_m6_kernel, GOOD, BAD, (1, 0.01, 0.02), want,
                [(0, 0.01, 0.02), (29, 0.01, 0.02), (1, 0.04, 0.02), (1, 0.01, 0.001), (1.5, 0.01, 0.02), ("x", 1, 1)])
    sys.exit(1 if wrong else 0)
