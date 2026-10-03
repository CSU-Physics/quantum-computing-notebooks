"""Accuracy test for the Module 5 check cell: correct solutions must pass with the same value, wrong ones must fail.

Run: PYTHONPATH=content/level1:checks python checks/test_m5_check.py
"""
import itertools
import math

import numpy as np

from m5_check import check_module5, personal_value

LABELS = ["00", "01", "10", "11"]


# ---------- correct implementations (two ways each)
def sn_a(p, shots):
    return np.sqrt(p * (1 - p) / shots)


def sn_b(p, shots):
    return math.sqrt(p - p * p) / math.sqrt(shots)


def ro_a(A0, A1):
    return np.kron(A1, A0)


def ro_b(A0, A1):
    m = np.zeros((4, 4))
    for i, j in itertools.product(range(4), range(4)):
        m[i, j] = A1[i >> 1, j >> 1] * A0[i & 1, j & 1]
    return m


def mit_a(counts, A):
    shots = sum(counts.values())
    p = np.array([counts.get(k, 0) / shots for k in LABELS])
    x = np.linalg.solve(A, p)
    return {k: float(v) for k, v in zip(LABELS, x)}


def mit_b(counts, A):
    p = np.array([counts.get(k, 0) for k in LABELS], dtype=float)
    x = np.linalg.inv(A) @ (p / p.sum())
    return dict(zip(LABELS, x))


# ---------- wrong implementations
def sn_variance(p, shots):            # forgets the square root
    return p * (1 - p) / shots


def sn_count(p, shots):               # spread of the count, not the fraction
    return np.sqrt(p * (1 - p) * shots)


def sn_no_1mp(p, shots):              # Poisson-style sqrt(p / shots)
    return np.sqrt(p / shots)


def sn_over_shots(p, shots):          # divides by shots outside the root
    return np.sqrt(p * (1 - p)) / shots


def ro_swapped(A0, A1):               # qubit order reversed
    return np.kron(A0, A1)


def ro_transposed(A0, A1):            # rows and columns swapped
    return np.kron(A1, A0).T


def ro_product(A0, A1):               # matrix product, not the tensor product
    return np.kron(A1 @ A0, np.eye(2)) / 1.0


def mit_multiply(counts, A):          # applies A instead of undoing it
    shots = sum(counts.values())
    p = np.array([counts.get(k, 0) / shots for k in LABELS])
    return dict(zip(LABELS, A @ p))


def mit_counts(counts, A):            # forgets to divide by shots
    p = np.array([counts.get(k, 0) for k in LABELS], dtype=float)
    return dict(zip(LABELS, np.linalg.solve(A, p)))


def mit_clip(counts, A):              # clips negative values and rescales
    x = np.array(list(mit_a(counts, A).values()))
    x = np.clip(x, 0, None)
    return dict(zip(LABELS, x / x.sum()))


def mit_keyerror(counts, A):          # fails when a result never appeared
    shots = sum(counts.values())
    p = np.array([counts[k] / shots for k in LABELS])
    return dict(zip(LABELS, np.linalg.solve(A, p)))


def mit_transposed(counts, A):        # solves with A transposed
    shots = sum(counts.values())
    p = np.array([counts.get(k, 0) / shots for k in LABELS])
    return dict(zip(LABELS, np.linalg.solve(A.T, p)))


GOOD = list(itertools.product([sn_a, sn_b], [ro_a, ro_b], [mit_a, mit_b]))
BAD = ([(f, ro_a, mit_a) for f in (sn_variance, sn_count, sn_no_1mp, sn_over_shots)]
       + [(sn_a, f, mit_a) for f in (ro_swapped, ro_transposed, ro_product)]
       + [(sn_a, ro_a, f) for f in (mit_multiply, mit_counts, mit_clip, mit_keyerror, mit_transposed)])
PARAMS = [(10, 3, 512), (2, 1, 100), (30, 8, 999)]

wrong = 0
values = {}
for combo in GOOD:
    for dep, ro, seed in PARAMS:
        ok, msgs, value = check_module5(*combo, dep, ro, seed)
        if not ok:
            wrong += 1
            print("WRONG: correct code rejected", [f.__name__ for f in combo], msgs[-1])
        values.setdefault((dep, ro, seed), set()).add(value)
for combo in BAD:
    ok, msgs, value = check_module5(*combo, 10, 3, 512)
    if ok:
        wrong += 1
        print("WRONG: wrong code accepted", [f.__name__ for f in combo])
    else:
        print("  rejected", [f.__name__ for f in combo], msgs[-1][:150])
for k, v in values.items():
    assert len(v) == 1, (k, v)
print(f"{len(GOOD)} correct combinations x {len(PARAMS)} parameter sets, {len(BAD)} wrong implementations: "
      f"{wrong} wrong verdicts. Value for (10, 3, 512): {personal_value(10, 3, 512)}")
assert wrong == 0
