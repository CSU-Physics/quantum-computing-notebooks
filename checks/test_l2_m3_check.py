"""Tests for checks/l2_m3_check.py: correct versions must pass, wrong ones must fail.
Run: PYTHONPATH=content/level1:checks python checks/test_l2_m3_check.py"""
import math
import sys
import time
from fractions import Fraction

from qsim import QuantumCircuit
from l2_m3_check import (check_l2_module3, c_amod15, qft_dagger, reference_order, reference_factors,
                         reference_order_finding, reference_period)


# ---------------- order
def ord_ok(a, N):
    for r in range(1, N + 1):
        if pow(a, r, N) == 1:
            return r


def ord_ok_loop(a, N):
    r, x = 1, a % N
    while x != 1:
        x, r = (x * a) % N, r + 1
    return r


def ord_off_by_one(a, N):
    return ord_ok(a, N) - 1


def ord_any_power(a, N):                   # returns a multiple of the order
    return 2 * ord_ok(a, N)


def ord_todo(a, N):
    raise NotImplementedError


# ---------------- factors
def fac_ok(a, r, N):
    if r % 2:
        return None
    h = pow(a, r // 2, N)
    if h == N - 1:
        return None
    return tuple(sorted((math.gcd(h - 1, N), math.gcd(h + 1, N))))


def fac_ok_list(a, r, N):                  # a sorted list is fine too
    if r % 2 == 1 or pow(a, r // 2, N) == N - 1:
        return None
    x = a ** (r // 2)
    return sorted([math.gcd(x + 1, N), math.gcd(x - 1, N)])


def fac_no_minus_one_check(a, r, N):
    if r % 2:
        return None
    h = pow(a, r // 2, N)
    return tuple(sorted((math.gcd(h - 1, N), math.gcd(h + 1, N))))


def fac_no_odd_check(a, r, N):
    h = pow(a, r // 2, N)
    if h == N - 1:
        return None
    return tuple(sorted((math.gcd(h - 1, N), math.gcd(h + 1, N))))


def fac_wrong_power(a, r, N):
    if r % 2:
        return None
    h = pow(a, r, N)
    return tuple(sorted((math.gcd(h - 1, N), math.gcd(h + 1, N))))


def fac_todo(a, r, N):
    raise NotImplementedError


# ---------------- order finding
def make_of(x_reg=True, h=True, inverse=True, powers=lambda k: 2 ** k, measure_reg=False, reversed_measure=False):
    def of(a, t):
        qc = QuantumCircuit(t + 4, t)
        if x_reg:
            qc.x(t)
        if h:
            for k in range(t):
                qc.h(k)
        for k in range(t):
            qc.compose(c_amod15(a, powers(k)), qubits=[k] + list(range(t, t + 4)), inplace=True)
        if inverse:
            qc.compose(qft_dagger(t), qubits=list(range(t)), inplace=True)
        if reversed_measure:
            for k in range(t):
                qc.measure(k, t - 1 - k)
        else:
            qc.measure(list(range(t)), list(range(t)))
        if measure_reg:
            qc.measure(t, 0)
        return qc
    return of


def of_ok_loop(a, t):                       # applies c_amod15(a, 1) 2^k times instead of power 2^k
    qc = QuantumCircuit(t + 4, t)
    qc.x(t)
    qc.h(range(t))
    for k in range(t):
        for _ in range(2 ** k):
            qc.compose(c_amod15(a, 1), qubits=[k, t, t + 1, t + 2, t + 3], inplace=True)
    qc.compose(qft_dagger(t), qubits=range(t), inplace=True)
    for k in range(t):
        qc.measure(k, k)
    return qc


def of_todo(a, t):
    raise NotImplementedError


# ---------------- period
def per_ok(counts, t, a, N):
    for key in sorted(counts, key=counts.get, reverse=True):
        r = Fraction(int(key, 2), 2 ** t).limit_denominator(N).denominator
        if pow(a, r, N) == 1:
            return r
    return None


def per_ok_smallest(counts, t, a, N):       # smallest valid candidate instead of the most frequent
    cands = {Fraction(int(k, 2), 2 ** t).limit_denominator(N).denominator for k in counts}
    good = sorted(r for r in cands if pow(a, r, N) == 1)
    return good[0] if good else None


def per_most_frequent_only(counts, t, a, N):
    key = max(counts, key=counts.get)
    return Fraction(int(key, 2), 2 ** t).limit_denominator(N).denominator


def per_no_limit(counts, t, a, N):          # forgets limit_denominator
    for key in sorted(counts, key=counts.get, reverse=True):
        r = Fraction(int(key, 2), 2 ** t).denominator
        if pow(a, r, N) == 1:
            return r
    return None


def per_largest(counts, t, a, N):            # returns the largest denominator, without checking it
    return max(Fraction(int(k, 2), 2 ** t).limit_denominator(N).denominator for k in counts)


def per_todo(counts, t, a, N):
    raise NotImplementedError


good_ord = [ord_ok, ord_ok_loop]
good_fac = [fac_ok, fac_ok_list]
good_of = [make_of(), of_ok_loop]
good_per = [per_ok, per_ok_smallest]
bad_ord = [ord_off_by_one, ord_any_power, ord_todo]
bad_fac = [fac_no_minus_one_check, fac_no_odd_check, fac_wrong_power, fac_todo]
bad_of = [make_of(x_reg=False), make_of(h=False), make_of(inverse=False), make_of(powers=lambda k: k + 1),
          make_of(measure_reg=True), make_of(reversed_measure=True), of_todo]
bad_per = [per_most_frequent_only, per_no_limit, per_largest, per_todo]
PARAMS = [(7, 512), (4, 101), (13, 999)]

assert [reference_order(a, 15) for a in (2, 4, 7, 8, 11, 13)] == [4, 2, 4, 4, 2, 4]
assert reference_factors(7, 4, 15) == (3, 5) and reference_factors(14, 2, 15) is None

wrong = 0
values = []
t0 = time.time()
for a, seed in PARAMS:
    for w in good_ord:
        for x in good_fac:
            for y in good_of:
                for z in good_per:
                    ok, msgs, v = check_l2_module3(w, x, y, z, a, seed)
                    if not ok:
                        wrong += 1; print("WRONG (should pass):", w.__name__, x.__name__, getattr(y, "__name__", y), z.__name__, msgs)
    values.append(check_l2_module3(ord_ok, fac_ok, make_of(), per_ok, a, seed)[2])
    groups = [(bad_ord, 0), (bad_fac, 1), (bad_of, 2), (bad_per, 3)]
    for bads, pos in groups:
        for i, b in enumerate(bads):
            fns = [ord_ok, fac_ok, make_of(), per_ok]
            fns[pos] = b
            ok, msgs, _ = check_l2_module3(*fns, a, seed)
            if ok:
                wrong += 1; print("WRONG (should fail):", pos, i, getattr(b, "__name__", b))
            elif a == 7:
                print("  ", pos, i, "->", [m for m in msgs if "passed" not in m][0][:150])
for bad in [(3, 512), (7, 99), (7.5, 512)]:
    if check_l2_module3(ord_ok, fac_ok, make_of(), per_ok, *bad)[0]:
        wrong += 1; print("WRONG: out-of-range parameters accepted", bad)
n_good = len(good_ord) * len(good_fac) * len(good_of) * len(good_per)
print(f"{n_good} correct combinations and {len(bad_ord) + len(bad_fac) + len(bad_of) + len(bad_per)} wrong versions "
      f"at {len(PARAMS)} parameter sets: {wrong} wrong verdicts ({time.time() - t0:.0f} s)")
print("values:", values)
sys.exit(1 if wrong else 0)
