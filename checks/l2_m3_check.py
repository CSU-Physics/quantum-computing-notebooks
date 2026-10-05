"""Check cell logic for the Level 2 Module 3 lab (Shor's algorithm and period finding for N = 15), on qsim 1.8.0.

check_l2_module3(order, factors_from_order, order_finding, period_from_counts, A, SEED)
returns (passed, messages, verification_value).

The learner writes four functions:
- order(a, N): the smallest r > 0 with a^r mod N = 1, by classical search;
- factors_from_order(a, r, N): the two factors gcd(a^(r/2) - 1, N) and gcd(a^(r/2) + 1, N), sorted, or None when r
  is odd or a^(r/2) mod N = N - 1 (then the order gives no factor);
- order_finding(a, t): t counting qubits (0 to t - 1) in superposition, a 4-qubit register (qubits t to t + 3)
  in |1>, counting qubit k controlling the prepared c_amod15(a, 2^k), the inverse QFT on the counting qubits, and
  counting qubit k measured into classical bit k;
- period_from_counts(counts, t, a, N): the order r read from measured results with continued fractions
  (Fraction(m, 2^t).limit_denominator(N)), checked with a^r mod N = 1; None if no result gives it.

The verification value is the number of shots, out of 4,000, in which the course's reference order-finding circuit
for A with 8 counting qubits returns a result whose continued fraction has the true order of A as its denominator,
run on qsim's AerSimulator with seed SEED. It is printed only when every test passes.
"""
import math
from fractions import Fraction

import numpy as np
from qsim import AerSimulator, QuantumCircuit, Statevector

A_VALUES = (2, 4, 7, 8, 11, 13)


# ---------------------------------------------------------------- prepared parts (also given to learners)
def c_amod15(a, power):
    """Prepared: controlled multiplication by a^power mod 15, on qubits [control, x0, x1, x2, x3].
    Hard-coded for N = 15 and a in (2, 4, 7, 8, 11, 13): swaps and X gates that permute the 15 residues."""
    if a not in A_VALUES:
        raise ValueError("c_amod15 works only for a = 2, 4, 7, 8, 11 or 13")
    U = QuantumCircuit(5)
    for _ in range(power):
        if a in (2, 13):
            U.cswap(0, 3, 4); U.cswap(0, 2, 3); U.cswap(0, 1, 2)
        if a in (7, 8):
            U.cswap(0, 1, 2); U.cswap(0, 2, 3); U.cswap(0, 3, 4)
        if a in (4, 11):
            U.cswap(0, 2, 4); U.cswap(0, 1, 3)
        if a in (7, 11, 13):
            for q in range(1, 5):
                U.cx(0, q)
    return U


def qft_dagger(t):
    """Prepared: the inverse QFT on t qubits (the inverse of Module 1's qft(t), gates in reverse order)."""
    qc = QuantumCircuit(t)
    for i in range(t // 2):
        qc.swap(i, t - 1 - i)
    for j in range(t):
        for k in range(j):
            qc.cp(-math.pi / 2 ** (j - k), k, j)
        qc.h(j)
    return qc


# ---------------------------------------------------------------- reference solutions
def reference_order(a, N):
    r, x = 1, a % N
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def reference_factors(a, r, N):
    if r % 2:
        return None
    h = pow(a, r // 2, N)
    if h == N - 1:
        return None
    return tuple(sorted((math.gcd(h - 1, N), math.gcd(h + 1, N))))


def reference_order_finding(a, t, x_reg=True, h=True, inverse=True):
    qc = QuantumCircuit(t + 4, t)
    if x_reg:
        qc.x(t)
    if h:
        qc.h(list(range(t)))
    for k in range(t):
        qc.compose(c_amod15(a, 2 ** k), qubits=[k] + list(range(t, t + 4)), inplace=True)
    if inverse:
        qc.compose(qft_dagger(t), qubits=list(range(t)), inplace=True)
    qc.measure(list(range(t)), list(range(t)))
    return qc


def reference_period(counts, t, a, N):
    for key in sorted(counts, key=counts.get, reverse=True):
        r = Fraction(int(key, 2), 2 ** t).limit_denominator(N).denominator
        if pow(a, r, N) == 1:
            return r
    return None


def counting_distribution(qc, t):
    """Exact probabilities of the counting-register results (final measurements removed)."""
    c = qc.copy()
    c.remove_final_measurements(inplace=True)
    p = Statevector(c).probabilities(list(range(t)))
    return {format(m, f"0{t}b"): float(v) for m, v in enumerate(p) if v > 1e-12}


def personal_value(a, seed):
    a, t = int(a), 8
    r = reference_order(a, 15)
    counts = AerSimulator(seed_simulator=int(seed)).run(reference_order_finding(a, t), shots=4000).result().get_counts()
    return int(sum(c for k, c in counts.items()
                   if Fraction(int(k, 2), 2 ** t).limit_denominator(15).denominator == r))


# ---------------------------------------------------------------- tests
ORDER_CASES = [(7, 15), (2, 15), (4, 15), (11, 15), (13, 15), (2, 21), (5, 21), (10, 21), (3, 7), (2, 35)]


def _test_order(order):
    for a, N in ORDER_CASES:
        try:
            r = order(a, N)
        except NotImplementedError:
            return False, ["order() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"order({a}, {N}) raised {type(e).__name__}: {e}"]
        want = reference_order(a, N)
        if r != want:
            hint = ""
            if isinstance(r, int) and r > 0 and pow(a, r, N) == 1:
                hint = f" {a}^{r} mod {N} is 1, but a smaller power already gives 1: return the smallest r."
            elif r == want - 1:
                hint = " Off by one: count the powers a^1, a^2, ... and return the first r with a^r mod N = 1."
            return False, [f"order({a}, {N}) returned {r!r}; it should be {want}.{hint}"]
    return True, [f"order(): passed ({len(ORDER_CASES)} cases, N = 7 to 35)."]


FACTOR_CASES = [(7, 4, 15), (2, 4, 15), (4, 2, 15), (11, 2, 15), (13, 4, 15), (2, 6, 21), (10, 6, 21), (5, 6, 21),
                (14, 2, 15), (2, 3, 7)]


def _norm(x):
    if x is None:
        return None
    try:
        return tuple(sorted(int(v) for v in x))
    except Exception:  # noqa: BLE001
        return ("unreadable", repr(x))


def _test_factors(factors_from_order):
    for a, r, N in FACTOR_CASES:
        try:
            got = factors_from_order(a, r, N)
        except NotImplementedError:
            return False, ["factors_from_order() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"factors_from_order({a}, {r}, {N}) raised {type(e).__name__}: {e}"]
        want = reference_factors(a, r, N)
        if _norm(got) != want:
            if want is None and r % 2:
                hint = " r is odd, so a^(r/2) is not a whole power: return None."
            elif want is None:
                hint = f" Here {a}^{r // 2} mod {N} = {N - 1}, so both gcds are trivial (1 and {N}): return None."
            else:
                hint = " Use math.gcd(h - 1, N) and math.gcd(h + 1, N) with h = pow(a, r // 2, N), and return them sorted."
            return False, [f"factors_from_order({a}, {r}, {N}) returned {got!r}; it should be {want!r}.{hint}"]
    return True, [f"factors_from_order(): passed ({len(FACTOR_CASES)} cases, including odd r and a^(r/2) = -1 mod N)."]


OF_CASES = [(7, 3), (2, 4), (4, 4), (11, 3), (13, 5), (8, 4), (7, 6)]


def _test_order_finding(order_finding):
    for a, t in OF_CASES:
        try:
            qc = order_finding(a, t)
        except NotImplementedError:
            return False, ["order_finding() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"order_finding({a}, {t}) raised {type(e).__name__}: {e}"]
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != t + 4 or qc.num_clbits != t:
            return False, [f"order_finding({a}, {t}) should return a circuit with {t + 4} qubits and {t} classical bits."]
        meas = [i for i in qc.data if i.name == "measure"]
        if sorted(i.qubits[0] for i in meas) != list(range(t)) or any(i.qubits[0] != i.clbits[0] for i in meas):
            return False, ["Measure each counting qubit k (0 to t - 1) into classical bit k, and do not measure the register."]
        got = counting_distribution(qc, t)
        ref = counting_distribution(reference_order_finding(a, t), t)
        keys = set(got) | set(ref)
        if 0.5 * sum(abs(got.get(k, 0) - ref.get(k, 0)) for k in keys) > 1e-6:
            hint = ""
            if len(got) == 2 ** t and max(got.values()) - min(got.values()) < 1e-9:
                hint = (" Every result is equally likely: check that every counting qubit gets an H first and that the "
                        "inverse QFT, qft_dagger(t), is applied to the counting qubits before measuring.")
            for kw, h in ((dict(x_reg=False), " Start the register in |1>: an X on qubit t."),
                          (dict(h=False), " Put an H on every counting qubit first."),
                          (dict(inverse=False), " Apply the inverse QFT, qft_dagger(t), to the counting qubits before measuring.")):
                alt = counting_distribution(reference_order_finding(a, t, **kw), t)
                if not hint and 0.5 * sum(abs(got.get(k, 0) - alt.get(k, 0)) for k in set(got) | set(alt)) < 1e-6:
                    hint = h
                    break
            return False, [f"For a = {a} with {t} counting qubits, the results do not match the expected distribution.{hint} "
                           f"Counting qubit k controls c_amod15(a, 2**k), composed on qubits [k] + list(range(t, t + 4))."]
    return True, [f"order_finding(): passed ({len(OF_CASES)} cases, every a, 3 to 6 counting qubits; each distribution exact)."]


def _peaks(a, t):
    r = reference_order(a, 15)
    return {format(s * 2 ** t // r, f"0{t}b"): 1000 // r for s in range(r)}


PERIOD_CASES = [
    (_peaks(7, 8), 8, 7, 15),
    (_peaks(4, 8), 8, 4, 15),
    (_peaks(13, 4), 4, 13, 15),
    ({"10000000": 600, "00000000": 400, "01000000": 3}, 8, 2, 15),   # the 1/4 peak is rare but present
    ({"00101011": 300, "01010101": 310, "00000000": 290, "10101011": 50, "11010101": 40}, 8, 2, 21),  # near s/6
    ({"00000000": 500, "10000000": 500}, 8, 7, 15),                  # no result gives the order 4
]
PERIOD_WANT = [4, 2, 4, 4, 6, None]


def _test_period(period_from_counts):
    for (counts, t, a, N), want in zip(PERIOD_CASES, PERIOD_WANT):
        try:
            got = period_from_counts(dict(counts), t, a, N)
        except NotImplementedError:
            return False, ["period_from_counts() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"period_from_counts() raised {type(e).__name__}: {e} (counts with {len(counts)} results, t = {t}, a = {a}, N = {N})."]
        if got != want:
            if want is None:
                hint = (" No result in these counts gives a valid order (7^1 and 7^2 are not 1 mod 15), so return None "
                        "instead of a wrong r.")
            else:
                hint = (" For each result m, take Fraction(m, 2**t).limit_denominator(N).denominator and keep the first "
                        "candidate r with pow(a, r, N) == 1.")
            return False, [f"period_from_counts() returned {got!r} for a = {a}, N = {N} (results {sorted(counts)[:4]}...); it should be {want!r}.{hint}"]
    return True, [f"period_from_counts(): passed ({len(PERIOD_CASES)} cases, including N = 21 and counts that hold no valid order)."]


def check_l2_module3(order, factors_from_order, order_finding, period_from_counts, A, SEED):
    try:
        a, seed = int(A), int(SEED)
        if float(A) != a or float(SEED) != seed:
            raise ValueError
    except (TypeError, ValueError):
        return False, ["A and SEED must be the whole numbers shown in your Canvas lab check."], None
    if a not in A_VALUES or not (100 <= seed <= 999):
        return False, ["A must be one of 2, 4, 7, 8, 11, 13 and SEED between 100 and 999. Copy them again from the lab check."], None
    msgs, ok = [], True
    for test, fn in ((_test_order, order), (_test_factors, factors_from_order),
                     (_test_order_finding, order_finding), (_test_period, period_from_counts)):
        good, m = test(fn)
        msgs += m
        ok = ok and good
    if not ok:
        return False, msgs, None
    return True, msgs, personal_value(a, seed)
