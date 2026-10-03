"""Accuracy test for the Module 6 check cell: correct solutions must pass with the same value, wrong ones must fail.

Run: PYTHONPATH=content/level1:checks python checks/test_m6_check.py
"""
import itertools

from qsim import QuantumCircuit

from m6_check import check_module6, personal_string, personal_value


# ---------- correct implementations (two ways each)
def or_a(s):
    n = len(s)
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if s[n - 1 - i] == "1":
            qc.cx(i, n)
    return qc


def or_b(s):
    n = len(s)
    qc = QuantumCircuit(n + 1)
    for i, bit in enumerate(reversed(s)):
        if bit == "1":
            qc.cx(i, n)
    return qc


def bv_a(oracle, n):
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    qc = qc.compose(oracle)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def bv_b(oracle, n):
    qc = QuantumCircuit(n + 1, n)
    for q in range(n):
        qc.h(q)
    qc.h(n)
    qc.z(n)                      # |-> made as Z H |0>
    qc.barrier()
    qc.compose(oracle, inplace=True)
    qc.barrier()
    for q in range(n):
        qc.h(q)
        qc.measure(q, q)
    return qc


def cl_a(query, n):
    return "".join(str(query("0" * j + "1" + "0" * (n - 1 - j))) for j in range(n))


def cl_b(query, n):
    bits = []
    for i in range(n):                          # qubit i: the character n - 1 - i
        x = ["0"] * n
        x[n - 1 - i] = "1"
        bits.append(str(query("".join(x))))
    return "".join(reversed(bits))


# ---------- wrong implementations
def or_reversed(s):                    # bit order reversed
    return or_a(s[::-1])


def or_cz(s):                          # CZ instead of CNOT, without H on the answer qubit
    n = len(s)
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if s[n - 1 - i] == "1":
            qc.cz(i, n)
    return qc


def or_backwards(s):                   # control and target swapped
    n = len(s)
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if s[n - 1 - i] == "1":
            qc.cx(n, i)
    return qc


def or_x(s):                           # X on the answer qubit, not controlled
    n = len(s)
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if s[n - 1 - i] == "1":
            qc.x(n)
    return qc


def bv_no_final_h(oracle, n):
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    qc = qc.compose(oracle)
    qc.measure(range(n), range(n))
    return qc


def bv_no_minus(oracle, n):            # answer qubit left in |+>, no kickback
    qc = QuantumCircuit(n + 1, n)
    qc.h(range(n + 1))
    qc = qc.compose(oracle)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def bv_measure_all(oracle, n):         # also measures the answer qubit
    qc = QuantumCircuit(n + 1, n + 1)
    qc.x(n)
    qc.h(range(n + 1))
    qc = qc.compose(oracle)
    qc.h(range(n))
    qc.measure(range(n + 1), range(n + 1))
    return qc


def bv_reversed(oracle, n):            # input qubit i into classical bit n - 1 - i
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    qc = qc.compose(oracle)
    qc.h(range(n))
    qc.measure(range(n), list(reversed(range(n))))
    return qc


def bv_twice(oracle, n):               # uses the oracle twice
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    qc = qc.compose(oracle).compose(oracle)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def bv_no_first_h(oracle, n):          # forgets H on the inputs before the oracle
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(n)
    qc = qc.compose(oracle)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def cl_reversed(query, n):
    return cl_a(query, n)[::-1]


def cl_brute(query, n):                # tries every input: 2^n queries
    answers = {"".join(x): query("".join(x)) for x in itertools.product("01", repeat=n)}
    return "".join(str(answers["0" * j + "1" + "0" * (n - 1 - j)]) for j in range(n))


def cl_offset(query, n):               # asks about the wrong position
    return "".join(str(query("0" * ((j + 1) % n) + "1" + "0" * (n - 1 - (j + 1) % n))) for j in range(n))


GOOD = list(itertools.product([or_a, or_b], [bv_a, bv_b], [cl_a, cl_b]))
BAD = ([(f, bv_a, cl_a) for f in (or_reversed, or_cz, or_backwards, or_x)]
       + [(or_a, f, cl_a) for f in (bv_no_final_h, bv_no_minus, bv_measure_all, bv_reversed, bv_twice, bv_no_first_h)]
       + [(or_a, bv_a, f) for f in (cl_reversed, cl_brute, cl_offset)])
PARAMS = [(512, 5), (100, 1), (999, 10)]

wrong = 0
values = {}
for combo in GOOD:
    for key, dep in PARAMS:
        ok, msgs, value = check_module6(*combo, key, dep)
        if not ok:
            wrong += 1
            print("WRONG: correct code rejected", [f.__name__ for f in combo], msgs[-1])
        values.setdefault((key, dep), set()).add(value)
for combo in BAD:
    ok, msgs, value = check_module6(*combo, 512, 5)
    if ok:
        wrong += 1
        print("WRONG: wrong code accepted", [f.__name__ for f in combo])
    else:
        print("  rejected", [f.__name__ for f in combo], msgs[-1][:150])
for k, v in values.items():
    assert len(v) == 1, (k, v)
print(f"{len(GOOD)} correct combinations x {len(PARAMS)} parameter sets, {len(BAD)} wrong implementations: "
      f"{wrong} wrong verdicts. Value for (512, 5): {personal_value(512, 5)} (hidden string {personal_string(512)})")
assert wrong == 0
