"""Tests for checks/l2_m2_check.py: correct versions must pass, wrong ones must fail.
Run: PYTHONPATH=content/level1:checks python checks/test_l2_m2_check.py"""
import sys

from qsim import QuantumCircuit
from l2_m2_check import check_l2_module2, reference_diffuser, reference_oracle


def zeros_of(marked):
    n = len(marked)
    return [q for q in range(n) if marked[n - 1 - q] == "0"]


# ---------------- oracle: correct versions
def or_ok(marked):                          # H, mcx, H on the top qubit
    n = len(marked); qc = QuantumCircuit(n); z = zeros_of(marked)
    if z: qc.x(z)
    if n == 2:
        qc.cz(0, 1)
    else:
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
    if z: qc.x(z)
    return qc


def or_ok_ccz_target0(marked):              # ccz for 3 qubits, otherwise target qubit 0
    n = len(marked); qc = QuantumCircuit(n); z = zeros_of(marked)
    for q in z: qc.x(q)
    if n == 3:
        qc.ccz(1, 2, 0)
    elif n == 2:
        qc.h(0); qc.cx(1, 0); qc.h(0)
    else:
        qc.h(0); qc.mcx(list(range(1, n)), 0); qc.h(0)
    for q in z: qc.x(q)
    return qc


def or_ok_negated(marked):                  # flips every sign except the marked one (overall sign -1)
    qc = or_ok(marked)
    n = len(marked)
    qc.x(0); qc.z(0); qc.x(0); qc.z(0)     # X Z X Z = -I on qubit 0
    return qc


# ---------------- oracle: wrong versions
def or_reversed(marked):
    return or_ok(marked[::-1])


def or_complement(marked):
    return or_ok("".join("1" if b == "0" else "0" for b in marked))


def or_no_undo(marked):
    n = len(marked); qc = QuantumCircuit(n); z = zeros_of(marked)
    if z: qc.x(z)
    qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
    return qc if z else or_ok(marked).compose(QuantumCircuit(n).x(0) or QuantumCircuit(n))


def or_no_mcz(marked):
    n = len(marked); qc = QuantumCircuit(n); z = zeros_of(marked)
    if z: qc.x(z); qc.x(z)
    return qc


def or_mcx_not_mcz(marked):                 # forgets the H gates around the mcx
    n = len(marked); qc = QuantumCircuit(n); z = zeros_of(marked)
    if z: qc.x(z)
    qc.mcx(list(range(n - 1)), n - 1)
    if z: qc.x(z)
    return qc


def or_todo(marked):
    raise NotImplementedError


# ---------------- diffuser
def mcz_any(qc, n):
    if n == 2:
        qc.cz(0, 1)
    else:
        qc.h(0); qc.mcx(list(range(1, n)), 0); qc.h(0)


def df_ok(n):
    return reference_diffuser(n)


def df_ok_target0(n):
    qc = QuantumCircuit(n)
    for q in range(n): qc.h(q)
    for q in range(n): qc.x(q)
    mcz_any(qc, n)
    for q in range(n): qc.x(q)
    for q in range(n): qc.h(q)
    return qc


def df_no_x(n):
    qc = QuantumCircuit(n); qc.h(list(range(n))); mcz_any(qc, n); qc.h(list(range(n))); return qc


def df_no_h(n):
    qc = QuantumCircuit(n); qc.x(list(range(n))); mcz_any(qc, n); qc.x(list(range(n))); return qc


def df_missing_last_h(n):
    qc = QuantumCircuit(n); qc.h(list(range(n))); qc.x(list(range(n))); mcz_any(qc, n); qc.x(list(range(n))); return qc


def df_todo(n):
    raise NotImplementedError


# ---------------- grover
def make_grover(oracle=reference_oracle, diffuser=reference_diffuser, start_h=True, extra=0, use_measure_all=False,
                reverse_measure=False, skip_diffuser=False):
    def g(marked, iterations):
        n = len(marked)
        qc = QuantumCircuit(n) if use_measure_all else QuantumCircuit(n, n)
        if start_h:
            qc.h(list(range(n)))
        for _ in range(iterations + extra):
            qc.compose(oracle(marked), inplace=True)
            if not skip_diffuser:
                qc.compose(diffuser(n), inplace=True)
        if use_measure_all:
            qc.measure_all()
        elif reverse_measure:
            for q in range(n): qc.measure(q, n - 1 - q)
        else:
            qc.measure(list(range(n)), list(range(n)))
        return qc
    return g


def gr_no_measure(marked, iterations):
    n = len(marked); qc = QuantumCircuit(n, n); qc.h(list(range(n))); return qc


def gr_todo(marked, iterations):
    raise NotImplementedError


good_or = [or_ok, or_ok_ccz_target0, or_ok_negated]
good_df = [df_ok, df_ok_target0]
good_gr = [make_grover(), make_grover(oracle=or_ok_ccz_target0, diffuser=df_ok_target0), make_grover(use_measure_all=True)]
bad_or = [or_reversed, or_complement, or_no_undo, or_no_mcz, or_mcx_not_mcz, or_todo]
bad_df = [df_no_x, df_no_h, df_missing_last_h, df_todo]
bad_gr = [make_grover(start_h=False), make_grover(extra=1), make_grover(extra=-1), make_grover(skip_diffuser=True),
          make_grover(reverse_measure=True), make_grover(oracle=or_reversed), gr_no_measure, gr_todo]
PARAMS = [(13, 512), (0, 101), (15, 999)]

wrong = 0
values = []
for mk, seed in PARAMS:
    for a in good_or:
        for b in good_df:
            for c in good_gr:
                ok, msgs, v = check_l2_module2(a, b, c, mk, seed)
                if not ok:
                    wrong += 1; print("WRONG (should pass):", a.__name__, b.__name__, msgs)
    ok, msgs, v = check_l2_module2(or_ok, df_ok, make_grover(), mk, seed)
    values.append(v)
    for a in bad_or:
        ok, msgs, _ = check_l2_module2(a, df_ok, make_grover(), mk, seed)
        if ok:
            wrong += 1; print("WRONG (should fail):", a.__name__)
        elif mk == 13:
            print("  ", a.__name__, "->", [m for m in msgs if "passed" not in m][0][:150])
    for b in bad_df:
        ok, msgs, _ = check_l2_module2(or_ok, b, make_grover(), mk, seed)
        if ok:
            wrong += 1; print("WRONG (should fail):", b.__name__)
        elif mk == 13:
            print("  ", b.__name__, "->", [m for m in msgs if "passed" not in m][0][:150])
    for i, c in enumerate(bad_gr):
        ok, msgs, _ = check_l2_module2(or_ok, df_ok, c, mk, seed)
        if ok:
            wrong += 1; print("WRONG (should fail): bad_gr", i)
        elif mk == 13:
            print("   bad_gr", i, "->", [m for m in msgs if "passed" not in m][0][:150])
for bad in [(16, 512), (3, 99), (2.5, 512)]:
    if check_l2_module2(or_ok, df_ok, make_grover(), *bad)[0]:
        wrong += 1; print("WRONG: out-of-range parameters accepted", bad)
note = check_l2_module2(or_ok_negated, df_ok, make_grover(), 13, 512)[1][0]
print("note for the negated oracle:", note[-120:])
n_good = len(good_or) * len(good_df) * len(good_gr)
print(f"{n_good} correct combinations and {len(bad_or) + len(bad_df) + len(bad_gr)} wrong versions "
      f"at {len(PARAMS)} parameter sets: {wrong} wrong verdicts")
print("values:", values)
sys.exit(1 if wrong else 0)
