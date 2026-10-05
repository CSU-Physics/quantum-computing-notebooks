"""Tests for checks/l2_m1_check.py: correct versions must pass, wrong ones must fail.
Run: PYTHONPATH=content/level1:checks python checks/test_l2_m1_check.py"""
import math
import sys

from qsim import QuantumCircuit
from l2_m1_check import check_l2_module1, reference_qft


# ---------------- qft3: correct versions
def q3_ok():
    qc = QuantumCircuit(3)
    qc.h(2); qc.cp(math.pi / 2, 1, 2); qc.cp(math.pi / 4, 0, 2)
    qc.h(1); qc.cp(math.pi / 2, 0, 1)
    qc.h(0)
    qc.swap(0, 2)
    return qc


def q3_ok_flipped_controls():               # cp is symmetric: control and target may be swapped
    qc = QuantumCircuit(3)
    qc.h(2); qc.cp(math.pi / 2, 2, 1); qc.cp(math.pi / 4, 2, 0)
    qc.h(1); qc.cp(math.pi / 2, 1, 0)
    qc.h(0)
    qc.swap(2, 0)
    return qc


def q3_ok_swap_first():                     # swap first, then the same gates on mirrored qubits
    qc = QuantumCircuit(3)
    qc.swap(0, 2)
    qc.h(0); qc.cp(math.pi / 2, 1, 0); qc.cp(math.pi / 4, 2, 0)
    qc.h(1); qc.cp(math.pi / 2, 2, 1)
    qc.h(2)
    return qc


# ---------------- qft3: wrong versions
def q3_noswap():
    qc = q3_ok(); qc.data = [i for i in qc.data if i.name != "swap"]; return qc


def q3_sign():
    qc = QuantumCircuit(3)
    qc.h(2); qc.cp(-math.pi / 2, 1, 2); qc.cp(-math.pi / 4, 0, 2)
    qc.h(1); qc.cp(-math.pi / 2, 0, 1)
    qc.h(0); qc.swap(0, 2)
    return qc


def q3_angles():
    qc = QuantumCircuit(3)
    qc.h(2); qc.cp(math.pi / 4, 1, 2); qc.cp(math.pi / 8, 0, 2)
    qc.h(1); qc.cp(math.pi / 4, 0, 1)
    qc.h(0); qc.swap(0, 2)
    return qc


def q3_noh():
    qc = q3_ok(); qc.data = [i for i in qc.data if not (i.name == "h" and i.qubits == (0,))]; return qc


def q3_measure():
    qc = QuantumCircuit(3, 3); qc.compose(q3_ok(), inplace=True); qc.measure([0, 1, 2], [0, 1, 2]); return qc


def q3_todo():
    raise NotImplementedError


# ---------------- qft(n)
def qft_ok(n):
    return reference_qft(n)


def qft_ok_swap_first(n):
    qc = QuantumCircuit(n)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(math.pi / 2 ** (k - j), k, j)
    return qc


def qft_noswap(n):
    qc = reference_qft(n); qc.data = [i for i in qc.data if i.name != "swap"]; return qc


def qft_inverse(n):
    return reference_qft(n).inverse()


def qft_fixed3(n):                          # ignores n
    return reference_qft(3)


def qft_angle_off(n):
    qc = QuantumCircuit(n)
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(math.pi / 2 ** (j - k + 1), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def qft_allswaps(n):                        # swaps every pair twice: no reversal
    qc = reference_qft(n)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


# ---------------- phase_estimation
def make_pe(qft_fn, x_target=True, power=lambda k, t: 2 ** k, inverse=True, h=True, measure_target=False):
    def pe(phase, t):
        qc = QuantumCircuit(t + 1, t)
        if x_target:
            qc.x(t)
        if h:
            for k in range(t):
                qc.h(k)
        for k in range(t):
            qc.cp(2 * math.pi * phase * power(k, t), k, t)
        q = qft_fn(t).inverse() if inverse else qft_fn(t)
        qc.compose(q, qubits=list(range(t)), inplace=True)
        if measure_target:
            qc.measure(t, 0)
        qc.measure(list(range(t)), list(range(t)))
        return qc
    return pe


def pe_ok_loop_power(phase, t):             # applies cp 2^k times instead of one big angle
    qc = QuantumCircuit(t + 1, t)
    qc.x(t)
    qc.h(range(t))
    for k in range(t):
        for _ in range(2 ** k):
            qc.cp(2 * math.pi * phase, k, t)
    qc.compose(reference_qft(t).inverse(), qubits=list(range(t)), inplace=True)
    for k in range(t):
        qc.measure(k, k)
    return qc


def pe_todo(phase, t):
    raise NotImplementedError


good_q3 = [q3_ok, q3_ok_flipped_controls, q3_ok_swap_first]
good_qft = [qft_ok, qft_ok_swap_first]
good_pe = [make_pe(reference_qft), make_pe(qft_ok_swap_first), pe_ok_loop_power]
bad_q3 = [q3_noswap, q3_sign, q3_angles, q3_noh, q3_measure, q3_todo]
bad_qft = [qft_noswap, qft_inverse, qft_fixed3, qft_angle_off, qft_allswaps]
bad_pe = [make_pe(reference_qft, x_target=False), make_pe(reference_qft, inverse=False),
          make_pe(reference_qft, power=lambda k, t: 2 ** (t - 1 - k)), make_pe(reference_qft, h=False),
          make_pe(reference_qft, power=lambda k, t: k + 1), make_pe(qft_noswap), pe_todo]
PARAMS = [(0.432, 512), (0.05, 101), (0.95, 999)]

wrong = 0
values = []
for phi, seed in PARAMS:
    for a in good_q3:
        for b in good_qft:
            for c in good_pe:
                ok, msgs, v = check_l2_module1(a, b, c, phi, seed)
                if not ok:
                    wrong += 1; print("WRONG (should pass):", a.__name__, b.__name__, getattr(c, "__name__", c), msgs)
    values.append(check_l2_module1(q3_ok, qft_ok, make_pe(reference_qft), phi, seed)[2])
    for a in bad_q3:
        if check_l2_module1(a, qft_ok, make_pe(reference_qft), phi, seed)[0]:
            wrong += 1; print("WRONG (should fail):", a.__name__)
    for b in bad_qft:
        if check_l2_module1(q3_ok, b, make_pe(reference_qft), phi, seed)[0]:
            wrong += 1; print("WRONG (should fail):", b.__name__)
    for i, c in enumerate(bad_pe):
        if check_l2_module1(q3_ok, qft_ok, c, phi, seed)[0]:
            wrong += 1; print("WRONG (should fail): bad_pe", i)
if check_l2_module1(q3_ok, qft_ok, make_pe(reference_qft), 0.99, 512)[0]:
    wrong += 1; print("WRONG: out-of-range PHI accepted")
n_good = len(good_q3) * len(good_qft) * len(good_pe)
print(f"{n_good} correct combinations and {len(bad_q3) + len(bad_qft) + len(bad_pe)} wrong versions "
      f"at {len(PARAMS)} parameter sets: {wrong} wrong verdicts")
print("values:", values)
sys.exit(1 if wrong else 0)
