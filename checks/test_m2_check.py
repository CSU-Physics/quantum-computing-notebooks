"""Accuracy test for the Module 2 check cell: correct solutions must pass with the same value, wrong ones must fail.

Run with: PYTHONPATH=content/level1:checks python checks/test_m2_check.py
"""
import math

import numpy as np
from m2_check import check_module2, personal_value

# ---------- correct implementations (two ways each)
def sfa_a(theta, phi):
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])

def sfa_b(theta, phi):
    return [complex(math.cos(theta / 2)), complex(math.cos(phi), math.sin(phi)) * math.sin(theta / 2)]

def bv_a(state):
    a, b = state[0], state[1]
    c = np.conj(a) * b
    return np.array([2 * c.real, 2 * c.imag, abs(a) ** 2 - abs(b) ** 2])

def bv_b(state):
    X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.array([[1, 0], [0, -1]])
    s = np.asarray(state)
    return tuple(float(np.real(np.vdot(s, M @ s))) for M in (X, Y, Z))

def mib_a(qc, basis):
    new = qc.copy()
    if basis == "X":
        new.h(0)
    elif basis == "Y":
        new.sdg(0)
        new.h(0)
    new.measure(0, 0)
    return new

def mib_c(qc, basis):
    new = qc.copy()
    if basis == "X":
        new.ry(-math.pi / 2, 0)   # Ry(-pi/2) sends |+> to |0>, the same job as H here
    elif basis == "Y":
        new.rx(math.pi / 2, 0)    # Rx(pi/2) sends |+i> to |0>
    new.measure(0, 0)
    return new

# ---------- wrong implementations
def sfa_noh(theta, phi):          # forgets the half angle
    return np.array([np.cos(theta), np.exp(1j * phi) * np.sin(theta)])

def sfa_swap(theta, phi):         # phase on the wrong amplitude
    return np.array([np.exp(1j * phi) * np.cos(theta / 2), np.sin(theta / 2)])

def sfa_deg(theta, phi):          # treats radians as degrees
    t, p = math.radians(theta), math.radians(phi)
    return np.array([np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)])

def sfa_minus(theta, phi):        # wrong sign of the phase
    return np.array([np.cos(theta / 2), np.exp(-1j * phi) * np.sin(theta / 2)])

def bv_noconj(state):             # no complex conjugate
    a, b = state[0], state[1]
    c = a * b
    return np.array([2 * c.real, 2 * c.imag, abs(a) ** 2 - abs(b) ** 2])

def bv_nofactor(state):           # forgets the factor 2
    a, b = state[0], state[1]
    c = np.conj(a) * b
    return np.array([c.real, c.imag, abs(a) ** 2 - abs(b) ** 2])

def bv_zflip(state):              # z = |b|^2 - |a|^2
    a, b = state[0], state[1]
    c = np.conj(a) * b
    return np.array([2 * c.real, 2 * c.imag, abs(b) ** 2 - abs(a) ** 2])

def bv_complex(state):            # returns the complex number instead of x and y
    a, b = state[0], state[1]
    return np.array([2 * np.conj(a) * b, 2 * np.conj(a) * b, abs(a) ** 2 - abs(b) ** 2])

def mib_s(qc, basis):             # S instead of S-dagger
    new = qc.copy()
    if basis == "X":
        new.h(0)
    elif basis == "Y":
        new.s(0); new.h(0)
    new.measure(0, 0)
    return new

def mib_order(qc, basis):         # H before S-dagger
    new = qc.copy()
    if basis == "X":
        new.h(0)
    elif basis == "Y":
        new.h(0); new.sdg(0)
    new.measure(0, 0)
    return new

def mib_nothing(qc, basis):       # ignores the basis
    new = qc.copy(); new.measure(0, 0); return new

def mib_inplace(qc, basis):       # changes the original circuit
    if basis == "X":
        qc.h(0)
    elif basis == "Y":
        qc.sdg(0); qc.h(0)
    qc.measure(0, 0)
    return qc

def mib_nomeasure(qc, basis):     # forgets the measurement
    new = qc.copy()
    if basis == "X":
        new.h(0)
    elif basis == "Y":
        new.sdg(0); new.h(0)
    return new

def mib_b(qc, basis):             # Ry(+pi/2) sends |+> to |1>, so 0 and 1 are swapped
    new = qc.copy()
    if basis == "Y":
        new.rz(-math.pi / 2, 0)
    if basis in ("X", "Y"):
        new.ry(math.pi / 2, 0)
    new.measure(0, 0)
    return new

def mib_xy_swapped(qc, basis):    # X and Y rotations swapped
    return mib_a(qc, {"X": "Y", "Y": "X"}.get(basis, basis))


PARAMS = [(73, 141, 512), (25, 300, 101), (160, 45, 999)]
wrong = 0

def expect(ok_expected, label, f1, f2, f3, params):
    global wrong
    ok, msgs, value = check_module2(f1, f2, f3, *params)
    verdict = "ok " if ok == ok_expected else "WRONG"
    if ok != ok_expected:
        wrong += 1
    print(f"  {verdict} {label} {params}: {'value ' + str(value) if ok else msgs[-1][:110]}")
    return value

print("Correct combinations (must pass with the reference value):")
for params in PARAMS:
    ref = personal_value(*params)
    for f1 in (sfa_a, sfa_b):
        for f2 in (bv_a, bv_b):
            for f3 in (mib_a, mib_c):
                v = expect(True, f"{f1.__name__}+{f2.__name__}+{f3.__name__}", f1, f2, f3, params)
                if v != ref:
                    wrong += 1
                    print("   WRONG value", v, "expected", ref)

print("Wrong implementations (must fail with no value):")
for f1, f2, f3, label in [
    (sfa_noh, bv_a, mib_a, "half angle forgotten"),
    (sfa_swap, bv_a, mib_a, "phase on |0>"),
    (sfa_deg, bv_a, mib_a, "radians treated as degrees"),
    (sfa_minus, bv_a, mib_a, "phase sign flipped"),
    (sfa_a, bv_noconj, mib_a, "no complex conjugate"),
    (sfa_a, bv_nofactor, mib_a, "factor 2 forgotten"),
    (sfa_a, bv_zflip, mib_a, "z sign flipped"),
    (sfa_a, bv_complex, mib_a, "complex x and y"),
    (sfa_a, bv_a, mib_s, "S instead of S-dagger"),
    (sfa_a, bv_a, mib_order, "H before S-dagger"),
    (sfa_a, bv_a, mib_nothing, "basis ignored"),
    (sfa_a, bv_a, mib_inplace, "changes the input circuit"),
    (sfa_a, bv_a, mib_nomeasure, "no measurement"),
    (sfa_a, bv_a, mib_xy_swapped, "X and Y swapped"),
    (sfa_a, bv_a, mib_b, "wrong rotations"),
]:
    expect(False, label, f1, f2, f3, PARAMS[0])

ok, msgs, value = check_module2(sfa_a, bv_a, mib_a, None, None, None)
print("Missing personal values:", "ok " if (not ok and value is None) else "WRONG", msgs[-1])
wrong += 0 if (not ok and value is None) else 1
print(f"\n{wrong} wrong verdicts")
