"""Accuracy test for the Module 3 check cell: correct solutions must pass with the same value, wrong ones must fail.

Run with: PYTHONPATH=content/level1:checks python checks/test_m3_check.py
"""
import math

import numpy as np
from m3_check import check_module3, personal_value

S2 = 1 / math.sqrt(2)
I2 = np.eye(2, dtype=complex)
PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
         "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
         "Z": np.array([[1, 0], [0, -1]], dtype=complex)}

# ---------- correct implementations (two ways each)
def gm_a(name):
    return {"X": [[0, 1], [1, 0]], "Y": [[0, -1j], [1j, 0]], "Z": [[1, 0], [0, -1]],
            "H": [[S2, S2], [S2, -S2]], "S": [[1, 0], [0, 1j]], "T": [[1, 0], [0, np.exp(1j * np.pi / 4)]]}[name]

def gm_b(name):
    if name in PAULI:
        return PAULI[name].copy()
    if name == "H":
        return (PAULI["X"] + PAULI["Z"]) / np.sqrt(2)
    return np.diag([1, np.exp(1j * np.pi / {"S": 2, "T": 4}[name])])

def rot_a(axis, theta):
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * PAULI[axis.upper()]

def rot_b(axis, theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if axis == "X":
        return np.array([[c, -1j * s], [-1j * s, c]])
    if axis == "Y":
        return np.array([[c, -s], [s, c]])
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])

def seq_a(mats):
    u = np.eye(2, dtype=complex)
    for g in mats:
        u = g @ u
    return u

def seq_b(mats):
    u = np.eye(2, dtype=complex)
    for g in reversed(mats):
        u = u @ np.asarray(g)
    return u

# ---------- wrong implementations
def gm_ysign(name):                       # Y with the signs swapped
    return np.array([[0, 1j], [-1j, 0]]) if name == "Y" else gm_a(name)

def gm_hnorm(name):                       # H without 1/sqrt(2)
    return np.array([[1, 1], [1, -1]]) if name == "H" else gm_a(name)

def gm_t_is_s(name):                      # T with angle pi/2
    return np.diag([1, 1j]) if name == "T" else gm_a(name)

def gm_s_conj(name):                      # S dagger instead of S
    return np.diag([1, -1j]) if name == "S" else gm_a(name)

def gm_t_global(name):                    # T written as Rz(pi/4): right up to a global phase only
    return np.diag([np.exp(-1j * np.pi / 8), np.exp(1j * np.pi / 8)]) if name == "T" else gm_a(name)

def rot_full(axis, theta):                # forgets the half angle
    return np.cos(theta) * I2 - 1j * np.sin(theta) * PAULI[axis.upper()]

def rot_plus(axis, theta):                # + i instead of - i
    return np.cos(theta / 2) * I2 + 1j * np.sin(theta / 2) * PAULI[axis.upper()]

def rot_phase(axis, theta):               # Rz written as the phase gate P(theta)
    return np.diag([1, np.exp(1j * theta)]) if axis == "Z" else rot_a(axis, theta)

def rot_noi(axis, theta):                 # forgets the i
    return np.cos(theta / 2) * I2 - np.sin(theta / 2) * PAULI[axis.upper()]

def seq_wrong_order(mats):                # multiplies left to right
    u = np.eye(2, dtype=complex)
    for g in mats:
        u = u @ g
    return u

def seq_inplace(mats):                    # right answer, but empties the list it was given
    u = np.eye(2, dtype=complex)
    while mats:
        u = mats.pop(0) @ u
    return u

def seq_first_only(mats):                 # starts from the first matrix and skips it in the loop... twice
    if not mats:
        return np.eye(2)
    u = np.asarray(mats[0], dtype=complex)
    for g in mats:
        u = g @ u
    return u

def seq_sum(mats):                        # adds instead of multiplying
    return sum(mats, np.zeros((2, 2), dtype=complex)) if mats else np.eye(2)

from qsim import QuantumCircuit


def _rt(undo, forward=("h", "t", "s", "h")):
    def f():
        qc = QuantumCircuit(1, 1)
        for g in forward:
            getattr(qc, g)(0)
        for g in undo:
            getattr(qc, g)(0)
        qc.measure(0, 0)
        return qc
    return f


rt_good = _rt(("h", "sdg", "tdg", "h"))
RT_BAD = {"undo left unfixed (T instead of T-dagger)": _rt(("h", "sdg", "t", "h")),
          "undo uses S instead of S-dagger": _rt(("h", "s", "tdg", "h")),
          "undo without the last H": _rt(("h", "sdg", "tdg")),
          "forward part deleted": _rt((), forward=()),
          "returns to |0> but is Z, not the identity": _rt(("h", "sdg", "tdg", "h", "z"))}

GOOD = [(g, r, s) for g in (gm_a, gm_b) for r in (rot_a, rot_b) for s in (seq_a, seq_b)]
BAD = [(gm_ysign, rot_a, seq_a), (gm_hnorm, rot_a, seq_a), (gm_t_is_s, rot_a, seq_a), (gm_s_conj, rot_a, seq_a),
       (gm_t_global, rot_a, seq_a), (gm_a, rot_full, seq_a), (gm_a, rot_plus, seq_a), (gm_a, rot_phase, seq_a),
       (gm_a, rot_noi, seq_a), (gm_a, rot_a, seq_wrong_order), (gm_a, rot_a, seq_inplace),
       (gm_a, rot_a, seq_first_only), (gm_a, rot_a, seq_sum)]

PARAMS = [(73, 141, 512), (20, 350, 100), (160, 10, 999)]
wrong_verdicts = 0
for theta, phi, seed in PARAMS:
    expected = personal_value(theta, phi, seed)
    for combo in GOOD:
        ok, msgs, value = check_module3(*combo, theta, phi, seed, rt_good)
        if not ok or value != expected:
            wrong_verdicts += 1
            print("WRONG: correct code rejected", [f.__name__ for f in combo], msgs[-1])
    for combo in BAD:
        ok, msgs, value = check_module3(*combo, theta, phi, seed, rt_good)
        if ok:
            wrong_verdicts += 1
            print("WRONG: wrong code accepted", [f.__name__ for f in combo])
        elif theta == 73:
            print(f"  rejected {[f.__name__ for f in combo]}: {msgs[-1][:110]}")
    for name, rt in RT_BAD.items():
        ok, msgs, value = check_module3(gm_a, rot_a, seq_a, theta, phi, seed, rt)
        if ok:
            wrong_verdicts += 1
            print("WRONG: broken round_trip accepted:", name)
        elif theta == 73:
            print(f"  rejected round_trip {name}: {msgs[-1][:100]}")
ok, msgs, value = check_module3(gm_a, rot_a, seq_a, None, None, None, rt_good)
assert not ok and value is None and "Enter THETA_DEG" in msgs[-1]
ok, msgs, value = check_module3(gm_a, rot_a, seq_a, 73, 141, 512)
assert not ok and value is None and "Step 7" in msgs[-1]
print(f"{len(GOOD)} correct combinations x {len(PARAMS)} parameter sets, {len(BAD)} wrong implementations, {len(RT_BAD)} broken round trips: "
      f"{wrong_verdicts} wrong verdicts. Value for (73, 141, 512): {personal_value(73, 141, 512)}")
assert wrong_verdicts == 0
