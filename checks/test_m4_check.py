"""Accuracy test for the Module 4 check cell: correct solutions must pass with the same value, wrong ones must fail.

Run with: PYTHONPATH=content/level1:checks python checks/test_m4_check.py
"""
import math

import numpy as np
from m4_check import check_module4, personal_value, personal_circuit
from qsim import QuantumCircuit, Statevector

# ---------- correct implementations (two ways each)
def ps_a(q0, q1):
    return np.kron(q1, q0)

def ps_b(q0, q1):
    a, b = q0
    c, d = q1
    return [c * a, c * b, d * a, d * b]          # |00>, |01>, |10>, |11> with qubit 0 on the right

def bell_a(name):
    qc = QuantumCircuit(2)
    if name in ("phi-", "psi-"):
        qc.x(0)
    if name in ("psi+", "psi-"):
        qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def bell_b(name):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    if name == "phi-":
        qc.z(0)
    elif name == "psi+":
        qc.x(1)
    elif name == "psi-":
        qc.x(1)
        qc.z(0)
    return qc

def corr_a(counts):
    same = counts.get("00", 0) + counts.get("11", 0)
    diff = counts.get("01", 0) + counts.get("10", 0)
    return (same - diff) / (same + diff)

def corr_b(counts):
    total = sum(counts.values())
    return sum((1 if k[0] == k[1] else -1) * v for k, v in counts.items()) / total

# ---------- wrong implementations
def ps_swapped(q0, q1):              # qubit order reversed
    return np.kron(q0, q1)

def ps_outer(q0, q1):                # a 2 x 2 table instead of 4 amplitudes, in the wrong order
    return np.outer(q0, q1)

def ps_add(q0, q1):                  # concatenates instead of a tensor product
    return np.concatenate([q0, q1])

def bell_measured(name):             # adds measurements
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def bell_all_phi(name):              # the same state for every name
    return bell_a("phi+")

def bell_no_h(name):                 # forgets H
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    return qc

def bell_psi_sign(name):             # psi- given the + sign
    return bell_a("psi+" if name == "psi-" else name)

def bell_three(name):                # 3 qubits
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def corr_only00(counts):             # counts only 00 as "same"
    total = sum(counts.values())
    return (2 * counts.get("00", 0) - total) / total

def corr_fraction(counts):           # returns the fraction of equal results, not the correlation
    total = sum(counts.values())
    return (counts.get("00", 0) + counts.get("11", 0)) / total

def corr_sign(counts):               # the sign the wrong way round
    return -corr_a(counts)

def corr_keyerror(counts):           # fails when a key is missing
    return (counts["00"] + counts["11"] - counts["01"] - counts["10"]) / sum(counts.values())

GOOD = [(p, b, c) for p in (ps_a, ps_b) for b in (bell_a, bell_b) for c in (corr_a, corr_b)]
BAD = [(ps_swapped, bell_a, corr_a), (ps_outer, bell_a, corr_a), (ps_add, bell_a, corr_a),
       (ps_a, bell_measured, corr_a), (ps_a, bell_all_phi, corr_a), (ps_a, bell_no_h, corr_a),
       (ps_a, bell_psi_sign, corr_a), (ps_a, bell_three, corr_a),
       (ps_a, bell_a, corr_only00), (ps_a, bell_a, corr_fraction), (ps_a, bell_a, corr_sign), (ps_a, bell_a, corr_keyerror)]

PARAMS = [(73, 141, 512), (20, 350, 100), (160, 10, 999)]
wrong = 0
for theta, phi, seed in PARAMS:
    expected = personal_value(theta, phi, seed)
    for combo in GOOD:
        ok, msgs, value = check_module4(*combo, theta, phi, seed)
        if not ok or value != expected:
            wrong += 1
            print("WRONG: correct code rejected", [f.__name__ for f in combo], msgs[-1])
    for combo in BAD:
        ok, msgs, value = check_module4(*combo, theta, phi, seed)
        if ok:
            wrong += 1
            print("WRONG: wrong code accepted", [f.__name__ for f in combo])
        elif theta == 73:
            print(f"  rejected {[f.__name__ for f in combo]}: {msgs[-1][:120]}")
ok, msgs, value = check_module4(ps_a, bell_a, corr_a, None, None, None)
assert not ok and value is None and "Enter THETA_DEG" in msgs[-1]

# The prediction printed by the check cell matches the exact probability of the personal circuit.
for t, p in [(73, 141), (20, 350), (160, 10), (90, 0)]:
    body = personal_circuit(t, p)
    body.remove_final_measurements()
    pr = Statevector(body).probabilities()
    exact = pr[0] + pr[3]
    assert abs(exact - (1 + math.sin(math.radians(t)) * math.cos(math.radians(p))) / 2) < 1e-12
print(f"{len(GOOD)} correct combinations x {len(PARAMS)} parameter sets, {len(BAD)} wrong implementations: "
      f"{wrong} wrong verdicts. Value for (73, 141, 512): {personal_value(73, 141, 512)}")
assert wrong == 0
