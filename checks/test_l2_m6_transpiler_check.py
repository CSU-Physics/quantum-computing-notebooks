"""Tests for checks/l2_m6_transpiler_check.py, with real Qiskit and with qsim (as in the browser).
Run: PYTHONPATH=content/level2:checks python checks/test_l2_m6_transpiler_check.py          (Qiskit if installed)
     PYTHONPATH=content/level2:checks python checks/test_l2_m6_transpiler_check.py qsim     (force qsim)"""
import json
import math
import sys
from pathlib import Path

QSIM = len(sys.argv) > 1 and sys.argv[1] == "qsim"
if QSIM:
    sys.modules["qiskit"] = None
import numpy as np
import l2_m6_transpiler_check as chk
from l2_m6_transpiler_check import check_l2_m6_transpiler, QuantumCircuit

print("checker uses", QuantumCircuit.__module__)
SAVED = json.loads((Path(__file__).resolve().parents[1] / "content" / "level2" / "m6_transpiler_runs.json").read_text())


def sig(p, shots): return math.sqrt(p * (1 - p) / shots)
def w3(k, shots, p): return abs(k / shots - p) <= 3 * sig(p, shots)
def dsig(f1, f2, shots): return math.sqrt(sig(f1, shots) ** 2 + sig(f2, shots) ** 2)
def sig_bad(p, shots): return math.sqrt(p * (1 - p) * shots)


def qft(qc, qubits):
    for name, qs, params in chk.reference_qft_ops(list(qubits)):
        getattr(qc, name)(*params, *qs)


def inverse_qft(qc, qubits):
    qubits = list(qubits)
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n):
        for k in range(j):
            qc.cp(-math.pi / 2 ** (j - k), qubits[k], qubits[j])
        qc.h(qubits[j])


def m_ok(bits):
    n = len(bits)
    qc = QuantumCircuit(n, n)
    for q in range(n):
        if bits[n - 1 - q] == "1":
            qc.x(q)
    qft(qc, range(n))
    qc.barrier()
    inverse_qft(qc, range(n))
    qc.measure(range(n), range(n))
    return qc


def m_ok_list(bits):                       # X gates as one list call
    n = len(bits)
    qc = QuantumCircuit(n, n)
    ones = [q for q, b in enumerate(reversed(bits)) if b == "1"]
    if ones:
        qc.x(ones)
    qft(qc, list(range(n)))
    qc.barrier()
    inverse_qft(qc, list(range(n)))
    for q in range(n):
        qc.measure(q, q)
    return qc


def m_no_barrier(bits):
    n = len(bits)
    qc = QuantumCircuit(n, n)
    for q in range(n):
        if bits[n - 1 - q] == "1":
            qc.x(q)
    qft(qc, range(n)); inverse_qft(qc, range(n))
    qc.measure(range(n), range(n))
    return qc


def m_two_barriers(bits):
    qc = m_ok(bits)
    out = QuantumCircuit(len(bits), len(bits))
    out.barrier()
    return out.compose(qc)


def m_reversed(bits): return m_ok(bits[::-1])


def m_no_x(bits):
    n = len(bits)
    qc = QuantumCircuit(n, n)
    qft(qc, range(n)); qc.barrier(); inverse_qft(qc, range(n))
    qc.measure(range(n), range(n))
    return qc


def m_x_after(bits):                         # X gates at the end: returns bits, but the QFT acts on |0...0>
    n = len(bits)
    qc = QuantumCircuit(n, n)
    qft(qc, range(n)); qc.barrier(); inverse_qft(qc, range(n))
    for q in range(n):
        if bits[n - 1 - q] == "1":
            qc.x(q)
    qc.measure(range(n), range(n))
    return qc


def m_qft_twice(bits):
    n = len(bits)
    qc = QuantumCircuit(n, n)
    for q in range(n):
        if bits[n - 1 - q] == "1":
            qc.x(q)
    qft(qc, range(n)); qc.barrier(); qft(qc, range(n))
    qc.measure(range(n), range(n))
    return qc


def m_todo(bits): raise NotImplementedError


def s_ok(counts, bits): return counts.get(bits, 0) / sum(counts.values())
def s_count(counts, bits): return counts.get(bits, 0)
def s_key(counts, bits): return counts[bits] / sum(counts.values())
def s_4000(counts, bits): return counts.get(bits, 0) / 4000


def sp_ok(v): return float(np.std(v, ddof=1))
def sp_manual(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
def sp_pop(v): return float(np.std(v))
def sp_var(v): return float(np.var(v, ddof=1))
def sp_sem(v): return float(np.std(v, ddof=1) / np.sqrt(len(v)))


results = []


def expect(label, args, want):
    passed, msgs, value = check_l2_m6_transpiler(*args)
    ok = passed == want
    results.append(ok)
    print(("OK   " if ok else "FAIL ") + f"{label}: passed={passed} value={value}")
    if not ok or not passed:
        print("      " + msgs[-1][:230])
    return value


sols = json.loads((Path(__file__).resolve().parents[1] / "l2_m6_transpiler_solutions.json").read_text())
for d in (sols[0], sols[57], sols[199]):
    b, s = d["bits"], d["seed"]
    print(f"\n--- BITS {b}, SEED {s}: stored {d['value']}")
    v = expect("all correct", (m_ok, s_ok, sp_ok, sig, w3, dsig, b, s, SAVED), True)
    assert v == d["value"]
    expect("list form, manual spread", (m_ok_list, s_ok, sp_manual, sig, w3, dsig, b, s, SAVED), True)
    for name, fns in [("no barrier", (m_no_barrier, s_ok, sp_ok)), ("two barriers", (m_two_barriers, s_ok, sp_ok)),
                      ("reversed bits", (m_reversed, s_ok, sp_ok)), ("no X", (m_no_x, s_ok, sp_ok)),
                      ("X at the end", (m_x_after, s_ok, sp_ok)), ("QFT twice", (m_qft_twice, s_ok, sp_ok)),
                      ("mirror todo", (m_todo, s_ok, sp_ok)), ("success count", (m_ok, s_count, sp_ok)),
                      ("success KeyError", (m_ok, s_key, sp_ok)), ("success /4000", (m_ok, s_4000, sp_ok)),
                      ("spread population", (m_ok, s_ok, sp_pop)), ("spread variance", (m_ok, s_ok, sp_var)),
                      ("spread of the mean", (m_ok, s_ok, sp_sem))]:
        expect(name, (*fns, sig, w3, dsig, b, s, SAVED), False)
    expect("statistics wrong", (m_ok, s_ok, sp_ok, sig_bad, w3, dsig, b, s, SAVED), False)
    expect("seed not in the saved runs", (m_ok, s_ok, sp_ok, sig, w3, dsig, b, 100 if s != 100 else 101, SAVED) if
           not any(x["bits"] == b and x["seed"] == (100 if s != 100 else 101) for x in sols) else
           (m_ok, s_ok, sp_ok, sig, w3, dsig, b, 1000, SAVED), False)
    expect("saved not loaded", (m_ok, s_ok, sp_ok, sig, w3, dsig, b, s, None), False)

print(f"\n{sum(results)} of {len(results)} verdicts as expected")
sys.exit(0 if all(results) else 1)
