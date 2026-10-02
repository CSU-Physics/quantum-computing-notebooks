"""Auto-grader accuracy test: run the GHZ check cell on 10 correct and 12 wrong submissions.

The pilot checklist asks for this before Level 1 opens. Run: python test_ghz_check.py
"""
import math

from qsim import QuantumCircuit

from ghz_check import check_ghz


def ghz(order=(0, 1, 2)):
    qc = QuantumCircuit(3)
    a, b, c = order
    qc.h(a)
    qc.cx(a, b)
    qc.cx(a, c)
    return qc


def with_measure(qc):
    qc.measure_all()
    return qc


def correct():
    out = {}
    out["H then two CNOTs from qubit 0, measure_all"] = with_measure(ghz())
    q = QuantumCircuit(3); q.h(0); q.cx(0, 1); q.cx(1, 2); out["CNOT chain 0-1-2"] = with_measure(q)
    out["control qubit 1"] = with_measure(ghz((1, 0, 2)))
    out["control qubit 2"] = with_measure(ghz((2, 1, 0)))
    q = QuantumCircuit(3); q.ry(math.pi / 2, 0); q.cx(0, 1); q.cx(0, 2); out["Ry(pi/2) instead of H"] = with_measure(q)
    q = ghz(); q.global_phase = 0.7; out["extra global phase"] = with_measure(q)
    q = QuantumCircuit(3, 3); q.h(0); q.cx(0, 1); q.cx(0, 2); q.measure(range(3), range(3)); out["explicit classical register"] = q
    q = QuantumCircuit(3); q.h(0); q.barrier(); q.cx(0, 1); q.barrier(); q.cx(0, 2); out["with barriers"] = with_measure(q)
    q = ghz(); q.x([0, 1, 2]); out["GHZ then X on all three"] = with_measure(q)
    q = QuantumCircuit(3); q.h(0); q.cx(0, 2); q.cx(0, 1); out["CNOT order swapped"] = with_measure(q)
    return out


def wrong():
    out = {}
    q = QuantumCircuit(3, 3); q.h(0); q.cx(0, 1); q.cx(0, 2); q.measure(0, 0); q.barrier(); q.measure(range(3), range(3))
    out["classical mixture: qubit 0 measured, then all three"] = q
    q = QuantumCircuit(3, 3); q.h(0); q.measure(0, 0); q.cx(0, 1); q.cx(0, 2); q.measure(range(3), range(3))
    out["classical mixture: measure before entangling"] = q
    q = ghz(); q.z(0); out["minus sign: |000> - |111>"] = with_measure(q)
    q = ghz(); q.s(0); out["phase i: |000> + i|111>"] = with_measure(q)
    q = QuantumCircuit(3); q.h(0); q.cx(0, 1); out["Bell pair, third qubit untouched"] = with_measure(q)
    q = QuantumCircuit(3); q.h([0, 1, 2]); out["H on all three"] = with_measure(q)
    q = QuantumCircuit(4); q.h(0); q.cx(0, 1); q.cx(0, 2); q.cx(0, 3); out["four-qubit GHZ"] = with_measure(q)
    out["no measurement"] = ghz()
    q = QuantumCircuit(3, 3); q.h(0); q.cx(0, 1); q.cx(0, 2); q.reset(2); q.cx(0, 2); q.measure(range(3), range(3))
    out["reset qubit 2 and re-entangle: mixed state"] = q
    q = QuantumCircuit(3); q.ry(math.pi / 3, 0); q.cx(0, 1); q.cx(0, 2); out["unequal superposition Ry(pi/3)"] = with_measure(q)
    q = QuantumCircuit(3, 3); q.h(0); q.cx(0, 1); q.cx(0, 2); q.measure(1, 1); q.measure(range(3), range(3))
    out["classical mixture: qubit 1 measured twice"] = q
    q = QuantumCircuit(3, 3); q.h(0); q.cx(0, 1); q.cx(0, 2); q.measure_all()
    out["six classical bits (QuantumCircuit(3, 3) plus measure_all)"] = q
    return out


def main():
    thetas = [0.37, 1.25, 2.84]
    errors = 0
    print("Correct submissions (must pass; value must equal cos(theta)):")
    for name, qc in correct().items():
        for th in thetas:
            ok, msgs, val = check_ghz(qc, th)
            good = ok and val is not None and abs(val - math.cos(th)) <= 0.002
            errors += not good
            if not good:
                print("   WRONG VERDICT", name, th, ok, val, msgs)
        print(f"  {'ok ' if all(check_ghz(qc, t)[0] for t in thetas) else 'BAD'} {name}")
    print("Wrong submissions (must fail and print no value):")
    for name, qc in wrong().items():
        ok, msgs, val = check_ghz(qc, 1.0)
        good = (not ok) and val is None
        errors += not good
        print(f"  {'ok ' if good else 'BAD'} {name}: {msgs[-1][:110]}")
    print(f"\n{errors} wrong verdicts")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
