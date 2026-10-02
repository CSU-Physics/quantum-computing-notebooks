"""Auto-grader accuracy test for the Module 0 check cell: 3 correct and 7 wrong circuits."""
import math
from qsim import QuantumCircuit
from m0_check import check_rotation


def cases(theta):
    ok, bad = {}, {}
    q = QuantumCircuit(1, 1); q.ry(theta, 0); q.measure(0, 0); ok["ry(THETA)"] = q
    q = QuantumCircuit(1, 1); q.ry(theta, 0); q.barrier(); q.measure(0, 0); q.global_phase = 1.1; ok["ry with barrier and global phase"] = q
    q = QuantumCircuit(1, 1); q.ry(theta / 2, 0); q.ry(theta / 2, 0); q.measure(0, 0); ok["two half rotations"] = q
    q = QuantumCircuit(1, 1); q.measure(0, 0); bad["unchanged starter"] = q
    q = QuantumCircuit(1, 1); q.ry(theta + 0.4, 0); q.measure(0, 0); bad["wrong angle"] = q
    q = QuantumCircuit(1, 1); q.rx(theta, 0); q.measure(0, 0); bad["rx instead of ry"] = q
    q = QuantumCircuit(1, 1); q.ry(theta, 0); bad["no measurement"] = q
    q = QuantumCircuit(2, 2); q.ry(theta, 0); q.measure(0, 0); bad["two qubits"] = q
    q = QuantumCircuit(1, 1); q.ry(theta, 0); q.measure(0, 0); q.measure(0, 0); bad["measured twice"] = q
    q = QuantumCircuit(1, 1); q.ry(theta, 0); q.reset(0); q.ry(theta, 0); q.measure(0, 0); bad["reset in the middle"] = q
    return ok, bad


def main():
    errors = 0
    for th in (0.37, 1.25, 2.50):
        ok, bad = cases(th)
        for name, qc in ok.items():
            p, m, v = check_rotation(qc, th)
            good = p and v is not None and abs(v - math.sin(th / 2) ** 2) <= 0.002
            errors += not good
            print(f"  {'ok ' if good else 'BAD'} accept {name} theta={th}: {v}")
        for name, qc in bad.items():
            p, m, v = check_rotation(qc, th)
            good = (not p) and v is None
            errors += not good
            print(f"  {'ok ' if good else 'BAD'} reject {name}: {m[-1][:80]}")
    print(f"\n{errors} wrong verdicts")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
