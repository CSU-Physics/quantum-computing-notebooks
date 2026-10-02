"""Check cell logic for the Module 0 lab (one rotated qubit), on qsim.

check_rotation(circuit, theta) returns (passed, messages, verification_value).
The verification value is P(1) = sin^2(theta / 2), computed from the learner's own state.
"""
import numpy as np
from qsim import QuantumCircuit, Statevector, state_fidelity


def _strip_final_measurements(qc):
    """Remove only the last measurement on each qubit; any earlier measurement or reset stays in."""
    keep = list(qc.data)
    done = set()
    for idx in range(len(keep) - 1, -1, -1):
        inst = keep[idx]
        if inst.name == "barrier":
            continue
        if inst.name == "measure" and inst.qubits[0] not in done:
            keep[idx] = None
            done.add(inst.qubits[0])
            continue
        done.update(inst.qubits)
        if len(done) == qc.num_qubits:
            break
    body = qc.copy_empty_like()
    for inst in keep:
        if inst is not None:
            body.append(inst)
    return body


def check_rotation(qc, theta):
    if not isinstance(qc, QuantumCircuit):
        return False, ["build_rotation() must return a QuantumCircuit."], None
    if qc.num_qubits != 1:
        return False, [f"The circuit has {qc.num_qubits} qubits; this task uses exactly 1."], None
    if not any(inst.name == "measure" for inst in qc.data):
        return False, ["Add a measurement at the end: qc.measure(0, 0)."], None
    body = _strip_final_measurements(qc)
    if any(inst.name in ("measure", "reset") for inst in body.data):
        return False, ["Measure the qubit only once, at the end."], None
    body.remove_final_measurements()
    state = Statevector(body)
    target = Statevector([np.cos(theta / 2), np.sin(theta / 2)])
    fid = state_fidelity(state, target)
    if fid < 0.99:
        return False, [f"The state is not Ry(THETA) applied to |0> (fidelity {fid:.3f}). "
                       "Use qc.ry(THETA, 0) with your THETA from Step 5."], None
    p1 = float(state.probabilities()[1])
    return True, [f"State check passed (fidelity {fid:.3f})."], round(p1, 3)
