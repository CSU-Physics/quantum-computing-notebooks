"""Check cell logic for the Level 1 final coding task (three-qubit GHZ state), on qsim.

check_ghz(circuit, theta) returns (passed, messages, verification_value).
The verification value is <XXX> after Rz(theta) on qubit 0, computed from the learner's own state:
cos(theta) for the GHZ state, 0 for a classical mixture of 000 and 111.
"""
import numpy as np
from qsim import QuantumCircuit, AerSimulator, Statevector, state_fidelity, simulate_density_matrix

SHOTS = 1000
SEED = 2026
FIDELITY_MIN = 0.99
# Three standard deviations of shot noise for p = 0.5 at 1,000 shots: 3 * sqrt(0.25 / 1000) = 0.047
COUNT_TOLERANCE = 0.047


def _target_ghz():
    sv = np.zeros(8, dtype=complex)
    sv[0] = sv[7] = 1 / np.sqrt(2)
    return Statevector(sv)


def _strip_final_measurements(qc):
    """Remove only the last measurement on each qubit, if nothing but barriers follows it.

    Any other measurement or reset stays in, so a mid-circuit measurement (which turns the state
    into a classical mixture) is still simulated.
    """
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


def check_ghz(qc, theta):
    msgs = []
    if not isinstance(qc, QuantumCircuit):
        return False, ["build_ghz() must return a QuantumCircuit."], None
    if qc.num_qubits != 3:
        return False, [f"The circuit has {qc.num_qubits} qubits; the GHZ task needs exactly 3."], None
    if not any(inst.name == "measure" for inst in qc.data):
        return False, ["Add measurements of all three qubits at the end of the circuit."], None
    if qc.num_clbits != 3:
        return False, [f"The circuit has {qc.num_clbits} classical bits; use 3. Use either QuantumCircuit(3, 3) with "
                       "qc.measure([0, 1, 2], [0, 1, 2]), or QuantumCircuit(3) with qc.measure_all(), not both."], None

    # Test 1: the circuit runs and gives the expected counts.
    try:
        counts = AerSimulator(seed_simulator=SEED).run(qc, shots=SHOTS).result().get_counts()
    except Exception as exc:  # noqa: BLE001
        return False, [f"The circuit did not run: {exc}"], None
    bad = {k: v for k, v in counts.items() if k.replace(" ", "") not in ("000", "111")}
    p000 = sum(v for k, v in counts.items() if k.replace(" ", "") == "000") / SHOTS
    counts_ok = not bad and abs(p000 - 0.5) <= COUNT_TOLERANCE
    if bad:
        msgs.append(f"Unexpected outcomes {sorted(bad)}: a GHZ state gives only 000 and 111.")
    if abs(p000 - 0.5) > COUNT_TOLERANCE:
        msgs.append(f"000 appeared with probability {p000:.3f}; a GHZ state gives about 0.5.")
    msgs.append(f"Test 1, counts: {'passed' if counts_ok else 'not passed'} ({dict(counts)})")

    # Test 2: the state before measurement is the GHZ state, up to global phase.
    rho = simulate_density_matrix(_strip_final_measurements(qc))
    fid = state_fidelity(rho, _target_ghz())
    if fid < FIDELITY_MIN:
        hint = ("Your counts look right, but equal counts of 000 and 111 are not enough: the three qubits must "
                "share one coherent state, (|000> + |111>)/sqrt(2), and each qubit is measured only once, at the end."
                if counts_ok else "The target before measurement is (|000> + |111>)/sqrt(2).")
        msgs.append(f"Test 2, state: not passed (fidelity with the GHZ state {fid:.3f}, needs at least {FIDELITY_MIN}). {hint}")
    else:
        msgs.append(f"Test 2, state: passed (fidelity {fid:.3f})")

    if not (counts_ok and fid >= FIDELITY_MIN):
        return False, msgs, None

    # Verification value: <XXX> after a phase rotation Rz(theta) on qubit 0, from the learner's own state.
    rz = QuantumCircuit(3)
    rz.rz(theta, 0)
    value = float(np.real(rho.evolve(rz).expectation_value("XXX")))
    return True, msgs, round(value, 3)
