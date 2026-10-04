"""Check cell logic for the Level 2 Module 0 lab (dynamic circuits and teleportation), on qsim 1.5.0.

check_l2_module0(active_reset, bob_corrections, THETA, SEED)
returns (passed, messages, verification_value).

The learner writes two functions:
- active_reset(qc, qubit, clbit): measure the qubit into the classical bit and, with qc.if_test,
  flip it back to |0> when the result was 1 (without qc.reset);
- bob_corrections(qc): add Bob's two classically controlled corrections to the teleportation
  circuit (X on qubit 2 if c1 = 1, then Z on qubit 2 if c0 = 1).

The verification value is the number of shots, out of 1,000, in which Bob's qubit is measured as 1
after the reference teleportation circuit sends ry(THETA)|0>, run on qsim's AerSimulator with seed
SEED. It is printed only when every test passes, and it depends on the seeded draw, so it cannot be
worked out by hand.
"""
import numpy as np
from qsim import AerSimulator, QuantumCircuit, Statevector, partial_trace, simulate_density_matrix, state_fidelity


def prepare(theta, phi=0.0):
    """The state to send: ry(theta) then rz(phi) on qubit 0."""
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    if phi:
        qc.rz(phi, 0)
    return qc


def teleport_circuit(prep, corrections, measure_bob=False, after=None):
    """Alice holds qubit 0 (the state) and qubit 1; Bob holds qubit 2.

    c0 and c1 hold Alice's two results. A one-qubit circuit 'after' is applied to Bob's qubit at the end;
    with measure_bob=True, Bob's qubit is then measured into c2.
    """
    qc = QuantumCircuit(3, 3 if measure_bob else 2)
    qc.compose(prep, qubits=[0], inplace=True)
    qc.barrier()
    qc.h(1)
    qc.cx(1, 2)                  # Alice and Bob share the Bell pair (|00> + |11>)/sqrt(2)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)                      # Alice's Bell measurement
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.barrier()
    if corrections is not None:
        corrections(qc)
    if after is not None:
        qc.compose(after, qubits=[2], inplace=True)
    if measure_bob:
        qc.measure(2, 2)
    return qc


def reference_corrections(qc):
    with qc.if_test((1, 1)):
        qc.x(2)
    with qc.if_test((0, 1)):
        qc.z(2)


def bob_state(qc):
    """Bob's qubit (qubit 2) at the end of the circuit, averaged over Alice's results."""
    return partial_trace(simulate_density_matrix(qc), [0, 1])


def personal_value(theta, seed):
    qc = teleport_circuit(prepare(float(theta)), reference_corrections, measure_bob=True)
    counts = AerSimulator(seed_simulator=int(seed)).run(qc, shots=1000).result().get_counts()
    return int(sum(v for k, v in counts.items() if k[0] == "1"))


def _test_active_reset(active_reset):
    msgs, ok = [], True
    rng = np.random.default_rng(31)
    for k in range(6):
        th, ph = rng.uniform(0.2, 3.0), rng.uniform(0, 6.2)
        qc = QuantumCircuit(2, 2)
        qc.ry(th, 0)
        qc.rz(ph, 0)
        qc.h(1)
        qc.cx(0, 1)              # qubit 0 is entangled with qubit 1 in some tests
        target = k % 2
        try:
            out = active_reset(qc, target, target)
        except NotImplementedError:
            return False, ["active_reset() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"active_reset() raised {type(e).__name__}: {e}"]
        if out is not None and out is not qc:
            qc = out
        names = [i.name for i in qc.data]
        if "reset" in names:
            return False, ["Use a measurement and qc.if_test, not qc.reset(): this lab is about feedforward."]
        if "if_else" not in names or "measure" not in names:
            return False, ["active_reset() should measure the qubit and then use qc.if_test((clbit, 1))."]
        rho = partial_trace(simulate_density_matrix(qc), [1 - target])
        p1 = float(np.real(np.asarray(rho)[1, 1]))
        if p1 > 1e-9:
            ok = False
            msgs.append(f"After active_reset() qubit {target} is |1> with probability {p1:.3f}; it should always be |0>.")
            break
    if ok:
        msgs.append("active_reset(): passed (6 states, some entangled; the qubit always ends in |0>).")
    return ok, msgs


def _test_corrections(bob_corrections):
    rng = np.random.default_rng(57)
    worst = 0.0
    for k in range(12):
        th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        prep = prepare(th, ph)
        try:
            qc = teleport_circuit(prep, bob_corrections)
        except NotImplementedError:
            return False, ["bob_corrections() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"bob_corrections() raised {type(e).__name__}: {e}"]
        added = qc.data[len(teleport_circuit(prep, None).data):]
        if not any(i.name == "if_else" for i in added):
            return False, ["bob_corrections() should use qc.if_test on Alice's results c0 and c1."]
        if any(i.name in ("measure", "reset") for i in added):
            return False, ["bob_corrections() should only add corrections, not measurements or resets."]
        f = state_fidelity(Statevector(prep), bob_state(qc))
        worst = max(worst, 1 - f)
    if worst > 1e-9:
        return False, [f"Bob's qubit does not always end in the state Alice sent (lowest fidelity {1 - worst:.3f}). "
                       "Check which result controls which correction: c1 (the Bell-pair half) controls X, "
                       "c0 (the state qubit) controls Z."]
    return True, ["bob_corrections(): passed (12 random states arrive with fidelity 1.000000)."]


def check_l2_module0(active_reset, bob_corrections, THETA, SEED):
    msgs = []
    try:
        theta, seed = float(THETA), int(SEED)
    except (TypeError, ValueError):
        return False, ["THETA and SEED must be the numbers shown in your Canvas lab check."], None
    if not (0.2 <= theta <= 3.0) or not (100 <= seed <= 999):
        return False, ["THETA and SEED are outside the range Canvas uses. Copy them again from the lab check."], None
    ok1, m1 = _test_active_reset(active_reset)
    msgs += m1
    ok2, m2 = _test_corrections(bob_corrections)
    msgs += m2
    if not (ok1 and ok2):
        return False, msgs, None
    return True, msgs, personal_value(theta, seed)
