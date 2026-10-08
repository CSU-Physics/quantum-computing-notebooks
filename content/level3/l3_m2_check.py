"""Check cell logic for the Level 3 Module 2 lab (the three-qubit repetition code).

check_l3_module2(encode, measure_syndrome, correct, THETA, P, Q) returns (passed, messages, verification_value).

Circuit layout used everywhere in the lab: data qubits 0, 1 and 2 (the state to protect starts on qubit 0),
ancilla qubits 3 and 4, classical bits 0 and 1 for the syndrome. The learner writes three functions that add
gates to a QuantumCircuit qc:
- encode(qc): alpha|0> + beta|1> on qubit 0 becomes alpha|000> + beta|111> on qubits 0, 1, 2;
- measure_syndrome(qc): records the parity of qubits 0 and 1 in classical bit 0 and the parity of qubits 1 and 2
  in classical bit 1, using ancillas 3 and 4 (and no measurement of the data qubits), then resets both ancillas;
- correct(qc): with qc.if_test on the two syndrome bits, applies X to the one data qubit the syndrome points to:
  (1, 0) -> qubit 0, (1, 1) -> qubit 1, (0, 1) -> qubit 2, (0, 0) -> nothing.

The verification value is round(1000 * F). F is the fidelity between the ideal encoded state
cos(THETA/2)|000> + sin(THETA/2)|111> and the data qubits after one round of the code: encode RY(THETA)|0>,
flip each data qubit independently with probability P, measure the syndrome with a readout error Q on each
syndrome bit, and correct. Every final error is either none (fidelity 1), X on all three qubits (a logical X,
fidelity sin^2 THETA) or a pattern that leaves the code space (fidelity 0), so
F = Prob(no final error) + sin^2(THETA) * Prob(final error XXX), summed over the 8 error patterns and the
4 readout outcomes. The tests also compute F by simulating the reference circuit with qsim.
"""
import itertools

import numpy as np

import qsim
from qsim import QuantumCircuit, NoiseModel, ReadoutError, pauli_error, simulate_density_matrix, partial_trace

SYNDROME_TO_QUBIT = {(1, 0): 0, (1, 1): 1, (0, 1): 2}


# ---------------------------------------------------------------- reference solutions
def reference_encode(qc):
    qc.cx(0, 1)
    qc.cx(0, 2)


def reference_measure_syndrome(qc):
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    qc.measure(3, 0)
    qc.measure(4, 1)
    qc.reset(3)
    qc.reset(4)


def reference_correct(qc):
    with qc.if_test((0, 1)):
        with qc.if_test((1, 1)):
            qc.x(1)
        with qc.if_test((1, 0)):
            qc.x(0)
    with qc.if_test((0, 0)):
        with qc.if_test((1, 1)):
            qc.x(2)


# ---------------------------------------------------------------- the verification value
def syndrome_of(error):
    """error = (e0, e1, e2), 1 where a data qubit is flipped; returns the syndrome (s1, s2)."""
    e0, e1, e2 = error
    return (e0 ^ e1, e1 ^ e2)


def personal_fidelity(theta, p, q):
    total = 0.0
    for err in itertools.product((0, 1), repeat=3):
        pe = np.prod([p if e else 1 - p for e in err])
        s = syndrome_of(err)
        for flips in itertools.product((0, 1), repeat=2):
            pf = np.prod([q if f else 1 - q for f in flips])
            meas = (s[0] ^ flips[0], s[1] ^ flips[1])
            fix = [0, 0, 0]
            if meas in SYNDROME_TO_QUBIT:
                fix[SYNDROME_TO_QUBIT[meas]] = 1
            final = tuple(a ^ b for a, b in zip(err, fix))
            if final == (0, 0, 0):
                total += pe * pf
            elif final == (1, 1, 1):
                total += pe * pf * np.sin(theta) ** 2
    return float(total)


def personal_value(theta, p, q):
    return int(round(1000 * personal_fidelity(float(theta), float(p), float(q))))


def codeword(theta, phi=0.0):
    v = np.zeros(8, dtype=complex)
    v[0] = np.cos(theta / 2)
    v[7] = np.exp(1j * phi) * np.sin(theta / 2)
    return v


def noise_model(p, q):
    """Bit flips with probability p on the data qubits (attached to an id gate on each) and readout error q on
    the two ancillas: the noise of the verification value and of lab Step 5."""
    nm = NoiseModel()
    if p > 0:
        for d in (0, 1, 2):
            nm.add_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["id"], [d])
    if q > 0:
        for a in (3, 4):
            nm.add_readout_error(ReadoutError([[1 - q, q], [q, 1 - q]]), [a])
    return nm


def one_round(theta, encode=reference_encode, measure_syndrome=reference_measure_syndrome, correct=reference_correct,
              phi=0.0, errors=None):
    """The circuit of one round: prepare RY(theta) (and RZ(phi)) on qubit 0, encode, an id gate on every data qubit
    (where the noise model puts its bit flips), optional fixed X errors, syndrome, correction."""
    qc = QuantumCircuit(5, 2)
    qc.ry(theta, 0)
    if phi:
        qc.rz(phi, 0)
    encode(qc)
    for d in (0, 1, 2):
        qc.id(d)
    for d in (errors or ()):
        qc.x(d)
    measure_syndrome(qc)
    correct(qc)
    return qc


def data_fidelity(qc, theta, nm=None, phi=0.0):
    rho = simulate_density_matrix(qc, nm)
    data = partial_trace(rho, [3, 4]).data
    v = codeword(theta, phi)
    return float(np.real(np.vdot(v, data @ v)))


# ---------------------------------------------------------------- helpers
def _call(fn, name, qc):
    n0 = len(qc.data)
    try:
        out = fn(qc)
    except NotImplementedError:
        return f"{name}() is not written yet."
    except qsim.CircuitError as e:
        return f"{name}() made an invalid circuit: {e}"
    except Exception as e:  # noqa: BLE001
        return f"{name}() raised {type(e).__name__}: {e}"
    if out is not None and out is not qc:
        return f"{name}(qc) should add gates to qc itself, not build and return a new circuit."
    if len(qc.data) == n0:
        return f"{name}(qc) added nothing to the circuit."
    return None


def _flat(insts):
    for i in insts:
        yield i
        if i.name == "if_else":
            yield from _flat(i.true_body)
            yield from _flat(i.false_body)


def _ops(qc, start):
    return list(_flat(qc.data[start:]))


# ---------------------------------------------------------------- tests
def _test_encode(encode):
    rng = np.random.default_rng(311)
    cases = [(0.0, 0.0), (np.pi, 0.0), (np.pi / 2, 0.0), (np.pi / 2, np.pi)] + \
        [(rng.uniform(0.2, 2.9), rng.uniform(-3, 3)) for _ in range(3)]
    for th, ph in cases:
        qc = QuantumCircuit(5, 2)
        qc.ry(th, 0)
        qc.rz(ph, 0)
        start = len(qc.data)
        err = _call(encode, "encode", qc)
        if err:
            return False, [err]
        ops = _ops(qc, start)
        if any(o.name in ("measure", "reset", "if_else") for o in ops):
            return False, ["encode() should use gates only: no measurements, resets or if_test blocks."]
        if any(q in (3, 4) for o in ops for q in o.qubits):
            return False, ["encode() should act on the data qubits 0, 1 and 2 only; qubits 3 and 4 are the ancillas."]
        got = qsim.Statevector(qc).data
        rz = np.array([np.exp(-1j * ph / 2), np.exp(1j * ph / 2)])
        a, b = rz * np.array([np.cos(th / 2), np.sin(th / 2)])
        want = np.zeros(32, dtype=complex)
        want[0], want[7] = a, b
        if abs(abs(np.vdot(want, got)) - 1) > 1e-8:
            names = [o.name for o in ops]
            if names.count("cx") == 1:
                hint = " Two CNOTs are needed: qubit 0 controls a CNOT onto qubit 1 and one onto qubit 2."
            elif "h" in names:
                hint = " The bit-flip code needs no H gates; those belong to the phase-flip code."
            else:
                hint = " Qubit 0 should be the control of both CNOTs; qubits 1 and 2 are the targets."
            return False, [f"encode() does not give a|000> + b|111> (tested with RY({th:.2f}) and RZ({ph:.2f})).{hint}"]
    return True, ["encode(): passed (7 input states)."]


def _test_syndrome(measure_syndrome):
    rng = np.random.default_rng(312)
    sim = qsim.AerSimulator()
    th = rng.uniform(0.5, 2.5)
    for errors in ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)):
        qc = QuantumCircuit(5, 2)
        qc.ry(th, 0)
        reference_encode(qc)
        for d in errors:
            qc.x(d)
        start = len(qc.data)
        err = _call(measure_syndrome, "measure_syndrome", qc)
        if err:
            return False, [err]
        ops = _ops(qc, start)
        if any(o.name == "measure" and o.qubits[0] in (0, 1, 2) for o in ops):
            return False, ["measure_syndrome() measures a data qubit. Measure only the ancillas 3 and 4: measuring a data "
                           "qubit destroys the superposition you are protecting."]
        if any(o.name == "if_else" for o in ops):
            return False, ["measure_syndrome() should only measure; the correction belongs in correct()."]
        e = [1 if d in errors else 0 for d in range(3)]
        want = syndrome_of(e)
        counts = sim.run(qc, shots=20, seed_simulator=7).result().get_counts()
        key = f"{want[1]}{want[0]}"
        if counts != {key: 20}:
            got = sorted(counts)
            swapped = f"{want[0]}{want[1]}"
            if counts == {swapped: 20} and want[0] != want[1]:
                hint = " The two classical bits are swapped: the parity of qubits 0 and 1 goes in classical bit 0."
            elif len(counts) > 1:
                hint = " The result is random, so the ancillas are not measuring parities. Each ancilla needs two CNOTs from data qubits."
            else:
                hint = " Ancilla 3 should get CNOTs from qubits 0 and 1, ancilla 4 from qubits 1 and 2."
            return False, [f"measure_syndrome() gives the counts {dict(counts)} for X errors on qubits {list(errors)}; "
                           f"expected the syndrome (bit 0, bit 1) = {want}, the counts key '{key}'.{hint}"]
        rho = simulate_density_matrix(qc).data
        v = np.zeros(8, dtype=complex)
        idx = int("".join(str(b) for b in reversed(e)), 2)
        v[idx], v[7 - idx] = np.cos(th / 2), np.sin(th / 2)
        full = np.zeros(32, dtype=complex)
        full[:8] = v                                  # ancillas (qubits 3 and 4) back in |00>
        f = float(np.real(np.vdot(full, rho @ full)))
        if f < 1 - 1e-8:
            anc = partial_trace(qsim.DensityMatrix(rho), [0, 1, 2]).data
            if np.real(anc[0, 0]) < 1 - 1e-8:
                return False, ["measure_syndrome() leaves an ancilla in |1>. Reset both ancillas (qc.reset(3) and qc.reset(4)) "
                               "after measuring them, so the next round starts from |0>."]
            return False, ["measure_syndrome() changes the encoded state. It should only copy parities onto the ancillas: "
                           "CNOTs with data qubits as controls and ancillas as targets."]
    return True, ["measure_syndrome(): passed (all 8 bit-flip patterns; ancillas reset; data state unchanged)."]


def _test_correct(correct, encode, measure_syndrome):
    rng = np.random.default_rng(313)
    for errors in ((), (0,), (1,), (2,)):
        th, ph = rng.uniform(0.4, 2.7), rng.uniform(-3, 3)
        qc = QuantumCircuit(5, 2)
        qc.ry(th, 0)
        qc.rz(ph, 0)
        reference_encode(qc)
        for d in errors:
            qc.x(d)
        reference_measure_syndrome(qc)
        start = len(qc.data)
        err = _call(correct, "correct", qc)
        if err:
            return False, [err]
        ops = _ops(qc, start)
        if any(o.name in ("measure", "reset") for o in ops):
            return False, ["correct() should not measure or reset; it reads the syndrome already in classical bits 0 and 1."]
        if not any(o.name == "if_else" for o in qc.data[start:]):
            return False, ["correct() needs qc.if_test blocks: which X to apply depends on the syndrome measured in this shot."]
        rho = simulate_density_matrix(qc)
        data = partial_trace(rho, [3, 4]).data
        rz = np.array([np.exp(-1j * ph / 2), np.exp(1j * ph / 2)])
        a, b = rz * np.array([np.cos(th / 2), np.sin(th / 2)])
        v = np.zeros(8, dtype=complex)
        v[0], v[7] = a, b
        f = float(np.real(np.vdot(v, data @ v)))
        if f < 1 - 1e-8:
            where = f"an X error on qubit {errors[0]}" if errors else "no error"
            return False, [f"After correct(), the state is not the encoded state again for {where} (fidelity {f:.3f}). "
                           "Syndrome (bit 0, bit 1) = (1, 0) points to qubit 0, (1, 1) to qubit 1, (0, 1) to qubit 2, "
                           "(0, 0) to no error. Nest one qc.if_test inside another to test both bits."]
    # the learner's three functions together
    for errors in ((), (0,), (1,), (2,)):
        th = rng.uniform(0.4, 2.7)
        try:
            qc = one_round(th, encode, measure_syndrome, correct, errors=errors)
        except Exception as e:  # noqa: BLE001
            return False, [f"Your three functions together raised {type(e).__name__}: {e}"]
        f = data_fidelity(qc, th)
        if f < 1 - 1e-8:
            return False, [f"encode(), measure_syndrome() and correct() each pass, but together they do not correct an X "
                           f"error on qubit {errors[0] if errors else '-'} (fidelity {f:.3f}). Check that they use the same "
                           "qubits and classical bits."]
    return True, ["correct(): passed (every single bit flip corrected, also with your encode() and measure_syndrome())."]


def check_l3_module2(encode, measure_syndrome, correct, THETA, P, Q):
    try:
        th, p, q = float(THETA), float(P), float(Q)
    except (TypeError, ValueError):
        return False, ["THETA, P and Q must be the numbers shown in your Canvas lab check."], None
    two = lambda x: abs(round(x, 2) - x) < 1e-9  # noqa: E731
    if not (0.3 <= th <= 2.8 and 0.01 <= p <= 0.30 and 0.0 <= q <= 0.10 and two(th) and two(p) and two(q)):
        return False, ["Copy your numbers again from the lab check: THETA (0.30 to 2.80), P (0.01 to 0.30) and Q "
                       "(0.00 to 0.10), each with two decimals."], None
    msgs, ok = [], True
    for good, m in (_test_encode(encode), _test_syndrome(measure_syndrome),
                    _test_correct(correct, encode, measure_syndrome)):
        msgs += m
        ok = ok and good
        if not ok:
            return False, msgs, None
    return True, msgs, personal_value(th, p, q)
