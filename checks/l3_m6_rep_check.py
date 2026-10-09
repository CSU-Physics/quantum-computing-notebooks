"""Check cell logic for the Level 3 Module 6 capstone "Repetition code: syndrome statistics and the logical error".

check_l3_m6_rep(fraction_sigma, expectation_sigma, combined_sigma, memory_circuit, detection_events, decode,
                ROUNDS, P2, READOUT) returns (passed, messages, verification_value).

The experiment (a memory experiment): encode a logical 0 or 1 in the three-qubit bit-flip code, measure the syndrome
for several rounds, then measure the three data qubits and decode by majority vote. Layout: data qubits 0, 1 and 2,
ancillas 3 and 4. memory_circuit(rounds, bit) has 2 * rounds + 3 classical bits:
- if bit is 1, X on qubit 0; then CX(0, 1) and CX(0, 2) (the encoding of Module 2);
- each round r = 0, 1, ...: an id gate on each data qubit (the data qubits wait while the ancillas are measured; the
  noise model puts the idle error on id), then CX(0, 3), CX(1, 3), CX(1, 4), CX(2, 4); measure ancilla 3 into
  classical bit 2r and ancilla 4 into classical bit 2r + 1; reset both ancillas;
- at the end, data qubit d is measured into classical bit 2 * rounds + d.

detection_events(key, rounds) returns rounds + 1 pairs: the syndrome of each round XOR the syndrome of the round
before (the round before the first counts as (0, 0)), and last the syndrome computed from the final data bits
(d0 XOR d1, d1 XOR d2) XOR the last measured syndrome. decode(key, rounds) returns the majority of the three final
data bits. Keys are in Qiskit's order: classical bit 0 is the rightmost character.

The verification value is round(100000 * P_L), where P_L is the exact probability that decode() gives 1 for a logical
0 after ROUNDS rounds, under the capstone noise model with two-qubit error P2 (one-qubit gates P2 / 10), idle error
P2 on each id gate and readout error READOUT (1 -> 0 with 2 * READOUT), from the reference circuit.
"""
import numpy as np

import qsim
from qsim import QuantumCircuit

from l3_m6_common import _try, apply_readout, capstone_noise_model, test_statistics

NAME = "repetition code"


# ---------------------------------------------------------------- reference solutions
def reference_memory_circuit(rounds, bit=0):
    qc = QuantumCircuit(5, 2 * rounds + 3)
    if bit:
        qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    for r in range(rounds):
        for d in (0, 1, 2):
            qc.id(d)
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)
        qc.measure(3, 2 * r)
        qc.measure(4, 2 * r + 1)
        qc.reset(3)
        qc.reset(4)
    for d in range(3):
        qc.measure(d, 2 * rounds + d)
    return qc


def _bit(key, c):
    return int(key[len(key) - 1 - c])


def reference_detection_events(key, rounds):
    syn = [(_bit(key, 2 * r), _bit(key, 2 * r + 1)) for r in range(rounds)]
    d = [_bit(key, 2 * rounds + j) for j in range(3)]
    syn.append((d[0] ^ d[1], d[1] ^ d[2]))
    prev, events = (0, 0), []
    for s in syn:
        events.append((s[0] ^ prev[0], s[1] ^ prev[1]))
        prev = s
    return events


def reference_decode(key, rounds):
    d = [_bit(key, 2 * rounds + j) for j in range(3)]
    return 1 if sum(d) >= 2 else 0


def bare_circuit(rounds, bit=0):
    """An unencoded qubit that waits as long: one id gate per round, then a measurement."""
    qc = QuantumCircuit(1, 1)
    if bit:
        qc.x(0)
    for _ in range(rounds):
        qc.id(0)
    qc.measure(0, 0)
    return qc


# ---------------------------------------------------------------- exact distributions
def record_distribution(qc, p2, readout, idle):
    """Exact probability of every recorded key of a circuit with mid-circuit measurements (no if_test), under the
    capstone noise model. Each mid-circuit measurement splits the state into branches (qsim's branch simulation);
    the final measurements are read from each branch's density matrix and then flipped by the readout confusion."""
    final = qc._final_measure_indices()
    meas = [qc.data[k] for k in sorted(final)]
    body = qc.copy_empty_like()
    body.data = [inst for k, inst in enumerate(qc.data) if k not in final]
    nm = capstone_noise_model(p2, readout, idle)
    branches = qsim._branch_states(body, nm if not nm.is_ideal() else None)
    qubits = [m.qubits[0] for m in meas]
    clbits = [m.clbits[0] for m in meas]
    out = {}
    for r, bits in branches:
        w = float(np.real(np.trace(r)))
        if w < 1e-15:
            continue
        probs = np.asarray(qsim.DensityMatrix(r / w).probabilities(qargs=qubits), dtype=float) * w
        probs = apply_readout(probs, readout)
        for o, p in enumerate(probs):
            if p < 1e-16:
                continue
            b = list(bits)
            for j, c in enumerate(clbits):
                b[c] = (o >> j) & 1
            key = "".join(str(x) for x in reversed(b))
            out[key] = out.get(key, 0.0) + float(p)
    return out


def logical_error(dist, rounds, bit=0, decode=reference_decode):
    return float(sum(p for k, p in dist.items() if decode(k, rounds) != bit))


def personal_error(rounds, p2, readout):
    dist = record_distribution(reference_memory_circuit(int(rounds), 0), p2, readout, p2)
    return logical_error(dist, int(rounds), 0)


def personal_value(rounds, p2, readout):
    return int(round(100000 * personal_error(int(rounds), float(p2), float(readout))))


# ---------------------------------------------------------------- tests
def _ops(qc):
    return [(inst.name, tuple(inst.qubits), tuple(inst.clbits)) for inst in qc.data]


def _test_circuit(memory_circuit):
    for rounds, bit in ((1, 0), (2, 1), (3, 0)):
        got, err = _try(memory_circuit, "memory_circuit", rounds, bit)
        if err:
            return False, [err]
        if not isinstance(got, QuantumCircuit):
            return False, [f"memory_circuit({rounds}, {bit}) returned {type(got).__name__}; return the QuantumCircuit."]
        if got.num_qubits != 5 or got.num_clbits != 2 * rounds + 3:
            return False, [f"memory_circuit({rounds}, {bit}) has {got.num_qubits} qubits and {got.num_clbits} classical bits; "
                           f"it needs 5 qubits and 2 * rounds + 3 = {2 * rounds + 3} classical bits."]
        ref = reference_memory_circuit(rounds, bit)
        names = [n for n, _, _ in _ops(got) if n != "barrier"]
        if names.count("id") != 3 * rounds:
            return False, [f"memory_circuit({rounds}, {bit}) has {names.count('id')} id gates; it needs one on each data qubit in "
                           f"every round ({3 * rounds}). The idle error of the noise model sits on them."]
        if names.count("reset") != 2 * rounds:
            return False, [f"memory_circuit({rounds}, {bit}) has {names.count('reset')} resets; reset both ancillas after "
                           "measuring them in every round, or the next round starts from the old syndrome."]
        if names.count("cx") != 2 + 4 * rounds:
            return False, [f"memory_circuit({rounds}, {bit}) has {names.count('cx')} CX gates; it needs 2 for the encoding and "
                           f"4 per round ({2 + 4 * rounds} in all). With noise, every extra gate changes the result."]
        meas = {cs[0]: qs[0] for n, qs, cs in _ops(got) if n == "measure"}
        want = {cs[0]: qs[0] for n, qs, cs in _ops(ref) if n == "measure"}
        if meas != want:
            return False, [f"memory_circuit({rounds}, {bit}): the measurements go to the wrong classical bits. Round r measures "
                           "ancilla 3 into bit 2r and ancilla 4 into bit 2r + 1; at the end data qubit d goes into bit 2 * rounds + d."]
        for p2, ro, idle in ((0.0, 0.0, 0.0), (0.03, 0.04, 0.02)):
            a = record_distribution(got, p2, ro, idle)
            b = record_distribution(ref, p2, ro, idle)
            tvd = 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))
            if tvd > 1e-9:
                if p2 == 0:
                    hint = (" Even without noise its results differ from the reference: check the encoding (X on qubit 0 for a "
                            "logical 1, then CX(0, 1) and CX(0, 2)) and the syndrome CNOTs (0 and 1 onto ancilla 3, 1 and 2 onto 4).")
                else:
                    hint = (" Without noise it is right, but with noise its results differ: the gates must come in the order of the "
                            "reference (id gates first in each round, then the four CNOTs, then measure and reset).")
                return False, [f"memory_circuit({rounds}, {bit}) does not give the expected results.{hint}"]
    return True, ["memory_circuit(): passed (1 to 3 rounds, logical 0 and 1, with and without noise)."]


_KEYS = [  # (key, rounds) with the clbit layout of memory_circuit
    ("000" + "00", 1), ("111" + "00", 1), ("001" + "01", 1), ("000" + "01", 1), ("011" + "01" + "11", 2),
    ("010" + "11" + "00" + "11", 3), ("100" + "00" + "10" + "00", 3), ("110" + "10" + "10", 2),
]


def _test_events(detection_events, decode):
    for key, rounds in _KEYS:
        got, err = _try(detection_events, "detection_events", key, rounds)
        if err:
            return False, [err]
        want = reference_detection_events(key, rounds)
        try:
            g = [tuple(int(x) for x in e) for e in got]
        except Exception:  # noqa: BLE001
            g = None
        if g != want:
            hint = ""
            syn = [(_bit(key, 2 * r), _bit(key, 2 * r + 1)) for r in range(rounds)]
            if g is not None and g[:rounds] == syn:
                hint = " These are the syndromes themselves; an event is a change: XOR each syndrome with the one before."
            elif g is not None and len(g) == rounds:
                hint = (" Add the last pair: the syndrome from the final data bits (d0 XOR d1, d1 XOR d2) compared with the "
                        "last measured syndrome.")
            elif g is not None and g[:rounds] == [(_bit(key, 2 * r + 1), _bit(key, 2 * r)) for r in range(rounds)]:
                hint = " The two syndrome bits are swapped: bit 2r (the rightmost of the pair) is the parity of qubits 0 and 1."
            return False, [f"detection_events({key!r}, {rounds}) gave {got!r}; expected {want}.{hint}"]
        got, err = _try(decode, "decode", key, rounds)
        if err:
            return False, [err]
        want = reference_decode(key, rounds)
        if isinstance(got, bool) or got not in (0, 1) or int(got) != want:
            hint = ""
            if got == 1 - want:
                hint = " Read the data bits from classical bits 2 * rounds to 2 * rounds + 2 (the leftmost three characters)."
            return False, [f"decode({key!r}, {rounds}) gave {got!r}; expected {want} (the majority of the three final data bits).{hint}"]
    return True, ["detection_events() and decode(): passed (eight keys, 1 to 3 rounds)."]


def check_l3_m6_rep(fraction_sigma, expectation_sigma, combined_sigma, memory_circuit, detection_events, decode,
                    ROUNDS, P2, READOUT):
    try:
        rounds, p2, ro = int(ROUNDS), float(P2), float(READOUT)
        ok = float(ROUNDS) == rounds and 1 <= rounds <= 5 and 0.005 <= p2 <= 0.030 and 0.005 <= ro <= 0.030
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return False, ["Copy your numbers again from your capstone quiz: ROUNDS (1 to 5), P2 and READOUT (0.005 to 0.030)."], None
    msgs = []
    for test in (lambda: test_statistics(fraction_sigma, expectation_sigma, combined_sigma),
                 lambda: _test_circuit(memory_circuit), lambda: _test_events(detection_events, decode)):
        good, m = test()
        msgs += m
        if not good:
            return False, msgs, None
    return True, msgs, personal_value(rounds, p2, ro)
