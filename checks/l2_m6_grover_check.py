"""Check cell logic for the Level 2 Module 6 project "Grover's search with noise".

check_l2_m6_grover(grover_circuit, shot_sigma, within_3_sigma, difference_sigma, MARKED, P2, READOUT, my_counts)
returns (passed, messages, verification_value).

The learner writes grover_circuit(marked, iterations): H on every qubit, `iterations` rounds of the oracle (X gates on
the 0 bits, the prepared mcz, the same X gates) and the diffuser (H, X, mcz, X, H on every qubit), then measure qubit
q into classical bit q. The prepared mcz is cz on 2 qubits and a 3-qubit controlled Z written with 6 cx and 7 t / tdg
gates, so that the course noise model (which puts errors on one- and two-qubit gates) acts on every part of it.

my_counts are the learner's counts from grover_circuit(marked string of MARKED, 2) run with the course noise model
for P2 and READOUT and 4,000 shots. The verification value is round(10000 * p), where p is the exact probability, under
that noise model, that the reference circuit for MARKED with 2 iterations records the marked string.
"""
from l2_m6_common import *          # noqa: F401,F403  (the notebook pastes l2_m6_common above this line)

GROVER_CASES = [("11", 1), ("01", 1), ("10", 2), ("101", 2), ("000", 1), ("110", 3), ("011", 0), ("111", 2)]


def reference_mcz_ops(qubits):
    """The prepared mcz as a list of (name, qubits, params)."""
    if len(qubits) == 2:
        return [("cz", [qubits[0], qubits[1]], [])]
    a, b, c = qubits
    return [("cx", [b, c], []), ("tdg", [c], []), ("cx", [a, c], []), ("t", [c], []), ("cx", [b, c], []),
            ("tdg", [c], []), ("cx", [a, c], []), ("t", [b], []), ("t", [c], []), ("cx", [a, b], []),
            ("t", [a], []), ("tdg", [b], []), ("cx", [a, b], [])]


def reference_grover_ops(marked, iterations):
    n = len(marked)
    qs = list(range(n))
    ops = [("h", [q], []) for q in qs]
    zeros = [q for q in qs if marked[n - 1 - q] == "0"]
    for _ in range(iterations):
        ops += [("x", [q], []) for q in zeros]
        ops += reference_mcz_ops(qs)
        ops += [("x", [q], []) for q in zeros]
        ops += [("h", [q], []) for q in qs] + [("x", [q], []) for q in qs]
        ops += reference_mcz_ops(qs)
        ops += [("x", [q], []) for q in qs] + [("h", [q], []) for q in qs]
    return ops


def reference_grover(marked, iterations):
    n = len(marked)
    return build_circuit(reference_grover_ops(marked, iterations), n, list(range(n)))


def expected_success(marked, iterations, p2, readout):
    """Exact probability that the reference circuit records the marked string under the course noise model."""
    n = len(marked)
    dist = exact_probabilities(reference_grover_ops(marked, iterations), n, list(range(n)), p2, readout)
    return float(dist[int(marked, 2)])


def _test_grover(grover_circuit):
    for marked, it in GROVER_CASES:
        n = len(marked)
        try:
            qc = grover_circuit(marked, it)
        except NotImplementedError:
            return False, ["grover_circuit() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"grover_circuit('{marked}', {it}) raised {type(e).__name__}: {e}"]
        if not hasattr(qc, "data") or getattr(qc, "num_qubits", None) != n:
            return False, [f"grover_circuit('{marked}', {it}) should return a QuantumCircuit on {n} qubits."]
        big = big_gates(qc)
        if big:
            return False, [f"grover_circuit('{marked}', {it}) uses {', '.join(big)} on 3 qubits. Use the prepared mcz(qc, qubits): "
                           "it is written with cx, t and tdg, and the course noise model puts errors only on one- and "
                           "two-qubit gates, so a 3-qubit gate would run without any noise."]
        mm = measurement_map(qc)
        if mm != {q: q for q in range(n)}:
            return False, [f"grover_circuit('{marked}', {it}) should measure qubit q into classical bit q, for every qubit "
                           f"(found {mm or 'no measurements'})."]
        try:
            got = ideal_probabilities(qc)
        except Exception as e:  # noqa: BLE001
            return False, [f"The circuit from grover_circuit('{marked}', {it}) cannot be simulated: {type(e).__name__}: {e}"]
        want = ideal_probabilities(reference_grover(marked, it))
        if np.abs(got - want).max() > 1e-9:
            hint = ""
            flipped = format(int(marked[::-1], 2), f"0{n}b")
            if flipped != marked and np.abs(got - ideal_probabilities(reference_grover(flipped, it))).max() < 1e-9:
                hint = " The circuit marks the reversed string: qubit q is bit marked[n - 1 - q] (qubit 0 is on the right)."
            elif it > 0 and abs(got[int(marked, 2)] - 1 / 2 ** n) < 1e-9:
                hint = (" The marked string is not amplified: check that the circuit runs `iterations` rounds, each with "
                        "the oracle (X gates, mcz, X gates) and the diffuser (H, X, mcz, X, H).")
            elif it > 1 and np.abs(got - ideal_probabilities(reference_grover(marked, it - 1))).max() < 1e-9:
                hint = " It looks like one iteration too few."
            return False, [f"grover_circuit('{marked}', {it}) gives probabilities different from the expected ones "
                           f"(P('{marked}') = {got[int(marked, 2)]:.4f}, expected {want[int(marked, 2)]:.4f}).{hint}"]
        n2, want2 = two_qubit_count(qc), ops_two_qubit_count(reference_grover_ops(marked, it))
        if n2 != want2:
            return False, [f"grover_circuit('{marked}', {it}) gives the right probabilities but has {n2} two-qubit gates; "
                           f"the project's circuit has {want2}. Use the prepared mcz once in each oracle and once in each "
                           "diffuser, and no other two-qubit gates: with noise, every extra gate changes the result."]
    return True, [f"grover_circuit(): passed ({len(GROVER_CASES)} cases, 2 and 3 qubits, 0 to 3 iterations)."]


def _test_counts(my_counts, marked, p):
    if not isinstance(my_counts, dict) or not my_counts:
        return False, ["my_counts is empty. Run the cell of Step 7 that makes it first."]
    total = counts_total(my_counts)
    if total != SHOTS:
        return False, [f"my_counts holds {total} shots; use shots=4000 (Step 7)."]
    keys = list(my_counts)
    if any(len(str(k).replace(" ", "")) != len(marked) for k in keys):
        return False, [f"The keys of my_counts should be {len(marked)}-bit strings such as '{marked}'."]
    k = int(my_counts.get(marked, 0))
    sigma = math.sqrt(p * (1 - p) / SHOTS)
    z = (k / SHOTS - p) / sigma
    if abs(z) > 3:
        return False, [f"In my_counts, {k} of 4000 shots found {marked}: a fraction {k / SHOTS:.4f}, {z:+.1f} standard "
                       f"deviations from the noise model's expected value {p:.4f}. Check that my_counts comes from "
                       f"grover_circuit('{marked}', 2) run with course_noise_model(P2, READOUT) and 4,000 shots. If it does, "
                       "a result this far out happens by chance about 3 times in 1,000: run Step 7 again with another seed."]
    return True, [f"my_counts: {k} of 4000 shots found {marked} ({k / SHOTS:.4f}), within 3 standard deviations "
                  f"({z:+.2f}) of the expected {p:.4f}."]


def personal_value(marked, p2, readout):
    return int(round(10000 * expected_success(format(int(marked), "03b"), 2, float(p2), float(readout))))


def check_l2_m6_grover(grover_circuit, shot_sigma, within_3_sigma, difference_sigma, MARKED, P2, READOUT, my_counts):
    try:
        m = int(MARKED)
        if m != float(MARKED) or not 0 <= m <= 7:
            raise ValueError
    except (TypeError, ValueError):
        return False, ["MARKED is a whole number from 0 to 7. Copy it again from your project quiz."], None
    ok, msg, p2, ro = personal_noise_ok(P2, READOUT)
    if not ok:
        return False, [msg], None
    msgs, passed = [], True
    good, mm = test_statistics(shot_sigma, within_3_sigma, difference_sigma)
    msgs += mm
    passed = passed and good
    good, mm = _test_grover(grover_circuit)
    msgs += mm
    passed = passed and good
    if not passed:
        return False, msgs, None
    marked = format(m, "03b")
    p = expected_success(marked, 2, p2, ro)
    good, mm = _test_counts(my_counts, marked, p)
    msgs += mm
    if not good:
        return False, msgs, None
    return True, msgs, int(round(10000 * p))
