"""Check cell logic for the Level 2 Module 6 project "Phase estimation of 1/3 with noise".

check_l2_m6_qpe(qpe_circuit, estimate_error, shot_sigma, within_3_sigma, difference_sigma, COUNT, P2, READOUT, my_counts)
returns (passed, messages, verification_value).

The learner writes:
- qpe_circuit(m): m counting qubits (0 to m - 1) and one target qubit (m). X on the target (|1> is the eigenvector of
  the phase gate P(2*pi/3) with eigenvalue e^(2*pi*i/3), so the phase is 1/3), H on every counting qubit,
  cp(2*pi * (1/3) * 2**k, k, m) for every counting qubit k, the prepared inverse_qft on the counting qubits, then
  measure counting qubit k into classical bit k. The target is not measured.
- estimate_error(counts, m): the mean distance, over all shots, between the estimate y / 2**m and 1/3, measured around
  the circle (a phase of 0.95 is 0.05 away from 0), where y is the measured m-bit number.

my_counts are the learner's counts from qpe_circuit(COUNT) with the course noise model for P2 and READOUT and 4,000
shots. The verification value is round(10000 * p), where p is the exact probability, under that noise model, that the
reference circuit with COUNT counting qubits gives the best m-bit estimate of 1/3.
"""
from l2_m6_common import *          # noqa: F401,F403  (the notebook pastes l2_m6_common above this line)

PHASE = 1 / 3


def reference_iqft_ops(qubits):
    qubits = list(qubits)
    n = len(qubits)
    ops = [("swap", [qubits[j], qubits[n - 1 - j]], []) for j in range(n // 2)]
    for j in range(n):
        for k in range(j):
            ops.append(("cp", [qubits[k], qubits[j]], [-math.pi / 2 ** (j - k)]))
        ops.append(("h", [qubits[j]], []))
    return ops


def reference_qpe_ops(m, phase=PHASE):
    ops = [("x", [m], [])] + [("h", [k], []) for k in range(m)]
    ops += [("cp", [k, m], [2 * math.pi * phase * 2 ** k]) for k in range(m)]
    return ops + reference_iqft_ops(range(m))


def reference_qpe(m, phase=PHASE):
    return build_circuit(reference_qpe_ops(m, phase), m + 1, list(range(m)))


def circular_distance(y, m, phase=PHASE):
    d = abs(y / 2 ** m - phase) % 1
    return min(d, 1 - d)


def best_estimate(m, phase=PHASE):
    return min(range(2 ** m), key=lambda y: circular_distance(y, m, phase))


def noisy_distribution(m, p2, readout):
    return exact_probabilities(reference_qpe_ops(m), m + 1, list(range(m)), p2, readout)


def expected_best(m, p2, readout):
    return float(noisy_distribution(m, p2, readout)[best_estimate(m)])


def expected_error(m, p2, readout):
    d = noisy_distribution(m, p2, readout)
    return float(sum(d[y] * circular_distance(y, m) for y in range(2 ** m)))


def _counting_probs(qc, m):
    probs = ideal_probabilities(qc)
    out = np.zeros(2 ** m)
    for idx, pr in enumerate(probs):
        out[idx % 2 ** m] += pr
    return out


def _test_qpe(qpe_circuit):
    for m in (2, 3, 4, 5):
        try:
            qc = qpe_circuit(m)
        except NotImplementedError:
            return False, ["qpe_circuit() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"qpe_circuit({m}) raised {type(e).__name__}: {e}"]
        if not hasattr(qc, "data") or getattr(qc, "num_qubits", None) != m + 1:
            return False, [f"qpe_circuit({m}) should return a QuantumCircuit on {m + 1} qubits ({m} counting qubits and the target)."]
        big = big_gates(qc)
        if big:
            return False, [f"qpe_circuit({m}) uses {', '.join(big)} on 3 or more qubits. Use only one- and two-qubit gates "
                           "(cp, swap, h, x): the course noise model puts errors only on those."]
        mm = measurement_map(qc)
        if mm != {k: k for k in range(m)}:
            hint = ""
            if m in mm:
                hint = " Do not measure the target qubit: it stays in |1>."
            return False, [f"qpe_circuit({m}) should measure counting qubit k into classical bit k, for k = 0 to {m - 1} "
                           f"(found {mm or 'no measurements'}).{hint}"]
        try:
            got = _counting_probs(qc, m)
        except Exception as e:  # noqa: BLE001
            return False, [f"The circuit from qpe_circuit({m}) cannot be simulated: {type(e).__name__}: {e}"]
        want = _counting_probs(reference_qpe(m), m)
        if np.abs(got - want).max() > 1e-9:
            y = int(np.argmax(got))
            hint = ""
            if np.abs(got - _counting_probs(reference_qpe(m, 1 - PHASE), m)).max() < 1e-9:
                hint = (" The circuit estimates 2/3 instead of 1/3: the controlled phases have the wrong sign, or the "
                        "QFT is not inverted (use the prepared inverse_qft).")
            elif got[0] > 0.999:
                hint = " The counting qubits end in |0...0>: check the X on the target qubit and the cp angles."
            elif abs(got[int(format(best_estimate(m), f'0{m}b')[::-1], 2)] - want[best_estimate(m)]) < 1e-9:
                hint = " The bits come out in reverse order: counting qubit k should control cp(2*pi*phase*2**k)."
            return False, [f"qpe_circuit({m}) gives a different distribution from the expected one (most likely y = {y}, "
                           f"expected y = {best_estimate(m)}).{hint}"]
        n2, want2 = two_qubit_count(qc), ops_two_qubit_count(reference_qpe_ops(m))
        if n2 != want2:
            return False, [f"qpe_circuit({m}) gives the right distribution but has {n2} two-qubit gates; the project's "
                           f"circuit has {want2}. Use one cp(2 * pi * (1/3) * 2**k, k, m) for each counting qubit k (not "
                           "2**k repeated gates) and the prepared inverse_qft: with noise, every extra gate changes the result."]
    return True, ["qpe_circuit(): passed (2 to 5 counting qubits)."]


ERROR_CASES = [({"011": 1000}, 3), ({"010": 500, "011": 500}, 3), ({"0101": 3, "1111": 1}, 4),
               ({"000": 1, "100": 1, "111": 2}, 3), ({"01011": 2500, "01010": 1000, "11011": 500}, 5)]


def _test_error(estimate_error):
    for counts, m in ERROR_CASES:
        total = sum(counts.values())
        want = sum(v * circular_distance(int(k, 2), m) for k, v in counts.items()) / total
        try:
            got = estimate_error(dict(counts), m)
        except NotImplementedError:
            return False, ["estimate_error() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"estimate_error({counts}, {m}) raised {type(e).__name__}: {e}"]
        try:
            gotf = float(got)
        except (TypeError, ValueError):
            return False, [f"estimate_error({counts}, {m}) should return a number; it returned {got!r}."]
        if abs(gotf - want) > 1e-9:
            plain = sum(v * abs(int(k, 2) / 2 ** m - PHASE) for k, v in counts.items()) / total
            rev = sum(v * circular_distance(int(k[::-1], 2), m) for k, v in counts.items()) / total
            unweighted = np.mean([circular_distance(int(k, 2), m) for k in counts])
            hint = ""
            if abs(gotf - plain) < 1e-9:
                hint = " Measure the distance around the circle: min(d, 1 - d), so that 0.875 is 0.458 away from 1/3, not 0.542."
            elif abs(gotf - rev) < 1e-9:
                hint = " Read each key with int(key, 2): the leftmost character is the highest bit."
            elif abs(gotf - unweighted) < 1e-9:
                hint = " Weight each result by its count, and divide by the total number of shots."
            return False, [f"estimate_error({counts}, {m}) returned {gotf:.6f}; expected {want:.6f}.{hint}"]
    return True, [f"estimate_error(): passed ({len(ERROR_CASES)} cases)."]


def _test_counts(my_counts, m, p):
    if not isinstance(my_counts, dict) or not my_counts:
        return False, ["my_counts is empty. Run the cell of Step 8 that makes it first."]
    total = counts_total(my_counts)
    if total != SHOTS:
        return False, [f"my_counts holds {total} shots; use shots=4000 (Step 8)."]
    if any(len(str(k).replace(" ", "")) != m for k in my_counts):
        return False, [f"The keys of my_counts should be {m}-bit strings: run qpe_circuit(COUNT) with COUNT = {m}."]
    best = format(best_estimate(m), f"0{m}b")
    k = int(my_counts.get(best, 0))
    sigma = math.sqrt(p * (1 - p) / SHOTS)
    z = (k / SHOTS - p) / sigma
    if abs(z) > 3:
        return False, [f"In my_counts, {k} of 4000 shots gave the best estimate {best}: a fraction {k / SHOTS:.4f}, "
                       f"{z:+.1f} standard deviations from the noise model's expected value {p:.4f}. Check that my_counts "
                       "comes from qpe_circuit(COUNT) run with course_noise_model(P2, READOUT) and 4,000 shots. If it does, "
                       "a result this far out happens by chance about 3 times in 1,000: run Step 8 again with another seed."]
    return True, [f"my_counts: {k} of 4000 shots gave the best estimate {best} ({k / SHOTS:.4f}), within 3 standard "
                  f"deviations ({z:+.2f}) of the expected {p:.4f}."]


def personal_value(count, p2, readout):
    return int(round(10000 * expected_best(int(count), float(p2), float(readout))))


def check_l2_m6_qpe(qpe_circuit, estimate_error, shot_sigma, within_3_sigma, difference_sigma, COUNT, P2, READOUT, my_counts):
    try:
        m = int(COUNT)
        if m != float(COUNT) or not 3 <= m <= 6:
            raise ValueError
    except (TypeError, ValueError):
        return False, ["COUNT is a whole number from 3 to 6. Copy it again from your project quiz."], None
    ok, msg, p2, ro = personal_noise_ok(P2, READOUT)
    if not ok:
        return False, [msg], None
    msgs, passed = [], True
    for good, mm in (test_statistics(shot_sigma, within_3_sigma, difference_sigma), _test_qpe(qpe_circuit),
                     _test_error(estimate_error)):
        msgs += mm
        passed = passed and good
        if not good:
            break
    if not passed:
        return False, msgs, None
    p = expected_best(m, p2, ro)
    good, mm = _test_counts(my_counts, m, p)
    msgs += mm
    if not good:
        return False, msgs, None
    return True, msgs, int(round(10000 * p))
