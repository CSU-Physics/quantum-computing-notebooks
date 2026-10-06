"""Check cell logic for the Level 2 Module 6 project "QAOA trained on a simulator, evaluated with noise".

check_l2_m6_qaoa(qaoa_circuit, mean_cut, shot_sigma, within_3_sigma, difference_sigma, LAYERS, P2, READOUT, my_counts)
returns (passed, messages, verification_value).

The learner writes:
- qaoa_circuit(gammas, betas, edges, n): as in Module 5 (H on every qubit; for each layer, rzz(2*gamma) on every edge
  and rx(2*beta) on every qubit), without measurements (the notebook's prepared measured(qc) adds them for runs);
- mean_cut(counts, edges): the average cut size over all shots, where bit q of a result (counted from the right) is
  node q's group.

The project graph has 5 nodes: a ring 0-1-2-3-4-0 and the chord (0, 2); its maximum cut is 5. The reference angles
(REFERENCE_ANGLES) were trained on the ideal simulator (a grid scan and COBYLA for 1 layer; COBYLA started from the
previous layer's angles for 2 and 3 layers) and rounded to 3 decimals.

my_counts are the learner's counts from qaoa_circuit with the reference angles for LAYERS layers, run with the course
noise model for P2 and READOUT and 4,000 shots. The verification value is round(1000 * C), where C is the exact
expected cut of the reference circuit under that noise model.
"""
from l2_m6_common import *          # noqa: F401,F403  (the notebook pastes l2_m6_common above this line)

PROJECT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
PROJECT_N = 5
REFERENCE_ANGLES = {1: ([2.817], [0.354]),
                    2: ([2.849, 2.627], [0.517, 0.309]),
                    3: ([2.905, 2.669, 2.619], [0.535, 0.401, 0.204])}


def cut_of(index, edges):
    """Cut size of the split whose node q is bit q of the integer index."""
    return sum(((index >> i) & 1) != ((index >> j) & 1) for i, j in edges)


def reference_qaoa_ops(gammas, betas, edges, n):
    ops = [("h", [q], []) for q in range(n)]
    for g, b in zip(gammas, betas):
        ops += [("rzz", [i, j], [2 * g]) for i, j in edges]
        ops += [("rx", [q], [2 * b]) for q in range(n)]
    return ops


def reference_qaoa(gammas, betas, edges, n):
    return build_circuit(reference_qaoa_ops(gammas, betas, edges, n), n, [])


def noisy_distribution(layers, p2, readout):
    g, b = REFERENCE_ANGLES[layers]
    return exact_probabilities(reference_qaoa_ops(g, b, PROJECT_EDGES, PROJECT_N), PROJECT_N, list(range(PROJECT_N)), p2, readout)


def _cuts():
    return np.array([cut_of(k, PROJECT_EDGES) for k in range(2 ** PROJECT_N)])


def expected_cut(layers, p2, readout):
    return float(noisy_distribution(layers, p2, readout) @ _cuts())


def cut_sigma(layers, p2, readout, shots=SHOTS):
    """The standard deviation of the mean cut of `shots` shots (the spread of one shot's cut, divided by sqrt(shots))."""
    d = noisy_distribution(layers, p2, readout)
    c = _cuts()
    mean = d @ c
    return float(math.sqrt(d @ (c - mean) ** 2 / shots))


def expected_max_cut_probability(layers, p2, readout):
    d = noisy_distribution(layers, p2, readout)
    c = _cuts()
    return float(d[c == c.max()].sum())


QAOA_CASES = [([0.4], [0.7], [(0, 1)], 2), ([2.817], [0.354], PROJECT_EDGES, 5),
              ([0.3, 0.9], [1.1, 0.2], [(0, 1), (1, 2), (0, 2)], 3), ([2.849, 2.627], [0.517, 0.309], PROJECT_EDGES, 5)]


def _test_qaoa(qaoa_circuit):
    for gammas, betas, edges, n in QAOA_CASES:
        label = f"qaoa_circuit({gammas}, {betas}, edges, {n})"
        try:
            qc = qaoa_circuit(list(gammas), list(betas), list(edges), n)
        except NotImplementedError:
            return False, ["qaoa_circuit() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"{label} raised {type(e).__name__}: {e}"]
        if not hasattr(qc, "data") or getattr(qc, "num_qubits", None) != n:
            return False, [f"{label} should return a QuantumCircuit on {n} qubits."]
        if measurement_map(qc):
            return False, [f"Leave measurements out of {label}, as in Module 5: the training uses the state itself, and "
                           "the prepared measured(qc) adds the measurements for the runs with shots."]
        try:
            got = ideal_probabilities(qc)
        except Exception as e:  # noqa: BLE001
            return False, [f"The circuit from {label} cannot be simulated: {type(e).__name__}: {e}"]
        want = ideal_probabilities(reference_qaoa(gammas, betas, edges, n))
        if np.abs(got - want).max() > 1e-9:
            half = ideal_probabilities(reference_qaoa([g / 2 for g in gammas], [b / 2 for b in betas], edges, n))
            hint = " Check each part: H on every qubit; then, for each layer, rzz(2 * gamma) on every edge and rx(2 * beta) on every qubit."
            if np.abs(got - half).max() < 1e-9:
                hint = " Use twice the angles: rzz(2 * gamma, i, j) and rx(2 * beta, q)."
            return False, [f"{label} gives probabilities different from the expected ones.{hint}"]
        n2, want2 = two_qubit_count(qc), len(edges) * len(gammas)
        if n2 != want2:
            return False, [f"{label} gives the right probabilities but has {n2} two-qubit gates; the project's circuit has "
                           f"{want2} (one rzz per edge in each layer). With noise, every extra gate changes the result."]
    return True, [f"qaoa_circuit(): passed ({len(QAOA_CASES)} cases, 1 and 2 layers, 2 to 5 nodes)."]


CUT_CASES = [({"01001": 10}, PROJECT_EDGES), ({"00000": 1, "11111": 1}, PROJECT_EDGES),
             ({"10": 3, "01": 1, "00": 4}, [(0, 1)]), ({"001": 2, "011": 1, "111": 1}, [(0, 1), (1, 2), (0, 2)]),
             ({"10110": 7, "00001": 2, "11000": 1}, PROJECT_EDGES)]


def _test_mean_cut(mean_cut):
    for counts, edges in CUT_CASES:
        want = sum(v * cut_of(int(k, 2), edges) for k, v in counts.items()) / sum(counts.values())
        try:
            got = mean_cut(dict(counts), list(edges))
        except NotImplementedError:
            return False, ["mean_cut() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"mean_cut({counts}, {edges}) raised {type(e).__name__}: {e}"]
        try:
            gotf = float(got)
        except (TypeError, ValueError):
            return False, [f"mean_cut({counts}, edges) should return a number; it returned {got!r}."]
        if abs(gotf - want) > 1e-9:
            rev = sum(v * cut_of(int(k[::-1], 2), edges) for k, v in counts.items()) / sum(counts.values())
            unw = np.mean([cut_of(int(k, 2), edges) for k in counts])
            tot = sum(v * cut_of(int(k, 2), edges) for k, v in counts.items())
            hint = ""
            if abs(gotf - rev) < 1e-9:
                hint = " Node q is the bit q places from the right: key[-1 - q], or bit q of int(key, 2)."
            elif abs(gotf - unw) < 1e-9:
                hint = " Weight each result by its count."
            elif abs(gotf - tot) < 1e-9:
                hint = " Divide by the total number of shots."
            return False, [f"mean_cut({counts}, {edges}) returned {gotf:.6f}; expected {want:.6f}.{hint}"]
    return True, [f"mean_cut(): passed ({len(CUT_CASES)} cases)."]


def _test_counts(my_counts, layers, p2, ro):
    if not isinstance(my_counts, dict) or not my_counts:
        return False, ["my_counts is empty. Run the cell of Step 8 that makes it first."]
    total = counts_total(my_counts)
    if total != SHOTS:
        return False, [f"my_counts holds {total} shots; use shots=4000 (Step 8)."]
    if any(len(str(k).replace(" ", "")) != PROJECT_N for k in my_counts):
        return False, ["The keys of my_counts should be 5-bit strings, one bit per node of the project graph."]
    c = sum(int(v) * cut_of(int(str(k).replace(" ", ""), 2), PROJECT_EDGES) for k, v in my_counts.items()) / SHOTS
    e, s = expected_cut(layers, p2, ro), cut_sigma(layers, p2, ro)
    z = (c - e) / s
    if abs(z) > 3:
        return False, [f"my_counts give a mean cut of {c:.4f}, {z:+.1f} standard deviations from the noise model's expected "
                       f"cut {e:.4f} for {layers} layer(s). Check that my_counts comes from qaoa_circuit with the reference "
                       "angles for LAYERS, run with course_noise_model(P2, READOUT) and 4,000 shots. If it does, a result this "
                       "far out happens by chance about 3 times in 1,000: run Step 8 again with another seed."]
    return True, [f"my_counts: mean cut {c:.4f}, within 3 standard deviations ({z:+.2f}) of the expected {e:.4f}."]


def personal_value(layers, p2, readout):
    return int(round(1000 * expected_cut(int(layers), float(p2), float(readout))))


def check_l2_m6_qaoa(qaoa_circuit, mean_cut, shot_sigma, within_3_sigma, difference_sigma, LAYERS, P2, READOUT, my_counts):
    try:
        layers = int(LAYERS)
        if layers != float(LAYERS) or layers not in REFERENCE_ANGLES:
            raise ValueError
    except (TypeError, ValueError):
        return False, ["LAYERS is 1, 2 or 3. Copy it again from your project quiz."], None
    ok, msg, p2, ro = personal_noise_ok(P2, READOUT)
    if not ok:
        return False, [msg], None
    msgs, passed = [], True
    for good, mm in (test_statistics(shot_sigma, within_3_sigma, difference_sigma), _test_qaoa(qaoa_circuit),
                     _test_mean_cut(mean_cut)):
        msgs += mm
        passed = passed and good
        if not good:
            break
    if not passed:
        return False, msgs, None
    good, mm = _test_counts(my_counts, layers, p2, ro)
    msgs += mm
    if not good:
        return False, msgs, None
    return True, msgs, int(round(1000 * expected_cut(layers, p2, ro)))
