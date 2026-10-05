"""Check cell logic for the Level 2 Module 6 project "A transpiler comparison" (Colab or browser).

check_l2_m6_transpiler(mirror_circuit, success_fraction, seed_spread, shot_sigma, within_3_sigma, difference_sigma,
                       BITS, SEED, saved) returns (passed, messages, verification_value).

The learner writes:
- mirror_circuit(bits): X on every qubit whose bit is 1 (qubit q is bits[n - 1 - q]), the prepared qft on all qubits,
  one barrier, the prepared inverse_qft, then measure qubit q into classical bit q. Without noise it returns `bits`
  every time; the barrier keeps the transpiler from cancelling the QFT against its inverse.
- success_fraction(counts, bits): the fraction of shots that returned `bits`;
- seed_spread(values): the sample standard deviation of a list of numbers (divide by len - 1).

`saved` is the course's saved-runs file (m6_transpiler_runs.json, made with qiskit 2.5.2, qiskit-aer 0.17.2 and
qiskit-ibm-runtime 0.50.0 on FakePittsburgh). The verification value is the number of shots, out of 4,000, in which the
reference mirror circuit for the 4-bit string of BITS, transpiled at optimization level 3 with seed_transpiler = SEED and
run on the FakePittsburgh noise model with seed_simulator = SEED, returned that string.
"""
from l2_m6_common import *          # noqa: F401,F403  (the notebook pastes l2_m6_common above this line)
from l2_m6_common import _name       # noqa: F401  (a private helper; this line is left out of the notebook too)

MIRROR_CASES = ["1", "10", "101", "0110", "10011"]


def reference_qft_ops(qubits):
    qubits = list(qubits)
    n = len(qubits)
    ops = []
    for j in reversed(range(n)):
        ops.append(("h", [qubits[j]], []))
        for k in reversed(range(j)):
            ops.append(("cp", [qubits[k], qubits[j]], [math.pi / 2 ** (j - k)]))
    ops += [("swap", [qubits[j], qubits[n - 1 - j]], []) for j in range(n // 2)]
    return ops


def reference_prefix_ops(bits):
    n = len(bits)
    return [("x", [q], []) for q in range(n) if bits[n - 1 - q] == "1"] + reference_qft_ops(range(n))


def _prefix(qc, stop):
    c = qc.copy_empty_like()
    for inst in list(qc.data)[:stop]:
        c.append(inst)
    return c


def _state(qc):
    return np.asarray(Statevector(qc).data, dtype=complex)


def _test_mirror(mirror_circuit):
    for bits in MIRROR_CASES:
        n = len(bits)
        label = f"mirror_circuit('{bits}')"
        try:
            qc = mirror_circuit(bits)
        except NotImplementedError:
            return False, ["mirror_circuit() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"{label} raised {type(e).__name__}: {e}"]
        if not hasattr(qc, "data") or getattr(qc, "num_qubits", None) != n:
            return False, [f"{label} should return a QuantumCircuit on {n} qubits."]
        mm = measurement_map(qc)
        if mm != {q: q for q in range(n)}:
            return False, [f"{label} should end by measuring qubit q into classical bit q, for every qubit (found {mm or 'no measurements'})."]
        names = [_name(i) for i in qc.data]
        barriers = [k for k, nm in enumerate(names) if nm == "barrier"]
        if len(barriers) != 1:
            return False, [f"{label} has {len(barriers)} barriers; put exactly one, between qft and inverse_qft. Without it, "
                           "the transpiler at levels 2 and 3 can cancel the QFT against its inverse and nothing is tested; "
                           "extra barriers stop it from optimizing inside the QFT."]
        try:
            before = _state(_prefix(qc, barriers[0]))
        except Exception as e:  # noqa: BLE001
            return False, [f"The part of {label} before the barrier cannot be simulated: {type(e).__name__}: {e}"]
        want = _state(build_circuit(reference_prefix_ops(bits), n, []))
        if abs(np.vdot(want, before)) ** 2 < 1 - 1e-9:
            hint = " Before the barrier: X on each qubit whose bit is 1 (qubit 0 is the rightmost bit), then qft(qc, range(n))."
            rev = _state(build_circuit(reference_prefix_ops(bits[::-1]), n, []))
            if abs(np.vdot(rev, before)) ** 2 > 1 - 1e-9 and bits != bits[::-1]:
                hint = " The X gates are on the reversed string: qubit q gets an X when bits[n - 1 - q] is '1'."
            return False, [f"The state of {label} at the barrier differs from the expected one.{hint}"]
        probs = ideal_probabilities(qc)
        if probs[int(bits, 2)] < 1 - 1e-9:
            return False, [f"Without noise, {label} returns '{bits}' with probability {probs[int(bits, 2)]:.4f}; it should be 1. "
                           "After the barrier, apply the prepared inverse_qft on the same qubits."]
        want2 = 2 * ops_two_qubit_count(reference_qft_ops(range(n)))
        if two_qubit_count(qc) != want2:
            return False, [f"{label} has {two_qubit_count(qc)} two-qubit gates; the project's circuit has {want2} "
                           "(the prepared qft and inverse_qft and nothing else). Every extra gate changes what the transpiler gets."]
    return True, [f"mirror_circuit(): passed ({len(MIRROR_CASES)} cases, 1 to 5 qubits)."]


SUCCESS_CASES = [({"1011": 3600, "1010": 300, "0011": 100}, "1011"), ({"01": 7, "10": 3}, "10"),
                 ({"000": 5, "111": 5}, "101"), ({"10110": 1}, "10110")]


def _test_success(success_fraction):
    for counts, bits in SUCCESS_CASES:
        want = counts.get(bits, 0) / sum(counts.values())
        try:
            got = success_fraction(dict(counts), bits)
        except NotImplementedError:
            return False, ["success_fraction() is not written yet."]
        except KeyError:
            return False, [f"success_fraction({counts}, '{bits}') raised KeyError: use counts.get(bits, 0), because a "
                           "result that never appeared has no entry."]
        except Exception as e:  # noqa: BLE001
            return False, [f"success_fraction({counts}, '{bits}') raised {type(e).__name__}: {e}"]
        try:
            gotf = float(got)
        except (TypeError, ValueError):
            return False, [f"success_fraction(counts, '{bits}') should return a number; it returned {got!r}."]
        if abs(gotf - want) > 1e-12:
            hint = ""
            if abs(gotf - counts.get(bits, 0)) < 1e-12:
                hint = " Divide by the total number of shots, sum(counts.values())."
            elif abs(gotf - counts.get(bits, 0) / 4000) < 1e-12 and sum(counts.values()) != 4000:
                hint = " Divide by the total of this counts dictionary, not by 4000."
            return False, [f"success_fraction({counts}, '{bits}') returned {gotf}; expected {want}.{hint}"]
    return True, [f"success_fraction(): passed ({len(SUCCESS_CASES)} cases)."]


SPREAD_CASES = [[0.9, 0.92, 0.94], [0.8589, 0.8, 0.88, 0.85, 0.83], [1, 1, 1, 1], [0.5, 0.7]]


def _test_spread(seed_spread):
    for vals in SPREAD_CASES:
        want = float(np.std(vals, ddof=1))
        try:
            got = seed_spread(list(vals))
        except NotImplementedError:
            return False, ["seed_spread() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"seed_spread({vals}) raised {type(e).__name__}: {e}"]
        try:
            gotf = float(got)
        except (TypeError, ValueError):
            return False, [f"seed_spread({vals}) should return a number; it returned {got!r}."]
        if abs(gotf - want) > 1e-12:
            hint = ""
            if abs(gotf - float(np.std(vals))) < 1e-12:
                hint = " Divide by len(values) - 1 (the sample standard deviation): np.std(values, ddof=1)."
            elif abs(gotf - float(np.var(vals, ddof=1))) < 1e-12:
                hint = " Take the square root of the variance."
            elif abs(gotf - want / math.sqrt(len(vals))) < 1e-12:
                hint = " That is the uncertainty of the mean; seed_spread() is the spread of the values themselves."
            return False, [f"seed_spread({vals}) returned {gotf}; expected {want:.6f}.{hint}"]
    return True, [f"seed_spread(): passed ({len(SPREAD_CASES)} cases)."]


def personal_value(bits, seed, saved):
    for d in saved.get("personal", []):
        if int(d["bits"]) == int(bits) and int(d["seed"]) == int(seed):
            return int(d["success"])
    return None


def check_l2_m6_transpiler(mirror_circuit, success_fraction, seed_spread, shot_sigma, within_3_sigma, difference_sigma,
                           BITS, SEED, saved):
    try:
        b, s = int(BITS), int(SEED)
        if b != float(BITS) or s != float(SEED) or not 0 <= b <= 15 or not 100 <= s <= 999:
            raise ValueError
    except (TypeError, ValueError):
        return False, ["BITS is a whole number from 0 to 15 and SEED from 100 to 999. Copy them again from your project quiz."], None
    if not isinstance(saved, dict) or "personal" not in saved:
        return False, ["The saved runs are not loaded: run the cell of Step 0 that loads m6_transpiler_runs.json."], None
    msgs, passed = [], True
    for good, mm in (test_statistics(shot_sigma, within_3_sigma, difference_sigma), _test_mirror(mirror_circuit),
                     _test_success(success_fraction), _test_spread(seed_spread)):
        msgs += mm
        passed = passed and good
        if not good:
            break
    if not passed:
        return False, msgs, None
    value = personal_value(b, s, saved)
    if value is None:
        return False, msgs + [f"BITS = {b} with SEED = {s} is not one of the course's saved runs. Copy both numbers again "
                              "from your project quiz."], None
    return True, msgs, value
