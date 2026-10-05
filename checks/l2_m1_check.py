"""Check cell logic for the Level 2 Module 1 lab (the quantum Fourier transform and phase estimation), on qsim 1.6.0.

check_l2_module1(qft3, qft, phase_estimation, PHI, SEED)
returns (passed, messages, verification_value).

The learner writes three functions:
- qft3(): the 3-qubit QFT, gate by gate;
- qft(n): the n-qubit QFT for any n;
- phase_estimation(phase, t): t counting qubits (0 to t - 1) and one target qubit (t) in |1>, an eigenstate of
  the phase gate P(2 pi phase); controlled powers; the inverse QFT on the counting qubits; counting qubit k
  measured into classical bit k.

The verification value is the number of shots, out of 1,000, in which the course's reference phase-estimation
circuit with 5 counting qubits returns the best 5-bit estimate of PHI, run on qsim's AerSimulator with seed SEED.
It is printed only when every test passes, and it depends on the seeded draw, so it cannot be worked out by hand.
"""
import math

import numpy as np
import qsim
from qsim import AerSimulator, Operator, QuantumCircuit


def dft_matrix(n):
    """The 2^n x 2^n discrete Fourier transform matrix: entry (k, j) = exp(2 pi i j k / 2^n) / sqrt(2^n)."""
    N = 2 ** n
    w = np.exp(2j * np.pi / N)
    return np.array([[w ** (j * k) for j in range(N)] for k in range(N)]) / math.sqrt(N)


def reference_qft(n):
    qc = QuantumCircuit(n)
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(math.pi / 2 ** (j - k), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def reference_phase_estimation(phase, t):
    qc = QuantumCircuit(t + 1, t)
    qc.x(t)
    for k in range(t):
        qc.h(k)
    for k in range(t):
        qc.cp(2 * math.pi * phase * 2 ** k, k, t)
    qc.compose(reference_qft(t).inverse(), qubits=list(range(t)), inplace=True)
    qc.measure(list(range(t)), list(range(t)))
    return qc


def exact_distribution(qc):
    """The exact probability of each result string of a circuit (qsim's branch simulator)."""
    out = {}
    for rho, bits in qsim._branch_states(qc, None):
        w = float(np.real(np.trace(rho)))
        if w > 1e-12:
            key = qsim._format_key(list(bits), qc._cregs)
            out[key] = out.get(key, 0.0) + w
    return out


def personal_value(phi, seed):
    t = 5
    best = format(round(phi * 2 ** t) % 2 ** t, f"0{t}b")
    counts = AerSimulator(seed_simulator=int(seed)).run(reference_phase_estimation(float(phi), t), shots=1000).result().get_counts()
    return int(counts.get(best, 0))


def _bit_reversal(n):
    N = 2 ** n
    perm = np.zeros((N, N))
    for i in range(N):
        perm[int(format(i, f"0{n}b")[::-1], 2), i] = 1
    return perm


def _diagnose_qft(m, n):
    """A hint for a wrong n-qubit QFT matrix m."""
    F = dft_matrix(n)
    if np.allclose(m, F.conj(), atol=1e-8):
        return "it is the inverse QFT: the controlled-phase angles need the opposite sign."
    if n > 1 and (np.allclose(m, _bit_reversal(n) @ F, atol=1e-8) or np.allclose(m, F @ _bit_reversal(n), atol=1e-8)):
        return "the qubit order is reversed: add the swaps at the end (qubit i with qubit n - 1 - i)."
    if Operator(m).equiv(Operator(F)):
        return "it matches the DFT only up to a global phase; the QFT should match it exactly."
    return "compare each controlled-phase angle with pi / 2^(j - k), and check that every qubit gets its H gate."


def _test_qft3(qft3):
    try:
        qc = qft3()
    except NotImplementedError:
        return False, ["qft3() is not written yet."]
    except Exception as e:  # noqa: BLE001
        return False, [f"qft3() raised {type(e).__name__}: {e}"]
    if not isinstance(qc, QuantumCircuit) or qc.num_qubits != 3:
        return False, ["qft3() should return a QuantumCircuit with 3 qubits."]
    if any(i.name in ("measure", "reset") for i in qc.data):
        return False, ["qft3() should contain gates only, no measurements."]
    m = Operator(qc).data
    if np.allclose(m, dft_matrix(3), atol=1e-8):
        return True, ["qft3(): passed (equal to the 8 x 8 DFT matrix)."]
    return False, ["qft3() does not equal the DFT matrix: " + _diagnose_qft(m, 3)]


def _test_qft(qft):
    for n in range(1, 6):
        try:
            qc = qft(n)
        except NotImplementedError:
            return False, ["qft(n) is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"qft({n}) raised {type(e).__name__}: {e}"]
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != n:
            return False, [f"qft({n}) should return a QuantumCircuit with {n} qubits."]
        if any(i.name in ("measure", "reset") for i in qc.data):
            return False, [f"qft({n}) should contain gates only, no measurements."]
        m = Operator(qc).data
        if not np.allclose(m, dft_matrix(n), atol=1e-8):
            return False, [f"qft({n}) does not equal the {2 ** n} x {2 ** n} DFT matrix: " + _diagnose_qft(m, n)]
    return True, ["qft(n): passed (n = 1 to 5, each equal to the DFT matrix)."]


CASES = [(1 / 8, 3), (1 / 3, 4), (0.7, 5), (0.05, 3), (5 / 16, 4), (0.9, 2), (0.432, 5)]


def _test_phase_estimation(phase_estimation):
    worst = 0.0
    for phase, t in CASES:
        try:
            qc = phase_estimation(phase, t)
        except NotImplementedError:
            return False, ["phase_estimation() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"phase_estimation({phase:.4g}, {t}) raised {type(e).__name__}: {e}"]
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != t + 1 or qc.num_clbits != t:
            return False, [f"phase_estimation(phase, {t}) should return a circuit with {t + 1} qubits and {t} classical bits."]
        meas = [i for i in qc.data if i.name == "measure"]
        if sorted(i.qubits[0] for i in meas) != list(range(t)) or any(i.qubits[0] != i.clbits[0] for i in meas):
            return False, ["Measure each counting qubit k (0 to t - 1) into classical bit k, and do not measure the target qubit."]
        got = exact_distribution(qc)
        ref = exact_distribution(reference_phase_estimation(phase, t))
        keys = set(got) | set(ref)
        d = 0.5 * sum(abs(got.get(k, 0) - ref.get(k, 0)) for k in keys)
        worst = max(worst, d)
        if d > 1e-6:
            best = max(got, key=got.get) if got else "none"
            hint = ""
            if set(got) == {"0" * t}:
                hint = " Every result is 0: is the target qubit in |1> (an X gate first), and are the counting qubits in superposition?"
            elif best == format((-round(phase * 2 ** t)) % 2 ** t, f"0{t}b"):
                hint = " The most likely result is the negative of the phase: use the inverse QFT, qft(t).inverse()."
            return False, [f"For phase {phase:.4g} with {t} counting qubits, the results do not match the expected "
                           f"distribution (most likely result {best}, expected "
                           f"{format(round(phase * 2 ** t) % 2 ** t, f'0{t}b')}).{hint} Check that counting qubit k "
                           f"controls the phase gate raised to the power 2^k."]
    return True, [f"phase_estimation(): passed ({len(CASES)} phases, 2 to 5 counting qubits; each distribution exact)."]


def check_l2_module1(qft3, qft, phase_estimation, PHI, SEED):
    try:
        phi, seed = float(PHI), int(SEED)
    except (TypeError, ValueError):
        return False, ["PHI and SEED must be the numbers shown in your Canvas lab check."], None
    if not (0.05 <= phi <= 0.95) or not (100 <= seed <= 999):
        return False, ["PHI and SEED are outside the range Canvas uses. Copy them again from the lab check."], None
    msgs, ok = [], True
    for test, fn in ((_test_qft3, qft3), (_test_qft, qft), (_test_phase_estimation, phase_estimation)):
        good, m = test(fn)
        msgs += m
        ok = ok and good
    if not ok:
        return False, msgs, None
    return True, msgs, personal_value(phi, seed)
