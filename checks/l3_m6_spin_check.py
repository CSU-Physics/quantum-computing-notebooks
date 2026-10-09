"""Check cell logic for the Level 3 Module 6 capstone "Spin-chain simulation with mitigation".

check_l3_m6_spin(fraction_sigma, expectation_sigma, combined_sigma, trotter_circuit, magnetization, correct_readout_z,
                 fold_rzz, extrapolate_linear, STEPS, P2, READOUT) returns (passed, messages, verification_value).

The model of Module 4 Track B: an open chain of N spins, H = -J sum Z_i Z_(i+1) - h sum X_i with J = 1, starting in
|00...0>; the quantity is the magnetization M = (1/N) sum <Z_i>.
- trotter_circuit(N, h, T, steps): the unitary of `steps` first-order Trotter steps of length dt = T / steps, each
  RZZ(-2 dt) on every neighbouring pair (0, 1), (1, 2), ... then RX(-2 h dt) on every qubit; no measurements;
- magnetization(probs): (1/N) sum_i <Z_i> from counts or probabilities (qubit i is character N - 1 - i of a key);
- correct_readout_z(z, readout): a measured <Z> (or magnetization) corrected for the capstone readout errors:
  measured = (1 - r0 - r1) true + (r1 - r0) with r0 = readout (0 read as 1) and r1 = 2 readout (1 read as 0);
- fold_rzz(qc, scale): every RZZ(theta) replaced by RZZ(theta) followed by (scale - 1) / 2 pairs RZZ(-theta),
  RZZ(theta); other gates unchanged (scale odd);
- extrapolate_linear(scales, values): the least-squares straight line through (scale, value), read at scale 0.

The verification value is round(10000 * M_mit) for N = 3, h = 0.7, T = 2.0 and the learner's STEPS, P2 and READOUT:
M_mit is the linear zero-noise extrapolation (scales 1, 3, 5) of the readout-corrected magnetization, all from exact
noisy probabilities.
"""
import numpy as np

from qsim import QuantumCircuit, Operator

from l3_m6_common import _num, _try, exact_probabilities, test_statistics

N_CAP, H_CAP, T_CAP = 3, 0.7, 2.0
SCALES = (1, 3, 5)


# ---------------------------------------------------------------- reference solutions
def reference_trotter_circuit(N, h, T, steps):
    qc = QuantumCircuit(N)
    dt = T / steps
    for _ in range(steps):
        for i in range(N - 1):
            qc.rzz(-2 * dt, i, i + 1)
        for i in range(N):
            qc.rx(-2 * h * dt, i)
    return qc


def measured(qc):
    out = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    out.compose(qc, inplace=True)
    for i in range(qc.num_qubits):
        out.measure(i, i)
    return out


def reference_magnetization(probs):
    total = sum(probs.values())
    n = len(next(iter(probs)))
    return float(sum(v * sum(1 - 2 * int(b) for b in k) for k, v in probs.items()) / total / n)


def reference_correct_readout_z(z, readout):
    r0, r1 = readout, 2 * readout
    return (z - (r1 - r0)) / (1 - r0 - r1)


def reference_fold_rzz(qc, scale):
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.name == "rzz":
            th = float(inst.params[0])
            a, b = inst.qubits
            out.rzz(th, a, b)
            for _ in range((scale - 1) // 2):
                out.rzz(-th, a, b)
                out.rzz(th, a, b)
        else:
            out.data.append(inst)
    return out


def reference_extrapolate_linear(scales, values):
    slope, intercept = np.polyfit(np.asarray(scales, dtype=float), np.asarray(values, dtype=float), 1)
    return float(intercept)


# ---------------------------------------------------------------- exact values
def exact_magnetization(N, h, T):
    """The exact M(T) (matrix exponential of the Hamiltonian), for comparison."""
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    X = np.array([[0, 1], [1, 0]])
    Z = np.diag([1.0, -1.0])

    def op(single, i):
        m = np.array([[1.0]])
        for q in reversed(range(N)):
            m = np.kron(m, single if q == i else np.eye(2))
        return m
    for i in range(N - 1):
        H -= op(Z, i) @ op(Z, i + 1)
    for i in range(N):
        H -= h * op(X, i)
    w, v = np.linalg.eigh(H)
    psi0 = np.zeros(dim)
    psi0[0] = 1
    psi = v @ (np.exp(-1j * w * T) * (v.conj().T @ psi0))
    mz = sum(op(Z, i) for i in range(N)) / N
    return float(np.real(np.vdot(psi, mz @ psi)))


def study(steps, p2, readout, N=N_CAP, h=H_CAP, T=T_CAP, trotter=reference_trotter_circuit, fold=reference_fold_rzz,
          mag=reference_magnetization, correct=reference_correct_readout_z, extrapolate=reference_extrapolate_linear):
    base = trotter(N, h, T, steps)
    raw = {s: mag(exact_probabilities(measured(fold(base, s)), p2, readout)) for s in SCALES}
    corrected = {s: correct(raw[s], readout) for s in SCALES}
    return {"ideal": mag(exact_probabilities(measured(base), 0.0, 0.0)), "raw": raw, "corrected": corrected,
            "mitigated": float(extrapolate(list(SCALES), [corrected[s] for s in SCALES]))}


def personal_value(steps, p2, readout):
    return int(round(10000 * study(int(steps), float(p2), float(readout))["mitigated"]))


# ---------------------------------------------------------------- tests
def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _test_trotter(trotter_circuit):
    for N, h, T, steps in ((2, 0.7, 1.0, 2), (3, 0.7, 2.0, 4), (3, 1.3, 0.6, 3)):
        got, err = _try(trotter_circuit, "trotter_circuit", N, h, T, steps)
        if err:
            return False, [err]
        if not isinstance(got, QuantumCircuit) or got.num_qubits != N:
            return False, [f"trotter_circuit({N}, {h}, {T}, {steps}) should return a QuantumCircuit on {N} qubits."]
        if any(i.name in ("measure", "reset") for i in got.data):
            return False, ["trotter_circuit() has measurements; return the unitary part only (the notebook adds them)."]
        ref = reference_trotter_circuit(N, h, T, steps)
        names = [i.name for i in got.data if i.name != "barrier"]
        if names.count("rzz") != steps * (N - 1) or names.count("rx") != steps * N or len(names) != steps * (2 * N - 1):
            return False, [f"trotter_circuit({N}, {h}, {T}, {steps}) has {names.count('rzz')} rzz and {names.count('rx')} rx gates "
                           f"({len(names)} in all); each of the {steps} steps needs {N - 1} rzz and {N} rx, and nothing else."]
        if not np.allclose(Operator(got).data, Operator(ref).data, atol=1e-9):
            U, V = Operator(got).data, Operator(ref).data
            hint = ""
            alt = QuantumCircuit(N)
            dt = T / steps
            for _ in range(steps):
                for i in range(N - 1):
                    alt.rzz(2 * dt, i, i + 1)
                for i in range(N):
                    alt.rx(2 * h * dt, i)
            if np.allclose(U, Operator(alt).data, atol=1e-9):
                hint = " The signs are flipped: RZZ(-2 dt) and RX(-2 h dt), because RZZ(theta) = exp(-i theta ZZ / 2)."
            else:
                alt = QuantumCircuit(N)
                for _ in range(steps):
                    for i in range(N - 1):
                        alt.rzz(-dt, i, i + 1)
                    for i in range(N):
                        alt.rx(-h * dt, i)
                if np.allclose(U, Operator(alt).data, atol=1e-9):
                    hint = " The angles are half the right size: RZZ(-2 dt) and RX(-2 h dt)."
            return False, [f"trotter_circuit({N}, {h}, {T}, {steps}) is not the first-order Trotter circuit.{hint}"]
        if _tvd(exact_probabilities(measured(got), 0.02, 0.0), exact_probabilities(measured(ref), 0.02, 0.0)) > 1e-9:
            return False, [f"trotter_circuit({N}, {h}, {T}, {steps}) is right without noise but not with it: in each step, all the "
                           "RZZ gates first, then the RX gates."]
    return True, ["trotter_circuit(): passed (2 and 3 spins, with and without noise)."]


def _test_magnetization(magnetization):
    cases = [({"000": 1.0}, 1.0), ({"111": 5}, -1.0), ({"001": 0.5, "100": 0.5}, 1 / 3), ({"01": 30, "11": 10}, -0.25),
             ({"0": 1, "1": 3}, -0.5)]
    for probs, want in cases:
        got, err = _try(magnetization, "magnetization", dict(probs))
        if err:
            return False, [err]
        g = _num(got)
        if g is None or abs(g - want) > 1e-12:
            hint = ""
            n = len(next(iter(probs)))
            tot = sum(probs.values())
            if g is not None and abs(g - want * n) < 1e-12:
                hint = " Divide by the number of spins: M is the average of the <Z_i>."
            elif g is not None and abs(g - sum((-1) ** k.count("1") * v for k, v in probs.items()) / tot) < 1e-12:
                hint = " That is the parity <Z Z ... Z>; M is the average of the single-qubit <Z_i>."
            return False, [f"magnetization({probs}) gave {got!r}; expected {want:.6g}.{hint}"]
    return True, ["magnetization(): passed."]


def _test_correct(correct_readout_z):
    for z, ro in ((0.5, 0.02), (-0.3, 0.01), (0.9, 0.03), (0.02, 0.02)):
        got, err = _try(correct_readout_z, "correct_readout_z", z, ro)
        if err:
            return False, [err]
        g, want = _num(got), reference_correct_readout_z(z, ro)
        if g is None or abs(g - want) > 1e-12:
            hint = ""
            if g is not None and abs(g - z / (1 - 3 * ro)) < 1e-12:
                hint = " Subtract the offset r1 - r0 = readout first, then divide by 1 - r0 - r1."
            elif g is not None and abs(g - (z + ro) / (1 - 3 * ro)) < 1e-12:
                hint = " The offset has the wrong sign: a 1 is misread more often than a 0, which pushes <Z> up, so subtract it."
            elif g is not None and abs(g - z / (1 - 2 * ro)) < 1e-12:
                hint = " With r0 = readout and r1 = 2 readout, the scale is 1 - r0 - r1 = 1 - 3 readout, and there is an offset."
            return False, [f"correct_readout_z({z}, {ro}) gave {got!r}; expected {want:.6g}.{hint}"]
    return True, ["correct_readout_z(): passed."]


def _test_fold(fold_rzz):
    base = reference_trotter_circuit(3, 0.7, 2.0, 3)
    before = len(base.data)
    n_rzz, n_rx = 6, 9
    for scale in (1, 3, 5):
        got, err = _try(fold_rzz, "fold_rzz", base, scale)
        if err:
            return False, [err]
        if len(base.data) != before:
            return False, ["fold_rzz() changed the circuit it was given; build a new one."]
        if not isinstance(got, QuantumCircuit):
            return False, [f"fold_rzz(qc, {scale}) should return a QuantumCircuit."]
        names = [i.name for i in got.data if i.name != "barrier"]
        if names.count("rzz") != scale * n_rzz or names.count("rx") != n_rx:
            return False, [f"fold_rzz(qc, {scale}) has {names.count('rzz')} rzz and {names.count('rx')} rx gates; each of the {n_rzz} "
                           f"rzz gates should become {scale}, and the {n_rx} rx gates stay as they are."]
        if not np.allclose(Operator(got).data, Operator(base).data, atol=1e-9):
            return False, [f"fold_rzz(qc, {scale}) does not do the same as qc: after RZZ(theta), add pairs RZZ(-theta), RZZ(theta)."]
        want = exact_probabilities(measured(reference_fold_rzz(base, scale)), 0.02, 0.0)
        if _tvd(exact_probabilities(measured(got), 0.02, 0.0), want) > 1e-9:
            return False, [f"fold_rzz(qc, {scale}): keep the gates in their places, each RZZ followed directly by its folds."]
    return True, ["fold_rzz(): passed (scales 1, 3, 5)."]


def _test_extrapolate(extrapolate_linear):
    for scales, values in (([1, 3, 5], [0.8, 0.7, 0.6]), ([1, 3, 5], [0.62, 0.55, 0.45]), ([1, 2, 3], [0.4, 0.38, 0.33]),
                           ([1, 3], [0.9, 0.84])):
        got, err = _try(extrapolate_linear, "extrapolate_linear", list(scales), list(values))
        if err:
            return False, [err]
        g, want = _num(got), reference_extrapolate_linear(scales, values)
        if g is None or abs(g - want) > 1e-9:
            hint = ""
            if g is not None and abs(g - values[0]) < 1e-12:
                hint = " That is the value at scale 1; read the line at scale 0."
            return False, [f"extrapolate_linear({scales}, {values}) gave {got!r}; expected {want:.6g}.{hint}"]
    return True, ["extrapolate_linear(): passed."]


def check_l3_m6_spin(fraction_sigma, expectation_sigma, combined_sigma, trotter_circuit, magnetization,
                     correct_readout_z, fold_rzz, extrapolate_linear, STEPS, P2, READOUT):
    try:
        steps, p2, ro = int(STEPS), float(P2), float(READOUT)
        ok = float(STEPS) == steps and 2 <= steps <= 12 and 0.005 <= p2 <= 0.030 and 0.005 <= ro <= 0.030
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return False, ["Copy your numbers again from your capstone quiz: STEPS (2 to 12), P2 and READOUT (0.005 to 0.030)."], None
    msgs = []
    for test in (lambda: test_statistics(fraction_sigma, expectation_sigma, combined_sigma),
                 lambda: _test_trotter(trotter_circuit), lambda: _test_magnetization(magnetization),
                 lambda: _test_correct(correct_readout_z), lambda: _test_fold(fold_rzz),
                 lambda: _test_extrapolate(extrapolate_linear)):
        good, m = test()
        msgs += m
        if not good:
            return False, msgs, None
    return True, msgs, personal_value(steps, p2, ro)
