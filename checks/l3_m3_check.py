"""Check cell logic for the Level 3 Module 3 lab (error suppression and mitigation).

check_l3_module3(idle_block, mitigate_readout, fold_cx, extrapolate_zero, DELTA, P_RO, P_CX)
returns (passed, messages, verification_value).

The experiment used everywhere in the lab: a Bell state (H on qubit 0, CNOT 0 -> 1), an idle period of
N_IDLE steps (one id gate on each qubit per step), then H on both qubits and a measurement of both, which
measures <XX>; ideally <XX> = 1. The noise: each idle step has thermal relaxation (T1 200 us, T2 150 us,
1 us per step) and a coherent Z rotation RZ(DELTA) (a qubit frequency slightly off); H and X gates have
0.1% depolarizing error; the CNOT has two-qubit depolarizing error P_CX; each qubit is misread 0 -> 1 with
probability P_RO and 1 -> 0 with probability 2 P_RO.

The learner writes four functions:
- idle_block(qc, n, dd): n idle steps on qubits 0 and 1; with dd=True, an X on both qubits after n // 2 steps
  and again after the last step (a spin echo, the simplest dynamical decoupling);
- mitigate_readout(probs, A): solves A p_true = p_measured, where A[i][j] is the probability of measuring the
  bit string with index i when the one with index j was prepared ('00' -> 0, '01' -> 1, '10' -> 2, '11' -> 3);
- fold_cx(qc, scale): a new circuit in which every CNOT is replaced by `scale` CNOTs in a row (scale odd), so the
  circuit does the same thing with `scale` times the CNOT noise;
- extrapolate_zero(scales, values, degree=1): fits a polynomial of that degree to the values and returns it at 0.

The verification value is round(1000 * (E_mitigated - E_raw)) for the learner's DELTA, P_RO and P_CX, where
E_raw is <XX> measured with no suppression or mitigation and E_mitigated combines dynamical decoupling, readout
mitigation and linear zero-noise extrapolation from scales 1, 3 and 5. All values are exact probabilities from
qsim's density-matrix simulation, so no random numbers are involved.
"""
import numpy as np

import qsim
from qsim import QuantumCircuit, NoiseModel, ReadoutError, coherent_unitary_error, depolarizing_error, \
    thermal_relaxation_error, simulate_density_matrix

N_IDLE, T1, T2, T_ID, P1 = 10, 200.0, 150.0, 1.0, 0.001
SCALES = (1, 3, 5)
KEYS = ("00", "01", "10", "11")


# ---------------------------------------------------------------- the experiment and its noise
def rz_matrix(delta):
    return np.diag([np.exp(-1j * delta / 2), np.exp(1j * delta / 2)])


def readout_matrix(p_ro):
    """Rows: prepared 0, 1; columns: measured 0, 1 (qiskit_aer.noise.ReadoutError convention)."""
    return np.array([[1 - p_ro, p_ro], [2 * p_ro, 1 - 2 * p_ro]])


def noise_model(delta, p_ro, p_cx):
    nm = NoiseModel()
    idle = thermal_relaxation_error(T1, T2, T_ID)
    if delta:
        idle = idle.compose(coherent_unitary_error(rz_matrix(delta)))
    nm.add_all_qubit_quantum_error(idle, ["id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(P1, 1), ["h", "x"])
    if p_cx:
        nm.add_all_qubit_quantum_error(depolarizing_error(p_cx, 2), ["cx"])
    if p_ro:
        nm.add_all_qubit_readout_error(ReadoutError(readout_matrix(p_ro)))
    return nm


def reference_idle_block(qc, n, dd=False):
    for k in range(n):
        if dd and k == n // 2:
            qc.x(0)
            qc.x(1)
        qc.id(0)
        qc.id(1)
    if dd:
        qc.x(0)
        qc.x(1)


def reference_mitigate_readout(probs, A):
    measured = np.array([probs.get(k, 0.0) for k in KEYS], dtype=float)
    true = np.linalg.solve(np.asarray(A, dtype=float), measured)
    return {k: float(v) for k, v in zip(KEYS, true)}


def reference_fold_cx(qc, scale):
    out = qc.copy_empty_like()
    for inst in qc.data:
        for _ in range(scale if inst.name == "cx" else 1):
            out.append(inst)
    return out


def reference_extrapolate_zero(scales, values, degree=1):
    return float(np.polyfit(np.asarray(scales, dtype=float), np.asarray(values, dtype=float), degree)[-1])


def experiment(dd=False, n_idle=N_IDLE, idle_block=reference_idle_block):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    idle_block(qc, n_idle, dd)
    qc.h(0)
    qc.h(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def exact_probabilities(qc, nm=None, p_ro=0.0):
    """Exact probabilities of the four results '00', '01', '10', '11' (qubit 0 on the right) of a two-qubit circuit
    that measures qubit 0 into bit 0 and qubit 1 into bit 1 at the end, including the readout error."""
    rho = simulate_density_matrix(qc, nm).data
    p = np.real(np.diag(rho)).clip(min=0)
    p = p / p.sum()
    if p_ro:
        R = readout_matrix(p_ro)
        p = (np.kron(R, R).T @ p)
    return {k: float(p[i]) for i, k in enumerate(KEYS)}


def parity_expectation(probs):
    """<ZZ> of the measured bits, which is <XX> of the state before the two H gates."""
    return float(sum(v * (-1) ** k.count("1") for k, v in probs.items()))


def calibration_matrix(nm, p_ro):
    """A[i][j]: probability of measuring KEYS[i] when KEYS[j] was prepared with X gates (noise included)."""
    A = np.zeros((4, 4))
    for j, key in enumerate(KEYS):
        qc = QuantumCircuit(2, 2)
        if key[1] == "1":
            qc.x(0)
        if key[0] == "1":
            qc.x(1)
        qc.measure(0, 0)
        qc.measure(1, 1)
        p = exact_probabilities(qc, nm, p_ro)
        for i, k in enumerate(KEYS):
            A[i, j] = p[k]
    return A


def pipeline(delta, p_ro, p_cx, idle_block=reference_idle_block, mitigate_readout=reference_mitigate_readout,
             fold_cx=reference_fold_cx, extrapolate_zero=reference_extrapolate_zero):
    """Raw <XX>, with DD, with DD and readout mitigation, the three ZNE points and the linear and quadratic
    zero-noise estimates."""
    nm = noise_model(delta, p_ro, p_cx)
    A = calibration_matrix(nm, p_ro)
    raw = parity_expectation(exact_probabilities(experiment(False, idle_block=idle_block), nm, p_ro))
    base = experiment(True, idle_block=idle_block)
    dd = parity_expectation(exact_probabilities(base, nm, p_ro))
    zne = [parity_expectation(mitigate_readout(exact_probabilities(fold_cx(base, s), nm, p_ro), A)) for s in SCALES]
    return {"raw": raw, "dd": dd, "dd_ro": zne[0], "zne_points": zne,
            "linear": extrapolate_zero(SCALES, zne, 1), "quadratic": extrapolate_zero(SCALES, zne, 2)}


def personal_value(delta, p_ro, p_cx):
    r = pipeline(float(delta), float(p_ro), float(p_cx))
    return int(round(1000 * (r["linear"] - r["raw"])))


# ---------------------------------------------------------------- helpers
def _try(fn, name, *args):
    try:
        return fn(*args), None
    except NotImplementedError:
        return None, f"{name}() is not written yet."
    except qsim.CircuitError as e:
        return None, f"{name}() made an invalid circuit: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"{name}() raised {type(e).__name__}: {e}"


def _bell_xx(qc_idle, delta):
    """<XX> after Bell, the given idle circuit body and H on both, with only the coherent error RZ(delta) on id."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(coherent_unitary_error(rz_matrix(delta)), ["id"])
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    for inst in qc_idle.data:
        qc.append(inst)
    qc.h(0)
    qc.h(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return parity_expectation(exact_probabilities(qc, nm))


# ---------------------------------------------------------------- tests
def _test_idle_block(idle_block):
    for n in (4, 10):
        for dd in (False, True):
            qc = QuantumCircuit(2, 2)
            out, err = _try(idle_block, "idle_block", qc, n, dd)
            if err:
                return False, [err]
            if out is not None and out is not qc:
                return False, ["idle_block(qc, n, dd) should add gates to qc itself, not build and return a new circuit."]
            names = [i.name for i in qc.data]
            if not names:
                return False, ["idle_block(qc, n, dd) added nothing to the circuit."]
            if set(names) - {"id", "x", "barrier"}:
                return False, [f"idle_block() should use only id gates (and X gates for dd=True); it added {sorted(set(names) - {'id', 'x', 'barrier'})}."]
            for q in (0, 1):
                ids = sum(1 for i in qc.data if i.name == "id" and i.qubits[0] == q)
                if ids != n:
                    return False, [f"idle_block(qc, {n}, dd={dd}) puts {ids} id gates on qubit {q}; it should put {n}, one per idle step."]
            xs = [i for i in qc.data if i.name == "x"]
            if not dd and xs:
                return False, ["With dd=False, idle_block() should only wait: id gates and no X gates."]
            if dd:
                for q in (0, 1):
                    nx = sum(1 for i in xs if i.qubits[0] == q)
                    if nx != 2:
                        return False, [f"With dd=True, qubit {q} gets {nx} X gates; the echo needs exactly two on each qubit: "
                                       "one halfway through the idle period and one at the end."]
                for delta in (0.07, 0.19):
                    e = _bell_xx(qc, delta)
                    if abs(e - 1) > 1e-8:
                        return False, [f"With dd=True and a coherent error RZ({delta}) on every idle step, <XX> is {e:.3f}, not 1: "
                                       f"the X gates do not cancel the rotation. Put the first X on both qubits after {n // 2} "
                                       f"of the {n} steps and the second after the last step, so both halves of the idle "
                                       "period are the same length."]
            else:
                for delta in (0.07, 0.19):
                    e = _bell_xx(qc, delta)
                    if abs(e - np.cos(2 * n * delta)) > 1e-8:
                        return False, [f"With dd=False, <XX> should be cos(2 n delta) = {np.cos(2 * n * delta):.3f}; "
                                       f"your idle block gives {e:.3f}."]
    return True, ["idle_block(): passed (4 and 10 steps, with and without the echo; the echo cancels a coherent Z rotation)."]


def _test_mitigate(mitigate_readout):
    rng = np.random.default_rng(321)
    for _ in range(6):
        A = np.eye(4) * rng.uniform(0.75, 0.95, 4)
        off = rng.uniform(0, 1, (4, 4)) * (1 - np.eye(4))
        A = A + off * (1 - A.diagonal()) / off.sum(axis=0)
        true = rng.dirichlet(np.ones(4))
        meas = A @ true
        probs = {k: float(v) for k, v in zip(KEYS, meas)}
        got, err = _try(mitigate_readout, "mitigate_readout", dict(probs), A.copy())
        if err:
            return False, [err]
        if not isinstance(got, dict) or set(got) != set(KEYS):
            return False, ["mitigate_readout(probs, A) should return a dictionary with the keys '00', '01', '10' and '11'."]
        g = np.array([float(got[k]) for k in KEYS])
        if np.allclose(g, true, atol=1e-8):
            continue
        if np.allclose(g, A @ meas, atol=1e-8):
            hint = " It multiplies by A; the measured probabilities are A times the true ones, so solve A p = measured instead."
        elif np.allclose(g, np.linalg.solve(A.T, meas), atol=1e-8):
            hint = " It uses A transposed. A[i][j] is P(measure i | prepared j), so the measured vector is A @ p_true."
        elif np.allclose(g, (np.linalg.solve(A, meas[[0, 2, 1, 3]]))[[0, 2, 1, 3]], atol=1e-8) or \
                np.allclose(g[[0, 2, 1, 3]], true, atol=1e-8):
            hint = " The '01' and '10' entries are swapped: index the vector in the order '00', '01', '10', '11'."
        elif np.allclose(g, meas, atol=1e-8):
            hint = " It returns the measured probabilities unchanged."
        else:
            hint = " Use np.linalg.solve(A, p_measured) with p_measured in the order '00', '01', '10', '11'."
        return False, ["mitigate_readout() does not give back the true probabilities." + hint]
    return True, ["mitigate_readout(): passed (6 random calibration matrices)."]


def _test_fold(fold_cx):
    for dd in (False, True):
        base = experiment(dd)
        before = [(i.name, tuple(i.qubits)) for i in base.data]
        for s in SCALES:
            got, err = _try(fold_cx, "fold_cx", base, s)
            if err:
                return False, [err]
            if not isinstance(got, QuantumCircuit):
                return False, ["fold_cx(qc, scale) should return a new QuantumCircuit."]
            if [(i.name, tuple(i.qubits)) for i in base.data] != before:
                return False, ["fold_cx() changed the circuit it was given. Build a new circuit (qc.copy_empty_like()) and "
                               "append to that one."]
            n_cx = sum(1 for i in got.data if i.name == "cx")
            if n_cx != s * sum(1 for i in base.data if i.name == "cx"):
                return False, [f"fold_cx(qc, {s}) gives {n_cx} CNOTs; every CNOT should become {s} CNOTs in a row."]
            want = reference_fold_cx(base, s)
            if [(i.name, tuple(i.qubits)) for i in got.data] != [(i.name, tuple(i.qubits)) for i in want.data]:
                return False, [f"fold_cx(qc, {s}) has the right number of CNOTs but changes the order of the other gates "
                               "or the CNOTs' qubits. Copy every instruction in order, repeating each CNOT."]
    return True, ["fold_cx(): passed (scales 1, 3 and 5, with and without the echo; the original circuit is unchanged)."]


def _test_extrapolate(extrapolate_zero):
    rng = np.random.default_rng(322)
    for _ in range(4):
        a, b, c = rng.uniform(0.5, 1), rng.uniform(-0.1, -0.01), rng.uniform(-0.005, 0.005)
        x = [1, 3, 5]
        y = [a + b * t + c * t * t for t in x]
        for degree in (1, 2):
            got, err = _try(extrapolate_zero, "extrapolate_zero", list(x), list(y), degree)
            if err:
                return False, [err]
            try:
                g = float(got)
            except (TypeError, ValueError):
                return False, ["extrapolate_zero() should return one number: the fitted value at scale 0."]
            want = reference_extrapolate_zero(x, y, degree)
            if abs(g - want) > 1e-8:
                if abs(g - y[0]) < 1e-8:
                    hint = " It returns the value at scale 1; extrapolate to scale 0."
                elif degree == 2 and abs(g - reference_extrapolate_zero(x, y, 1)) < 1e-8:
                    hint = " It ignores degree: pass it to np.polyfit."
                else:
                    hint = " np.polyfit(scales, values, degree) returns the coefficients highest power first; the value at 0 is the last one."
                return False, [f"extrapolate_zero() gives {g:.4f} for degree {degree}; the fit at 0 is {want:.4f}." + hint]
    return True, ["extrapolate_zero(): passed (linear and quadratic fits)."]


def check_l3_module3(idle_block, mitigate_readout, fold_cx, extrapolate_zero, DELTA, P_RO, P_CX):
    try:
        d, r, c = float(DELTA), float(P_RO), float(P_CX)
    except (TypeError, ValueError):
        return False, ["DELTA, P_RO and P_CX must be the numbers shown in your Canvas lab check."], None
    two = lambda x: abs(round(x, 2) - x) < 1e-9  # noqa: E731
    if not (0.02 <= d <= 0.15 and 0.01 <= r <= 0.06 and 0.01 <= c <= 0.08 and two(d) and two(r) and two(c)):
        return False, ["Copy your numbers again from the lab check: DELTA (0.02 to 0.15), P_RO (0.01 to 0.06) and P_CX "
                       "(0.01 to 0.08), each with two decimals."], None
    msgs, ok = [], True
    for good, m in (_test_idle_block(idle_block), _test_mitigate(mitigate_readout), _test_fold(fold_cx),
                    _test_extrapolate(extrapolate_zero)):
        msgs += m
        ok = ok and good
        if not ok:
            return False, msgs, None
    try:
        mine = pipeline(0.05, 0.02, 0.02, idle_block, mitigate_readout, fold_cx, extrapolate_zero)["linear"]
    except Exception as e:  # noqa: BLE001
        return False, msgs + [f"Your four functions together raised {type(e).__name__}: {e}"], None
    ref = pipeline(0.05, 0.02, 0.02)["linear"]
    if abs(mine - ref) > 1e-8:
        return False, msgs + [f"Each function passes, but together they give {mine:.4f} instead of {ref:.4f} for the lab's "
                              "example noise. Check that they use the same bit order and scales."], None
    return True, msgs, personal_value(d, r, c)
