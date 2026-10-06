"""Shared parts of the Level 2 Module 6 project checks (the mini-project).

The project notebooks include this code in their check cell, followed by the project's own check. It holds:
- the course noise model as numbers (two-qubit depolarizing error P2, one-qubit error P2 / 10, readout error READOUT:
  a 0 is recorded as 1 with probability READOUT and a 1 as 0 with probability 2 * READOUT);
- an exact noisy simulator for the course's reference circuits, written with NumPy only, so that the verification
  values do not depend on qsim, Qiskit or Aer and are the same in every browser and in Colab;
- the tests of the three statistics functions that every project asks for:
  shot_sigma(p, shots), within_3_sigma(k, shots, p) and difference_sigma(f1, f2, shots).
"""
import math

import numpy as np

try:                                    # real Qiskit, if it is installed
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError:                     # the browser: the course simulator
    from qsim import QuantumCircuit, Statevector

ONE_QUBIT_GATES = ["h", "x", "rx", "ry", "rz", "p", "t", "tdg", "s", "sdg", "sx"]
TWO_QUBIT_GATES = ["cx", "cz", "cp", "rzz", "swap"]
SHOTS = 4000

# ---------------------------------------------------------------- an exact noisy simulator (NumPy only)
_S2 = 1 / math.sqrt(2)
_FIXED = {
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "t": np.diag([1, np.exp(1j * math.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
    "s": np.diag([1, 1j]),
    "sdg": np.diag([1, -1j]),
}


def _matrix(name, params):
    """The gate's matrix in Qiskit's convention (for two-qubit gates, the first qubit is the right-hand factor)."""
    if name in _FIXED:
        return _FIXED[name]
    if name == "rx":
        c, s = math.cos(params[0] / 2), math.sin(params[0] / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])
    if name == "rz":
        return np.diag([np.exp(-0.5j * params[0]), np.exp(0.5j * params[0])])
    if name == "p":
        return np.diag([1, np.exp(1j * params[0])])
    if name == "cx":                     # control = first qubit (the low bit of the 4 x 4 index)
        return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
    if name == "cz":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if name == "cp":
        return np.diag([1, 1, 1, np.exp(1j * params[0])])
    if name == "rzz":
        a = np.exp(-0.5j * params[0])
        b = np.exp(0.5j * params[0])
        return np.diag([a, b, b, a])
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    raise ValueError(f"unknown gate {name}")


def _apply(rho, u, qubits, n):
    """rho -> U rho U^dagger, with U acting on the given qubits (qubits[0] is the low bit of U's index)."""
    k = len(qubits)
    t = rho.reshape([2] * (2 * n))
    # axis of qubit q in the row index is n - 1 - q; in the column index 2n - 1 - q
    ut = u.reshape([2] * (2 * k))           # axes: out bits (high to low), in bits (high to low)
    row_axes = [n - 1 - q for q in reversed(qubits)]
    col_axes = [2 * n - 1 - q for q in reversed(qubits)]
    t = np.tensordot(ut, t, axes=(list(range(k, 2 * k)), row_axes))
    t = np.moveaxis(t, list(range(k)), row_axes)
    t = np.tensordot(t, ut.conj(), axes=(col_axes, list(range(k, 2 * k))))
    t = np.moveaxis(t, list(range(2 * n - k, 2 * n)), col_axes)
    return t.reshape(2 ** n, 2 ** n)


def _depolarize(rho, lam, qubits, n):
    """rho -> (1 - lam) rho + lam (I / 2^k on the given qubits) x (rho traced over them)."""
    if lam == 0:
        return rho
    k = len(qubits)
    t = rho.reshape([2] * (2 * n))
    row_axes = [n - 1 - q for q in qubits]
    col_axes = [2 * n - 1 - q for q in qubits]
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sub = list(letters[:2 * n])
    for ra, ca in zip(row_axes, col_axes):
        sub[ca] = sub[ra]
    keep = [i for i in range(2 * n) if i not in row_axes + col_axes]
    red = np.einsum("".join(sub) + "->" + "".join(sub[i] for i in keep), t)
    eye = np.eye(2 ** k).reshape([2] * (2 * k))      # identity: row bit i pairs with column bit i
    full = np.multiply.outer(red, eye)
    full = np.moveaxis(full, list(range(len(keep), 2 * n)), row_axes[::-1] + col_axes[::-1])
    return (1 - lam) * rho + lam * full.reshape(2 ** n, 2 ** n) / 2 ** k


def exact_probabilities(ops, n, measured, p2, readout):
    """Exact probabilities of the recorded results under the course noise model.

    ops: list of (name, qubits, params) applied to |0...0>; measured: the qubits measured at the end, measured[j] into
    classical bit j. Returns an array over the 2^len(measured) recorded results (bit j of the index is classical bit j).
    """
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1
    for name, qubits, params in ops:
        rho = _apply(rho, _matrix(name, params), list(qubits), n)
        if name in TWO_QUBIT_GATES:
            rho = _depolarize(rho, p2, list(qubits), n)
        elif name in ONE_QUBIT_GATES:
            rho = _depolarize(rho, p2 / 10, list(qubits), n)
    probs = np.real(np.diag(rho)).clip(min=0)
    probs = probs / probs.sum()
    m = len(measured)
    idx = np.arange(2 ** n)
    code = np.zeros(2 ** n, dtype=int)
    for j, q in enumerate(measured):
        code |= ((idx >> q) & 1) << j
    dist = np.zeros(2 ** m)
    np.add.at(dist, code, probs)
    conf = np.array([[1 - readout, readout], [2 * readout, 1 - 2 * readout]])
    t = dist.reshape([2] * m)
    for j in range(m):
        ax = m - 1 - j
        t = np.moveaxis(np.tensordot(t, conf, axes=([ax], [0])), -1, ax)
    dist = t.reshape(-1).clip(min=0)
    return dist / dist.sum()


def build_circuit(ops, n, measured):
    """The ops as a QuantumCircuit (Qiskit or qsim), with the measurements at the end."""
    qc = QuantumCircuit(n, len(measured)) if measured else QuantumCircuit(n)
    for name, qubits, params in ops:
        getattr(qc, name)(*params, *qubits)
    for j, q in enumerate(measured):
        qc.measure(q, j)
    return qc


# ---------------------------------------------------------------- helpers for the checks
def _op(inst):
    return getattr(inst, "operation", inst)


def _name(inst):
    return getattr(_op(inst), "name", "")


def _qubit_indices(qc, inst):
    qs = getattr(inst, "qubits", ())
    try:
        return [qc.find_bit(q).index for q in qs]
    except Exception:  # noqa: BLE001
        return [int(q) for q in qs]


def _clbit_indices(qc, inst):
    cs = getattr(inst, "clbits", ())
    try:
        return [qc.find_bit(c).index for c in cs]
    except Exception:  # noqa: BLE001
        return [int(c) for c in cs]


def measurement_map(qc):
    """{qubit: classical bit} for the measurements in a circuit."""
    return {_qubit_indices(qc, i)[0]: _clbit_indices(qc, i)[0] for i in qc.data if _name(i) == "measure"}


def big_gates(qc):
    """Names of gates on 3 or more qubits (the course noise model puts no error on them)."""
    return sorted({_name(i) for i in qc.data if _name(i) not in ("barrier", "measure") and len(_qubit_indices(qc, i)) >= 3})


def two_qubit_count(qc):
    """The number of two-qubit gates in a circuit (barriers not counted)."""
    return sum(1 for i in qc.data if _name(i) not in ("barrier", "measure") and len(_qubit_indices(qc, i)) == 2)


def ops_two_qubit_count(ops):
    return sum(1 for name, qubits, params in ops if len(qubits) == 2)


def ideal_probabilities(qc):
    """Probabilities of all qubits' values (index bit q = qubit q) for a circuit without its final measurements."""
    c = qc.copy()
    c.remove_final_measurements()
    return np.abs(np.asarray(Statevector(c).data, dtype=complex)) ** 2


def counts_total(counts):
    try:
        return int(sum(int(v) for v in counts.values()))
    except Exception:  # noqa: BLE001
        return -1


# ---------------------------------------------------------------- the statistics functions
def _close(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


def test_statistics(shot_sigma, within_3_sigma, difference_sigma):
    """Returns (passed, messages)."""
    msgs = []
    cases = [(0.5, 100), (0.687, 4000), (0.1, 1000), (0.0, 50), (1.0, 50), (0.9206, 4000)]
    for p, n in cases:
        want = math.sqrt(p * (1 - p) / n)
        try:
            got = shot_sigma(p, n)
        except NotImplementedError:
            return False, ["shot_sigma() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"shot_sigma({p}, {n}) raised {type(e).__name__}: {e}"]
        if not _close(got, want):
            hint = ""
            if _close(got, math.sqrt(p * (1 - p) * n)):
                hint = " That is the uncertainty of the count k; divide by shots for the fraction: sqrt(p * (1 - p) / shots)."
            elif _close(got, p * (1 - p) / n):
                hint = " Take the square root: sqrt(p * (1 - p) / shots)."
            elif _close(got, math.sqrt(p / n)) or _close(got, math.sqrt(1 / n)):
                hint = " Use p * (1 - p), the binomial variance of one shot."
            return False, [f"shot_sigma({p}, {n}) returned {got}; expected {want:.6g}.{hint}"]
    msgs.append("shot_sigma(): passed (6 cases).")

    w_cases = [(2748, 4000, 0.687, True), (2655, 4000, 0.687, False), (2841, 4000, 0.687, False),
               (2700, 4000, 0.687, True), (2821, 4000, 0.687, True), (55, 100, 0.5, True), (66, 100, 0.5, False),
               (34, 100, 0.5, False), (3700, 4000, 0.9206, True), (3620, 4000, 0.9206, False)]
    for k, n, p, want in w_cases:
        try:
            got = within_3_sigma(k, n, p)
        except NotImplementedError:
            return False, msgs + ["within_3_sigma() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, msgs + [f"within_3_sigma({k}, {n}, {p}) raised {type(e).__name__}: {e}"]
        if bool(got) != want or not isinstance(got, (bool, np.bool_)):
            d = abs(k / n - p) / max(math.sqrt(p * (1 - p) / n), 1e-300)
            hint = ""
            if not isinstance(got, (bool, np.bool_)):
                hint = " Return True or False."
            elif k > n * p and not want:
                hint = " Check both directions: a result can be too high as well as too low (use abs())."
            elif 2 < d <= 3:
                hint = " The course rule allows up to 3 standard deviations."
            else:
                hint = " Compare the fraction k / shots with p, using shot_sigma(p, shots)."
            return False, msgs + [f"within_3_sigma({k}, {n}, {p}) returned {got}; expected {want} "
                                  f"(the result is {d:.2f} standard deviations from p).{hint}"]
    msgs.append("within_3_sigma(): passed (10 cases).")

    d_cases = [(0.687, 0.634, 4000), (0.5, 0.5, 100), (0.9, 0.1, 1000), (0.2, 0.25, 4000)]
    for f1, f2, n in d_cases:
        want = math.sqrt(f1 * (1 - f1) / n + f2 * (1 - f2) / n)
        try:
            got = difference_sigma(f1, f2, n)
        except NotImplementedError:
            return False, msgs + ["difference_sigma() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, msgs + [f"difference_sigma({f1}, {f2}, {n}) raised {type(e).__name__}: {e}"]
        if not _close(got, want):
            s1, s2 = math.sqrt(f1 * (1 - f1) / n), math.sqrt(f2 * (1 - f2) / n)
            hint = ""
            if _close(got, s1 + s2):
                hint = " Uncertainties add in quadrature: sqrt(sigma1**2 + sigma2**2), not sigma1 + sigma2."
            elif _close(got, abs(s1 - s2)):
                hint = " The uncertainty of a difference is larger than either one: sqrt(sigma1**2 + sigma2**2)."
            elif _close(got, want ** 2):
                hint = " Take the square root."
            return False, msgs + [f"difference_sigma({f1}, {f2}, {n}) returned {got}; expected {want:.6g}.{hint}"]
    msgs.append("difference_sigma(): passed (4 cases).")
    return True, msgs


def personal_noise_ok(P2, READOUT):
    """Checks the personal noise numbers from Canvas; returns (ok, message, p2, readout)."""
    try:
        p2, ro = float(P2), float(READOUT)
    except (TypeError, ValueError):
        return False, "P2 and READOUT must be the numbers shown in your Canvas project quiz.", None, None
    if not (0.005 - 1e-12 <= p2 <= 0.030 + 1e-12) or abs(round(p2, 3) - p2) > 1e-9:
        return False, "P2 is a number from 0.005 to 0.030 with three decimals. Copy it again from the project quiz.", None, None
    if not (0.005 - 1e-12 <= ro <= 0.030 + 1e-12) or abs(round(ro, 3) - ro) > 1e-9:
        return False, "READOUT is a number from 0.005 to 0.030 with three decimals. Copy it again from the project quiz.", None, None
    return True, "", round(p2, 3), round(ro, 3)
