"""Check cell logic for the Level 3 Module 4 Track A lab (quantum-kernel classifier).

check_l3_module4a(feature_map, kernel_matrix, kernel_entry_shots, A, B, C)
returns (passed, messages, verification_value).

The learner writes three functions:
- feature_map(x, reps=2): the ZZ feature map on len(x) qubits, as Qiskit's zz_feature_map(len(x), reps=reps) with
  its default full entanglement. Each repetition: H on every qubit; P(2 x_i) on qubit i; then for every pair i < j:
  CX(i, j), P(2 (pi - x_i)(pi - x_j)) on qubit j, CX(i, j).
- kernel_matrix(XA, XB): the matrix K[a, b] = |<phi(XA[a]) | phi(XB[b])>|^2, from statevectors.
- kernel_entry_shots(x1, x2, shots, rng): an estimate of K(x1, x2) from `shots` measurements of the overlap circuit
  feature_map(x1) followed by feature_map(x2).inverse(): the fraction of shots in which every qubit reads 0.

The verification value is round(1000 * K(x, x')) for x = (A, B) and x' = (B, C), computed with the reference feature
map after every test passes (statevectors, no random numbers).
"""
import numpy as np

import qsim
from qsim import QuantumCircuit, Statevector


# ---------------------------------------------------------------- reference solutions
def reference_feature_map(x, reps=2):
    x = [float(v) for v in x]
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.p(2 * x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def reference_states(X, feature_map=reference_feature_map):
    return np.array([Statevector(feature_map(x)).data for x in X])


def reference_kernel_matrix(XA, XB):
    A, B = reference_states(XA), reference_states(XB)
    return np.abs(A.conj() @ B.T) ** 2


def reference_kernel_entry_shots(x1, x2, shots, rng):
    qc = reference_feature_map(x1).compose(reference_feature_map(x2).inverse())
    p0 = float(abs(Statevector(qc).data[0]) ** 2)
    return rng.binomial(shots, min(max(p0, 0.0), 1.0)) / shots


def personal_value(a, b, c):
    s1, s2 = reference_states([(float(a), float(b)), (float(b), float(c))])
    return int(round(1000 * abs(np.vdot(s1, s2)) ** 2))


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


_RNG = np.random.default_rng(424242)
_X2 = [tuple(_RNG.uniform(0, 2 * np.pi, 2)) for _ in range(4)]
_X3 = [tuple(_RNG.uniform(0, 2 * np.pi, 3)) for _ in range(2)]


# ---------------------------------------------------------------- tests
def _test_feature_map(feature_map):
    for x in _X2 + _X3:
        qc, err = _try(feature_map, "feature_map", x)
        if err:
            return False, [err]
        if not isinstance(qc, QuantumCircuit):
            return False, ["feature_map(x) should return a QuantumCircuit."]
        if qc.num_qubits != len(x):
            return False, [f"feature_map(x) should use one qubit per feature: {len(x)} features, {qc.num_qubits} qubits."]
        mine, ref = Statevector(qc).data, Statevector(reference_feature_map(x)).data
        if abs(abs(np.vdot(ref, mine)) - 1) > 1e-8:
            ops = qc.count_ops()
            hint = ""
            if abs(abs(np.vdot(Statevector(reference_feature_map(x, reps=1)).data, mine)) - 1) < 1e-8:
                hint = " It looks like one repetition; the default must be reps=2."
            elif ops.get("h", 0) != 2 * len(x):
                hint = " Each of the two repetitions starts with an H on every qubit."
            elif ops.get("cx", 0) != 2 * len(x) * (len(x) - 1):
                hint = " Every pair of qubits i < j needs CX(i, j), a phase on qubit j, and CX(i, j) again, in each repetition."
            else:
                hint = " Check the angles: P(2 x_i) on qubit i, and P(2 (pi - x_i)(pi - x_j)) on qubit j for each pair."
            return False, [f"feature_map({tuple(round(float(v), 3) for v in x)}) gives a different state from the ZZ feature map." + hint]
    qc, err = _try(feature_map, "feature_map", _X2[0], 1)
    if err:
        return False, ["feature_map(x, reps) should accept the number of repetitions: " + err]
    if abs(abs(np.vdot(Statevector(reference_feature_map(_X2[0], 1)).data, Statevector(qc).data)) - 1) > 1e-8:
        return False, ["feature_map(x, reps=1) should repeat the pattern once (the default is reps=2)."]
    return True, ["feature_map(): passed (2 and 3 qubits, 1 and 2 repetitions, same states as Qiskit's zz_feature_map)."]


def _test_kernel_matrix(kernel_matrix):
    XA, XB = _X2[:3], _X2[1:]
    K, err = _try(kernel_matrix, "kernel_matrix", XA, XB)
    if err:
        return False, [err]
    K = np.asarray(K, dtype=float) if K is not None else None
    if K is None or K.shape != (3, 3):
        return False, [f"kernel_matrix(XA, XB) should return a len(XA) x len(XB) array; it returned shape "
                       f"{None if K is None else K.shape} for 3 x 3."]
    ref = reference_kernel_matrix(XA, XB)
    if np.allclose(K, ref, atol=1e-8):
        pass
    elif np.allclose(K, np.sqrt(ref), atol=1e-8):
        return False, ["kernel_matrix() returns |<phi(a)|phi(b)>|, not its square: the kernel is |<phi(a)|phi(b)>|^2."]
    elif np.allclose(K, ref.T, atol=1e-8):
        return False, ["kernel_matrix(XA, XB) is transposed: row a should belong to XA[a] and column b to XB[b]."]
    else:
        return False, ["kernel_matrix() does not match |<phi(a)|phi(b)>|^2 for the ZZ feature map."]
    Ks, err = _try(kernel_matrix, "kernel_matrix", _X2, _X2)
    if err:
        return False, [err]
    Ks = np.asarray(Ks, dtype=float)
    if not np.allclose(np.diag(Ks), 1, atol=1e-8):
        return False, ["The kernel of a point with itself must be 1; check that you use the same feature map for both."]
    return True, ["kernel_matrix(): passed (a 3 x 3 block and a symmetric 4 x 4 matrix with 1s on the diagonal)."]


def _test_shots(kernel_entry_shots):
    x1, x2 = _X2[0], _X2[1]
    exact = reference_kernel_matrix([x1], [x2])[0, 0]
    vals = []
    for seed in range(3):
        v, err = _try(kernel_entry_shots, "kernel_entry_shots", x1, x2, 1000, np.random.default_rng(seed))
        if err:
            return False, [err]
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False, ["kernel_entry_shots() should return one number: the fraction of shots that read all zeros."]
        if abs(v * 1000 - round(v * 1000)) > 1e-9:
            return False, ["kernel_entry_shots(x1, x2, 1000, rng) should be a whole number of shots divided by 1000; "
                           "draw the number of all-zero results with rng.binomial(shots, p)."]
        vals.append(v)
    if len(set(vals)) == 1:
        return False, ["kernel_entry_shots() gives the same value for different random generators: sample with rng, "
                       "for example rng.binomial(shots, p)."]
    big, err = _try(kernel_entry_shots, "kernel_entry_shots", x1, x2, 2_000_000, np.random.default_rng(7))
    if err:
        return False, [err]
    big = float(big)
    if abs(big - exact) > 0.003:
        hint = ""
        if abs(big - (1 - exact)) < 0.003:
            hint = " It looks like the fraction of shots that are not all zeros."
        elif abs(big - np.sqrt(exact)) < 0.003:
            hint = " It averages to |<phi|phi'>| instead of its square: use the probability of all zeros."
        else:
            hint = (" Build the overlap circuit feature_map(x1).compose(feature_map(x2).inverse()) and take the probability "
                    "of all zeros, Statevector(qc).data[0] squared in magnitude.")
        return False, [f"kernel_entry_shots() averages to {big:.4f} with many shots; the kernel is {exact:.4f}." + hint]
    return True, ["kernel_entry_shots(): passed (whole numbers of shots, random from shot to shot, and the right "
                  "average with 2,000,000 shots)."]


def check_l3_module4a(feature_map, kernel_matrix, kernel_entry_shots, A, B, C):
    try:
        a, b, c = float(A), float(B), float(C)
    except (TypeError, ValueError):
        return False, ["A, B and C must be the numbers shown in your Canvas track quiz."], None
    two = lambda v: abs(round(v, 2) - v) < 1e-9  # noqa: E731
    if not all(0.10 <= v <= 3.00 and two(v) for v in (a, b, c)):
        return False, ["Copy your numbers again from the track quiz: A, B and C are each between 0.10 and 3.00, with two "
                       "decimals."], None
    msgs = []
    for test in (lambda: _test_feature_map(feature_map), lambda: _test_kernel_matrix(kernel_matrix),
                 lambda: _test_shots(kernel_entry_shots)):
        good, m = test()
        msgs += m
        if not good:
            return False, msgs, None
    return True, msgs, personal_value(a, b, c)
