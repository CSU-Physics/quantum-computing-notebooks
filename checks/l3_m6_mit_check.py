"""Check cell logic for the Level 3 Module 6 capstone "Mitigation study: one expectation value".

check_l3_m6_mit(fraction_sigma, expectation_sigma, combined_sigma, ghz_circuit, parity_expectation, mitigate_readout,
                fold_global, zne_coefficients, bootstrap_sigma, NQ, P2, READOUT) returns (passed, messages, value).

The expectation value: <X X ... X> of an NQ-qubit GHZ state, ideally 1. ghz_circuit(n) is the unitary part, without
measurements: H on qubit 0, CX(i, i + 1) for i = 0 .. n - 2, then H on every qubit (the H gates turn the X measurement
into an ordinary Z measurement). The notebook adds the measurements (qubit i into classical bit i).
- parity_expectation(probs): sum over results of (-1)^(number of 1s) * weight, divided by the total weight (works for
  counts and for probabilities);
- mitigate_readout(probs, readout): the probabilities corrected for readout errors, qubit by qubit (a tensor product of
  2 x 2 inverses), for the capstone readout model (0 -> 1 with READOUT, 1 -> 0 with 2 * READOUT);
- fold_global(qc, scale): global folding: qc followed by (scale - 1) / 2 copies of (qc inverse, qc), scale odd;
- zne_coefficients(scales, degree): weights c_i with estimate(0) = sum c_i E_i for a least-squares polynomial of that
  degree through (scale_i, E_i); for degree = len(scales) - 1 this is Richardson extrapolation;
- bootstrap_sigma(counts, estimator, n_boot=200, seed=0): the standard deviation of estimator(resampled counts) over
  n_boot multinomial resamplings of the shots.

The verification value is round(10000 * (E_mit - E_raw)) for the learner's NQ, P2 and READOUT: E_raw is the exact raw
<X...X>; E_mit applies readout mitigation at scales 1, 3 and 5 of global folding and extrapolates linearly to 0. All
values are exact (density matrices and the readout confusion), so no random numbers are involved.
"""
import math

import numpy as np

from qsim import QuantumCircuit, Operator

from l3_m6_common import _num, _try, apply_readout, confusion, exact_probabilities, test_statistics

SCALES = (1, 3, 5)


# ---------------------------------------------------------------- reference solutions
def reference_ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.h(i)
    return qc


def measured(qc):
    out = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    out.compose(qc, inplace=True)
    for i in range(qc.num_qubits):
        out.measure(i, i)
    return out


def reference_parity_expectation(probs):
    total = sum(probs.values())
    return float(sum((-1) ** k.count("1") * v for k, v in probs.items()) / total)


def reference_mitigate_readout(probs, readout):
    n = len(next(iter(probs)))
    v = np.zeros(2 ** n)
    for k, p in probs.items():
        v[int(k, 2)] = p
    v = v / v.sum()
    inv = np.linalg.inv(confusion(readout).T)          # measured = C^T true, for each bit
    for j in range(n):
        t = v.reshape(2 ** (n - 1 - j), 2, 2 ** j)
        v = np.einsum("ij,ajb->aib", inv, t).reshape(-1)
    return {format(i, f"0{n}b"): float(x) for i, x in enumerate(v)}


def reference_fold_global(qc, scale):
    out = qc.copy()
    inv = qc.inverse()
    for _ in range((scale - 1) // 2):
        out.compose(inv, inplace=True)
        out.compose(qc, inplace=True)
    return out


def reference_zne_coefficients(scales, degree):
    v = np.vander(np.asarray(scales, dtype=float), degree + 1, increasing=True)
    return [float(c) for c in np.linalg.pinv(v)[0]]


def reference_bootstrap_sigma(counts, estimator, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(counts)
    k = np.array([counts[x] for x in keys], dtype=float)
    shots = int(k.sum())
    vals = []
    for _ in range(n_boot):
        draw = rng.multinomial(shots, k / shots)
        vals.append(estimator({x: int(c) for x, c in zip(keys, draw) if c > 0}))
    return float(np.std(vals, ddof=1))


# ---------------------------------------------------------------- the verification value
def study(n, p2, readout, ghz=reference_ghz_circuit, fold=reference_fold_global, mitigate=reference_mitigate_readout,
          parity=reference_parity_expectation, coefficients=reference_zne_coefficients):
    """Exact raw and readout-mitigated values at each scale, and the extrapolations."""
    raw, ro = {}, {}
    for s in SCALES:
        probs = exact_probabilities(measured(fold(ghz(n), s)), p2, readout)
        raw[s] = parity(probs)
        ro[s] = parity(mitigate(probs, readout))
    lin = coefficients(list(SCALES), 1)
    quad = coefficients(list(SCALES), 2)
    return {"raw": raw, "ro": ro,
            "zne_lin": float(sum(c * ro[s] for c, s in zip(lin, SCALES))),
            "zne_quad": float(sum(c * ro[s] for c, s in zip(quad, SCALES))),
            "zne_lin_raw": float(sum(c * raw[s] for c, s in zip(lin, SCALES)))}


def personal_gain(n, p2, readout):
    r = study(int(n), float(p2), float(readout))
    return r["zne_lin"] - r["raw"][1]


def personal_value(n, p2, readout):
    return int(round(10000 * personal_gain(n, p2, readout)))


# ---------------------------------------------------------------- tests
def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _test_ghz(ghz_circuit):
    for n in (2, 3, 4):
        got, err = _try(ghz_circuit, "ghz_circuit", n)
        if err:
            return False, [err]
        if not isinstance(got, QuantumCircuit) or got.num_qubits != n:
            return False, [f"ghz_circuit({n}) should return a QuantumCircuit on {n} qubits."]
        if any(i.name in ("measure", "reset") for i in got.data):
            return False, [f"ghz_circuit({n}) has measurements; return the unitary part only (the notebook adds them), so it can be folded."]
        ref = reference_ghz_circuit(n)
        ideal = exact_probabilities(measured(got), 0, 0)
        if _tvd(ideal, exact_probabilities(measured(ref), 0, 0)) > 1e-9:
            hint = ""
            par = reference_parity_expectation(ideal)
            if abs(par) < 1e-6:
                hint = " Its <X...X> is 0: did you leave out the H gate on every qubit at the end?"
            return False, [f"ghz_circuit({n}) does not give an ideal <X...X> of 1 with the expected results.{hint}"]
        if _tvd(exact_probabilities(measured(got), 0.03, 0.0), exact_probabilities(measured(ref), 0.03, 0.0)) > 1e-9:
            return False, [f"ghz_circuit({n}) is right without noise but not with it: use H on qubit 0, the chain CX(0, 1), "
                           "CX(1, 2), ..., then H on every qubit, and no other gates."]
    return True, ["ghz_circuit(): passed (2 to 4 qubits, with and without noise)."]


def _test_parity(parity_expectation):
    cases = [({"00": 600, "11": 400}, 1.0), ({"01": 1, "10": 1}, -1.0), ({"000": 0.25, "011": 0.25, "001": 0.5}, 0.0),
             ({"0": 3, "1": 1}, 0.5), ({"111": 10, "000": 30, "100": 60}, -0.4)]
    for probs, want in cases:
        got, err = _try(parity_expectation, "parity_expectation", dict(probs))
        if err:
            return False, [err]
        g = _num(got)
        if g is None or abs(g - want) > 1e-12:
            hint = ""
            if g is not None and abs(g - sum((-1) ** k.count("1") * v for k, v in probs.items())) < 1e-12:
                hint = " Divide by the total weight, so it works for counts as well as probabilities."
            return False, [f"parity_expectation({probs}) gave {got!r}; expected {want}.{hint}"]
    return True, ["parity_expectation(): passed."]


def _test_mitigate(mitigate_readout):
    rng = np.random.default_rng(616)
    for n, ro in ((1, 0.02), (2, 0.03), (3, 0.015), (3, 0.05)):
        true = rng.dirichlet(np.ones(2 ** n))
        meas = apply_readout(true, ro)
        probs = {format(i, f"0{n}b"): float(p) for i, p in enumerate(meas)}
        got, err = _try(mitigate_readout, "mitigate_readout", dict(probs), ro)
        if err:
            return False, [err]
        try:
            g = np.array([float(got.get(format(i, f"0{n}b"), 0.0)) for i in range(2 ** n)])
        except Exception:  # noqa: BLE001
            return False, [f"mitigate_readout() returned {type(got).__name__}; return a dict {{'bits': probability}}."]
        if np.max(np.abs(g - true)) > 1e-9:
            hint = ""
            swapped = reference_mitigate_readout(probs, ro)
            c_sym = np.array([[1 - ro, ro], [ro, 1 - ro]])
            v = np.array([probs[format(i, f"0{n}b")] for i in range(2 ** n)])
            for j in range(n):
                t = v.reshape(2 ** (n - 1 - j), 2, 2 ** j)
                v = np.einsum("ij,ajb->aib", np.linalg.inv(c_sym.T), t).reshape(-1)
            if np.max(np.abs(g - v)) < 1e-9:
                hint = " A 1 is misread as 0 with probability 2 * readout, not readout."
            elif np.max(np.abs(g - np.array([swapped[format(i, f'0{n}b')] for i in range(2 ** n)])[::-1])) < 1e-9:
                hint = " The bit order is reversed: in the keys, qubit 0 is the rightmost character."
            else:
                c = confusion(ro)
                v = np.array([probs[format(i, f"0{n}b")] for i in range(2 ** n)])
                for j in range(n):
                    t = v.reshape(2 ** (n - 1 - j), 2, 2 ** j)
                    v = np.einsum("ij,ajb->aib", np.linalg.inv(c), t).reshape(-1)
                if np.max(np.abs(g - v)) < 1e-9:
                    hint = (" The matrix is transposed: measured[recorded] = sum over true of C[true][recorded] * true, so solve "
                            "with C transposed (rows: recorded bit).")
            return False, [f"mitigate_readout() on {n} qubit(s) with readout {ro} does not recover the true probabilities.{hint}"]
    return True, ["mitigate_readout(): passed (1 to 3 qubits)."]


def _asymmetric_circuit():
    qc = QuantumCircuit(3)
    qc.ry(0.7, 0)
    qc.cx(0, 1)
    qc.rz(0.3, 1)
    qc.ry(1.1, 2)
    qc.cx(1, 2)
    qc.h(0)
    return qc


def _test_fold(fold_global):
    for base in (reference_ghz_circuit(3), _asymmetric_circuit()):
        before = len(base.data)
        for scale in (1, 3, 5):
            got, err = _try(fold_global, "fold_global", base, scale)
            if err:
                return False, [err]
            if len(base.data) != before:
                return False, ["fold_global() changed the circuit it was given; build a new one (qc.copy())."]
            if not isinstance(got, QuantumCircuit):
                return False, [f"fold_global(qc, {scale}) should return a QuantumCircuit."]
            n_ops = len([i for i in got.data if i.name != "barrier"])
            if n_ops != scale * before:
                return False, [f"fold_global(qc, {scale}) has {n_ops} gates; global folding repeats the whole circuit {scale} times "
                               f"({scale * before}): qc, then (scale - 1) / 2 copies of qc.inverse() followed by qc."]
            if not np.allclose(Operator(got).data, Operator(base).data, atol=1e-9):
                return False, [f"fold_global(qc, {scale}) does not do the same as qc: each inverse must come before a copy of qc."]
            want = exact_probabilities(measured(reference_fold_global(base, scale)), 0.02, 0.01)
            if _tvd(exact_probabilities(measured(got), 0.02, 0.01), want) > 1e-9:
                return False, [f"fold_global(qc, {scale}) has the right gates in a different order; with noise the order matters: "
                               "qc, qc.inverse(), qc, ..."]
    return True, ["fold_global(): passed (scales 1, 3, 5, two circuits)."]


def _test_coefficients(zne_coefficients):
    for scales, deg in (([1, 3, 5], 1), ([1, 3, 5], 2), ([1, 2, 3], 1), ([1, 3], 1), ([1, 2, 3, 4], 2)):
        got, err = _try(zne_coefficients, "zne_coefficients", list(scales), deg)
        if err:
            return False, [err]
        want = reference_zne_coefficients(scales, deg)
        try:
            g = [float(x) for x in got]
        except Exception:  # noqa: BLE001
            g = None
        if g is None or len(g) != len(want) or max(abs(a - b) for a, b in zip(g, want)) > 1e-9:
            hint = ""
            if g is not None and len(g) == deg + 1:
                hint = (" These look like polynomial coefficients. Return one weight per scale: the values c_i with "
                        "estimate = sum c_i E_i (the first row of the pseudo-inverse of the Vandermonde matrix).")
            return False, [f"zne_coefficients({scales}, {deg}) gave {got!r}; expected {[round(x, 6) for x in want]}.{hint}"]
    return True, ["zne_coefficients(): passed (linear and quadratic, 2 to 4 scales)."]


def _test_bootstrap(bootstrap_sigma):
    counts = {"000": 1800, "011": 400, "101": 300, "111": 1500}
    keep = dict(counts)
    est = reference_parity_expectation
    E = est(counts)
    want = math.sqrt((1 - E * E) / 4000)
    got, err = _try(bootstrap_sigma, "bootstrap_sigma", counts, est, 400, 7)
    if err:
        return False, [err]
    if counts != keep:
        return False, ["bootstrap_sigma() changed the counts it was given; resample into new dictionaries."]
    g = _num(got)
    if g is None or not (0.8 * want < g < 1.25 * want):
        hint = ""
        if g is not None and abs(g - want * math.sqrt(4000)) < 0.25 * want * math.sqrt(4000):
            hint = " Each resample must have as many shots as the original (4,000), drawn with the original fractions."
        elif g is not None and g < 1e-12:
            hint = " All resamples gave the same value: draw new counts each time (rng.multinomial(shots, fractions))."
        return False, [f"bootstrap_sigma() gave {got!r}; for these counts it should be close to {want:.4f}.{hint}"]
    g2, err = _try(bootstrap_sigma, "bootstrap_sigma", counts, est, 400, 7)
    if err:
        return False, [err]
    if _num(g2) != g:
        return False, ["bootstrap_sigma() with the same seed gave different results; make the generator with "
                       "np.random.default_rng(seed)."]
    return True, ["bootstrap_sigma(): passed."]


def check_l3_m6_mit(fraction_sigma, expectation_sigma, combined_sigma, ghz_circuit, parity_expectation,
                    mitigate_readout, fold_global, zne_coefficients, bootstrap_sigma, NQ, P2, READOUT):
    try:
        n, p2, ro = int(NQ), float(P2), float(READOUT)
        ok = float(NQ) == n and 2 <= n <= 5 and 0.005 <= p2 <= 0.030 and 0.005 <= ro <= 0.030
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return False, ["Copy your numbers again from your capstone quiz: NQ (2 to 5), P2 and READOUT (0.005 to 0.030)."], None
    msgs = []
    for test in (lambda: test_statistics(fraction_sigma, expectation_sigma, combined_sigma),
                 lambda: _test_ghz(ghz_circuit), lambda: _test_parity(parity_expectation),
                 lambda: _test_mitigate(mitigate_readout), lambda: _test_fold(fold_global),
                 lambda: _test_coefficients(zne_coefficients), lambda: _test_bootstrap(bootstrap_sigma)):
        good, m = test()
        msgs += m
        if not good:
            return False, msgs, None
    return True, msgs, personal_value(n, p2, ro)
