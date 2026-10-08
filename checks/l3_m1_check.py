"""Check cell logic for the Level 3 Module 1 lab (noise and open quantum systems).

check_l3_module1(density_matrix, bloch_vector, apply_channel, relax, THETA, T1, T2, TIME)
returns (passed, messages, verification_value).

The learner writes four functions, with NumPy only:
- density_matrix(psi): the density matrix |psi><psi| of a state vector on any number of qubits;
- bloch_vector(rho): the Bloch vector (Tr(rho X), Tr(rho Y), Tr(rho Z)) of a one-qubit density matrix, as 3 real numbers;
- apply_channel(rho, kraus): the channel rho -> sum_k K rho K^dagger for a list of Kraus matrices;
- relax(rho, t, T1, T2): one qubit after a time t of T1 and T2 relaxation: amplitude damping with
  gamma = 1 - exp(-t/T1), then phase damping with lam = 1 - exp(-2t/T2 + t/T1) (needs T2 <= 2 T1).
  This is qiskit_aer.noise.thermal_relaxation_error(T1, T2, t) with no excited-state population.

The verification value is round(1000 * F), where F = <psi|relax(|psi><psi|, TIME, T1, T2)|psi> for
|psi> = RY(THETA)|0>, computed with the reference solutions below. In Bloch-vector form it is
F = (1 + sin^2(THETA) exp(-TIME/T2) + cos(THETA) (1 - (1 - cos(THETA)) exp(-TIME/T1))) / 2.
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------- reference solutions
def reference_density_matrix(psi):
    v = np.asarray(psi, dtype=complex).reshape(-1)
    return np.outer(v, v.conj())


def reference_bloch_vector(rho):
    r = np.asarray(rho, dtype=complex)
    return np.array([np.trace(r @ P).real for P in (X, Y, Z)])


def reference_apply_channel(rho, kraus):
    r = np.asarray(rho, dtype=complex)
    return sum(np.asarray(k, dtype=complex) @ r @ np.asarray(k, dtype=complex).conj().T for k in kraus)


def amplitude_damping_kraus(gamma):
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]


def phase_damping_kraus(lam):
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]


def reference_relax(rho, t, T1, T2):
    gamma = 1 - np.exp(-t / T1)
    lam = 1 - np.exp(-2 * t / T2 + t / T1)
    out = reference_apply_channel(rho, amplitude_damping_kraus(gamma))
    return reference_apply_channel(out, phase_damping_kraus(lam))


def ry_state(theta):
    return np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)


def personal_fidelity(theta, T1, T2, time):
    psi = ry_state(theta)
    rho = reference_relax(reference_density_matrix(psi), time, T1, T2)
    return float(np.real(np.vdot(psi, rho @ psi)))


def personal_value(theta, T1, T2, time):
    return int(round(1000 * personal_fidelity(float(theta), float(T1), float(T2), float(time))))


# ---------------------------------------------------------------- helpers
def _rand_state(rng, n):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def _rand_rho(rng, n, rank=None):
    d = 2 ** n
    k = rank or d
    a = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
    r = a @ a.conj().T
    return r / np.trace(r).real


def _rand_kraus(rng, n, m):
    """m Kraus matrices on n qubits from a random isometry, so that sum K^dagger K = I."""
    d = 2 ** n
    a = rng.normal(size=(m * d, d)) + 1j * rng.normal(size=(m * d, d))
    q, _ = np.linalg.qr(a)
    return [q[i * d:(i + 1) * d, :] for i in range(m)]


def _as_matrix(x, shape, name):
    try:
        m = np.asarray(x, dtype=complex)
    except Exception:  # noqa: BLE001
        return None, f"{name} should return a NumPy array, not {type(x).__name__}."
    if m.shape != shape:
        return None, f"{name} returned an array of shape {m.shape}; it should be {shape}."
    return m, None


def _call(fn, name, *args):
    try:
        return fn(*args), None
    except NotImplementedError:
        return None, f"{name}() is not written yet."
    except Exception as e:  # noqa: BLE001
        return None, f"{name}() raised {type(e).__name__}: {e}"


# ---------------------------------------------------------------- tests
def _test_density_matrix(density_matrix):
    rng = np.random.default_rng(301)
    cases = [np.array([1, 0], dtype=complex), np.array([1, 1], dtype=complex) / np.sqrt(2),
             np.array([1, 1j], dtype=complex) / np.sqrt(2)] + [_rand_state(rng, n) for n in (1, 1, 2, 3)]
    for psi in cases:
        out, err = _call(density_matrix, "density_matrix", psi.copy())
        if err:
            return False, [err]
        d = len(psi)
        got, err = _as_matrix(out, (d, d), f"density_matrix() for a {int(np.log2(d))}-qubit state")
        if err:
            return False, [err]
        want = reference_density_matrix(psi)
        if not np.allclose(got, want, atol=1e-9):
            if np.allclose(got, np.outer(psi, psi), atol=1e-9):
                hint = " Take the complex conjugate of the second factor: np.outer(psi, psi.conj())."
            elif np.allclose(got, np.outer(psi.conj(), psi), atol=1e-9):
                hint = " The conjugate is on the wrong factor: |psi><psi| is np.outer(psi, psi.conj())."
            elif np.allclose(got, np.abs(np.diag(psi)) ** 2, atol=1e-9):
                hint = " Keep the off-diagonal terms: |psi><psi| is the outer product, not just the probabilities."
            else:
                hint = " |psi><psi| is the outer product of the column psi with its conjugate row: np.outer(psi, psi.conj())."
            return False, [f"density_matrix() gives the wrong matrix for psi = {np.round(psi, 3)}.{hint}"]
    return True, ["density_matrix(): passed (7 states on 1 to 3 qubits)."]


def _test_bloch(bloch_vector):
    rng = np.random.default_rng(302)
    cases = [reference_density_matrix(np.array([1, 0])), reference_density_matrix(np.array([1, 1j]) / np.sqrt(2)),
             I2 / 2] + [_rand_rho(rng, 1) for _ in range(4)] + [_rand_rho(rng, 1, 1) for _ in range(2)]
    for rho in cases:
        out, err = _call(bloch_vector, "bloch_vector", rho.copy())
        if err:
            return False, [err]
        try:
            got = np.asarray(out, dtype=complex).reshape(-1)
        except Exception:  # noqa: BLE001
            return False, ["bloch_vector() should return three numbers, for example a NumPy array of length 3."]
        if got.shape != (3,):
            return False, [f"bloch_vector() returned {got.shape[0]} numbers; it should return 3: (x, y, z)."]
        if np.max(np.abs(got.imag)) > 1e-9:
            return False, ["bloch_vector() returned complex numbers. Each component is real: take .real of each trace."]
        want = reference_bloch_vector(rho)
        got = got.real
        if not np.allclose(got, want, atol=1e-9):
            if np.allclose(got, want * [1, -1, 1], atol=1e-9):
                hint = (" The y component has the wrong sign. Y = [[0, -1j], [1j, 0]], and y = Tr(rho Y); "
                        "check the matrix you used for Y.")
            elif np.allclose(got, want[::-1], atol=1e-9):
                hint = " The order is (x, y, z): Tr(rho X), Tr(rho Y), Tr(rho Z)."
            elif np.allclose(got, want / 2, atol=1e-9):
                hint = " Each component is the full trace Tr(rho P), not half of it."
            else:
                hint = " Each component is a trace: x = Tr(rho X), y = Tr(rho Y), z = Tr(rho Z) (np.trace(rho @ X).real)."
            return False, [f"bloch_vector() gives {np.round(got, 3)} for a state whose Bloch vector is {np.round(want, 3)}.{hint}"]
    return True, ["bloch_vector(): passed (9 one-qubit states, pure and mixed)."]


def _test_channel(apply_channel):
    rng = np.random.default_rng(303)
    for n, m in ((1, 1), (1, 2), (1, 4), (2, 3), (2, 1)):
        kraus = _rand_kraus(rng, n, m)
        rho = _rand_rho(rng, n)
        out, err = _call(apply_channel, "apply_channel", rho.copy(), [k.copy() for k in kraus])
        if err:
            return False, [err]
        d = 2 ** n
        got, err = _as_matrix(out, (d, d), "apply_channel()")
        if err:
            return False, [err]
        want = reference_apply_channel(rho, kraus)
        if not np.allclose(got, want, atol=1e-9):
            rev = sum(k.conj().T @ rho @ k for k in kraus)
            notconj = sum(k @ rho @ k.T for k in kraus)
            if np.allclose(got, rev, atol=1e-9):
                hint = " The order is reversed: each term is K rho K^dagger (K on the left), not K^dagger rho K."
            elif np.allclose(got, notconj, atol=1e-9):
                hint = " Use the conjugate transpose K.conj().T (the dagger), not only the transpose K.T."
            elif len(kraus) > 1 and np.allclose(got, want / len(kraus), atol=1e-9):
                hint = " Add the terms, do not average them: the Kraus matrices already carry the probabilities."
            elif len(kraus) > 1 and np.allclose(got, kraus[0] @ rho @ kraus[0].conj().T, atol=1e-9):
                hint = " Only the first Kraus matrix was used: add K rho K^dagger over all of them."
            else:
                hint = " Return the sum over all Kraus matrices K of K @ rho @ K.conj().T."
            return False, [f"apply_channel() gives the wrong result for {m} Kraus matrices on {n} qubit(s).{hint}"]
    return True, ["apply_channel(): passed (5 random channels on 1 and 2 qubits)."]


RELAX_CASES = [(10.0, 100.0, 80.0), (50.0, 100.0, 80.0), (30.0, 120.0, 240.0), (200.0, 300.0, 150.0), (0.0, 100.0, 50.0),
               (75.0, 250.0, 400.0)]


def _test_relax(relax):
    rng = np.random.default_rng(304)
    for t, T1, T2 in RELAX_CASES:
        rho = _rand_rho(rng, 1)
        out, err = _call(relax, "relax", rho.copy(), t, T1, T2)
        if err:
            return False, [err]
        got, err = _as_matrix(out, (2, 2), "relax()")
        if err:
            return False, [err]
        want = reference_relax(rho, t, T1, T2)
        if not np.allclose(got, want, atol=1e-9):
            gamma = 1 - np.exp(-t / T1)
            wrong_lam = reference_apply_channel(reference_apply_channel(rho, amplitude_damping_kraus(gamma)),
                                                phase_damping_kraus(1 - np.exp(-t / T2)))
            ad_only = reference_apply_channel(rho, amplitude_damping_kraus(gamma))
            swapped = reference_apply_channel(reference_apply_channel(rho, amplitude_damping_kraus(1 - np.exp(-t / T2))),
                                              phase_damping_kraus(max(0.0, min(1.0, 1 - np.exp(-2 * t / T1 + t / T2)))))
            if np.allclose(got, wrong_lam, atol=1e-9):
                hint = (" Amplitude damping already shrinks the coherences by exp(-t/(2 T1)), so phase damping must supply only "
                        "the rest: lam = 1 - exp(-2 t / T2 + t / T1), not 1 - exp(-t / T2).")
            elif np.allclose(got, ad_only, atol=1e-9):
                hint = " After amplitude damping, apply phase damping too, with lam = 1 - exp(-2 t / T2 + t / T1)."
            elif np.allclose(got, swapped, atol=1e-9):
                hint = " T1 and T2 are swapped: gamma uses T1, and lam uses both, as 1 - exp(-2 t / T2 + t / T1)."
            else:
                hint = (" Apply amplitude_damping_kraus(1 - exp(-t/T1)) with apply_channel, then phase_damping_kraus(lam) "
                        "with lam = 1 - exp(-2 t / T2 + t / T1).")
            return False, [f"relax() gives the wrong state for t = {t}, T1 = {T1}, T2 = {T2}.{hint}"]
    return True, ["relax(): passed (6 cases, including T2 > T1 and t = 0)."]


def check_l3_module1(density_matrix, bloch_vector, apply_channel, relax, THETA, T1, T2, TIME):
    try:
        th, t1, t2, tm = float(THETA), float(T1), float(T2), float(TIME)
    except (TypeError, ValueError):
        return False, ["THETA, T1, T2 and TIME must be the numbers shown in your Canvas lab check."], None
    ok_range = (0.3 <= th <= 2.8 and abs(round(th, 2) - th) < 1e-9 and 100 <= t1 <= 400 and 50 <= t2 <= 2 * t1
                and 20 <= tm <= 200 and t1 == int(t1) and t2 == int(t2) and tm == int(tm))
    if not ok_range:
        return False, ["Copy your numbers again from the lab check: THETA has two decimals (0.30 to 2.80); T1, T2 and TIME "
                       "are whole numbers of microseconds (T1 100 to 400, T2 50 up to 2 T1, TIME 20 to 200)."], None
    msgs, ok = [], True
    for test, fn in ((_test_density_matrix, density_matrix), (_test_bloch, bloch_vector),
                     (_test_channel, apply_channel), (_test_relax, relax)):
        good, m = test(fn)
        msgs += m
        ok = ok and good
    if not ok:
        return False, msgs, None
    return True, msgs, personal_value(th, t1, t2, tm)
