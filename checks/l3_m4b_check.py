"""Check cell logic for the Level 3 Module 4 Track B lab (Trotter simulation of a transverse-field Ising chain).

check_l3_module4b(ising_hamiltonian, trotter_step, trotter_step2, H_FIELD, T, STEPS)
returns (passed, messages, verification_value).

The model: an open chain of N spins, H = -J sum_i Z_i Z_{i+1} - h sum_i X_i, qubit i is spin i. In Qiskit's Pauli
labels qubit 0 is the rightmost character, so the term Z_0 Z_1 on 4 qubits is "IIZZ" and X_0 is "IIIX".

The learner writes three functions:
- ising_hamiltonian(N, J, h): H as a SparsePauliOp;
- trotter_step(qc, N, J, h, dt): one first-order (Lie-Trotter) step added to qc, in this order: RZZ(-2 J dt) on
  every neighbouring pair (i, i + 1), then RX(-2 h dt) on every qubit. Since RZZ(theta) = exp(-i theta Z Z / 2) and
  RX(theta) = exp(-i theta X / 2), the step is exp(+i J dt sum Z Z) exp(+i h dt sum X), an approximation of
  exp(-i H dt);
- trotter_step2(qc, N, J, h, dt): one second-order (symmetric, Strang) step: RX(-h dt) on every qubit, RZZ(-2 J dt)
  on every pair, RX(-h dt) on every qubit.

The verification value is round(1000 * M) where M = (1/N) sum_i <Z_i> after STEPS first-order steps from |0000> to
time T, for N = 4, J = 1 and h = H_FIELD, computed with the reference functions after every test passes
(statevectors, no random numbers).
"""
import numpy as np

import qsim
from qsim import QuantumCircuit, SparsePauliOp, Statevector

N_LAB, J_LAB = 4, 1.0


# ---------------------------------------------------------------- reference solutions
def _label(N, ops):
    chars = ["I"] * N
    for q, p in ops.items():
        chars[N - 1 - q] = p
    return "".join(chars)


def reference_ising_hamiltonian(N, J, h):
    terms = [(_label(N, {i: "Z", i + 1: "Z"}), -J) for i in range(N - 1)]
    terms += [(_label(N, {i: "X"}), -h) for i in range(N)]
    return SparsePauliOp.from_list(terms)


def reference_trotter_step(qc, N, J, h, dt):
    for i in range(N - 1):
        qc.rzz(-2 * J * dt, i, i + 1)
    for i in range(N):
        qc.rx(-2 * h * dt, i)


def reference_trotter_step2(qc, N, J, h, dt):
    for i in range(N):
        qc.rx(-h * dt, i)
    for i in range(N - 1):
        qc.rzz(-2 * J * dt, i, i + 1)
    for i in range(N):
        qc.rx(-h * dt, i)


def _x_first(qc, N, J, h, dt):
    for i in range(N):
        qc.rx(-2 * h * dt, i)
    for i in range(N - 1):
        qc.rzz(-2 * J * dt, i, i + 1)


def _zz_outside(qc, N, J, h, dt):
    for i in range(N - 1):
        qc.rzz(-J * dt, i, i + 1)
    for i in range(N):
        qc.rx(-2 * h * dt, i)
    for i in range(N - 1):
        qc.rzz(-J * dt, i, i + 1)


def magnetization_op(N):
    return SparsePauliOp.from_list([(_label(N, {i: "Z"}), 1.0 / N) for i in range(N)])


def trotter_circuit(N, J, h, t, n_steps, step=reference_trotter_step):
    qc = QuantumCircuit(N)
    for _ in range(int(n_steps)):
        step(qc, N, J, h, t / n_steps)
    return qc


def magnetization(qc, N):
    return float(np.real(Statevector(qc).expectation_value(magnetization_op(N))))


def exact_state(H, t):
    """exp(-i H t)|0...0> by diagonalizing H."""
    E, V = np.linalg.eigh(H.to_matrix())
    psi0 = np.zeros(V.shape[0], dtype=complex)
    psi0[0] = 1
    return V @ (np.exp(-1j * E * t) * (V.conj().T @ psi0))


def personal_value(h, t, steps):
    qc = trotter_circuit(N_LAB, J_LAB, float(h), float(t), int(steps))
    return int(round(1000 * magnetization(qc, N_LAB)))


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


def _unitary(qc):
    n = qc.num_qubits
    cols = []
    for k in range(2 ** n):
        init = np.zeros(2 ** n, dtype=complex)
        init[k] = 1
        cols.append(Statevector(init).evolve(qc).data)
    return np.array(cols).T


def _same_up_to_phase(U, V):
    k = np.argmax(np.abs(V))
    if abs(U.flat[k]) < 1e-12:
        return False
    ph = V.flat[k] / U.flat[k]
    return abs(abs(ph) - 1) < 1e-8 and np.allclose(U * ph, V, atol=1e-8)


# ---------------------------------------------------------------- tests
def _test_hamiltonian(ising_hamiltonian):
    for N, J, h in ((2, 1.0, 0.5), (3, 0.7, 1.3), (4, 1.0, 1.0)):
        H, err = _try(ising_hamiltonian, "ising_hamiltonian", N, J, h)
        if err:
            return False, [err]
        if not isinstance(H, SparsePauliOp):
            return False, ["ising_hamiltonian() should return a SparsePauliOp, for example SparsePauliOp.from_list(terms)."]
        if H.num_qubits != N:
            return False, [f"ising_hamiltonian({N}, ...) has {H.num_qubits} qubits; it should have one per spin."]
        mine, ref = H.to_matrix(), reference_ising_hamiltonian(N, J, h).to_matrix()
        if np.allclose(mine, ref, atol=1e-10):
            continue
        hint = ""
        if np.allclose(mine, -ref, atol=1e-10):
            hint = " The overall sign is wrong: both coefficients are negative, -J and -h."
        elif np.allclose(mine, reference_ising_hamiltonian(N, -J, h).to_matrix(), atol=1e-10):
            hint = " The ZZ terms have the wrong sign: their coefficient is -J."
        elif np.allclose(mine, reference_ising_hamiltonian(N, J, -h).to_matrix(), atol=1e-10):
            hint = " The X terms have the wrong sign: their coefficient is -h."
        elif N > 2 and np.allclose(mine, (reference_ising_hamiltonian(N, J, h)
                                          + SparsePauliOp.from_list([(_label(N, {0: "Z", N - 1: "Z"}), -J)])).to_matrix(), atol=1e-10):
            hint = " It includes a ZZ term between the first and last spin; the chain is open (N - 1 pairs)."
        else:
            hint = (" Check the terms: N - 1 terms Z_i Z_(i+1) with coefficient -J and N terms X_i with coefficient -h. "
                    "In a Pauli label qubit 0 is the rightmost character.")
        return False, [f"ising_hamiltonian({N}, {J}, {h}) is not -J sum Z_i Z_(i+1) - h sum X_i." + hint]
    return True, ["ising_hamiltonian(): passed (2, 3 and 4 spins)."]


def _test_step(fn, name, ref_fn, other_fn):
    for N, J, h, dt in ((2, 1.0, 0.6, 0.3), (3, 0.8, 1.1, 0.2), (4, 1.0, 0.5, 0.25)):
        qc = QuantumCircuit(N)
        out, err = _try(fn, name, qc, N, J, h, dt)
        if err:
            return False, [err]
        if out is not None and out is not qc:
            return False, [f"{name}(qc, N, J, h, dt) should add gates to qc itself, not build and return a new circuit."]
        if not qc.data:
            return False, [f"{name}() added nothing to the circuit."]
        extra = set(i.name for i in qc.data) - {"rzz", "rx", "barrier"}
        if extra:
            return False, [f"{name}() should use only rzz and rx gates; it added {sorted(extra)}."]
        ref = QuantumCircuit(N)
        ref_fn(ref, N, J, h, dt)
        U, R = _unitary(qc), _unitary(ref)
        if _same_up_to_phase(U, R):
            continue
        hint = ""
        flip = QuantumCircuit(N)
        ref_fn(flip, N, -J, -h, dt)
        if _same_up_to_phase(U, _unitary(flip)):
            hint = " The angles have the wrong sign: exp(-i H dt) with H = -J ZZ - h X needs RZZ(-2 J dt) and RX(-2 h dt)."
        else:
            for alt, text in other_fn:
                oth = QuantumCircuit(N)
                alt(oth, N, J, h, dt)
                if _same_up_to_phase(U, _unitary(oth)):
                    hint = text
                    break
        if not hint:
            half = QuantumCircuit(N)
            ref_fn(half, N, J / 2, h / 2, dt)
            dbl = QuantumCircuit(N)
            ref_fn(dbl, N, 2 * J, 2 * h, dt)
            if _same_up_to_phase(U, _unitary(half)):
                hint = " The angles are half as large as they should be: RZZ(theta) is exp(-i theta ZZ / 2), so use -2 J dt."
            elif _same_up_to_phase(U, _unitary(dbl)):
                hint = " The angles are twice as large as they should be."
            else:
                hint = " Check the pairs (i, i + 1) for i = 0 .. N - 2 and the angle of each gate."
        return False, [f"{name}() gives a different step from the expected one for N = {N}." + hint]
    return True, [f"{name}(): passed (2, 3 and 4 spins; the same unitary as the expected step)."]


def _test_order(trotter_step, trotter_step2):
    """Error of the learner's steps against the exact evolution: first order ~ dt, second order ~ dt^2."""
    N, J, h, t = 3, 1.0, 0.7, 1.0
    psi = exact_state(reference_ising_hamiltonian(N, J, h), t)
    errs = {}
    for name, step in (("trotter_step", trotter_step), ("trotter_step2", trotter_step2)):
        e = []
        for n in (8, 16):
            qc = QuantumCircuit(N)
            for _ in range(n):
                step(qc, N, J, h, t / n)
            e.append(1 - abs(np.vdot(psi, Statevector(qc).data)) ** 2)
        errs[name] = np.log2(e[0] / e[1])
    if not (1.6 < errs["trotter_step"] < 2.6 and 3.4 < errs["trotter_step2"] < 4.6):
        return False, [f"Halving the step should divide the infidelity by about 4 (first order) and 16 (second order); "
                       f"yours give factors 2^{errs['trotter_step']:.1f} and 2^{errs['trotter_step2']:.1f}."]
    return True, []


def check_l3_module4b(ising_hamiltonian, trotter_step, trotter_step2, H_FIELD, T, STEPS):
    try:
        h, t, s = float(H_FIELD), float(T), float(STEPS)
    except (TypeError, ValueError):
        return False, ["H_FIELD, T and STEPS must be the numbers shown in your Canvas track quiz."], None
    two = lambda v: abs(round(v, 2) - v) < 1e-9  # noqa: E731
    if not (0.30 <= h <= 1.20 and 0.50 <= t <= 2.50 and two(h) and two(t) and s == int(s) and 3 <= s <= 12):
        return False, ["Copy your numbers again from the track quiz: H_FIELD (0.30 to 1.20) and T (0.50 to 2.50) with two "
                       "decimals, and STEPS a whole number from 3 to 12."], None
    msgs = []
    tests = (lambda: _test_hamiltonian(ising_hamiltonian),
             lambda: _test_step(trotter_step, "trotter_step", reference_trotter_step, [
                 (_x_first, " The order is different: put all the RZZ gates first, then all the RX gates."),
                 (reference_trotter_step2, " This is the second-order step; trotter_step() is the first-order one.")]),
             lambda: _test_step(trotter_step2, "trotter_step2", reference_trotter_step2, [
                 (reference_trotter_step, " This is the first-order step; the second-order step is RX(-h dt), "
                                          "RZZ(-2 J dt), RX(-h dt)."),
                 (_zz_outside, " This symmetric step is also second order, but the lab puts the RX gates outside: "
                               "RX(-h dt), RZZ(-2 J dt), RX(-h dt).")]),
             lambda: _test_order(trotter_step, trotter_step2))
    for test in tests:
        good, m = test()
        msgs += m
        if not good:
            return False, msgs, None
    return True, msgs, personal_value(h, t, int(s))
