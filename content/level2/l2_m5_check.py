"""Check cell logic for the Level 2 Module 5 lab (variational algorithms: VQE and QAOA).

check_l2_module5(ansatz, maxcut_hamiltonian, qaoa_circuit, GAMMA, BETA) returns (passed, messages, verification_value).

The learner writes three functions:
- ansatz(layers): a parameterized 2-qubit circuit with 2*layers parameters. Layer 1 is ry(θ[0]) on qubit 0 and
  ry(θ[1]) on qubit 1; each further layer k adds cx(0, 1), then ry(θ[2k]) on qubit 0 and ry(θ[2k+1]) on qubit 1
  (the structure of Qiskit's real_amplitudes(2, reps=layers - 1));
- maxcut_hamiltonian(edges, n): the SparsePauliOp sum of Z_i Z_j over the edges (i, j) of a graph on n nodes;
- qaoa_circuit(gammas, betas, edges, n): H on every qubit, then for each layer k: rzz(2*gammas[k]) on every edge
  and rx(2*betas[k]) on every qubit. No measurements.

The verification value is round(1000 * C), where C is the expected cut size of the reference QAOA circuit with one
layer, gamma = GAMMA and beta = BETA, on the course graph (a square with one diagonal). It is computed exactly from
the state vector, so it is the same in every browser and in Qiskit.
"""
import numpy as np

try:                                    # real Qiskit, if it is installed
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp, Statevector
except ImportError:                     # the browser: the course simulator
    from qsim import QuantumCircuit, ParameterVector, SparsePauliOp, Statevector

COURSE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]


# ---------------------------------------------------------------- reference solutions
def reference_ansatz(layers):
    theta = ParameterVector("θ", 2 * layers)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    for k in range(1, layers):
        qc.cx(0, 1)
        qc.ry(theta[2 * k], 0)
        qc.ry(theta[2 * k + 1], 1)
    return qc


def reference_maxcut_hamiltonian(edges, n):
    return SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0) for i, j in edges], num_qubits=n)


def reference_qaoa_circuit(gammas, betas, edges, n):
    qc = QuantumCircuit(n)
    qc.h(list(range(n)))
    for g, b in zip(gammas, betas):
        for i, j in edges:
            qc.rzz(2 * g, i, j)
        for q in range(n):
            qc.rx(2 * b, q)
    return qc


def _state(qc):
    return np.asarray(Statevector(qc).data, dtype=complex)


def _fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2


def expected_cut(state, edges, n):
    """Expected number of cut edges for a state vector (qubit q is bit q of the index)."""
    idx = np.arange(2 ** n)
    cut = sum(((idx >> i) & 1) ^ ((idx >> j) & 1) for i, j in edges)
    return float(np.abs(state) ** 2 @ cut)


def _matrix(op):
    if hasattr(op, "to_matrix"):
        return np.asarray(op.to_matrix(), dtype=complex)
    raise TypeError


# ---------------------------------------------------------------- tests
def _test_ansatz(ansatz):
    rng = np.random.default_rng(55)
    for layers in (1, 2, 3):
        try:
            qc = ansatz(layers)
        except NotImplementedError:
            return False, ["ansatz() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"ansatz({layers}) raised {type(e).__name__}: {e}"]
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != 2:
            return False, [f"ansatz({layers}) should return a QuantumCircuit on 2 qubits."]
        if any(getattr(getattr(i, "operation", i), "name", "") == "measure" for i in qc.data):
            return False, ["Leave measurements out of the ansatz: the Estimator works with the state itself."]
        if qc.num_parameters != 2 * layers:
            return False, [f"ansatz({layers}) has {qc.num_parameters} parameters; it should have {2 * layers} "
                           "(two per layer). Use theta = ParameterVector('θ', 2 * layers)."]
        ref = reference_ansatz(layers)
        for _ in range(4):
            vals = rng.uniform(-np.pi, np.pi, 2 * layers)
            try:
                got = _state(qc.assign_parameters(vals))
            except Exception as e:  # noqa: BLE001
                return False, [f"ansatz({layers}).assign_parameters(...) raised {type(e).__name__}: {e}"]
            want = _state(ref.assign_parameters(vals))
            if _fidelity(got, want) < 1 - 1e-9:
                hint = ""
                # common mistakes, checked on the same values
                swapped = _state(_swap_variant(layers).assign_parameters(vals))
                no_cx = _state(_swap_variant(layers, cx=False).assign_parameters(vals))
                if _fidelity(got, swapped) > 1 - 1e-9:
                    hint = " The CNOT points the wrong way: use qc.cx(0, 1) (control 0, target 1)."
                elif _fidelity(got, no_cx) > 1 - 1e-9:
                    hint = " Each layer after the first starts with qc.cx(0, 1); without it the qubits never become entangled."
                elif layers > 1:
                    hint = (" Check the order: layer 1 is ry(θ[0]) on qubit 0 and ry(θ[1]) on qubit 1; each further "
                            "layer k is cx(0, 1), then ry(θ[2k]) on qubit 0 and ry(θ[2k+1]) on qubit 1.")
                else:
                    hint = " Layer 1 is ry(θ[0]) on qubit 0 and ry(θ[1]) on qubit 1, and nothing else."
                return False, [f"ansatz({layers}) gives a different state from the expected one for some parameter "
                               f"values.{hint}"]
    return True, ["ansatz(): passed (1, 2 and 3 layers, 4 sets of random parameter values each)."]


def _swap_variant(layers, cx=True):
    """The ansatz with the CNOT reversed (or, with cx=False, left out): two common mistakes."""
    theta = ParameterVector("θ", 2 * layers)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    for k in range(1, layers):
        if cx:
            qc.cx(1, 0)
        qc.ry(theta[2 * k], 0)
        qc.ry(theta[2 * k + 1], 1)
    return qc


HAM_CASES = [([(0, 1)], 2), ([(0, 1), (1, 2)], 3), ([(0, 2)], 3), (COURSE_EDGES, 4), ([(0, 3), (1, 3), (2, 3)], 4)]


def _test_hamiltonian(maxcut_hamiltonian):
    for edges, n in HAM_CASES:
        try:
            h = maxcut_hamiltonian(list(edges), n)
        except NotImplementedError:
            return False, ["maxcut_hamiltonian() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"maxcut_hamiltonian({edges}, {n}) raised {type(e).__name__}: {e}"]
        if not isinstance(h, SparsePauliOp):
            return False, ["maxcut_hamiltonian() should return a SparsePauliOp."]
        if h.num_qubits != n:
            return False, [f"maxcut_hamiltonian({edges}, {n}) acts on {h.num_qubits} qubits; it should act on {n}."]
        got, want = _matrix(h), _matrix(reference_maxcut_hamiltonian(edges, n))
        if not np.allclose(got, want, atol=1e-9):
            dim = 2 ** n
            cut_form = (len(edges) * np.eye(dim) - want) / 2
            if np.allclose(got, cut_form, atol=1e-9) or np.allclose(got, -cut_form, atol=1e-9):
                hint = (" You built the cut-counting operator (I - ZZ)/2. In this lab the cost Hamiltonian is the sum of "
                        "Z_i Z_j alone, with coefficient 1, so that its lowest energy belongs to the maximum cut.")
            elif np.allclose(got, -want, atol=1e-9):
                hint = " The sign is reversed: use coefficient +1 for each Z_i Z_j term."
            else:
                hint = (" Put a Z on qubits i and j of each edge and I elsewhere, with qubit 0 as the rightmost letter; "
                        "SparsePauliOp.from_sparse_list([('ZZ', [i, j], 1.0) for i, j in edges], num_qubits=n) does this.")
            return False, [f"maxcut_hamiltonian({edges}, {n}) is not the expected operator.{hint}"]
    return True, [f"maxcut_hamiltonian(): passed ({len(HAM_CASES)} graphs on 2 to 4 nodes)."]


QAOA_CASES = [([0.4], [0.3], [(0, 1)], 2), ([1.1], [0.7], COURSE_EDGES, 4), ([0.25, 0.8], [0.6, 0.2], [(0, 1), (1, 2)], 3),
              ([0.9, 0.3], [0.15, 1.2], COURSE_EDGES, 4), ([0.5], [0.5], [(0, 3), (1, 3), (2, 3)], 4)]


def _test_qaoa(qaoa_circuit):
    for gammas, betas, edges, n in QAOA_CASES:
        try:
            qc = qaoa_circuit(list(gammas), list(betas), list(edges), n)
        except NotImplementedError:
            return False, ["qaoa_circuit() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"qaoa_circuit({gammas}, {betas}, {edges}, {n}) raised {type(e).__name__}: {e}"]
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != n:
            return False, [f"qaoa_circuit(...) should return a QuantumCircuit on {n} qubits."]
        if any(getattr(getattr(i, "operation", i), "name", "") == "measure" for i in qc.data):
            return False, ["Leave measurements out of qaoa_circuit(): the Estimator and Statevector work with the state itself."]
        try:
            got = _state(qc)
        except Exception as e:  # noqa: BLE001
            return False, [f"The circuit from qaoa_circuit(...) cannot be simulated: {type(e).__name__}: {e}"]
        want = _state(reference_qaoa_circuit(gammas, betas, edges, n))
        if _fidelity(got, want) < 1 - 1e-9:
            half = _state(reference_qaoa_circuit([g / 2 for g in gammas], [b / 2 for b in betas], edges, n))
            half_g = _state(reference_qaoa_circuit([g / 2 for g in gammas], betas, edges, n))
            half_b = _state(reference_qaoa_circuit(gammas, [b / 2 for b in betas], edges, n))
            no_h = QuantumCircuit(n)
            for g, b in zip(gammas, betas):
                for i, j in edges:
                    no_h.rzz(2 * g, i, j)
                for q in range(n):
                    no_h.rx(2 * b, q)
            if _fidelity(got, half) > 1 - 1e-9:
                hint = " Use twice the angles: rzz(2 * gamma, i, j) and rx(2 * beta, q)."
            elif _fidelity(got, half_g) > 1 - 1e-9:
                hint = " The cost layer needs rzz(2 * gamma, i, j), not rzz(gamma, i, j)."
            elif _fidelity(got, half_b) > 1 - 1e-9:
                hint = " The mixer needs rx(2 * beta, q), not rx(beta, q)."
            elif _fidelity(got, _state(no_h)) > 1 - 1e-9:
                hint = " Start with H on every qubit (the equal superposition), before the first layer."
            elif len(gammas) > 1:
                hint = " With two layers, apply layer 1 (gammas[0], betas[0]) completely, then layer 2."
            else:
                hint = " Check each part: H on every qubit; rzz(2 * gamma) on every edge; rx(2 * beta) on every qubit."
            return False, [f"qaoa_circuit({gammas}, {betas}, {edges}, {n}) gives a different state from the expected one.{hint}"]
    return True, [f"qaoa_circuit(): passed ({len(QAOA_CASES)} cases, 1 and 2 layers, 2 to 4 nodes)."]


def personal_value(gamma, beta):
    """round(1000 * expected cut) for the reference one-layer QAOA on the course graph."""
    state = _state(reference_qaoa_circuit([float(gamma)], [float(beta)], COURSE_EDGES, 4))
    return int(round(1000 * expected_cut(state, COURSE_EDGES, 4)))


def check_l2_module5(ansatz, maxcut_hamiltonian, qaoa_circuit, GAMMA, BETA):
    try:
        g, b = float(GAMMA), float(BETA)
    except (TypeError, ValueError):
        return False, ["GAMMA and BETA must be the numbers shown in your Canvas lab check."], None
    if not (0.1 <= g <= 1.5 and 0.1 <= b <= 1.5) or abs(round(g, 2) - g) > 1e-9 or abs(round(b, 2) - b) > 1e-9:
        return False, ["GAMMA and BETA are numbers from 0.10 to 1.50 with two decimals. Copy them again from the lab check."], None
    msgs, ok = [], True
    for test, fn in ((_test_ansatz, ansatz), (_test_hamiltonian, maxcut_hamiltonian), (_test_qaoa, qaoa_circuit)):
        good, m = test(fn)
        msgs += m
        ok = ok and good
    if not ok:
        return False, msgs, None
    return True, msgs, personal_value(g, b)
