"""Tests of checks/l3_m4b_check.py: the reference Hamiltonian and Trotter steps against Qiskit (SparsePauliOp,
PauliEvolutionGate with LieTrotter and SuzukiTrotter), the noisy magnetization against Qiskit Aer's density matrix,
and the check's verdicts on correct and wrong versions of the learner's functions. Run with Qiskit 2.5.2 and Aer
0.17.2:  cd checks && python test_l3_m4b_check.py"""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "content" / "level3"), str(Path(__file__).resolve().parent)]
import l3_m4b_check as c  # noqa: E402
from qsim import QuantumCircuit, SparsePauliOp, NoiseModel, depolarizing_error, simulate_density_matrix  # noqa: E402

from qiskit import QuantumCircuit as QQC, transpile  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp as QOp, Operator  # noqa: E402
from qiskit.synthesis import LieTrotter, SuzukiTrotter  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel as ANM, depolarizing_error as adep  # noqa: E402


def qiskit_tfim(N, J, h, x_first=False):
    zz = [("ZZ", [i, i + 1], -J) for i in range(N - 1)]
    xx = [("X", [i], -h) for i in range(N)]
    return QOp.from_sparse_list(xx + zz if x_first else zz + xx, num_qubits=N)


for N, J, h in ((2, 1.0, 0.5), (3, 0.7, 1.3), (4, 1.0, 0.5)):
    d = np.abs(c.reference_ising_hamiltonian(N, J, h).to_matrix() - qiskit_tfim(N, J, h).to_matrix()).max()
    assert d < 1e-12
print("Hamiltonian: matches Qiskit's SparsePauliOp for 2, 3 and 4 spins")

worst = 0.0
for N, J, h, t, n in ((3, 1.0, 0.7, 1.0, 4), (4, 1.0, 0.5, 2.0, 6), (4, 0.8, 1.2, 1.5, 3)):
    for step, synth, x_first in ((c.reference_trotter_step, LieTrotter(reps=n), False),
                                 (c.reference_trotter_step2, SuzukiTrotter(order=2, reps=n), True)):
        qq = QQC(N)
        qq.append(PauliEvolutionGate(qiskit_tfim(N, J, h, x_first), time=t, synthesis=synth), range(N))
        U_q = Operator(qq.decompose()).data          # the synthesized gates, not the exact exponential
        U_m = c._unitary(c.trotter_circuit(N, J, h, t, n, step))
        k = np.argmax(np.abs(U_q))
        worst = max(worst, np.abs(U_m * (U_q.flat[k] / U_m.flat[k]) - U_q).max())
print(f"Trotter steps vs Qiskit PauliEvolutionGate (LieTrotter, SuzukiTrotter order 2): max difference {worst:.1e}")
assert worst < 1e-10

N, J, h, t = 4, 1.0, 0.5, 2.0
psi = c.exact_state(c.reference_ising_hamiltonian(N, J, h), t)
U = Operator(PauliEvolutionGate(qiskit_tfim(N, J, h), time=t)).data   # exact exponential
d = np.abs(psi - U[:, 0]).max()
print(f"exact evolution by diagonalization vs Qiskit's exact operator: max difference {d:.1e}")
assert d < 1e-10

p, n = 0.005, 5
nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["rzz"])
rho = simulate_density_matrix(c.trotter_circuit(N, J, h, t, n), nm)
mine = float(np.real(np.trace(rho.data @ c.magnetization_op(N).to_matrix())))
qq = QQC(N)
for _ in range(n):
    for i in range(N - 1):
        qq.rzz(-2 * J * t / n, i, i + 1)
    for i in range(N):
        qq.rx(-2 * h * t / n, i)
qq.save_density_matrix()
anm = ANM()
anm.add_all_qubit_quantum_error(adep(p, 2), ["rzz"])
sim = AerSimulator(method="density_matrix", noise_model=anm, basis_gates=["rzz", "rx"])
rho_a = sim.run(qq).result().data()["density_matrix"]
aer = float(np.real(rho_a.expectation_value(QOp.from_sparse_list([("Z", [i], 1 / N) for i in range(N)], N))))
print(f"noisy magnetization (p = {p}, {n} steps): qsim {mine:.10f}, Aer {aer:.10f}")
assert abs(mine - aer) < 1e-10


# ---------------------------------------------------------------- learner-style versions
def ham_good(N, J, h):
    terms = []
    for i in range(N - 1):
        label = ["I"] * N
        label[i] = label[i + 1] = "Z"
        terms.append(("".join(label)[::-1], -J))
    for i in range(N):
        label = ["I"] * N
        label[i] = "X"
        terms.append(("".join(label)[::-1], -h))
    return SparsePauliOp.from_list(terms)


def ham_sparse(N, J, h):
    return SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], -J) for i in range(N - 1)] + [("X", [i], -h) for i in range(N)], N)


def step_good(qc, N, J, h, dt):
    for i in range(N - 1):
        qc.rzz(-2 * J * dt, i, i + 1)
    for i in range(N):
        qc.rx(-2 * h * dt, i)


def step2_good(qc, N, J, h, dt):
    for i in range(N):
        qc.rx(-h * dt, i)
    for i in range(N - 1):
        qc.rzz(-2 * J * dt, i, i + 1)
    for i in range(N):
        qc.rx(-h * dt, i)


def ham_sign(N, J, h):
    return ham_good(N, -J, -h)


def ham_ring(N, J, h):
    H = ham_good(N, J, h)
    label = ["I"] * N
    label[0] = label[N - 1] = "Z"
    return SparsePauliOp.from_list(H.to_list() + [("".join(label)[::-1], -J)])


def ham_noreverse(N, J, h):
    terms = [("".join("Z" if k in (i, i + 1) else "I" for k in range(N)), -J) for i in range(N - 1)]
    terms += [("".join("X" if k == 0 else "I" for k in range(N)), -h) for i in range(N)]
    return SparsePauliOp.from_list(terms)


def ham_matrix(N, J, h):
    return ham_good(N, J, h).to_matrix()


def step_sign(qc, N, J, h, dt):
    step_good(qc, N, -J, -h, dt)


def step_half(qc, N, J, h, dt):
    step_good(qc, N, J / 2, h / 2, dt)


def step_xfirst(qc, N, J, h, dt):
    for i in range(N):
        qc.rx(-2 * h * dt, i)
    for i in range(N - 1):
        qc.rzz(-2 * J * dt, i, i + 1)


def step_returns_new(qc, N, J, h, dt):
    new = QuantumCircuit(N)
    step_good(new, N, J, h, dt)
    return new


def step_cx(qc, N, J, h, dt):
    for i in range(N - 1):
        qc.cx(i, i + 1)
        qc.rz(-2 * J * dt, i + 1)
        qc.cx(i, i + 1)
    for i in range(N):
        qc.rx(-2 * h * dt, i)


def step2_first(qc, N, J, h, dt):
    step_good(qc, N, J, h, dt)


def step2_zzout(qc, N, J, h, dt):
    for i in range(N - 1):
        qc.rzz(-J * dt, i, i + 1)
    for i in range(N):
        qc.rx(-2 * h * dt, i)
    for i in range(N - 1):
        qc.rzz(-J * dt, i, i + 1)


def step2_fullx(qc, N, J, h, dt):
    for i in range(N):
        qc.rx(-2 * h * dt, i)
    for i in range(N - 1):
        qc.rzz(-2 * J * dt, i, i + 1)
    for i in range(N):
        qc.rx(-2 * h * dt, i)


def raise_(*a, **k):
    raise NotImplementedError


GOOD = {"reference": (c.reference_ising_hamiltonian, c.reference_trotter_step, c.reference_trotter_step2),
        "loops": (ham_good, step_good, step2_good), "sparse list": (ham_sparse, step_good, step2_good)}
BAD = {"signs of H": (ham_sign, step_good, step2_good), "closed ring": (ham_ring, step_good, step2_good),
       "labels not reversed": (ham_noreverse, step_good, step2_good), "matrix not SparsePauliOp": (ham_matrix, step_good, step2_good),
       "step angle signs": (ham_good, step_sign, step2_good), "step half angles": (ham_good, step_half, step2_good),
       "X before ZZ": (ham_good, step_xfirst, step2_good), "returns new circuit": (ham_good, step_returns_new, step2_good),
       "cx + rz instead of rzz": (ham_good, step_cx, step2_good), "step2 is first order": (ham_good, step_good, step2_first),
       "step2 with ZZ outside": (ham_good, step_good, step2_zzout), "step2 full X twice": (ham_good, step_good, step2_fullx),
       "hamiltonian not written": (raise_, step_good, step2_good), "step not written": (ham_good, raise_, step2_good),
       "step2 not written": (ham_good, step_good, raise_)}

wrong = 0
for name, fns in GOOD.items():
    ok, msgs, val = c.check_l3_module4b(*fns, "0.50", "2.00", "8")
    print(f"GOOD {name:26s} -> {ok} {val}")
    wrong += (not ok) or val != c.personal_value(0.5, 2.0, 8)
for name, fns in BAD.items():
    ok, msgs, val = c.check_l3_module4b(*fns, "0.50", "2.00", "8")
    print(f"BAD  {name:26s} -> {ok} | {msgs[-1][:160]}")
    wrong += ok
for args in [("0.2", "1", "5"), ("0.5", "1.234", "5"), ("0.5", "1", "2.5"), ("0.5", "1", "13"), ("x", "1", "5")]:
    ok, msgs, _ = c.check_l3_module4b(*GOOD["reference"], *args)
    wrong += ok
print("wrong verdicts:", wrong)
assert wrong == 0
