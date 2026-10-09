"""Build the Level 3 Module 4 Track B browser notebook (Trotter simulation of a 4-spin transverse-field Ising chain
against the exact evolution: the Hamiltonian, first- and second-order Trotter steps, how the error scales with the
step size, and the trade-off between Trotter error and gate noise).
The check code is checks/l3_m4b_check.py, published next to the notebook by l3_hidden."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level3"
from l3_hidden import hidden_check  # noqa: E402
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L3-M4B-lab-trotter-ising.ipynb"
VERSION = "2026-10-09"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 4, Track B lab: simulating a spin chain with Trotter steps

**Quantum Computing Advanced · Module 4 · Track B · about 110 minutes** · notebook version {VERSION}

Simulating how a quantum system changes in time is the application that Feynman proposed for quantum computers. In this lab you simulate four interacting spins, the **transverse-field Ising chain**, and compare every result with the exact answer:

1. write `ising_hamiltonian(N, J, h)`: the energy of the chain as a sum of Pauli terms;
2. compute the exact evolution and watch the magnetization change;
3. write `trotter_step(...)`: one first-order **Trotter step** made of RZZ and RX gates;
4. measure how the error shrinks as the steps get smaller;
5. write `trotter_step2(...)`: a second-order step, and see how much faster its error shrinks;
6. add gate noise and find the number of steps that gives the smallest total error;
7. count what this would cost on real hardware;
8. get your verification value.

The **Module 4 Track B quiz** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser."""),
code("""import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import QuantumCircuit, SparsePauliOp, Statevector, NoiseModel, depolarizing_error, simulate_density_matrix

N, J = 4, 1.0            # four spins; coupling J = 1 sets the unit of energy (and 1/J the unit of time)
print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),

md("""## Step 1: the Hamiltonian

The chain has N spins in a row, one qubit each. Neighbouring spins prefer to point the same way along Z (coupling J), and a transverse magnetic field h tries to turn them along X:

H = −J (Z₀Z₁ + Z₁Z₂ + Z₂Z₃) − h (X₀ + X₁ + X₂ + X₃).

The chain is **open**: spin 3 is not coupled back to spin 0, so there are N − 1 = 3 ZZ terms and N = 4 X terms.

Write `ising_hamiltonian(N, J, h)` and return a `SparsePauliOp`. In a Pauli label **qubit 0 is the rightmost character**, as in Qiskit: on 4 qubits, Z₀Z₁ is `"IIZZ"` and X₃ is `"XIII"`. `SparsePauliOp.from_list([("IIZZ", -J), ...])` builds the operator from (label, coefficient) pairs."""),
code("""def ising_hamiltonian(N, J, h):
    \"\"\"H = -J sum_i Z_i Z_(i+1) - h sum_i X_i on an open chain of N qubits, as a SparsePauliOp.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete ising_hamiltonian() first.")


H = ising_hamiltonian(N, J, 0.5)
for label, coeff in H.to_list():
    print(f"{label}  {coeff.real:+.2f}")
energies = np.linalg.eigvalsh(H.to_matrix())
print(f"lowest energies: {np.round(energies[:3], 4)}")"""),
md("""**What to notice.** Seven terms: three ZZ couplings with coefficient −1 and four X terms with −0.5. The matrix is 16 × 16, small enough to handle exactly. With 50 spins it would have 2⁵⁰ rows: that is why we want a quantum computer to do the evolution."""),

md("""## Step 2: the exact evolution

All spins start up along Z, the state |0000⟩. The state at time t is |ψ(t)⟩ = e^(−iHt)|0000⟩. For a small chain we can compute it exactly by diagonalizing H (`exact_state`, given). We follow the **magnetization** M = (⟨Z₀⟩ + ⟨Z₁⟩ + ⟨Z₂⟩ + ⟨Z₃⟩)/4, which starts at 1. Run the cell."""),
code("""def exact_state(H, t):
    \"\"\"exp(-i H t)|0...0>, by diagonalizing H.\"\"\"
    E, V = np.linalg.eigh(H.to_matrix())
    psi0 = np.zeros(V.shape[0], dtype=complex)
    psi0[0] = 1
    return V @ (np.exp(-1j * E * t) * (V.conj().T @ psi0))


M_OP = SparsePauliOp.from_list([("IIIZ", 0.25), ("IIZI", 0.25), ("IZII", 0.25), ("ZIII", 0.25)])


def magnetization(psi):
    \"\"\"(1/4) sum_i <Z_i> for a state vector, a circuit, or a density matrix.\"\"\"
    if isinstance(psi, QuantumCircuit):
        psi = Statevector(psi).data
    psi = np.asarray(getattr(psi, "data", psi))
    if psi.ndim == 2:
        return float(np.real(np.trace(psi @ M_OP.to_matrix())))
    return float(np.real(psi.conj() @ M_OP.to_matrix() @ psi))


times = np.linspace(0, 3, 61)
plt.figure(figsize=(7, 3.8))
for h, col in ((0.5, "#99004C"), (1.0, "#24313D")):
    Hh = ising_hamiltonian(N, J, h)
    plt.plot(times, [magnetization(exact_state(Hh, t)) for t in times], color=col, label=f"exact, h = {h}")
plt.xlabel("time t (units of 1/J)")
plt.ylabel("magnetization M")
plt.legend()
plt.show()
H = ising_hamiltonian(N, J, 0.5)
print(f"exact magnetization at t = 2 for h = 0.5: {magnetization(exact_state(H, 2.0)):.3f}")"""),
md("""**What to notice.** With a weak field (h = 0.5) the spins stay mostly up and M oscillates between about 0.67 and 1. With h = J = 1, the critical point of the infinite chain, the field turns the spins much more and M swings down to about 0. From here on we use h = 0.5 and look at t = 2, where the exact value is 0.670."""),

md("""## Step 3: one Trotter step

A quantum computer applies gates, not e^(−iHt) directly. H has two parts that do not commute: the ZZ part and the X part. The **Trotter** idea is to apply them one after the other for a short time dt and repeat:

e^(−iH dt) ≈ e^(+iJ dt Σ ZᵢZᵢ₊₁) e^(+ih dt Σ Xᵢ), repeated n = t/dt times.

Each factor is made of simple gates, because the terms inside it commute. With RZZ(θ) = e^(−iθ ZZ/2) and RX(θ) = e^(−iθ X/2), one step is:

- **RZZ(−2J dt)** on each neighbouring pair (0, 1), (1, 2), (2, 3), then
- **RX(−2h dt)** on each qubit.

Write `trotter_step(qc, N, J, h, dt)`, which adds one step to the circuit `qc`, with the RZZ gates first. `trotter_circuit` (given) repeats it n times."""),
code("""def trotter_step(qc, N, J, h, dt):
    \"\"\"Add one first-order Trotter step to qc: RZZ(-2 J dt) on every pair (i, i + 1), then RX(-2 h dt) on every qubit.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete trotter_step() first.")


def trotter_circuit(N, J, h, t, n_steps, step=None):
    step = step or trotter_step
    qc = QuantumCircuit(N)
    for _ in range(n_steps):
        step(qc, N, J, h, t / n_steps)
    return qc


print(trotter_circuit(N, J, 0.5, 2.0, 1).draw())
plt.figure(figsize=(7, 3.8))
plt.plot(times, [magnetization(exact_state(H, t)) for t in times], color="#99004C", label="exact")
for dt, style in ((0.5, "o"), (0.2, "s")):
    ts = np.arange(0, 3.0001, dt)
    plt.plot(ts, [magnetization(trotter_circuit(N, J, 0.5, t, max(1, round(t / dt)))) for t in ts], style,
             color="#24313D", alpha=0.7, label=f"Trotter, dt = {dt}")
plt.xlabel("time t")
plt.ylabel("magnetization M")
plt.legend()
plt.title("h = 0.5")
plt.show()"""),
md("""**What to notice.** With dt = 0.5 the Trotter points drift away from the exact curve as time goes on; with dt = 0.2 they follow it closely. The error comes from the parts of H that do not commute and grows with the total time."""),

md("""## Step 4: how fast does the error shrink?

To measure the error of the whole simulation we compare states. The **infidelity** 1 − |⟨ψ_exact|ψ_Trotter⟩|² is 0 when they agree. Run the cell: it fixes t = 2 and doubles the number of steps from 4 to 32."""),
code("""psi_exact = exact_state(H, 2.0)
steps = np.array([4, 8, 16, 32])


def infidelities(step=None):
    return np.array([1 - abs(np.vdot(psi_exact, Statevector(trotter_circuit(N, J, 0.5, 2.0, n, step)).data)) ** 2
                     for n in steps])


inf1 = infidelities()
slope1 = np.polyfit(np.log(steps), np.log(inf1), 1)[0]
for n, e in zip(steps, inf1):
    print(f"first order, {n:2d} steps (dt = {2.0 / n:.4f}): infidelity {e:.5f}")
print(f"fitted slope of log(infidelity) against log(steps): {slope1:.2f}")"""),
md("""**What to notice.** Each doubling of the steps divides the infidelity by about 4: the slope is close to −2. The state error of a first-order step is proportional to dt, and the infidelity is its square. Halving the error of the state costs twice as many gates."""),

md("""## Step 5: a second-order step

A symmetric arrangement cancels the leading error. Split the X part in two halves around the ZZ part:

- **RX(−h dt)** on each qubit, then **RZZ(−2J dt)** on each pair, then **RX(−h dt)** on each qubit.

Write `trotter_step2(qc, N, J, h, dt)`. It has the same RZZ gates as before, the two-qubit gates that cost the most on hardware, and only more single-qubit gates."""),
code("""def trotter_step2(qc, N, J, h, dt):
    \"\"\"Add one second-order (symmetric) Trotter step: RX(-h dt) on every qubit, RZZ(-2 J dt) on every pair,
    RX(-h dt) on every qubit.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete trotter_step2() first.")


inf2 = infidelities(trotter_step2)
slope2 = np.polyfit(np.log(steps), np.log(inf2), 1)[0]
for n, e1, e2 in zip(steps, inf1, inf2):
    print(f"{n:2d} steps: infidelity first order {e1:.5f}   second order {e2:.7f}")
print(f"fitted slopes: first order {slope1:.2f}, second order {slope2:.2f}")
print("two-qubit gates per step: first order", trotter_circuit(N, J, 0.5, 2.0, 1).count_ops().get("rzz", 0),
      "  second order", trotter_circuit(N, J, 0.5, 2.0, 1, trotter_step2).count_ops().get("rzz", 0))
plt.figure(figsize=(5.5, 3.8))
plt.loglog(steps, inf1, "o-", color="#24313D", label="first order")
plt.loglog(steps, inf2, "s-", color="#99004C", label="second order")
plt.xlabel("number of Trotter steps")
plt.ylabel("infidelity at t = 2")
plt.legend()
plt.show()"""),
md("""**What to notice.** The second-order slope is about −4: each doubling of the steps divides the infidelity by about 16. With 8 steps the second-order step is already about 40 times more accurate than the first-order one, for the same three RZZ gates per step. (Consecutive half RX steps can even be merged, so the extra cost is small.)"""),

md("""## Step 6: Trotter error against gate noise

On hardware every RZZ gate adds noise. Here each RZZ gets a **0.5% two-qubit depolarizing error**, about right for an RZZ made of two CNOTs on a current device. More steps mean less Trotter error but more noisy gates, so there is a best number of steps. The cell computes the error of the magnetization at t = 2 for 1 to 16 steps, exactly (density matrices, no shots)."""),
code("""P2 = 0.005
noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(P2, 2), ["rzz"])
M_exact = magnetization(exact_state(H, 2.0))
ns = np.arange(1, 17)
err = {}
for name, step in (("first order", trotter_step), ("second order", trotter_step2)):
    vals = [magnetization(simulate_density_matrix(trotter_circuit(N, J, 0.5, 2.0, int(n), step), noise)) for n in ns]
    err[name] = np.abs(np.array(vals) - M_exact)
    best = int(ns[np.argmin(err[name])])
    print(f"{name}: smallest error {err[name].min():.4f} with {best} steps ({3 * best} noisy RZZ gates)")
plt.figure(figsize=(6, 3.8))
for (name, e), col in zip(err.items(), ("#24313D", "#99004C")):
    plt.semilogy(ns, e, "o-", color=col, label=name)
plt.xlabel("number of Trotter steps")
plt.ylabel("|M - M_exact| at t = 2")
plt.title(f"two-qubit error {100 * P2:.1f}% per RZZ")
plt.legend()
plt.show()"""),
md("""**What to notice.** Both curves fall, reach a minimum and rise again. With few steps the Trotter error dominates; with many, the noise of the extra gates does. The first-order step does best with 5 steps (error 0.043); the second-order step does best with 6 steps and reaches an error about five times smaller (0.008). A better formula buys accuracy without more two-qubit gates, and mitigation (Module 3) can then remove part of the remaining noise."""),

md("""## Step 7: toward hardware

On an IBM computer each RZZ is usually built from two CNOT (or CZ) gates with a single-qubit rotation between them, and the qubits must sit next to each other on the chip. The cell counts the two-qubit gates of the best schedules from Step 6."""),
code("""for name, step in (("first order", trotter_step), ("second order", trotter_step2)):
    best = int(ns[np.argmin(err[name])])
    rzz = trotter_circuit(N, J, 0.5, 2.0, best, step).count_ops().get("rzz", 0)
    print(f"{name}: {best} steps, {rzz} RZZ gates, about {2 * rzz} CNOTs on hardware")"""),
md("""**What to notice.** 15 to 18 RZZ gates, about 30 to 36 CNOTs, for four spins: fine for today's hardware. IBM's 2023 "utility" experiment (Kim et al., Nature 618, 500) ran the same kind of Trotter circuits for a 127-spin Ising model with up to 60 layers and used zero-noise extrapolation to recover accurate magnetizations. For the largest circuits no exact answer was available to compare with, which is the point of using a quantum computer. In the capstone you can run a 2- or 3-spin version with mitigation, and optionally on real hardware."""),

md("""## Step 8: your personal check

Open the **Module 4 Track B quiz** in Canvas. Question 1 shows your own H_FIELD (0.30 to 1.20), T (0.50 to 2.50) and STEPS (3 to 12). Type them below and run the next two cells. The check cell tests `ising_hamiltonian()`, `trotter_step()` and `trotter_step2()`. Only if every test passes does it print your **verification value**: 1,000 times the magnetization after STEPS first-order Trotter steps from |0000⟩ to time T, for four spins with J = 1 and h = H_FIELD, rounded to a whole number. For example, 0.50, 2.00, 8 gives 665."""),
code("""H_FIELD = 0.0    # your number from Canvas, for example 0.50
T = 0.0          # for example 2.00
STEPS = 0        # for example 8"""),
code(hidden_check(['l3_m4b_check'], 'check_l3_module4b', '''

passed, messages, value = check_l3_module4b(ising_hamiltonian, trotter_step, trotter_step2, H_FIELD, T, STEPS)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

md("""## What you should notice

- A Hamiltonian made of non-commuting parts can be simulated by alternating short evolutions of each part: **Trotter steps** made of RZZ and RX gates.
- The first-order error falls in proportion to the step size; a symmetric **second-order** step falls much faster for the same number of two-qubit gates.
- On noisy hardware, more steps are not always better: the best number of steps balances **Trotter error** against **gate noise**.
- Small chains can be checked exactly; the reason to use a quantum computer is the large chains that cannot.

**Next in Canvas:** the Module 4 Track B quiz and the time log."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m4bc{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
