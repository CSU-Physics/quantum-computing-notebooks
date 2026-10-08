"""Build the Level 3 Module 1 browser notebook (noise and open quantum systems: density matrices, channels,
T1 and T2, device data and one-qubit tomography). The check code is checks/l3_m1_check.py, published next to
the notebook by l3_hidden. Device data: content/level3/l3_m1_device_data.json (tools/make_l3_m1_data.py)."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level3"
from l3_hidden import hidden_check  # noqa: E402
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L3-M1-lab-noise-density-matrices.ipynb"
VERSION = "2026-10-08"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 1 lab: noise and open quantum systems

**Quantum Computing Advanced · Module 1 · about 95 minutes** · notebook version {VERSION}

A real qubit is never alone: it exchanges energy with its surroundings and loses its phase. A state vector cannot describe the result, which is a **mixture**. In this lab you describe noise the way the rest of this course does, with **density matrices** and **noise channels**:

1. write `density_matrix(psi)` and see why a mixture is not a superposition;
2. write `bloch_vector(rho)` and place pure and mixed states in the **Bloch ball**;
3. write `apply_channel(rho, kraus)`, the general rule for noise, and try amplitude damping and dephasing;
4. write `relax(rho, t, T1, T2)`: what T1 and T2 do to a qubit over time, checked against the simulator's own `thermal_relaxation_error`;
5. measure T1 in a simulated experiment, and read T1 and T2 from the data of a real IBM computer;
6. reconstruct a qubit's state from measurements (**state tomography**);
7. get your verification value.

The **Module 1 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order. You write your functions with NumPy only: `np.outer`, `np.trace`, `@` for matrix products and `.conj().T` for the conjugate transpose (the dagger)."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. It also loads the device data used in Step 7."""),
code("""import json
import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import (QuantumCircuit, Statevector, DensityMatrix, partial_trace, AerSimulator, NoiseModel,
                  thermal_relaxation_error, amplitude_damping_error, ReadoutError)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

device = json.load(open("l3_m1_device_data.json"))
print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)
print("Device data:", device["backend"], "with", device["num_qubits"], "qubits, calibration of", device["calibration_date"])"""),

md("""## Step 1: the density matrix of a pure state

For a state vector |ψ⟩, the **density matrix** is the outer product ρ = |ψ⟩⟨ψ|: the column ψ times the row ψ†, its conjugate transpose. Its diagonal holds the measurement probabilities; its off-diagonal entries, the **coherences**, hold the phases between the basis states.

**Your task:** write `density_matrix(psi)` for a state vector on any number of qubits. One line does it: `np.outer(psi, psi.conj())`, or a column vector times its dagger. The cell after it compares your matrix with qsim's `DensityMatrix`."""),
code("""def density_matrix(psi):
    \"\"\"The density matrix |psi><psi| of a state vector psi (a NumPy array).\"\"\"
    psi = np.asarray(psi, dtype=complex)
    # YOUR CODE HERE
    raise NotImplementedError("Complete density_matrix() first.")"""),
code("""plus = np.array([1, 1]) / np.sqrt(2)
plus_i = np.array([1, 1j]) / np.sqrt(2)
rho_plus = density_matrix(plus)
print("rho for |+>:\\n", np.round(rho_plus, 3))
print("rho for |+i>:\\n", np.round(density_matrix(plus_i), 3))
print("trace:", np.trace(rho_plus).real, "  purity Tr(rho^2):", round(np.trace(rho_plus @ rho_plus).real, 6))
print("same as qsim's DensityMatrix:", np.allclose(rho_plus, DensityMatrix(Statevector(plus)).data),
      np.allclose(density_matrix(plus_i), DensityMatrix(Statevector(plus_i)).data))"""),
md("""**What to notice.** Every density matrix is Hermitian (equal to its own dagger), has trace 1, and has no negative eigenvalues. For a pure state the **purity** Tr(ρ²) is 1. The coherences of |+⟩ are +1/2; those of |+i⟩ are ∓i/2: the phase between |0⟩ and |1⟩ lives there."""),

md("""## Step 2: a mixture is not a superposition

Suppose a qubit is |0⟩ half of the time and |1⟩ the other half, and you do not know which. Its density matrix is the **mixture** ρ = ½|0⟩⟨0| + ½|1⟩⟨1|, an average of density matrices. The cell compares it with |+⟩, which also gives 0 and 1 half of the time. An expectation value is ⟨P⟩ = Tr(ρP).

*Predict first:* which measurement tells the two apart?"""),
code("""zero, one = np.array([1, 0]), np.array([0, 1])
minus = np.array([1, -1]) / np.sqrt(2)
rho_mix = 0.5 * density_matrix(zero) + 0.5 * density_matrix(one)

for name, rho in [("|+>", rho_plus), ("50/50 mixture of |0> and |1>", rho_mix)]:
    ez, ex = np.trace(rho @ Z).real, np.trace(rho @ X).real
    print(f"{name:30s} <Z> = {ez:+.2f}   <X> = {ex:+.2f}   purity = {np.trace(rho @ rho).real:.2f}")

other = 0.5 * density_matrix(plus) + 0.5 * density_matrix(minus)
print("\\nA 50/50 mixture of |+> and |-> has the same density matrix:", np.allclose(other, rho_mix))

bell = Statevector(np.array([1, 0, 0, 1]) / np.sqrt(2))
half = partial_trace(bell, [1]).data          # keep qubit 0, trace out qubit 1
print("Half of a Bell pair:\\n", np.round(half, 3))"""),
md("""**What to notice.** Both give ⟨Z⟩ = 0, but |+⟩ has ⟨X⟩ = 1 and the mixture ⟨X⟩ = 0: only a measurement in the X basis tells them apart. The mixture is the **completely mixed state** I/2, with purity 1/2, the smallest possible for one qubit. Two different recipes (|0⟩ or |1⟩; |+⟩ or |−⟩) give the same density matrix, so no measurement can tell which recipe was used. Half of a Bell pair is also I/2: entanglement makes each qubit, on its own, completely mixed."""),

md("""## Step 3: the Bloch ball

Every one-qubit density matrix can be written ρ = (I + xX + yY + zZ)/2, where the **Bloch vector** (x, y, z) = (Tr(ρX), Tr(ρY), Tr(ρZ)). Pure states lie on the surface of the Bloch sphere (length 1); mixed states lie inside it, in the **Bloch ball**, and the purity is (1 + |r|²)/2.

**Your task:** write `bloch_vector(rho)`, returning a NumPy array of the three real numbers (x, y, z). Use `np.trace(rho @ X).real` and so on."""),
code("""def bloch_vector(rho):
    \"\"\"The Bloch vector (Tr(rho X), Tr(rho Y), Tr(rho Z)) of a one-qubit density matrix, as 3 real numbers.\"\"\"
    rho = np.asarray(rho, dtype=complex)
    # YOUR CODE HERE
    raise NotImplementedError("Complete bloch_vector() first.")"""),
code("""rho_p = 0.75 * density_matrix(zero) + 0.25 * density_matrix(plus)     # |0> three times in four, |+> once
states = {"|0>": density_matrix(zero), "|+>": rho_plus, "|+i>": density_matrix(plus_i),
          "mixture of |0> and |1>": rho_mix, "rho_p (3/4 |0>, 1/4 |+>)": rho_p, "half of a Bell pair": half}
for name, rho in states.items():
    r = np.asarray(bloch_vector(rho), dtype=float)
    purity = np.trace(rho @ rho).real
    print(f"{name:26s} r = {np.round(r, 3)}   |r| = {np.linalg.norm(r):.3f}   purity = {purity:.4f}   (1 + |r|^2)/2 = {(1 + r @ r) / 2:.4f}")"""),
md("""**What to notice.** The three pure states have |r| = 1. The completely mixed state is at the centre. rho_p is a mixture of two pure states that are not orthogonal, so it sits inside the ball between them: its Bloch vector is the same mixture of their Bloch vectors, (0.75)(0, 0, 1) + (0.25)(1, 0, 0). The lab check asks for its length."""),

md("""## Step 4: noise channels

A **channel** is the most general thing that can happen to a state, noise included. Every channel can be written with **Kraus matrices** K₁, K₂, ...: ρ → Σₖ Kₖ ρ Kₖ†, where Σₖ Kₖ†Kₖ = I so that the trace stays 1. A unitary gate U is the channel with one Kraus matrix, U.

Two channels describe most single-qubit noise. They are prepared below, with the same matrices as Qiskit Aer and qsim:

- **amplitude damping** with probability γ: |1⟩ loses its energy and falls to |0⟩ with probability γ (energy relaxation);
- **phase damping** with parameter λ: the coherences shrink by √(1 − λ), with no change to the populations (dephasing).

**Your task:** write `apply_channel(rho, kraus)`, the sum of K @ rho @ K.conj().T over the Kraus matrices. It must work for any number of qubits."""),
code("""def amplitude_damping_kraus(gamma):
    \"\"\"Prepared: energy relaxation with probability gamma.\"\"\"
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]


def phase_damping_kraus(lam):
    \"\"\"Prepared: dephasing; the coherences shrink by sqrt(1 - lam).\"\"\"
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]


def apply_channel(rho, kraus):
    \"\"\"The channel rho -> sum over K in kraus of K rho K^dagger.\"\"\"
    rho = np.asarray(rho, dtype=complex)
    # YOUR CODE HERE
    raise NotImplementedError("Complete apply_channel() first.")"""),
code("""ad = amplitude_damping_kraus(0.3)
print("Sum of K^dagger K for amplitude damping:\\n", np.round(sum(K.conj().T @ K for K in ad), 6))
print("same Kraus matrices as qsim:", all(np.allclose(a, b) for a, b in zip(ad, amplitude_damping_error(0.3).kraus)))

after = apply_channel(density_matrix(one), ad)
print("\\n|1> after amplitude damping with gamma = 0.3:\\n", np.round(after, 3), "  Bloch vector", np.round(bloch_vector(after), 3))
after = apply_channel(rho_plus, ad)
print("|+> after amplitude damping with gamma = 0.3: Bloch vector", np.round(bloch_vector(after), 3))
after = apply_channel(rho_plus, phase_damping_kraus(0.36))
print("|+> after phase damping with lam = 0.36:      Bloch vector", np.round(bloch_vector(after), 3))"""),
md("""**What to notice.** Amplitude damping moves |1⟩ toward |0⟩: 30% of the population falls down. It also shrinks the coherences of |+⟩, by √(1 − γ) ≈ 0.837, and pulls the Bloch vector up toward the north pole. Phase damping shrinks the x and y components only (by √(1 − 0.36) = 0.8) and leaves z alone: the qubit keeps its energy but loses its phase."""),

md("""## Step 5: T1 and T2

On a real qubit, both processes run all the time. Two numbers describe them:

- **T1**, the energy relaxation time: the population of |1⟩ decays as e^(−t/T1);
- **T2**, the coherence time: the coherences (the x and y of the Bloch vector) decay as e^(−t/T2).

Amplitude damping over a time t, with γ = 1 − e^(−t/T1), already shrinks the coherences by √(1 − γ) = e^(−t/(2T1)). Phase damping must supply the rest, so λ = 1 − e^(−2t/T2 + t/T1). This only works when **T2 ≤ 2 T1**: energy loss alone already limits T2 to 2 T1.

**Your task:** write `relax(rho, t, T1, T2)`: amplitude damping with γ = 1 − e^(−t/T1), then phase damping with λ = 1 − e^(−2t/T2 + t/T1), using your `apply_channel`. The next cell compares it with qsim's `thermal_relaxation_error`, which Qiskit Aer also provides."""),
code("""def relax(rho, t, T1, T2):
    \"\"\"One qubit after a time t of relaxation with times T1 and T2 (same units, T2 <= 2 T1).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete relax() first.")"""),
code("""T1, T2 = 100.0, 80.0                  # microseconds
for t in (10.0, 50.0, 150.0):
    lib = apply_channel(rho_plus, thermal_relaxation_error(T1, T2, t).kraus)
    print(f"t = {t:5.0f} us: relax() matches thermal_relaxation_error: {np.allclose(relax(rho_plus, t, T1, T2), lib)}")

times = np.linspace(0, 300, 61)
p1 = [np.real(relax(density_matrix(one), t, T1, T2)[1, 1]) for t in times]       # starts in |1>
ex = [bloch_vector(relax(rho_plus, t, T1, T2))[0] for t in times]                 # starts in |+>
ez = [bloch_vector(relax(rho_plus, t, T1, T2))[2] for t in times]

t = 50.0
print(f"\\nat t = {t:.0f} us: P(1) for |1> = {np.real(relax(density_matrix(one), t, T1, T2)[1, 1]):.4f}"
      f"   <X> for |+> = {bloch_vector(relax(rho_plus, t, T1, T2))[0]:.4f}")

plt.figure(figsize=(7, 3.5))
plt.plot(times, p1, label="P(1), starting in |1>  (T1 decay)")
plt.plot(times, ex, label="<X>, starting in |+>  (T2 decay)")
plt.plot(times, ez, ":", label="<Z>, starting in |+>")
plt.xlabel("time (microseconds)"); plt.ylabel("value"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
md("""**What to notice.** P(1) falls as e^(−t/T1) and ⟨X⟩ as e^(−t/T2), here faster, because T2 = 80 µs is shorter than T1 = 100 µs. A qubit that starts in |+⟩ ends at |0⟩ (z = 1): relaxation does not just scramble a state, it pulls every state toward the ground state. The lab check asks for ⟨X⟩ at t = 50 µs."""),

md("""## Step 6: measure T1 in a simulated experiment

On hardware, T1 is measured like this: flip the qubit to |1⟩, wait a time t, measure, and repeat for many waiting times. The cell does it on the simulator. Each `id` gate stands for a 10 µs wait, and the noise model attaches `thermal_relaxation_error(T1, T2, 10)` to it, the way Qiskit Aer does. Each point is 1,000 shots, so each has shot noise.

The prepared fit uses ln P(1) = −t/T1: a straight line through the logarithms, whose slope gives T1."""),
code("""DT = 10.0                                            # microseconds per id gate
nm = NoiseModel()
nm.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, DT), ["id"])
sim = AerSimulator(noise_model=nm)

waits = np.arange(0, 31, 2)                          # 0, 2, ..., 30 id gates: 0 to 300 us
p1_measured = []
for k in waits:
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    for _ in range(int(k)):
        qc.id(0)
    qc.measure(0, 0)
    counts = sim.run(qc, shots=1000, seed_simulator=int(100 + k)).result().get_counts()
    p1_measured.append(counts.get("1", 0) / 1000)
t_wait = waits * DT
p1_measured = np.array(p1_measured)

use = p1_measured > 0.05                             # the logarithm of a few counts is too noisy
slope, intercept = np.polyfit(t_wait[use], np.log(p1_measured[use]), 1)
T1_fit = -1 / slope
print(f"fitted T1 = {T1_fit:.1f} us   (the simulator was given T1 = {T1:.0f} us)")

plt.figure(figsize=(7, 3.3))
plt.plot(t_wait, p1_measured, "o", label="measured P(1), 1,000 shots each")
plt.plot(times, np.exp(-times / T1), label="exp(-t / T1)")
plt.plot(times, np.exp(intercept + slope * times), "--", label=f"fit: T1 = {T1_fit:.1f} us")
plt.xlabel("wait (microseconds)"); plt.ylabel("P(1)"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
md("""**What to notice.** The fitted T1 is close to 100 µs but not exactly 100: shot noise moves every point a little, and the points at long waits, where only a few shots still give 1, are the noisiest. That is why the fit leaves out points below P(1) = 0.05. A real T1 measurement also has readout errors, which this simulation leaves out (you correct them in Module 3)."""),

md("""## Step 7: T1 and T2 of a real IBM computer

The file loaded in Step 0 holds the calibration data of **FakePittsburgh**, a model of the 156-qubit IBM computer ibm_pittsburgh that comes with Qiskit: for every qubit, its T1, its T2 and its readout error, from one calibration on the date printed in Step 0. The cell summarizes them."""),
code("""t1 = np.array([q["t1_us"] for q in device["qubits"]])
t2 = np.array([q["t2_us"] for q in device["qubits"]])
print(f"median T1 = {np.median(t1):.1f} us   (from {t1.min():.1f} to {t1.max():.1f})")
print(f"median T2 = {np.median(t2):.1f} us   (from {t2.min():.1f} to {t2.max():.1f})")
print(f"one sx gate takes {device['sx_duration_ns']} ns, so about {np.median(t1) * 1000 / device['sx_duration_ns']:,.0f} of them fit in the median T1")
odd = [q for q in device["qubits"] if q["t2_us"] > 2 * q["t1_us"]]
print("qubits with T2 > 2 T1:", odd)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].hist(t1, bins=20, alpha=0.7, label="T1"); ax[0].hist(t2, bins=20, alpha=0.6, label="T2")
ax[0].set_xlabel("microseconds"); ax[0].set_ylabel("qubits"); ax[0].legend()
ax[1].plot(t1, t2, ".", alpha=0.7); ax[1].plot([0, 500], [0, 1000], "k--", lw=1, label="T2 = 2 T1")
ax[1].set_xlabel("T1 (us)"); ax[1].set_ylabel("T2 (us)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()"""),
md("""**What to notice.** The qubits of one chip differ a lot: most have T1 between about 150 and 450 µs, and a few are much worse. Every qubit but one lies below the line T2 = 2 T1. That one, qubit 36, has a very short T1 and a T2 slightly above 2 T1, which no single relaxation channel allows: the two numbers come from separate measurements with their own uncertainties, made at different moments. The lab check asks what the simulator does with such values. The median T1 is also on the lab check."""),

md("""## Step 8: one-qubit state tomography

How do you find the state of a qubit you did not prepare yourself? Measure many copies in three bases. Each basis gives one component of the Bloch vector, r = (N₀ − N₁)/N, and then ρ = (I + xX + yY + zZ)/2. This is **state tomography**. To measure in the X basis, apply H first; for Y, apply S† and then H.

The prepared function below does it on the simulator. It reconstructs a state made by `ry(1.1)` then `rz(0.7)`, first with 1,000 shots per basis, then with 10,000 shots per basis and a 3% readout error on each bit. The fidelity of a pure state |ψ⟩ with ρ is ⟨ψ|ρ|ψ⟩."""),
code("""def tomography(prep, shots, seed, noise_model=None):
    \"\"\"Prepared: estimate the Bloch vector of the one-qubit state made by the circuit prep.\"\"\"
    sim = AerSimulator(noise_model=noise_model)
    r = []
    for k, basis in enumerate("XYZ"):
        qc = QuantumCircuit(1, 1)
        qc.compose(prep, inplace=True)
        if basis == "X":
            qc.h(0)
        elif basis == "Y":
            qc.sdg(0); qc.h(0)
        qc.measure(0, 0)
        counts = sim.run(qc, shots=shots, seed_simulator=seed + k).result().get_counts()
        r.append((counts.get("0", 0) - counts.get("1", 0)) / shots)
    return np.array(r)


prep = QuantumCircuit(1)
prep.ry(1.1, 0); prep.rz(0.7, 0)
psi = Statevector(prep).data
r_true = bloch_vector(density_matrix(psi))

r_est = tomography(prep, 1000, seed=11)
rho_est = (I2 + r_est[0] * X + r_est[1] * Y + r_est[2] * Z) / 2
print("true Bloch vector:          ", np.round(r_true, 3))
print("estimated, 1,000 shots each:", np.round(r_est, 3), f"  length |r| = {np.linalg.norm(r_est):.3f}")
print(f"<psi|rho_est|psi> = {np.vdot(psi, rho_est @ psi).real:.4f}   eigenvalues of rho_est: {np.round(np.linalg.eigvalsh(rho_est), 4)}")

ro = NoiseModel()
ro.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.03, 0.97]]))
r_ro = tomography(prep, 10000, seed=21, noise_model=ro)
print("estimated, 10,000 shots each, 3% readout error:", np.round(r_ro, 3), f"  length |r| = {np.linalg.norm(r_ro):.3f}")"""),
md("""**What to notice.** With 1,000 shots each component is off by a few hundredths (shot noise, about 1/√1000 ≈ 0.03). Here the errors happen to push the estimate slightly outside the ball, |r| ≈ 1.01, which no state can have: ρ_est has a small negative eigenvalue, and ⟨ψ|ρ_est|ψ⟩ comes out just above 1. Careful tomography then finds the closest physical state (for example by maximum likelihood). More shots shrink that random error, but not a **systematic** one: with a 3% readout error, every component shrinks by the factor 1 − 2 × 0.03 = 0.94, so the reconstructed state looks mixed (|r| ≈ 0.94) although the qubit was pure. Module 3 corrects readout errors."""),

md("""## Step 9: your personal check

Open the **Module 1 lab check** in Canvas. Question 1 shows your own THETA (0.30 to 2.80), T1 and T2 (whole microseconds) and TIME (20 to 200 µs). Type them below and run the next two cells. The check cell tests `density_matrix()`, `bloch_vector()`, `apply_channel()` and `relax()`. Only if every test passes does it print your **verification value**: 1,000 times the fidelity ⟨ψ|ρ|ψ⟩ between the state |ψ⟩ = RY(THETA)|0⟩ and the same state after a time TIME of relaxation with your T1 and T2, rounded to a whole number."""),
code("""THETA = 0.0    # your number from Canvas, for example 1.25
T1 = 0         # microseconds, for example 200
T2 = 0         # microseconds, for example 150
TIME = 0       # microseconds, for example 60"""),
code(hidden_check(['l3_m1_check'], 'check_l3_module1', '''

passed, messages, value = check_l3_module1(density_matrix, bloch_vector, apply_channel, relax, THETA, T1, T2, TIME)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

md("""## What you should notice

- A density matrix describes every state a qubit can be in, mixtures included. Pure states have purity 1 and lie on the Bloch sphere; mixtures lie inside the Bloch ball.
- A mixture and a superposition can give the same probabilities in one basis and differ in another; and different mixtures can share one density matrix, which is all that any measurement can see.
- Noise is a channel: ρ → Σ K ρ K†. Amplitude damping (T1) loses energy toward |0⟩; dephasing destroys coherence. Together they are T1 and T2 relaxation, with T2 ≤ 2 T1.
- T1 and T2 are measured, with shot noise, and they differ from qubit to qubit on the same chip; hundreds of microseconds on today's IBM devices, thousands of gate times.
- Tomography rebuilds a state from three bases. Shot noise makes random errors that more shots shrink; readout errors make a systematic one that more shots cannot remove.

**Next in Canvas:** the lab check, the quiz and the time log."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m1c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
shutil.copyfile(ROOT / "content" / "level2" / "qsim.py", OUT / "qsim.py")
print("wrote", OUT / NAME, "and synced qsim.py")
