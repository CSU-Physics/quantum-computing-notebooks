"""Build the Level 1 Module 5 browser notebook (JupyterLite, Pyodide kernel) from checks/m5_check.py."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level1"
CHECK = (ROOT / "checks" / "m5_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L1-M5-lab-noise-and-hardware.ipynb"
COLAB = ("https://colab.research.google.com/github/CSU-Physics/quantum-computing-notebooks/blob/main/"
         "colab/level1/QC-L1-M5-real-hardware-colab.ipynb")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 5 lab: simulators, noise and real hardware

**Quantum Computing Foundations · Module 5 · about 75 minutes**

An ideal simulator gives the Bell circuit's results exactly as the theory says: only 00 and 11. A real quantum computer also gives a few 01 and 10. In this lab you find out where those come from. You write three functions:

1. `shot_noise(p, shots)`: how much a measured fraction varies from run to run, just from having a finite number of shots;
2. `two_qubit_readout(A0, A1)`: the table of readout errors for two qubits, built from each qubit's own table;
3. `mitigate(counts, A)`: a correction that removes the effect of readout errors from measured counts.

You also look at a real result from an IBM quantum computer, build a noise model from that computer's calibration data, and compare the two. The **Module 5 lab check** asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. qsim 1.3.0 adds noise models with the same names as Qiskit Aer's (`NoiseModel`, `depolarizing_error`, `ReadoutError`, `thermal_relaxation_error`)."""),
code("""import numpy as np
import matplotlib.pyplot as plt   # used for the plots below
import qsim
from qsim import (QuantumCircuit, Operator, AerSimulator, transpile, plot_histogram,
                  simulate_density_matrix, NoiseModel, ReadoutError, depolarizing_error,
                  thermal_relaxation_error)

LABELS = ["00", "01", "10", "11"]


def bell():
    \"\"\"The Module 4 Bell circuit, measured: H on qubit 0, CNOT 0 -> 1.\"\"\"
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def different(counts):
    \"\"\"How many shots gave different results on the two qubits (01 or 10).\"\"\"
    return counts.get("01", 0) + counts.get("10", 0)


print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),
md("""## Step 1: shot noise

Even an ideal Bell circuit does not give exactly 500 zeros in 1,000 shots: each shot is a random draw. If a result has probability p and you take N shots, the fraction you measure varies from run to run with a **standard deviation**

σ = √( p (1 − p) / N ).

This course uses the **3-sigma rule**: if a measured fraction is more than 3σ away from the prediction, we treat the difference as real rather than bad luck, because under the model a difference that large happens by chance only about 3 times in 1,000. It is strong evidence against the prediction, under the model's assumptions, but it is not proof of any particular cause, such as a hardware fault.

Complete `shot_noise(p, shots)`, then run the cell. It runs the ideal Bell circuit 40 times with different seeds and checks how many runs fall inside the 3σ band around 0.5. Canvas asks for the half-width of the band, 3σ, for p = 0.5 and 1,000 shots."""),
code("""def shot_noise(p, shots):
    # YOUR CODE: replace the next line. Return the standard deviation of a measured fraction.
    raise NotImplementedError("Complete shot_noise() first.")


ideal = AerSimulator()
fractions = []
for seed in range(1, 41):
    counts = ideal.run(bell(), shots=1000, seed_simulator=seed).result().get_counts()
    fractions.append(counts.get("00", 0) / 1000)

band = 3 * shot_noise(0.5, 1000)
inside = sum(abs(f - 0.5) <= band for f in fractions)
print(f"3-sigma band for p = 0.5 and 1,000 shots: 0.5 ± {band:.3f}")
print(f"{inside} of 40 runs gave a fraction of 00 inside the band")
for shots in (100, 1000, 10000):
    print(f"  {shots:>6} shots: 3 sigma = {3 * shot_noise(0.5, shots):.4f}")

fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(range(1, 41), fractions, "o", color="#99004C")
ax.axhline(0.5, color="#24313D", lw=1)
ax.axhspan(0.5 - band, 0.5 + band, color="#E4DCE0")
ax.set_xlabel("run (seed)")
ax.set_ylabel("fraction of 00")
ax.set_title("40 ideal runs of 1,000 shots, with the 3-sigma band")
plt.show()"""),
md("""Four times as many shots halve σ, because N is under a square root. So shot noise never disappears, but it can be made as small as you like by running longer. Noise in the **hardware** is different: more shots measure it more precisely, but do not make it go away."""),
md("""## Step 2: what the hardware really runs

A quantum computer has only a few **native gates**. IBM's Heron processors use `rz`, `sx` (√X), `x` and the two-qubit `cz`, and each qubit is wired to only a few neighbours. Before a circuit runs, Qiskit's **transpiler** rewrites it in native gates on qubits that are connected.

The cell below is the Bell circuit as Qiskit's transpiler wrote it for the IBM computer `ibm_pittsburgh` (optimization level 1). It checks with `Operator` (Module 3) that the two circuits are the same gate, and counts the gates."""),
code("""from math import pi

hw = QuantumCircuit(2, 2)
for q in (0, 1):
    hw.rz(pi / 2, q)
    hw.sx(q)
    hw.rz(pi / 2, q)
hw.cz(0, 1)
hw.rz(pi / 2, 1)
hw.sx(1)
hw.rz(pi / 2, 1)
hw.measure([0, 1], [0, 1])
print(hw.draw())

gates = hw.copy()
gates.remove_final_measurements()
b = bell()
b.remove_final_measurements()
print("same matrix exactly (==):       ", Operator(gates) == Operator(b))
print("same gate up to a global phase: ", Operator(gates).equiv(Operator(b)))
print("gates in the Bell circuit:   ", dict(b.count_ops()))
print("gates the hardware runs:     ", dict(gates.count_ops()))"""),
md("""The `rz` gates are free on IBM hardware: they are done in software by changing the phase of the next pulses, so they add no error. The three `sx` gates and the `cz` are real pulses, and each one adds a little error. A longer circuit has more chances to go wrong. That is why the transpiler tries to use as few two-qubit gates as possible."""),
md("""## Step 3: a real result from an IBM quantum computer

IBM's free course *Use a quantum computer today* runs this Bell circuit on the IBM computer `ibm_pittsburgh` (a 156-qubit Heron processor) with 1,000 shots, and publishes the result. The counts are in the cell below.

Run it. It compares the result with the ideal simulator and uses your `shot_noise()` to ask two questions:

1. Are the 01 and 10 results more than shot noise could explain?
2. Is the difference between the 507 zeros and the 463 ones more than shot noise could explain?"""),
code("""ibm = {"00": 507, "01": 15, "10": 15, "11": 463}    # ibm_pittsburgh, 1,000 shots (IBM Quantum Learning)
shots = sum(ibm.values())

p_diff = different(ibm) / shots
print(f"different results: {different(ibm)} of {shots} = {p_diff:.3f}")
print("an ideal Bell state never gives them (p = 0, so sigma = 0): every one comes from noise")

same = ibm["00"] + ibm["11"]
p00 = ibm["00"] / same
band = 3 * shot_noise(0.5, same)
print(f"\\nof the {same} equal results, 00 was {p00:.3f} of them")
print(f"ideal: 0.5 ± {band:.3f} (3 sigma), so the imbalance is",
      "MORE than shot noise" if abs(p00 - 0.5) > band else "within shot noise")

ideal_counts = AerSimulator(seed_simulator=7).run(bell(), shots=1000).result().get_counts()
plot_histogram([ideal_counts, ibm], title="ideal simulator (red) and ibm_pittsburgh (grey)")"""),
md("""So the 30 different results are certainly noise, but the counts alone cannot say which kind of noise. The 507 against 463 imbalance is within shot noise, so by itself it is not evidence of anything. The next steps build the two main kinds of noise one at a time.

The counts were published by IBM in the lesson "Build and run your first quantum program" (IBM Quantum Learning, CC BY-SA 4.0)."""),
md("""## Step 4: readout errors

Sometimes the qubit is in |1⟩ but the measurement records 0, or the other way round. For one qubit, the **readout matrix** A lists the probabilities, with the true state in the columns and the recorded result in the rows:

A = [[P(read 0 | was 0), P(read 0 | was 1)], [P(read 1 | was 0), P(read 1 | was 1)]] = [[1 − e0, e1], [e0, 1 − e1]],

where e0 = P(read 1 | was 0) and e1 = P(read 0 | was 1). The prepared function `single_qubit_readout(e0, e1)` builds it. Then the recorded probabilities are A times the true ones.

For two qubits, each qubit has its own A, and the two-qubit matrix is their **tensor product** in Qiskit's order, just like a two-qubit state in Module 4. Complete `two_qubit_readout(A0, A1)` (A0 for qubit 0, A1 for qubit 1).

The numbers below are the readout errors of qubits 0 and 1 of `ibm_pittsburgh` in a calibration snapshot from 17 April 2026 (from Qiskit's `FakePittsburgh`). Notice that e1 is larger than e0: a qubit in |1⟩ can relax to |0⟩ during the measurement."""),
code("""def single_qubit_readout(e0, e1):
    \"\"\"Prepared for you: columns are the true state (0, 1), rows the recorded result (0, 1).\"\"\"
    return np.array([[1 - e0, e1],
                     [e0, 1 - e1]])


def two_qubit_readout(A0, A1):
    # YOUR CODE: replace the next line. Return the 4 x 4 matrix for the order 00, 01, 10, 11.
    raise NotImplementedError("Complete two_qubit_readout() first.")


READ = {0: (0.0034, 0.0146), 1: (0.0020, 0.0083)}   # (e0, e1) for qubits 0 and 1 of ibm_pittsburgh
A0 = single_qubit_readout(*READ[0])
A1 = single_qubit_readout(*READ[1])
A = two_qubit_readout(A0, A1)
print("A (rows: recorded 00, 01, 10, 11; columns: true 00, 01, 10, 11) =")
print(np.array2string(np.asarray(A), precision=5, suppress_small=True))

p_true = np.array([0.5, 0, 0, 0.5])               # an ideal Bell state
p_read = A @ p_true
print("\\nideal Bell state, recorded with these readout errors:", {k: round(float(v), 4) for k, v in zip(LABELS, p_read)})
print(f"different results from readout errors alone: {p_read[1] + p_read[2]:.4f}")

# The same with a qsim noise model that has readout errors only.
ro_model = NoiseModel()
for q in (0, 1):
    e0, e1 = READ[q]
    ro_model.add_readout_error(ReadoutError([[1 - e0, e0], [e1, 1 - e1]]), [q])
c = AerSimulator(seed_simulator=7, noise_model=ro_model).run(bell(), shots=100000).result().get_counts()
print("qsim with readout errors, 100,000 shots:", {k: round(v / 100000, 4) for k, v in c.items()})"""),
md("""Readout errors alone give about 1.4% different results. IBM's real run had 3.0%, so readout is only part of the story. Note that Qiskit's `ReadoutError` writes the same numbers by rows: `[[P(0|0), P(1|0)], [P(0|1), P(1|1)]]`, which is the transpose of A."""),
md("""## Step 5: gate errors

Every gate is a little imperfect. The simplest model is the **depolarizing error**: with probability λ the gate scrambles the qubits completely (into a random state), and otherwise it works. For the Bell circuit, a depolarizing error of λ on the CNOT gives different results with probability λ/2, because a scrambled pair of qubits gives each of the four results a quarter of the time.

Run the cell. It adds a depolarizing error to the CNOT in qsim, from λ = 0 to 0.3, and compares the fraction of different results with λ/2."""),
code("""lams = np.arange(0, 0.31, 0.05)
measured = []
for lam in lams:
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["cx"])
    c = AerSimulator(seed_simulator=7, noise_model=model).run(bell(), shots=4000).result().get_counts()
    measured.append(different(c) / 4000)
    print(f"lambda = {lam:.2f}: different results {different(c) / 4000:.4f}   (predicted {lam / 2:.4f})")

fig, ax = plt.subplots(figsize=(5.5, 3))
ax.plot(lams, lams / 2, color="#24313D", label="predicted, lambda / 2")
ax.plot(lams, measured, "o", color="#99004C", label="qsim, 4,000 shots")
ax.set_xlabel("depolarizing error on the CNOT, lambda")
ax.set_ylabel("fraction of 01 and 10")
ax.legend()
plt.show()"""),
md("""## Step 6: a model of the real device

IBM publishes calibration data for every device: each gate's error rate, each qubit's readout errors, and its **T1** (how long |1⟩ lasts before it relaxes to |0⟩) and **T2** (how long a superposition keeps its phase). Qiskit's `FakePittsburgh` keeps a snapshot of `ibm_pittsburgh`'s data from 17 April 2026. The cell below turns the numbers for qubits 0 and 1 into a qsim noise model, the same way Qiskit Aer does it: a depolarizing error for each gate's error rate, relaxation during each gate (T1 and T2), and the readout errors from Step 4.

Run it. It runs the hardware circuit from Step 2 with this model, 1,000 shots with seed 7, and compares the result with IBM's. Canvas asks how many shots gave different results here."""),
code("""T1 = {0: 66.8e-6, 1: 310.7e-6}        # seconds
T2 = {0: 112.4e-6, 1: 219.3e-6}
SX_ERROR = {0: 7.31e-4, 1: 2.09e-4}   # error per sx (or x) gate, 32 ns long
CZ_ERROR = 5.39e-3                    # error per cz gate, 88 ns long

device = NoiseModel(basis_gates=["rz", "sx", "x", "cz"])
for q in (0, 1):
    one = depolarizing_error(2 * SX_ERROR[q], 1).compose(thermal_relaxation_error(T1[q], T2[q], 32e-9))
    device.add_quantum_error(one, ["sx", "x"], [q])
    e0, e1 = READ[q]
    device.add_readout_error(ReadoutError([[1 - e0, e0], [e1, 1 - e1]]), [q])
relax = thermal_relaxation_error(T1[1], T2[1], 88e-9).tensor(thermal_relaxation_error(T1[0], T2[0], 88e-9))
device.add_quantum_error(depolarizing_error(4 / 3 * CZ_ERROR, 2).compose(relax), ["cz"], [0, 1])
print(device)

model_counts = AerSimulator(seed_simulator=7, noise_model=device).run(hw, shots=1000).result().get_counts()
print("\\nmodel, 1,000 shots:", model_counts, "  different:", different(model_counts))
print("ibm_pittsburgh:     ", ibm, "  different:", different(ibm))

# The model's exact prediction: gate noise from the density matrix, then the readout matrix A.
p_gates = simulate_density_matrix(gates, noise_model=device).probabilities()
p_model = A @ p_gates
print(f"\\nmodel prediction: different results with probability {p_model[1] + p_model[2]:.4f}"
      f" (gates alone {p_gates[1] + p_gates[2]:.4f}, readout adds the rest)")
plot_histogram([model_counts, ibm], title="noise model from calibration data (red) and ibm_pittsburgh (grey)")"""),
md("""The model predicts about 1.9% different results (Qiskit Aer's own model of the same snapshot gives 1.8%), and about three quarters of that comes from readout. IBM's run had 3.0%. Is the model wrong? Use the 3-sigma rule: at p = 0.019 and 1,000 shots, σ = √(0.019 × 0.981 / 1000) ≈ 0.0043, so 3σ ≈ 0.013. The gap, 0.030 − 0.019 = 0.011, is inside it. With only 1,000 shots, these counts **cannot tell** whether the device was noisier on the day of the run than its April snapshot says. To find out you would need more shots, and the calibration from the same day and the same physical qubits, which IBM's lesson does not give."""),
md("""## Step 7: readout mitigation

If you know the readout matrix A, you can undo its effect. The recorded probabilities are p_read = A p_true, so the best estimate of the true ones solves

A x = p_read,  that is,  x = `np.linalg.solve(A, p_read)`.

This is **readout error mitigation**. (Qiskit Runtime's TREX method corrects readout errors too, with a refined procedure that does not need the full matrix.) The corrected numbers are **quasi-probabilities**: they add up to 1, but one can come out slightly negative when a count is small. Keep them as they are; do not clip them.

Complete `mitigate(counts, A)`: turn the counts into probabilities in the order 00, 01, 10, 11, solve, and return a dictionary such as `{"00": 0.49, "01": 0.004, "10": 0.006, "11": 0.5}`. Then run the cell. Canvas asks for the mitigated fraction of different results for IBM's run."""),
code("""def mitigate(counts, A):
    # YOUR CODE: replace the next line. Return {"00": ..., "01": ..., "10": ..., "11": ...}.
    raise NotImplementedError("Complete mitigate() first.")


for name, c in (("noise model", model_counts), ("ibm_pittsburgh", ibm)):
    m = mitigate(c, A)
    raw = different(c) / sum(c.values())
    print(f"{name:15s} raw different {raw:.3f}  ->  mitigated {m['01'] + m['10']:.3f}   ",
          {k: round(v, 4) for k, v in m.items()})

# Mitigation removes a bias but adds spread: 200 repeated runs of the model.
raw_d, mit_d = [], []
for seed in range(200):
    c = AerSimulator(seed_simulator=seed, noise_model=device).run(hw, shots=1000).result().get_counts()
    raw_d.append(different(c) / 1000)
    m = mitigate(c, A)
    mit_d.append(m["01"] + m["10"])
print(f"\\n200 runs of the model: raw {np.mean(raw_d):.4f} ± {np.std(raw_d):.4f},"
      f"  mitigated {np.mean(mit_d):.4f} ± {np.std(mit_d):.4f}")
print(f"gate errors alone, from Step 6: {p_gates[1] + p_gates[2]:.4f}")"""),
md("""For the model, mitigation removes the bias that readout adds: on average the mitigated value matches the gate errors alone (0.005). It does not remove the shot noise, and the spread from run to run is a little larger after mitigation, because the correction also scales up the random part. With larger readout errors the extra spread is larger, so mitigated results need more shots for the same precision.

For IBM's run, this step is an **illustration**, not a measurement. IBM's lesson does not say which physical qubits the run used, and it does not give that day's calibration. If we apply the April snapshot's readout matrix for qubits 0 and 1 anyway, it takes out about 1.3 points and leaves about 1.7%, against 0.5% predicted for the gates. The spread of a mitigated value at 1,000 shots is about 0.005 (from the 200 runs above), so 3σ is about 0.014, and the gap of 0.012 is again inside it. The honest conclusion: the run clearly shows noise; readout errors of the snapshot's size would account for part of it; and these 1,000 shots, without the run's own calibration, are not enough to say whether the rest is more than the calibration predicts. Canvas asks you to recognise exactly this kind of conclusion."""),
md(f"""## Step 8 (optional): run it on a real quantum computer

If you have set up an IBM Quantum account (see **Set Up Your Tools** in Start Here), you can run the Bell circuit on a real IBM computer yourself, for free. It uses Qiskit, which cannot run in the browser, so it runs in Google Colab:

**[Open the Module 5 hardware notebook in Colab]({COLAB})**

It takes about 15 minutes plus the queue, and uses only a few seconds of the 10 minutes of quantum time the free Open Plan gives each 28 days. This step is **optional and not graded**. The Module 5 run required for the badge is the simulator work in this notebook, checked by the lab check.

When it finishes, paste your counts and your device's readout errors below and run the cell. It repeats the analysis of Steps 3 and 7 for your own run."""),
code("""my_counts = None          # for example: my_counts = {"00": 488, "01": 9, "10": 14, "11": 489}
my_read = None            # for example: my_read = {0: (0.004, 0.012), 1: (0.003, 0.010)}  (e0, e1) from Colab

if my_counts is None:
    print("Optional: paste your counts from the Colab notebook to analyse your own run.")
else:
    n = sum(my_counts.values())
    p = different(my_counts) / n
    print(f"different results: {different(my_counts)} of {n} = {p:.3f}")
    same = my_counts.get("00", 0) + my_counts.get("11", 0)
    p00 = my_counts.get("00", 0) / same
    band = 3 * shot_noise(0.5, same)
    print(f"00 among equal results: {p00:.3f}; the 3-sigma band is 0.5 ± {band:.3f}, so this is",
          "within shot noise" if abs(p00 - 0.5) <= band else "outside the band (evidence of an imbalance, under the model)")
    if my_read is None:
        print("No mitigation yet: paste the readout errors (e0, e1) of the two physical qubits your run used, from the "
              "Colab notebook, into my_read. The April Pittsburgh values are not used for your own run, because they "
              "describe a different day and possibly different qubits.")
        m = None
    else:
        my_A = two_qubit_readout(single_qubit_readout(*my_read[0]), single_qubit_readout(*my_read[1]))
        m = mitigate(my_counts, my_A)
        print(f"mitigated different results: {m['01'] + m['10']:.3f}")
    plot_histogram([my_counts, ibm], title="your run (red) and IBM's published run (grey)")"""),
md("""## Step 9: your personal noisy circuit

Open the Canvas quiz **Module 5 lab check**. Question 1 gives you three numbers: **DEP_PCT** (a depolarizing error on the CNOT, in percent), **RO_PCT** (a readout error on each qubit, in percent) and **SEED**. Type them below in place of `None` and run the cell. It shows the noise model your Bell circuit will be run with in the check cell."""),
code("""DEP_PCT = None   # for example: DEP_PCT = 10
RO_PCT = None    # for example: RO_PCT = 3
SEED = None      # for example: SEED = 512

if None in (DEP_PCT, RO_PCT, SEED):
    print("Enter DEP_PCT, RO_PCT and SEED from Canvas first.")
else:
    lam, e = DEP_PCT / 100, RO_PCT / 100
    print(f"depolarizing error on the CNOT: lambda = {lam}")
    print(f"readout error on each qubit:    e0 = e1 = {e}")
    print(f"different results from the CNOT error alone:  {lam / 2:.4f}")
    print(f"probability that exactly one readout flips:   {2 * e * (1 - e):.4f}")"""),
md("""## Step 10: run the check cell

**Do not edit this cell.** Run it. It tests your three functions on cases you have not seen. If they all pass, it runs the Bell circuit 1,000 times with your noise model and your SEED and prints your **verification value**: the number of shots in which the two qubits gave **different** results. Type it into Canvas question 1.

If a test does not pass, read its message, fix that function, run its cell again, then run this cell again."""),
code("# CHECK CELL: do not edit\n" + CHECK + '''

ok, messages, value = check_module5(shot_noise, two_qubit_readout, mitigate, DEP_PCT, RO_PCT, SEED)
for m in messages:
    print(m)
if ok:
    print(f"\\nAll tests passed. Your verification value for DEP_PCT = {DEP_PCT}, RO_PCT = {RO_PCT}, SEED = {SEED}: {value}")
    print("Type this number into Canvas question 1.")
else:
    print("\\nNot passed yet: no verification value.")
'''),
md("""## Finish

1. Answer the five questions of the **Module 5 lab check** in Canvas and submit.
2. Take the **Module 5 quiz**.
3. Fill in the short **Module 5 time log**. It also asks whether you ran the optional hardware notebook.

Your notebook is saved in this browser as you work. It is not saved anywhere else, so to keep a copy choose **File > Download**. In Module 6 you use these circuits for a first quantum algorithm, Bernstein-Vazirani."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m5{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
