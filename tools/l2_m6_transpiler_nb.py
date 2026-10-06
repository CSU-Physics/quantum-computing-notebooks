"""Cells of the Module 6 project notebook "A transpiler comparison" (Colab and browser versions)."""
from l2_m6_text import md, code, common_intro, step1_statistics, limits_and_summary, check_source, COLAB_BASE

SAVED_URL = "https://csu-physics.github.io/quantum-computing-notebooks/files/level2/m6_transpiler_runs.json"
COLAB_URL = COLAB_BASE + "QC-L2-M6-project-transpiler-colab.ipynb"


def cells(colab):
    version = ("This is the **Google Colab version**: it installs real Qiskit, runs the transpiler and the noise model "
               "itself, and checks its results against the course's saved runs. (No Google account? Use the browser "
               "version on the Canvas projects page; it gives the same answers.)" if colab else
               "This is the **browser version**, for learners without a Google account. The transpiler and Qiskit Aer "
               "cannot run in a browser, so this version reads their results from a file saved with the same versions "
               f"that the [Colab version]({COLAB_URL}) installs (qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime "
               "0.50.0). You write the same functions and get the same quiz answers; the difference is that here you "
               "**analyse** the transpiler's saved runs instead of making them, and in Step 7b you make one noisy run "
               "of your own circuit in the browser, with a simpler noise model. If you can use Colab, its version runs "
               "the whole experiment itself.")
    c = [common_intro(
        "a transpiler comparison" + (" (Colab version)" if colab else " (browser version)"), 180,
        "Before a circuit runs on an IBM computer, the transpiler rewrites it in the computer's native gates, places it "
        "on physical qubits and adds swaps where qubits are not connected (Module 4). Its optimization level and its "
        "random seed change the result. **How much does the optimization level change the success of a circuit on a "
        "model of an IBM computer, how much does the transpiler's random seed change it, and which of these differences "
        "can 4,000 shots detect?** " + version,
        """1. Write three statistics functions (Step 1) and meet the computer (Step 2).
2. Write `mirror_circuit(bits)`: a circuit that should return its input, and see why it needs a barrier (Step 3).
3. Look at one run, with its uncertainty, and write `success_fraction(counts, bits)` (Step 4).
4. Compare 10 transpiler seeds at each optimization level and write `seed_spread(values)` (Step 5).
5. Test which optimization levels differ significantly (Step 6), and how success falls with the number of qubits (Step 7).
6. Get your verification value (Step 8), then look at the limits.""",
        "Your project quiz gives you two numbers: BITS (a 4-bit string, 0 to 15) and SEED (100 to 999).")]
    if colab:
        c += [
            md("""## Step 0: install Qiskit and load the saved runs

This takes about a minute. The versions are fixed so that your results match the course's saved runs exactly; the second cell downloads them."""),
            code("%pip install -q qiskit==2.5.2 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.50.0"),
            code(f"""import json, math, urllib.request
import numpy as np
import matplotlib.pyplot as plt
import qiskit, qiskit_aer, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

backend = FakePittsburgh()
noisy = AerSimulator.from_backend(backend)
saved = json.load(urllib.request.urlopen("{SAVED_URL}"))
print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__, "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__)
print("saved runs made with", saved["versions"])"""),
        ]
    else:
        c += [
            md("""## Step 0: start Python and load the saved runs

Run the cell below. The first time, Python can take up to a minute to start in your browser, and Matplotlib takes a few more seconds to load."""),
            code("""import json, math
import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import QuantumCircuit, Statevector

saved = json.load(open("m6_transpiler_runs.json"))
print("Ready. qsim", qsim.__version__, "| saved runs made with", saved["versions"])"""),
        ]
    c += step1_statistics(0.933, ", and the check cell tests them")
    c += [
        md("""## Step 2: the computer and the saved runs

The computer is **FakePittsburgh**, a model of the IBM computer `ibm_pittsburgh` that comes with Qiskit (Module 4): 156 qubits on a heavy-hex lattice (each qubit connected to at most three others), native gates `cz`, `rz`, `sx` and `x`, and a noise model built from a snapshot of its calibration (median `cz` error 0.0015, median readout error 0.0045).

The saved runs hold, for 3, 4, 5 and 6 qubits, the four optimization levels 0 to 3 and the transpiler seeds 1 to 10: the number of two-qubit gates after transpiling, the physical qubits chosen, the counts of 4,000 shots on the noise model (with `seed_simulator` equal to the transpiler seed), and the noise model's **exact** probability of the right answer, `p_exact`, computed from the density matrix without shots. """
           + ("The prepared `run_live` repeats any of them here." if colab else "The prepared `saved_run(n, level, seed)` returns one of them.")),
        code("""def saved_run(n, level, seed):
    \"\"\"Prepared: the saved run for n qubits, an optimization level and a transpiler seed (1 to 10).\"\"\"
    return next(r for r in saved["runs"] if r["n"] == n and r["level"] == level and r["seed"] == seed)


r = saved_run(4, 3, 1)
print("4 qubits, level 3, seed 1: input", r["bits"], "|", r["two_qubit"], "two-qubit gates | depth", r["depth"],
      "| physical qubits", r["layout"], "| p_exact", r["p_exact"])"""
             + ("""


def run_live(bits, level, seed):
    \"\"\"Prepared: transpile mirror_circuit(bits) for FakePittsburgh and run 4,000 shots on its noise model.\"\"\"
    pm = generate_preset_pass_manager(backend=backend, optimization_level=level, seed_transpiler=seed)
    isa = pm.run(mirror_circuit(bits))
    counts = noisy.run(isa, shots=4000, seed_simulator=seed).result().get_counts()
    two_q = sum(1 for i in isa.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
    return {"two_qubit": two_q, "depth": isa.depth(), "layout": list(isa.layout.final_index_layout()), "counts": counts}"""
                if colab else "")),

        md("""## Step 3: a circuit that should return its input

To compare transpiler settings, you need a circuit whose right answer is known for any size. A **mirror circuit** does a computation and then undoes it: here, prepare the input string, apply the QFT, then the inverse QFT. Without noise it returns the input every time, so any other result is an error.

**Your task:** write `mirror_circuit(bits)`:

1. X on every qubit q whose bit is 1 (qubit q is `bits[n - 1 - q]`, qubit 0 on the right);
2. the prepared `qft(qc, range(n))`;
3. **one barrier**, `qc.barrier()`;
4. the prepared `inverse_qft(qc, range(n))`;
5. measure qubit q into classical bit q.

The QFT and its inverse use `cp` and `swap` gates: for n qubits, n(n − 1)/2 + ⌊n/2⌋ two-qubit gates each, so 16 in all for 4 qubits."""),
        code("""def qft(qc, qubits):
    \"\"\"Prepared: the quantum Fourier transform on the given qubits (Qiskit's convention, with the final swaps).\"\"\"
    qubits = list(qubits)
    n = len(qubits)
    for j in reversed(range(n)):
        qc.h(qubits[j])
        for k in reversed(range(j)):
            qc.cp(np.pi / 2 ** (j - k), qubits[k], qubits[j])
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])


def inverse_qft(qc, qubits):
    \"\"\"Prepared: the inverse of qft(): the same gates in reverse order, with negative angles.\"\"\"
    qubits = list(qubits)
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n):
        for k in range(j):
            qc.cp(-np.pi / 2 ** (j - k), qubits[k], qubits[j])
        qc.h(qubits[j])


def mirror_circuit(bits):
    \"\"\"Prepare |bits>, QFT, one barrier, inverse QFT, measure qubit q into classical bit q.\"\"\"
    n = len(bits)
    qc = QuantumCircuit(n, n)
    # YOUR CODE HERE
    raise NotImplementedError("Complete mirror_circuit() first.")
    return qc"""),
        code("""qc = mirror_circuit("1011")
c = qc.copy()
c.remove_final_measurements()
print("without noise, mirror_circuit('1011') gives:", {str(k): round(float(v), 6) for k, v in Statevector(c).probabilities_dict().items() if v > 1e-9})
print("gates:", dict(qc.count_ops()))"""),
        md("""**Why the barrier?** The transpiler's optimization passes look for gates that cancel, and a QFT followed by its inverse cancels completely. The table shows the saved 4-qubit circuit **without** the barrier, transpiled with seed 1: at level 3 every two-qubit gate is gone, only the three X gates remain, and the circuit tests nothing. A barrier tells the transpiler not to move or merge gates across it; it optimizes each half on its own."""
           + (" The cell transpiles your own circuit at level 3 too." if colab else "")),
        code("""print("level   two-qubit gates without the barrier   with the barrier (seed 1)")
for nb in saved["no_barrier"]:
    print(f"  {nb['level']}                  {nb['two_qubit']:3d}                              {saved_run(4, nb['level'], 1)['two_qubit']:3d}")"""
             + ("""
mine = run_live("1011", 3, 1)
print("\\nyour mirror_circuit('1011') at level 3, seed 1:", mine["two_qubit"], "two-qubit gates (saved run:", saved_run(4, 3, 1)["two_qubit"], ")")"""
                if colab else "")),

        md("""## Step 4: one run and its uncertainty

**Your task:** write `success_fraction(counts, bits)`: the fraction of shots that returned `bits`. Use `counts.get(bits, 0)`, because a result that never appeared has no entry, and divide by the total number of shots in `counts`.

The cell takes the 4-qubit run at level 3 with seed 1 (input 1011), computes its success fraction and σ, and compares it with the noise model's exact value by the 3σ rule. The project quiz asks for this success fraction."""),
        code("""def success_fraction(counts, bits):
    \"\"\"The fraction of shots in counts that returned the string bits.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete success_fraction() first.")"""),
        code(("""live = run_live("1011", 3, 1)
counts = live["counts"]
print("this notebook's live run matches the saved run:", counts == saved_run(4, 3, 1)["counts"])
""" if colab else """counts = saved_run(4, 3, 1)["counts"]
""") + """f = success_fraction(counts, "1011")
p = saved_run(4, 3, 1)["p_exact"]
print(f"success fraction {f:.4f}; noise model's exact value {p:.4f}")
print(f"sigma = {shot_sigma(p, 4000):.4f}; within 3 sigma: {within_3_sigma(counts.get('1011', 0), 4000, p)}")
top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
print("most frequent results:", top)"""),
        md("""**What to notice.** About 93% of the shots return 1011. The wrong results are mostly one bit away from it: single errors, from a two-qubit gate or a readout. The difference from the exact value is a fraction of σ = 0.004: one run with 4,000 shots pins down this circuit's success to within about ±0.012 (3σ)."""),

        md("""## Step 5: ten transpiler seeds

The transpiler makes random choices when it places qubits and adds swaps; `seed_transpiler` fixes them. Different seeds can give different physical qubits and different numbers of gates, so **the same circuit can succeed more or less often depending on the seed**. Is that variation larger than shot noise?

**Your task:** write `seed_spread(values)`: the **sample standard deviation** of a list of numbers, √(Σ(x − mean)² / (len − 1)). It measures how much results scatter from run to run. (`np.std(values, ddof=1)` does the same.)

The cell takes the 4-qubit runs at each level with seeds 1 to 10 and prints the mean success, the spread of the measured fractions, the spread of the exact values (the part due to the transpiler's choices alone), and the typical shot noise σ."""
           + (" With `LIVE = True` it first runs the 40 transpilations and simulations here (about 2 minutes) and checks them against the saved runs." if colab else "")),
        code("""def seed_spread(values):
    \"\"\"The sample standard deviation of a list of numbers (divide by len(values) - 1).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete seed_spread() first.")"""),
        code(("""LIVE = True
if LIVE:
    same = all(run_live("1011", lvl, s)["counts"] == saved_run(4, lvl, s)["counts"] for lvl in range(4) for s in range(1, 11))
    print("40 live runs identical to the saved runs:", same)

""" if colab else "") + """print("level   two-qubit gates   mean success   spread (measured)   spread (exact)   shot sigma")
summary = {}
for lvl in range(4):
    runs = [saved_run(4, lvl, s) for s in range(1, 11)]
    fr = [success_fraction(r["counts"], "1011") for r in runs]
    ex = [r["p_exact"] for r in runs]
    tq = [r["two_qubit"] for r in runs]
    summary[lvl] = (np.mean(fr), seed_spread(fr))
    print(f"  {lvl}       {min(tq):3d} to {max(tq):3d}         {np.mean(fr):.4f}         {seed_spread(fr):.4f}            {seed_spread(ex):.4f}         {shot_sigma(np.mean(fr), 4000):.4f}")"""),
        md("""**What to notice.**

- At **level 1**, every seed gives the same 54 two-qubit gates and the same exact value, so the measured spread (0.004) is just shot noise.
- At **level 0** the spread of the measured fractions (0.011) is clearly larger than shot noise (0.007), and the exact values alone spread by 0.010: the seeds give different routings (60 to 66 two-qubit gates) on different qubits.
- At levels 2 and 3 the seeds give 32 to 38 two-qubit gates, and the transpiler's choices add a spread about as large as shot noise.

**The lesson for uncertainty:** σ from one run says how well that run measured **its** circuit. If the question is about a transpiler setting, not one transpiled circuit, the uncertainty must include the variation between seeds, and one run is not enough to measure it."""),

        md("""## Step 6: which optimization levels differ significantly?

To compare two levels, use the **mean over the 10 seeds**. Its uncertainty is the **standard error of the mean**, SEM = seed_spread / √10, which includes both shot noise and the seed-to-seed variation. For two means, σ_diff = √(SEM₁² + SEM₂²), the same rule as `difference_sigma`. The cell compares levels 1 and 3, and 2 and 3, for 4, 5 and 6 qubits."""),
        code("""def level_mean(n, level):
    bits = saved["widths"][str(n)]
    fr = [success_fraction(saved_run(n, level, s)["counts"], bits) for s in range(1, 11)]
    return np.mean(fr), seed_spread(fr) / math.sqrt(len(fr))


for n in (4, 5, 6):
    for a, b in ((1, 3), (2, 3)):
        (ma, sa), (mb, sb) = level_mean(n, a), level_mean(n, b)
        d, sd = mb - ma, math.sqrt(sa ** 2 + sb ** 2)
        print(f"{n} qubits: level {a} {ma:.4f} ± {sa:.4f}, level {b} {mb:.4f} ± {sb:.4f}: difference {d:+.4f} = {d / sd:+.1f} sigma_diff,",
              "significant" if abs(d) > 3 * sd else "not significant")"""),
        md("""**What to notice.** Level 3 beats level 1 by 3 to 5 percentage points at every size, many σ_diff: a clear result. Level 3 against level 2 is subtler: significantly better for 4 and 5 qubits (by 0.010 and 0.023, with the same gate counts: level 3 chooses physical qubits with lower errors), but for 6 qubits the two levels give the same success within 0.001, far below σ_diff. "Level 3 is better" is true for some circuits and not shown for others."""),

        md("""## Step 7: success against the number of qubits

The cell shows the mean success at level 3 for 3 to 6 qubits, with its SEM, and the mean number of two-qubit gates after transpiling. The project quiz asks for the mean success with 6 qubits.

If every two-qubit gate failed with the same probability ε and nothing else went wrong, the success would be about (1 − ε)^N₂ for N₂ two-qubit gates. The last column inverts this, ε ≈ 1 − success^(1/N₂): an **effective error per two-qubit gate**, which also absorbs the readout and one-qubit errors."""),
        code("""print("qubits   two-qubit gates (mean)   mean success ± SEM    effective error per two-qubit gate")
ns, means, sems = [3, 4, 5, 6], [], []
for n in ns:
    m, se = level_mean(n, 3)
    n2 = np.mean([saved_run(n, 3, s)["two_qubit"] for s in range(1, 11)])
    means.append(m); sems.append(se)
    print(f"  {n}            {n2:5.1f}                {m:.4f} ± {se:.4f}            {1 - m ** (1 / n2):.4f}")

plt.figure(figsize=(6.5, 3.3))
plt.errorbar(ns, means, yerr=[3 * s for s in sems], fmt="C0o-", capsize=4, label="level 3, mean of 10 seeds, ±3 SEM")
plt.xticks(ns); plt.xlabel("qubits"); plt.ylabel("success fraction"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.** The success falls from 0.968 with 3 qubits to 0.861 with 6, while the two-qubit gates grow from about 17 to 92: the QFT has a `cp` between every pair of qubits, so its gate count grows as n², and on a heavy-hex chip most pairs need swaps. The effective error per two-qubit gate is about 0.002, close to the computer's median `cz` error (0.0015) plus a share of the readout errors."""),
        *([] if colab else [
            md("""### Step 7b (browser version, not graded): a noisy run of your own circuit

Everything above analyses runs that Qiskit's transpiler and Qiskit Aer made on the FakePittsburgh model; they cannot run in a browser. This step runs **your** `mirror_circuit()` for 4 qubits yourself, here, on a simpler noise model: the course model of the other three projects (a depolarizing error P2 after each two-qubit gate and P2/10 after each one-qubit gate; a 0 read as 1 with probability READOUT and a 1 read as 0 with probability 2·READOUT), with P2 = 0.002 (an average two-qubit gate error of 0.75 · P2 = 0.0015, the computer's median `cz` error) and READOUT = 0.0045, the computer's median readout error.

Your circuit is **not transpiled** here: every qubit can interact with every other, so no swaps are added, and each `cp` counts as one two-qubit gate (on the real chip it becomes two `cz` gates). Compare the two results and say which differences between the two models could explain the gap."""),
            code("""import math
from qsim import AerSimulator, NoiseModel, ReadoutError, depolarizing_error


def simple_noise_model(p2, readout):
    \"\"\"Prepared: the course noise model of the other Module 6 projects.\"\"\"
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["h", "x", "rx", "ry", "rz", "p", "t", "tdg", "s", "sdg", "sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz", "cp", "rzz", "swap"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [2 * readout, 1 - 2 * readout]]))
    return nm


bits4 = saved["widths"]["4"]
mine = mirror_circuit(bits4)
counts = AerSimulator(noise_model=simple_noise_model(0.002, 0.0045)).run(mine, shots=4000, seed_simulator=1).result().get_counts()
f = success_fraction(counts, bits4)
n2 = sum(1 for i in mine.data if len(i.qubits) == 2 and i.name != "barrier")
m4, se4 = level_mean(4, 3)
print(f"your circuit, simple model, no routing ({n2} two-qubit gates): {f:.4f} ± {shot_sigma(f, 4000):.4f}")
print(f"saved FakePittsburgh runs, level 3, mean of 10 seeds:          {m4:.4f} ± {se4:.4f}")
print(f"difference {f - m4:+.4f} = {(f - m4) / math.sqrt(shot_sigma(f, 4000) ** 2 + se4 ** 2):.1f} times its standard deviation")"""),
            md("""**What to notice.** The two numbers are close (about 0.946 and 0.937; the difference is about 2 standard deviations, so not significant), yet the models are very different. Your untranspiled circuit has 16 two-qubit gates; transpiled for FakePittsburgh at level 3 the same circuit has about 35, because each `cp` becomes two `cz` gates and swaps are added. On the other hand, the simple model reads a 1 wrongly twice as often as a 0, and three of the four bits are 1, while FakePittsburgh's errors differ from qubit to qubit and include decoherence while qubits wait. These differences pull in opposite directions. Agreement between two models is therefore not evidence that either is right: it can hide compensating errors."""),
        ]),

        md("""## Step 8: your verification value

Open your **project quiz for this experiment** in Canvas. Question 1 shows your BITS (0 to 15: a 4-bit input string, for example 11 is 1011) and SEED (100 to 999). Type them below and run the next cell. The check cell tests your three statistics functions, `mirror_circuit()`, `success_fraction()` and `seed_spread()`. Only if everything passes does it print your **verification value**: in the course's saved runs, the number of shots out of 4,000 in which the mirror circuit for BITS, transpiled at level 3 with `seed_transpiler = SEED` and run on the noise model with `seed_simulator = SEED`, returned BITS."""
           + (" The cell after it repeats that run live here." if colab else "")),
        code("""BITS = -1     # your number from Canvas, for example 11 (the string 1011)
SEED = 0      # your seed from Canvas, for example 512"""),
        code(check_source("l2_m6_transpiler_check.py", '''passed, messages, value = check_l2_m6_transpiler(mirror_circuit, success_fraction, seed_spread, shot_sigma,
                                                 within_3_sigma, difference_sigma, BITS, SEED, saved)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''', colab=colab)),
    ]
    if colab:
        c += [code("""if passed:
    w = format(int(BITS), "04b")
    live = run_live(w, 3, int(SEED))["counts"].get(w, 0)
    print("this notebook's live run:", live, "| saved run:", value,
          "| they match" if live == value else "| they differ: use the saved value printed above in Canvas")""")]
    c += limits_and_summary(
        """- **One model of one computer.** FakePittsburgh is a snapshot of one calibration. The real `ibm_pittsburgh` is recalibrated daily, its errors drift, and the best qubits today may not be the best tomorrow; a level-3 advantage that comes from choosing good qubits depends on that snapshot.
- **What the noise model leaves out.** Gate and readout errors and decoherence are in it; crosstalk between neighbouring qubits, leakage out of the qubit states, and errors that repeat the same way every time are not. So results on the real computer can differ from its noise model, often for the worse, and change from day to day.
- **One kind of circuit.** A mirror circuit returns its input, which makes success easy to define. Circuits whose answer is a distribution (QAOA, phase estimation of 1/3) need other measures, and transpiler settings can rank differently for them.
- **Ten seeds.** The seed-to-seed spread is itself estimated from 10 values, so it is uncertain too (by roughly a quarter of its size).""",
        "the Module 6 project quiz for this experiment, the transpiler comparison,")
    return c
