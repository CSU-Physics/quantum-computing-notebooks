"""Shared cells for the Level 3 Module 6 capstone notebooks (see make_l3_m6.py)."""
import nbformat as nbf

VERSION = "2026-10-10"
COLAB_BASE = "https://colab.research.google.com/github/CSU-Physics/quantum-computing-notebooks/blob/main/colab/level3/"
HARDWARE_COLAB = COLAB_BASE + "QC-L3-M6-hardware-colab.ipynb"
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


def intro(title, question, plan, personal):
    return md(f"""# Capstone: {title}

**Quantum Computing Advanced · Module 6 · about 5 hours** · notebook version {VERSION}

**The question of this experiment.** {question}

You run the experiment on a **noisy simulator**, a simulator that adds the errors of a real quantum computer described by a **noise model**. You compare what you measure with what the noise model predicts, say how certain every result is, and say what the experiment cannot tell you. The capstone brings together the code, the mitigation and the application of Modules 2, 3 and 4, and you write more of the code yourself than in a lab.

**The plan.**

{plan}

**How it is graded.** Your capstone quiz in Canvas (the one for this experiment, 20 points, pass at 16) asks for your verification value from the check cell (6 points), three results from this notebook and four questions about uncertainty, interpretation and limits (2 points each). {personal} At the end you write a short **summary** of your experiment (required, not graded), which you submit in the Course Completion module.

**Saving your work.** JupyterLite keeps your changes in this browser, so you can stop and come back. To keep a copy elsewhere, use **File → Download**. Run each code cell with **Shift + Enter**, in order; if you come back later, run the cells from the top again.""")


def step0(extra_imports=""):
    return [
        md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser, and Matplotlib takes a few more seconds to load."""),
        code(f"""import math
import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import QuantumCircuit, NoiseModel, ReadoutError, depolarizing_error, simulate_density_matrix{extra_imports}

print("Ready. qsim", qsim.__version__, "| NumPy", np.__version__)"""),
    ]


KERNEL_STEP1 = """In this capstone two kinds of numbers have an uncertainty, and they get it from different places. A **kernel entry** is a fraction of shots, so its uncertainty comes from the shots: the formulas below are for these. A classifier's **accuracy** is a fraction of **test points**: its uncertainty comes from how many points you test, not from the shots, and two classifiers tested on the same points are compared point by point (Step 6), not with the 3σ rule below.
"""
KERNEL_3SIGMA = """

**Accuracies are reported differently.** Give an accuracy as a count, for example 4 of 4 or 18 of 20. Do not put an observed 4 of 4 into √(p (1 − p) / N): p = 1 gives σ = 0, a certainty that 4 points cannot give. If you need an interval, use one made for small counts, such as the Wilson interval (Step 6)."""


def step1_statistics(track=None):
    first = (KERNEL_STEP1 if track == "kernel" else
             "Every number you measure in this capstone comes from shots, so every number has an uncertainty. Three formulas cover all of them.\n")
    return [
        md("## Step 1: how certain is a result" + ("?" if track == "kernel" else " from shots?") + """

""" + first + """
**A fraction.** If a result has probability p and you take S shots, the fraction f of shots with that result has standard deviation

**σ = √( p (1 − p) / S )**.

**An expectation value of ±1 results**, such as ⟨Z⟩ or ⟨X X X⟩: each shot gives +1 or −1, and the average E has standard deviation

**σ = √( (1 − E²) / S )**.

This is twice the σ of the fraction of +1 results, because E = 2 f − 1. For E = 0.8 and S = 4,000 it is √(0.36 / 4000) = 0.0095.

**A combination of independent results.** Mitigation combines several measured values: an estimate Ê = c₁ E₁ + c₂ E₂ + ... from independent runs with uncertainties σ₁, σ₂, ... has

**σ = √( (c₁ σ₁)² + (c₂ σ₂)² + ... )**.

Uncertainties add **in quadrature**. The difference of two results is the special case c = (1, −1): σ_diff = √(σ₁² + σ₂²). Large coefficients, as in an extrapolation, make the combination much less certain than any single run: this is the price of mitigation you met in Module 3.

**The course rule: 3σ.** A measured value agrees with an expected value when they differ by at most 3σ. A difference larger than 3σ_diff is **significant**: shot noise alone would almost never produce it. A smaller difference may be real, but your data cannot show it. Two cautions: the rule treats the spread as normal (bell-shaped), which is good when many shots give each result; and when you compare many settings, chance alone makes one "significant" difference likely somewhere (with 20 comparisons, about 5%). Treat a single surprising difference among many as a question to test again with new shots.""" + (KERNEL_3SIGMA if track == "kernel" else "") + """

**Your task:** write the three functions below. Every later step uses them.

- `fraction_sigma(p, shots)`: √(p (1 − p) / shots).
- `expectation_sigma(E, shots)`: √((1 − E²) / shots).
- `combined_sigma(coefficients, sigmas)`: √(Σ (cᵢ σᵢ)²) for two lists of the same length."""),
        code("""def fraction_sigma(p, shots):
    \"\"\"The standard deviation of a fraction measured with `shots` shots, when the true probability is p.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete fraction_sigma() first.")


def expectation_sigma(E, shots):
    \"\"\"The standard deviation of the average of `shots` results of +1 or -1 with expectation value E.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete expectation_sigma() first.")


def combined_sigma(coefficients, sigmas):
    \"\"\"The standard deviation of sum(c_i * E_i) for independent E_i with standard deviations sigma_i.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete combined_sigma() first.")"""),
        code("""print("fraction_sigma(0.0613, 4000) =", round(fraction_sigma(0.0613, 4000), 5), "  (expected 0.00379)")
print("expectation_sigma(0.8, 4000) =", round(expectation_sigma(0.8, 4000), 5), "  (expected 0.00949)")
print("combined_sigma([1, -1], [0.01, 0.02]) =", round(combined_sigma([1, -1], [0.01, 0.02]), 5), "  (expected 0.02236)")
print("combined_sigma([1.875, -1.25, 0.375], [0.012, 0.015, 0.02]) =",
      round(combined_sigma([1.875, -1.25, 0.375], [0.012, 0.015, 0.02]), 5), "  (expected 0.03023)")"""),
    ]


NOISE_CODE = """ONE_QUBIT_GATES = ["h", "x", "y", "z", "rx", "ry", "rz", "p", "s", "sdg", "t", "tdg", "sx"]
TWO_QUBIT_GATES = ["cx", "cz", "cp", "rzz", "swap"]
P2_COURSE, READOUT_COURSE = 0.01, 0.02


def confusion(readout):
    \"\"\"Prepared. Rows: the true bit (0, 1); columns: the recorded bit.\"\"\"
    return np.array([[1 - readout, readout], [2 * readout, 1 - 2 * readout]])


def capstone_noise_model(p2=P2_COURSE, readout=READOUT_COURSE, idle=0.0):
    \"\"\"Prepared: depolarizing error p2 after two-qubit gates, p2/10 after one-qubit gates, idle on id gates, readout.\"\"\"
    nm = NoiseModel()
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ONE_QUBIT_GATES)
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), TWO_QUBIT_GATES)
    if idle > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(idle, 1), ["id"])
    if readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError(confusion(readout).tolist()))
    return nm


def apply_readout(probs, readout):
    \"\"\"Prepared. probs over m recorded bits (index bit j = classical bit j); each bit is flipped by the confusion.\"\"\"
    probs = np.asarray(probs, dtype=float)
    m = int(round(math.log2(len(probs))))
    c = confusion(readout)
    for j in range(m):
        t = probs.reshape(2 ** (m - 1 - j), 2, 2 ** j)
        probs = np.einsum("aib,ij->ajb", t, c).reshape(-1)
    return probs


def exact_probabilities(qc, p2=P2_COURSE, readout=READOUT_COURSE, idle=0.0):
    \"\"\"Prepared: the noise model's exact probability of each recorded result, {"bits": p}, Qiskit's key order.
    qc must end with its measurements (each classical bit receives one qubit). No shots are taken.\"\"\"
    measured = {inst.clbits[0]: inst.qubits[0] for inst in qc.data if inst.name == "measure"}
    body = qc.copy()
    body.remove_final_measurements()
    nm = capstone_noise_model(p2, 0.0, idle)
    rho = simulate_density_matrix(body, nm if not nm.is_ideal() else None)   # the exact mixed state after the noisy gates
    m = len(measured)
    probs = rho.probabilities(qargs=[measured[j] for j in range(m)])
    probs = apply_readout(probs, readout)                                       # then the readout errors
    return {format(i, f"0{m}b"): float(p) for i, p in enumerate(probs) if p > 1e-15}


def sample_counts(probs, shots, seed):
    \"\"\"Prepared: `shots` random results drawn from exact probabilities, as a quantum computer gives them.\"\"\"
    keys = sorted(probs)
    p = np.array([probs[k] for k in keys])
    draw = np.random.default_rng(seed).multinomial(shots, p / p.sum())
    return {k: int(c) for k, c in zip(keys, draw) if c > 0}


print(capstone_noise_model())"""


def step2_noise(extra=""):
    return [
        md(f"""## Step 2: the capstone noise model

The noise model of the capstone is the one you used in Level 2 Module 6, with one addition:

- after every **two-qubit gate**, a **depolarizing error** with parameter P2 (with probability P2 the two qubits are replaced by a completely random state);
- after every **one-qubit gate**, the same on one qubit with P2 / 10;
- an **idle error** on `id` gates, used only by the repetition-code capstone (a qubit that waits while others are measured);
- at **readout**, a 0 is recorded as 1 with probability READOUT and a 1 as 0 with probability 2 · READOUT.

The course values are **P2 = 0.01** and **READOUT = 0.02**, several times worse than IBM's best devices, so that the effects show clearly in a few thousand shots. Your capstone quiz gives you your own P2 and READOUT (0.005 to 0.030 each) for the check cell.

The prepared cell builds the model and three tools:

- `exact_probabilities(qc, p2, readout)`: the noise model's **exact** probability of every recorded result, from the density matrix (Module 1) and the readout confusion; no shots, no randomness. This is the expected value the 3σ rule compares with.
- `sample_counts(probs, shots, seed)`: random shots drawn from those probabilities. A run of the noisy simulator gives counts with exactly these statistics; drawing them from the exact probabilities is much faster in the browser, and the seed makes your run repeatable.
- `apply_readout(probs, readout)`: the readout errors alone, applied to probabilities.{extra}"""),
        code(NOISE_CODE),
    ]


def personal_step(step_no, names, ranges, example, func, args, value_text):
    lines = "\n".join(f"{n} = 0    # your number from Canvas" for n in names)
    return [
        md(f"""## Step {step_no}: your personal check

Open your capstone quiz in Canvas. Question 1 shows your own numbers: {ranges}. Type them below and run the next two cells. The check cell tests the three statistics functions of Step 1 and every function you wrote for this capstone. Only if every test passes does it print your **verification value**: {value_text} For example, {example}."""),
        code(lines),
    ]


def check_cell(hidden_check, project_module, func, args):
    call = f"""

passed, messages, value = {func}({args})
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")"""
    return code(hidden_check(["l3_m6_common", project_module], func, call))


def ending(limits, quiz_name, track=None):
    result = ("Your main numbers: kernel entries with their uncertainty from the shots (for example 0.558 ± 0.016), and each accuracy as a count of test points "
              "(for example 4 of 4, or 18 of 20), with a small-sample interval such as Wilson's if you give one."
              if track == "kernel" else "Your main numbers, each with its uncertainty (for example 0.976 ± 0.010).")
    interp = ("What do the numbers say about the question? For the classifiers, what does the point-by-point comparison on the same test points show? "
              "Which kernel entries differ by more than 3σ?"
              if track == "kernel" else "What do the numbers say about the question? Which differences are significant (more than 3σ)?")
    return [
        md("## Limits of this experiment\n\n" + limits + """

These limits are not failures of your experiment: every experiment has them. Saying what a result does and does not show is part of reporting it."""),
        md(f"""## Optional: run it on a real IBM quantum computer

If you have an IBM Quantum account with an Open Plan instance (see **Set Up Your Tools** in Start Here), the [Module 6 hardware notebook in Colab]({HARDWARE_COLAB}) runs this capstone's reference circuits on a real computer and compares them with the ideal values and with this noise model. Your API key stays in Colab's **Secrets**; never type it into a cell. The Open Plan gives up to 10 minutes of quantum computer time every 28 days; the hardware notebook uses well under a minute of it, but the queue can take from seconds to hours. This step is not graded, and the badge does not depend on it."""),
        md("""## Your capstone summary (required, not graded)

Write five short answers here (double-click this cell to edit it). They are how a result is reported in a lab notebook or a paper, and they are good preparation for the capstone quiz. Then paste them into **Your Capstone Summary** in the Course Completion module in Canvas: it is not graded, but it is required for the badge.

1. **Question.** What did you want to find out?
2. **Method.** Which circuits, which noise model (P2, READOUT), how many shots, which mitigation or decoding?
3. **Result.** """ + result + """
4. **Interpretation.** """ + interp + """
5. **Limits.** What does this experiment not show?"""),
        md(f"""**Next in Canvas:** {quiz_name}, then the Module 6 time log."""),
    ]
