"""Shared cells for the Level 2 Module 6 project notebooks (see make_l2_m6.py)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-10-05"
COLAB_BASE = "https://colab.research.google.com/github/CSU-Physics/quantum-computing-notebooks/blob/main/colab/level2/"
HARDWARE_COLAB = COLAB_BASE + "QC-L2-M6-hardware-colab.ipynb"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


def _body(path):
    """A check file without its docstring and without the lines that import l2_m6_common."""
    src = (ROOT / "checks" / path).read_text()
    body = src.split('"""', 2)[2].lstrip()
    return "\n".join(line for line in body.splitlines() if not line.startswith("from l2_m6_common import")).strip() + "\n"


def check_source(project_file, call):
    """The check cell: l2_m6_common, then the project's check, then the call that prints the result."""
    return ("# The course's check code. You do not need to read it; it is here so that you can.\n"
            + _body("l2_m6_common.py") + "\n\n" + _body(project_file) + "\n\n" + call.strip())


def common_intro(title, minutes, question, plan, personal):
    return md(f"""# Module 6 project: {title}

**Quantum Computing Intermediate · Module 6 · about {minutes} minutes** · notebook version {VERSION}

**The question of this experiment.** {question}

In this project you run an experiment on a **noisy simulator**: a simulator that adds the errors of a real quantum computer, described by a **noise model**. You compare what you measure with what the noise model predicts, say how certain each result is, and say what the experiment cannot tell you. These are the skills of course outcome 5.

**The plan.**

{plan}

**How it is graded.** Your project quiz in Canvas (the one for this experiment, 20 points, pass at 16) asks for your verification value from the check cell, three results from this notebook, and four questions about uncertainty, interpretation and limits. {personal}

**Saving your work.** JupyterLite keeps your changes in this browser, so you can stop and come back. To keep a copy elsewhere, use **File → Download**. Run each code cell with **Shift + Enter**, in order; if you come back later, run the cells from the top again.""")


def step0_browser(extra=""):
    return [
        md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser, and Matplotlib takes a few more seconds to load."""),
        code(f"""import math
import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import (QuantumCircuit, Statevector, AerSimulator, NoiseModel, depolarizing_error, ReadoutError,
                  simulate_density_matrix){extra}

print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),
    ]


def step1_statistics(example_p, example_name):
    return [
        md(f"""## Step 1: how certain is a result from shots?

A circuit that gives a result with probability p, run for S shots, gives that result k times. The number k is random: it follows the **binomial distribution**. The fraction f = k / S has

- mean p, and
- standard deviation **σ = √(p (1 − p) / S)**. This spread is called **shot noise**.

For example, with S = 4,000 shots and p = {example_p}, σ = {(example_p * (1 - example_p) / 4000) ** 0.5:.4f}: two runs of the same circuit typically differ by about this much. Four times as many shots halve σ.

**The course rule: 3σ.** A measured fraction f agrees with an expected value p when |f − p| ≤ 3σ. For a noisy simulator, p must be the **noise model's** expected value (computed exactly, without shots, in Step 2), not the ideal value from a circuit without noise. A correct run lands outside 3σ only about 3 times in 1,000 (0.27%).

**Comparing two results.** Two fractions f₁ and f₂, each from S independent shots, differ by f₂ − f₁, with standard deviation

**σ_diff = √( f₁(1 − f₁)/S + f₂(1 − f₂)/S )**.

Uncertainties of independent results add **in quadrature** (their squares add), so a difference is less certain than either result alone. A difference larger than 3σ_diff is **significant**: shot noise alone would almost never produce it. A smaller difference may be real, but this experiment cannot show it.

**Your task:** write the three functions below. They are used in every later step{example_name}.

- `shot_sigma(p, shots)`: σ = √(p (1 − p) / shots), the standard deviation of a fraction.
- `within_3_sigma(k, shots, p)`: `True` if the fraction k / shots is within 3 · shot_sigma(p, shots) of p (in either direction), otherwise `False`.
- `difference_sigma(f1, f2, shots)`: σ_diff for two fractions from `shots` shots each."""),
        code("""def shot_sigma(p, shots):
    \"\"\"The standard deviation of a fraction measured with `shots` shots, when the true probability is p.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete shot_sigma() first.")


def within_3_sigma(k, shots, p):
    \"\"\"True if k successes out of `shots` agree with the probability p by the course's 3-sigma rule.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete within_3_sigma() first.")


def difference_sigma(f1, f2, shots):
    \"\"\"The standard deviation of f2 - f1, for two fractions from `shots` independent shots each.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete difference_sigma() first.")"""),
        code("""print("shot_sigma(0.687, 4000) =", round(shot_sigma(0.687, 4000), 5), "  (expected 0.00733)")
print("within_3_sigma(2748, 4000, 0.687) =", within_3_sigma(2748, 4000, 0.687), "  (expected True: exactly 0.687)")
print("within_3_sigma(2655, 4000, 0.687) =", within_3_sigma(2655, 4000, 0.687), "  (expected False: 3.2 sigma too low)")
print("difference_sigma(0.687, 0.634, 4000) =", round(difference_sigma(0.687, 0.634, 4000), 5), "  (expected 0.01057)")"""),
    ]


def step1_test():
    return []


def step2_noise():
    return [
        md("""## Step 2: the course noise model

A **noise model** says what can go wrong in each operation. The course model has three parts, each with a number:

- after every **two-qubit gate** (`cx`, `cz`, `cp`, `rzz`, `swap`), a **depolarizing error** with parameter P2: with probability P2 the two qubits are replaced by a completely random state. For a density matrix, ρ → (1 − P2) ρ + P2 · I/4 on those two qubits. (An average gate error, the number IBM reports, of 0.75 · P2.)
- after every **one-qubit gate**, the same on one qubit with parameter P2 / 10: one-qubit gates are about ten times better.
- at **readout**: a qubit in |0⟩ is recorded as 1 with probability READOUT, and a qubit in |1⟩ is recorded as 0 with probability 2 · READOUT. The second is larger because an excited qubit can decay to |0⟩ during the measurement.

The course values are **P2 = 0.01** and **READOUT = 0.02**. They are several times worse than IBM's best current computers (the model of `ibm_pittsburgh` used in Module 4 has a median two-qubit gate error of 0.0015 and a median readout error of 0.0045), so that the effects of noise show clearly in a few thousand shots. In Canvas, your project quiz gives you your own P2 (0.005 to 0.030) and READOUT (0.005 to 0.030): a model of "your" computer.

The noise model applies errors by **gate name**. A gate on three qubits, such as `ccx` or `mcx`, would get no error at all, so every circuit in this project uses only one- and two-qubit gates.

The prepared cell builds the noise model with the same functions as Qiskit Aer (`qiskit_aer.noise`), and defines `expected_probabilities(qc, p2, readout)`. It gives the noise model's **exact** probability of every recorded result, with no shots and no randomness: it follows the **density matrix**, the mixed state that the noisy gates produce, then applies the readout errors to the recorded bits. A run with shots, `AerSimulator(noise_model=...).run(qc, shots=4000)`, samples from these same probabilities and adds shot noise. The 3σ rule compares the two."""),
        code("""ONE_QUBIT_GATES = ["h", "x", "rx", "ry", "rz", "p", "t", "tdg", "s", "sdg", "sx"]
TWO_QUBIT_GATES = ["cx", "cz", "cp", "rzz", "swap"]


def course_noise_model(p2=0.01, readout=0.02):
    \"\"\"Prepared: depolarizing error p2 after two-qubit gates and p2/10 after one-qubit gates; readout errors.\"\"\"
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ONE_QUBIT_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), TWO_QUBIT_GATES)
    nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [2 * readout, 1 - 2 * readout]]))
    return nm


def expected_probabilities(qc, p2=0.01, readout=0.02):
    \"\"\"Prepared: the noise model's exact probability of each recorded result, as a dict {"bits": probability}.

    qc must end with its measurements (classical bit j receives one qubit). No shots are taken.
    \"\"\"
    measured = {inst.clbits[0]: inst.qubits[0] for inst in qc.data if inst.name == "measure"}
    body = qc.copy()
    body.remove_final_measurements()
    rho = simulate_density_matrix(body, course_noise_model(p2, readout))   # the exact mixed state after the noisy gates
    m = len(measured)
    probs = rho.probabilities(qargs=[measured[j] for j in range(m)])      # index bit j = classical bit j
    confusion = np.array([[1 - readout, readout], [2 * readout, 1 - 2 * readout]])   # rows: true bit; columns: recorded bit
    for j in range(m):                                                     # each recorded bit can be flipped
        t = probs.reshape(2 ** (m - 1 - j), 2, 2 ** j)
        probs = np.einsum("aib,ij->ajb", t, confusion).reshape(-1)
    return {format(i, f"0{m}b"): float(p) for i, p in enumerate(probs) if p > 1e-15}


print(course_noise_model())"""),
    ]


def limits_and_summary(limit_bullets, next_quiz):
    return [
        md("## Limits of this experiment\n\n" + limit_bullets + """

These limits are not failures of your experiment: every experiment has them. Saying what a result does and does not show is part of reporting it."""),
        md(f"""## Optional: run it on a real IBM quantum computer

If you have an IBM Quantum account with an Open Plan instance (see **Set Up Your Tools** in Start Here), the [Module 6 hardware notebook in Colab]({HARDWARE_COLAB}) runs the project's reference circuit on a real computer and compares the result with the ideal value, with this course's noise model, and with IBM's own noise model of that computer. Your API key stays in Colab's **Secrets**; never type it into a cell. The Open Plan gives up to 10 minutes of quantum computer time every 28 days, and one run of 4,000 shots uses a few seconds of it, but the queue can take from seconds to hours. This step is not graded, and the badge does not depend on it."""),
        md("""## Optional: a short summary of your experiment

Not graded. Writing five short answers is a good way to prepare for the project quiz, and it is how results are reported in a lab notebook or a paper. Double-click this cell to edit it.

1. **Question.** What did you want to find out?
2. **Method.** Which circuit, which noise model (P2, READOUT), how many shots?
3. **Result.** Your main numbers, each with its uncertainty (for example 0.687 ± 0.007, with σ from Step 1).
4. **Interpretation.** What do the numbers say about the question? Which differences are significant (more than 3σ)?
5. **Limits.** What does this experiment not show?"""),
        md(f"""**Next in Canvas:** {next_quiz} and the Module 6 time log."""),
    ]
