# Quantum Computing Notebooks (SemiAcademy)

Lab notebooks for the SemiAcademy quantum computing micro-credentials. The labs run **inside Canvas, in the learner's own browser**: GitHub Pages serves a JupyterLite site, and each Canvas lab page embeds one notebook. Python runs in the browser through Pyodide, so learners install nothing and need no account.

Site: https://csu-physics.github.io/quantum-computing-notebooks/

| Level | Notebook | Canvas page (course 51) | Lab address |
|---|---|---|---|
| 1 | `content/level1/QC-L1-M0-lab-first-qubit.ipynb` | Module 0 Lab: Notebooks, NumPy and Your First Qubit | `lab/index.html?path=level1/QC-L1-M0-lab-first-qubit.ipynb` |
| 1 | `content/level1/QC-L1-M1-lab-state-vectors.ipynb` | Module 1 Lab: State Vectors and Measurement | `lab/index.html?path=level1/QC-L1-M1-lab-state-vectors.ipynb` |
| 1 | `content/level1/QC-L1-final-coding-task-GHZ.ipynb` | Final Coding Task: GHZ State | `lab/index.html?path=level1/QC-L1-final-coding-task-GHZ.ipynb` |

## Why qsim and not Qiskit

Qiskit and Qiskit Aer have no build for the browser (they need compiled Rust and C++ code). The labs therefore use **qsim** (`content/level1/qsim.py`), a small NumPy simulator written for the course. It keeps Qiskit's names (`QuantumCircuit`, `h`, `cx`, `ry`, `measure`, `measure_all`, `AerSimulator`, `transpile`, `Statevector`, `state_fidelity`, `plot_histogram`), its bit order and its error messages, so learner code carries over to Qiskit by changing the import lines.

Checked against Qiskit 2.5.2 and Qiskit Aer 0.17.2 (`tests/test_qsim_vs_qiskit.py`):

- state vectors on 300 random circuits agree to 4e-16;
- density matrices with mid-circuit measurement and reset agree with exact Qiskit evolution (Kraus channels) to better than 1e-15;
- measurement counts agree within shot noise.

The one exception is Module 5's real-hardware run, which needs Qiskit and IBM Quantum. That notebook runs in Google Colab (folder `colab/`), and its simulator version runs in the browser.

## The check cells

Each graded lab ends with a check cell. It tests the learner's circuit and, only if the tests pass, prints a verification value for the learner's own Canvas parameter. A Canvas formula question then checks that value. The check code is in `checks/`, with accuracy tests:

- `checks/test_ghz_check.py`: 10 correct and 12 wrong GHZ circuits, including classical mixtures with perfect counts. 0 wrong verdicts, in CPython and in Pyodide 0.27.7.
- `checks/test_m0_check.py`: 3 correct and 7 wrong one-qubit circuits at three angles. 0 wrong verdicts.
- `checks/test_m1_check.py`: 8 correct combinations of `probabilities()`, `normalize()` and `simulate()` and 10 wrong ones. 0 wrong verdicts. The Module 1 verification value is a seeded count (NumPy's `default_rng(SEED).choice`), so its Canvas answers are precomputed for 200 (A, B, SEED) sets; do not regenerate them from the formula in the Canvas editor.

Run them with `PYTHONPATH=content/level1 python checks/test_ghz_check.py`.

## Editing a notebook

1. Edit `tools/make_notebooks.py` (Module 0 and the GHZ task) or `tools/make_m1.py` (Module 1), or the check files, then run the script. It writes the notebooks into `content/level1/` with outputs cleared.
2. Commit. The workflow in `.github/workflows/deploy.yml` rebuilds the site and deploys it to GitHub Pages in about a minute.

Learners keep their own edited copy in their browser's storage. A learner who has already changed and saved a notebook keeps that copy after an update, until they clear this site's data in their browser. So fix problems before a cohort starts, not during it.

## Rules for this repository

- **Public**, because GitHub Pages serves it to learners.
- **No answer keys or solution notebooks.** They stay with the course team. `.gitignore` blocks file names containing "SOLUTION", "solution" or "answer-key".
- **No outputs in the learner notebooks.**

Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). Questions: mhadizadeh@centralstate.edu
