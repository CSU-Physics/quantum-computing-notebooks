# Quantum Computing Notebooks (SemiAcademy)

Lab notebooks for the SemiAcademy quantum computing micro-credentials. The labs run **inside Canvas, in the learner's own browser**: GitHub Pages serves a JupyterLite site, and each Canvas lab page embeds one notebook. Python runs in the browser through Pyodide, so learners install nothing and need no account.

Site: https://csu-physics.github.io/quantum-computing-notebooks/

| Level | Notebook | Canvas page (course 51) | Lab address |
|---|---|---|---|
| 1 | `content/level1/QC-L1-M0-lab-first-qubit.ipynb` | Module 0 Lab: Notebooks, NumPy and Your First Qubit | `lab/index.html?path=level1/QC-L1-M0-lab-first-qubit.ipynb` |
| 1 | `content/level1/QC-L1-M1-lab-state-vectors.ipynb` | Module 1 Lab: State Vectors and Measurement | `lab/index.html?path=level1/QC-L1-M1-lab-state-vectors.ipynb` |
| 1 | `content/level1/QC-L1-M2-lab-bloch-sphere.ipynb` | Module 2 Lab: The Bloch Sphere and Measurement Bases | `lab/index.html?path=level1/QC-L1-M2-lab-bloch-sphere.ipynb` |
| 1 | `content/level1/QC-L1-M3-lab-single-qubit-gates.ipynb` | Module 3 Lab: Single-Qubit Gates as Matrices | `lab/index.html?path=level1/QC-L1-M3-lab-single-qubit-gates.ipynb` |
| 1 | `content/level1/QC-L1-M4-lab-bell-states.ipynb` | Module 4 Lab: Two Qubits, CNOT and the Bell States | `lab/index.html?path=level1/QC-L1-M4-lab-bell-states.ipynb` |
| 1 | `content/level1/QC-L1-final-coding-task-GHZ.ipynb` | Final Coding Task: GHZ State | `lab/index.html?path=level1/QC-L1-final-coding-task-GHZ.ipynb` |

## Course media (slides and explorers)

The folder `media/` holds self-contained HTML pages that Canvas embeds next to the readings. The workflow copies it into the site, so each file is served at `https://csu-physics.github.io/quantum-computing-notebooks/media/...`.

| File | What it is | Canvas page (course 51) |
|---|---|---|
| `media/level1/m1-slides.html` | Module 1 slide deck, 19 slides. Arrow keys or the buttons move between slides; **Print / PDF** gives one slide per page. `#s7` opens slide 7. | Module 1: From Bits to Qubits |
| `media/level1/m1-explore.html` | Module 1 explorer: a state builder (`#state`) and a shot simulator (`#shots`). | Module 1: From Bits to Qubits |
| `media/level1/m2-slides.html` | Module 2 slide deck, 19 slides, with Bloch spheres drawn in the page. | Module 2: One Qubit |
| `media/level1/m2-explore.html` | Module 2 explorer: a draggable Bloch sphere with θ, φ and a global phase (`#bloch`), and measurement in three bases with a mystery-state game (`#tomo`). | Module 2: One Qubit |
| `media/level1/m3-slides.html` | Module 3 slide deck, 19 slides, with rotation arcs on the Bloch sphere and the interference fringe from the lab. | Module 3: Single-Qubit Gates |
| `media/level1/m3-explore.html` | Module 3 explorer: a gate playground with animated turns and the matrix of the whole sequence (`#gates`), the circuit H, phase, H with its interference fringe (`#interf`), and an undo challenge (`#undo`). | Module 3: Single-Qubit Gates |
| `media/level1/m4-slides.html` | Module 4 slide deck, 19 slides: two-qubit states, Qiskit's bit order, CNOT, the Bell states, correlations and no signalling. | Module 4: Two Qubits and Entanglement |
| `media/level1/m4-explore.html` | Module 4 explorer: a two-qubit circuit builder with a live product-or-entangled test and each qubit's Bloch vector (`#build`), and a correlation lab for Alice and Bob with a classical mix for comparison (`#corr`). | Module 4: Two Qubits and Entanglement |

They use no outside libraries and no network calls, and they are licensed CC BY 4.0 (see each page's credit line). To change one, edit the HTML and commit; the site updates in about a minute. The Module 2, 3 and 4 pages are assembled by `tools/build_m2_slides.py`, `tools/build_m3_slides.py` and `tools/build_m4_slides.py` from `tools/mK_slides_body.html`, `tools/mK_explore_src.html` and the shared Bloch-sphere drawing code `tools/bloch.js` (rotation arcs were added for Module 3; the Module 2 files built before that are unchanged); edit those and run the script.

## Why qsim and not Qiskit

Qiskit and Qiskit Aer have no build for the browser (they need compiled Rust and C++ code). The labs therefore use **qsim** (`content/level1/qsim.py`), a small NumPy simulator written for the course. It keeps Qiskit's names (`QuantumCircuit`, `h`, `cx`, `ry`, `measure`, `measure_all`, `AerSimulator`, `transpile`, `Statevector`, `Operator`, `state_fidelity`, `plot_histogram`, `plot_bloch_vector`, `plot_bloch_multivector`), its bit order and its error messages, so learner code carries over to Qiskit by changing the import lines.

Checked against Qiskit 2.5.2 and Qiskit Aer 0.17.2 (`tests/test_qsim_vs_qiskit.py`):

- state vectors on 300 random circuits agree to 4e-16;
- density matrices with mid-circuit measurement and reset agree with exact Qiskit evolution (Kraus channels) to better than 1e-15;
- measurement counts agree within shot noise;
- Bloch vectors (the per-qubit reduced states behind `plot_bloch_multivector`) agree with Qiskit's partial trace on 200 random circuits to better than 1e-15 (`tests/test_bloch_vs_qiskit.py`);
- `Operator` matrices agree with `qiskit.quantum_info.Operator` on 300 random 1- to 3-qubit circuits to 6e-16, and `==` and `equiv` (equal up to a global phase) give the same answers as Qiskit in 600 of 600 cases (`tests/test_operator_vs_qiskit.py`).

The one exception is Module 5's real-hardware run, which needs Qiskit and IBM Quantum. That notebook runs in Google Colab (folder `colab/`), and its simulator version runs in the browser.

## The check cells

Each graded lab ends with a check cell. It tests the learner's circuit and, only if the tests pass, prints a verification value for the learner's own Canvas parameter. A Canvas formula question then checks that value. The check code is in `checks/`, with accuracy tests:

- `checks/test_ghz_check.py`: 10 correct and 12 wrong GHZ circuits, including classical mixtures with perfect counts. 0 wrong verdicts, in CPython and in Pyodide 0.27.7.
- `checks/test_m0_check.py`: 3 correct and 7 wrong one-qubit circuits at three angles. 0 wrong verdicts.
- `checks/test_m1_check.py`: 8 correct combinations of `probabilities()`, `normalize()` and `simulate()` and 10 wrong ones. 0 wrong verdicts. The Module 1 verification value is a seeded count (NumPy's `default_rng(SEED).choice`), so its Canvas answers are precomputed for 200 (A, B, SEED) sets; do not regenerate them from the formula in the Canvas editor.
- `checks/test_m2_check.py`: 8 correct combinations of `state_from_angles()`, `bloch_vector()` and `measure_in_basis()` at three parameter sets, and 15 wrong ones (half angle, phase sign, no conjugate, S instead of S-dagger, H before S-dagger, X and Y swapped, changing the input circuit, and others). 0 wrong verdicts. The Module 2 verification value is a seeded Y-basis count from `AerSimulator`, so its Canvas answers are also precomputed (200 sets); do not regenerate them in the Canvas editor.

- `checks/test_m3_check.py`: 8 correct combinations of `gate_matrix()`, `rotation()` and `sequence_matrix()` at three parameter sets, and 13 wrong ones (Y with the signs swapped, H without 1/√2, S-dagger for S, T equal only up to a global phase, no half angle, +i or no i in the rotation, Rz written as the phase gate, matrices multiplied in the wrong order, changing the input list, and others). 0 wrong verdicts. The Module 3 verification value is a seeded count from `AerSimulator` for the circuit ry(θ), rx(φ), so its Canvas answers are also precomputed (200 sets); do not regenerate them in the Canvas editor.
- `checks/test_m4_check.py`: 8 correct combinations of `product_state()`, `bell_circuit()` and `correlation()` at three parameter sets, and 12 wrong ones (qubits in the wrong order, concatenating instead of the tensor product, measuring inside `bell_circuit()`, the same state for every name, no H, psi− with the + sign, three qubits, counting only 00, the fraction instead of the correlation, the sign reversed, failing on a missing key). 0 wrong verdicts. The Module 4 verification value is a seeded count of equal results from `AerSimulator`, so its Canvas answers are also precomputed (200 sets); do not regenerate them in the Canvas editor.

Run them with `PYTHONPATH=content/level1 python checks/test_ghz_check.py`.

## Editing a notebook

1. Edit `tools/make_notebooks.py` (Module 0 and the GHZ task), `tools/make_m1.py` (Module 1), `tools/make_m2.py` (Module 2), `tools/make_m3.py` (Module 3) or `tools/make_m4.py` (Module 4), or the check files, then run the script. It writes the notebooks into `content/level1/` with outputs cleared.
2. Commit. The workflow in `.github/workflows/deploy.yml` rebuilds the site and deploys it to GitHub Pages in about a minute.

Learners keep their own edited copy in their browser's storage. A learner who has already changed and saved a notebook keeps that copy after an update, until they clear this site's data in their browser. So fix problems before a cohort starts, not during it.

## Rules for this repository

- **Public**, because GitHub Pages serves it to learners.
- **No answer keys or solution notebooks.** They stay with the course team. `.gitignore` blocks file names containing "SOLUTION", "solution" or "answer-key".
- **No outputs in the learner notebooks.**

Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). Questions: mhadizadeh@centralstate.edu
