"""Make the two data sets of the Level 3 Module 4 Track A lab (quantum-kernel classifier):
content/level3/l3_m4a_data.json.

- "moons": scikit-learn's two interleaved half-moons (60 points, noise 0.15, random_state 1), both features scaled
  to [0, pi]. An ordinary data set with no quantum structure.
- "adhoc": 60 points in [0, 2 pi)^2 labelled by the quantum feature map itself, as in Havlicek et al. (Nature 567,
  209, 2019): with |phi(x)> the state of Qiskit's zz_feature_map(2, reps=2) and V a fixed random two-qubit unitary,
  f(x) = <phi(x)| V^dagger Z0 Z1 V |phi(x)>; label 1 if f > 0.3, label 0 if f < -0.3, and points with |f| <= 0.3
  are dropped (a gap between the classes). Built so that the quantum kernel fits it; it says nothing about real data.

Each set is split 40 training / 20 test points, 20 and 10 of each class, in a fixed random order.
Run with Qiskit 2.5.2, scikit-learn and SciPy (not needed by learners)."""
import json
from pathlib import Path

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from scipy.stats import unitary_group
from sklearn.datasets import make_moons

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level3" / "l3_m4a_data.json"
GAP, V_SEED, DATA_SEED = 0.3, 3, 2026


def split(X, y, rng):
    """40 training and 20 test points, 20 + 10 of each class, shuffled."""
    tr, te = [], []
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        tr += list(idx[:20])
        te += list(idx[20:30])
    tr, te = rng.permutation(tr), rng.permutation(te)
    r = lambda a: [[round(float(v), 6) for v in row] for row in a]  # noqa: E731
    return {"X_train": r(X[tr]), "y_train": [int(v) for v in y[tr]], "X_test": r(X[te]), "y_test": [int(v) for v in y[te]]}


rng = np.random.default_rng(DATA_SEED)
X, y = make_moons(60, noise=0.15, random_state=1)
X = (X - X.min(0)) / (X.max(0) - X.min(0)) * np.pi
moons = split(X, y, rng)

fm = zz_feature_map(2, reps=2)
V = unitary_group.rvs(4, random_state=V_SEED)
O = V.conj().T @ np.diag([1, -1, -1, 1]) @ V
pts, labels = [], []
r2 = np.random.default_rng(V_SEED)
while min(labels.count(0), labels.count(1)) < 30:
    x = r2.uniform(0, 2 * np.pi, 2)
    s = Statevector(fm.assign_parameters(x)).data
    f = float(np.real(s.conj() @ O @ s))
    if abs(f) <= GAP:
        continue
    c = int(f > 0)
    if labels.count(c) >= 30:
        continue
    pts.append(x)
    labels.append(c)
adhoc = split(np.array(pts), np.array(labels), rng)

data = {"about": "Level 3 Module 4 Track A data (tools/make_l3_m4a_data.py). Features are angles in radians.",
        "moons": dict(moons, description="scikit-learn make_moons(60, noise=0.15, random_state=1), scaled to [0, pi]"),
        "adhoc": dict(adhoc, description="labels from the ZZ feature map (reps 2) and a fixed random unitary, gap 0.3")}
OUT.write_text(json.dumps(data, indent=1))
print("wrote", OUT, {k: (len(v["X_train"]), len(v["X_test"]), sum(v["y_train"]), sum(v["y_test"])) for k, v in data.items() if k != "about"})
