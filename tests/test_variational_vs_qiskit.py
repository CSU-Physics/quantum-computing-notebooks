"""qsim 1.9.0 against Qiskit 2.5.2: Parameter, ParameterVector and assign_parameters, rzz, SparsePauliOp and
StatevectorEstimator (the tools of Level 2, Module 5)."""
import random, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "content/level1"))
import qsim
from qiskit import QuantumCircuit as QQC
from qiskit.circuit import Parameter as QP, ParameterVector as QPV
from qiskit.quantum_info import Operator as QOp, Statevector as QSV, SparsePauliOp as QSPO
from qiskit.primitives import StatevectorEstimator as QEst

rng = random.Random(2028)
nrng = np.random.default_rng(2028)


def twin_circuit(n, k):
    """The same random parameterized circuit in qsim and in Qiskit; returns (a, b, values in parameter order)."""
    names = rng.sample(["a", "b", "γ", "β", "phi"], 2)
    qv, sv = QPV("θ", k), qsim.ParameterVector("θ", k)
    qp, sp = [QP(x) for x in names], [qsim.Parameter(x) for x in names]
    a, b = qsim.QuantumCircuit(n), QQC(n)
    for q in range(n):
        a.h(q); b.h(q)
    for _ in range(rng.randint(4, 16)):
        g = rng.choice(["rx", "ry", "rz", "p", "rzz", "rzz", "cx", "cp", "crz"])
        # a linear expression in one or two parameters, or a plain number
        kind = rng.choice(["vec", "par", "lin", "num"])
        if kind == "vec":
            i = rng.randrange(k); ea, eb = sv[i], qv[i]
        elif kind == "par":
            i = rng.randrange(2); ea, eb = sp[i], qp[i]
        elif kind == "lin":
            i, j = rng.randrange(k), rng.randrange(2); c1, c0 = rng.choice([2, -1, 0.5, 3]), rng.uniform(-1, 1)
            ea, eb = c1 * sv[i] + sp[j] - c0, c1 * qv[i] + qp[j] - c0
        else:
            ea = eb = rng.uniform(-3, 3)
        if g in ("rx", "ry", "rz", "p"):
            q = rng.randrange(n); getattr(a, g)(ea, q); getattr(b, g)(eb, q)
        elif g == "cx":
            c, t = rng.sample(range(n), 2); a.cx(c, t); b.cx(c, t)
        else:
            c, t = rng.sample(range(n), 2); getattr(a, g)(ea, c, t); getattr(b, g)(eb, c, t)
    return a, b


worst = 0.0
order_ok = 0
for trial in range(300):
    n, k = rng.randint(2, 4), rng.randint(1, 4)
    a, b = twin_circuit(n, k)
    names_a = [str(p) for p in a.parameters]
    names_b = [p.name for p in b.parameters]
    assert names_a == names_b, (names_a, names_b)
    assert a.num_parameters == b.num_parameters
    order_ok += 1
    vals = list(nrng.uniform(-np.pi, np.pi, a.num_parameters))
    sa = qsim.Statevector(a.assign_parameters(vals)).data
    sb = QSV(b.assign_parameters(vals)).data
    worst = max(worst, np.abs(sa - sb).max())
    # dict binding, by Parameter (and by ParameterVector where the circuit uses every element)
    da = dict(zip(a.parameters, vals)); db = dict(zip(b.parameters, vals))
    worst = max(worst, np.abs(qsim.Statevector(a.assign_parameters(da)).data - QSV(b.assign_parameters(db)).data).max())
    # inverses
    if a.num_parameters:
        ia = a.assign_parameters(vals).inverse(); ib = b.assign_parameters(vals).inverse()
        worst = max(worst, np.abs(qsim.Operator(ia).data - QOp(ib).data).max())
        worst = max(worst, np.abs(qsim.Operator(a.inverse().assign_parameters(vals)).data - QOp(b.inverse().assign_parameters(vals)).data).max())
print(f"300 random parameterized circuits (rx, ry, rz, p, rzz, cp, crz with linear expressions): same parameter "
      f"order as Qiskit in {order_ok}; largest state or operator difference {worst:.1e}")
assert worst < 1e-10

# rzz alone
for th in np.linspace(-7, 7, 29):
    a = qsim.QuantumCircuit(2); a.rzz(th, 0, 1); b = QQC(2); b.rzz(th, 0, 1)
    assert np.allclose(qsim.Operator(a).data, QOp(b).data, atol=1e-12)

# errors that learners will meet
t = qsim.ParameterVector("θ", 2)
c = qsim.QuantumCircuit(1); c.ry(t[0], 0); c.rx(t[1], 0)
for bad, exc in [(lambda: qsim.Statevector(c), qsim.CircuitError), (lambda: c.assign_parameters([1.0]), ValueError),
                 (lambda: c.assign_parameters({qsim.Parameter("zz"): 1}), qsim.CircuitError), (lambda: float(t[0]), TypeError)]:
    try:
        bad(); raise AssertionError("no error")
    except exc:
        pass
print("rzz on 29 angles and the error cases: as in Qiskit")

# ---------------------------------------------------------------- SparsePauliOp
worst = 0.0
for trial in range(300):
    n, m = rng.randint(1, 4), rng.randint(1, 6)
    labels = ["".join(rng.choice("IXYZ") for _ in range(n)) for _ in range(m)]
    coeffs = list(nrng.normal(size=m) + (1j * nrng.normal(size=m) if trial % 3 == 0 else 0))
    a, b = qsim.SparsePauliOp(labels, coeffs), QSPO(labels, coeffs)
    worst = max(worst, np.abs(a.to_matrix() - b.to_matrix()).max())
    assert list(a.paulis) == [str(p) for p in b.paulis] and a.num_qubits == b.num_qubits and a.size == b.size
    s1 = a.simplify(); s2 = b.simplify()
    assert sorted(s1.paulis) == sorted(str(p) for p in s2.paulis)
    worst = max(worst, np.abs(s1.to_matrix() - s2.to_matrix()).max())
    a2, b2 = qsim.SparsePauliOp(labels[::-1], coeffs[::-1]), QSPO(labels[::-1], coeffs[::-1])
    for x, y in [(a + a2, b + b2), (a - a2, b - b2), (2.5 * a, 2.5 * b), (a * -1j, b * -1j), (-a, -b), (a / 4, b / 4)]:
        worst = max(worst, np.abs(x.to_matrix() - y.to_matrix()).max())
    assert a.equiv(a2) == b.equiv(b2)
    # from_sparse_list
    nn = rng.randint(2, 5)
    terms = []
    for _ in range(rng.randint(1, 4)):
        qs = rng.sample(range(nn), rng.randint(1, nn))
        terms.append(("".join(rng.choice("XYZ") for _ in qs), qs, rng.uniform(-2, 2)))
    x, y = qsim.SparsePauliOp.from_sparse_list(terms, nn), QSPO.from_sparse_list(terms, nn)
    assert list(x.paulis) == [str(p) for p in y.paulis]
    worst = max(worst, np.abs(x.to_matrix() - y.to_matrix()).max())
    # expectation values of random states
    v = nrng.normal(size=2 ** n) + 1j * nrng.normal(size=2 ** n); v /= np.linalg.norm(v)
    worst = max(worst, abs(qsim.Statevector(v).expectation_value(a) - QSV(v).expectation_value(b)))
print(f"300 random SparsePauliOps: matrices, sums, scaling, simplify, from_sparse_list and expectation values; "
      f"largest difference from Qiskit {worst:.1e}")
assert worst < 1e-10
h2 = [("II", -1.052373245772859), ("IZ", 0.39793742484318045), ("ZI", -0.39793742484318045),
      ("ZZ", -0.01128010425623538), ("XX", 0.18093119978423156)]
assert repr(qsim.SparsePauliOp.from_list(h2)) == repr(QSPO.from_list(h2))
assert repr(qsim.SparsePauliOp(["ZZ", "XI"], [1, 0.5])) == repr(QSPO(["ZZ", "XI"], [1, 0.5]))
assert repr(sum([qsim.SparsePauliOp("ZZ"), qsim.SparsePauliOp("XX")])) == repr(sum([QSPO("ZZ"), QSPO("XX")]))
print("repr of SparsePauliOp: identical to Qiskit on 3 cases")

# ---------------------------------------------------------------- StatevectorEstimator
worst, shapes = 0.0, set()
ea, eb = qsim.StatevectorEstimator(), QEst()
for trial in range(200):
    n, k = rng.randint(2, 3), rng.randint(1, 3)
    a, b = twin_circuit(n, k)
    P = a.num_parameters
    if P == 0:
        continue
    labs = [["".join(rng.choice("IXYZ") for _ in range(n)) for _ in range(rng.randint(1, 3))] for _ in range(3)]
    obs_a = [qsim.SparsePauliOp(l, list(np.arange(1, len(l) + 1) * 0.5)) for l in labs]
    obs_b = [QSPO(l, list(np.arange(1, len(l) + 1) * 0.5)) for l in labs]
    form = trial % 5
    if form == 0:
        pa, pb, va = obs_a[0], obs_b[0], list(nrng.uniform(-3, 3, P))
    elif form == 1:
        pa, pb, va = obs_a, obs_b, list(nrng.uniform(-3, 3, P))
    elif form == 2:
        pa, pb, va = obs_a[0], obs_b[0], nrng.uniform(-3, 3, (4, P))
    elif form == 3:
        pa, pb, va = [[o] for o in obs_a], [[o] for o in obs_b], nrng.uniform(-3, 3, (2, P))
    else:
        pa, pb, va = str(labs[0][0]), str(labs[0][0]), nrng.uniform(-3, 3, (3, P))
    ra = ea.run([(a, pa, va)]).result()[0].data.evs
    rb = eb.run([(b, pb, va)]).result()[0].data.evs
    assert np.shape(ra) == np.shape(rb), (np.shape(ra), np.shape(rb))
    shapes.add(np.shape(ra))
    worst = max(worst, np.abs(np.asarray(ra) - np.asarray(rb)).max())
print(f"StatevectorEstimator on 200 PUBs with result shapes {sorted(shapes)}: largest difference from Qiskit {worst:.1e}")
assert worst < 1e-10
