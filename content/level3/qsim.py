"""qsim: a small quantum circuit simulator for the SemiAcademy quantum computing labs.

It runs in the browser (JupyterLite / Pyodide), where Qiskit cannot be installed, and it uses
Qiskit's names on purpose: QuantumCircuit, h, x, cx, ry, measure, AerSimulator, transpile,
Statevector, state_fidelity and plot_histogram work as they do in Qiskit for the circuits in
this course. Code you write here runs with real Qiskit when you change the import lines.

Conventions follow Qiskit:
- qubit 0 is the rightmost bit of a result such as "011" and the least significant bit of a
  state vector index;
- a Pauli label such as "XZ" puts qubit 0 on the right.

Everything is exact linear algebra with NumPy. It is meant for small circuits (up to about
10 qubits), which is all this course needs.

QuantumCircuit.compose (version 1.4.0) joins circuits as in Qiskit.

Coherent errors (version 1.11.0): coherent_unitary_error(U) adds the same small unitary each time a gate
runs, as in qiskit_aer.noise; Level 3 Module 3 uses it for an idle qubit whose frequency is slightly off.

Faster density matrices (version 1.10.0): noisy runs and DensityMatrix work as before, much faster (an 8-qubit
noisy circuit about 75 times), so that the Module 6 projects can scan noise levels in the browser.

Statevector.probabilities(qargs) and probabilities_dict(qargs) (version 1.6.0) give the probabilities
of measuring only some qubits, as in Qiskit; qargs[0] is the rightmost bit of a result.

Variational algorithms (version 1.9.0): Parameter, ParameterVector and QuantumCircuit.assign_parameters,
the two-qubit gate rzz(theta, q1, q2), SparsePauliOp (Hamiltonians as sums of Pauli strings) and
StatevectorEstimator (exact expectation values) work as in Qiskit; Module 5 builds VQE and QAOA with them.
Parameter expressions must be linear, such as 2*theta or theta + 0.5.

The controlled swap (version 1.8.0): cswap(control, target1, target2), also called the Fredkin gate, works
as in Qiskit; the Module 3 order-finding circuits for N = 15 use it.

Multi-controlled gates (version 1.7.0): ccz(c1, c2, target) and mcx(control_qubits, target_qubit) work
as in Qiskit; Grover oracles and diffusers on 3 and 4 qubits use them.

Dynamic circuits (version 1.5.0): `with qc.if_test((clbit, value)):` applies the gates inside the
block only when that classical bit holds that value, as in Qiskit 2. An `else` block works too:
`with qc.if_test((0, 1)) as else_: ...` then `with else_: ...`.

Noise (version 1.3.0): NoiseModel, depolarizing_error, pauli_error, amplitude_damping_error,
phase_damping_error, thermal_relaxation_error and ReadoutError work as in qiskit_aer.noise, and
AerSimulator(noise_model=...) runs a circuit with them.
"""
import math
from collections import Counter

import numpy as np

__all__ = [
    "CircuitError", "QuantumCircuit", "Statevector", "DensityMatrix", "AerSimulator",
    "transpile", "state_fidelity", "plot_histogram", "simulate_density_matrix",
    "plot_bloch_vector", "plot_bloch_multivector", "bloch_vectors", "Operator",
    "NoiseModel", "QuantumError", "ReadoutError", "depolarizing_error", "pauli_error",
    "amplitude_damping_error", "phase_damping_error", "thermal_relaxation_error", "partial_trace",
    "coherent_unitary_error",
    "Parameter", "ParameterVector", "ParameterExpression", "SparsePauliOp", "StatevectorEstimator",
]
__version__ = "1.11.0"


class CircuitError(Exception):
    """Raised for a circuit that cannot be built, with Qiskit's wording."""


# ---------------------------------------------------------------- gate matrices
_S2 = 1 / math.sqrt(2)
_FIXED = {
    "id": np.eye(2, dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
}


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def _p(t):
    return np.array([[1, 0], [0, np.exp(1j * t)]], dtype=complex)


_PARAM = {"rx": _rx, "ry": _ry, "rz": _rz, "p": _p}


def _rzz(t):
    """exp(-i t/2 Z(x)Z), as Qiskit's RZZGate."""
    a, b = np.exp(-1j * t / 2), np.exp(1j * t / 2)
    return np.diag([a, b, b, a]).astype(complex)


# ---------------------------------------------------------------- parameters
class ParameterExpression:
    """A value that is linear in one or more Parameters, such as 2*theta + 0.5 (Qiskit's ParameterExpression).

    Use qc.assign_parameters(...) to replace the parameters with numbers before simulating."""

    __slots__ = ("_terms", "_const")

    def __init__(self, terms=None, const=0.0):
        self._terms = dict(terms or {})
        self._const = float(const)

    @property
    def parameters(self):
        return set(p for p, c in self._terms.items() if c != 0)

    def _combine(self, other, sign):
        if isinstance(other, ParameterExpression):
            terms = dict(self._terms)
            for p, c in other._terms.items():
                terms[p] = terms.get(p, 0.0) + sign * c
            return ParameterExpression(terms, self._const + sign * other._const)
        return ParameterExpression(self._terms, self._const + sign * float(other))

    def __add__(self, other): return self._combine(other, 1)
    def __radd__(self, other): return self._combine(other, 1)
    def __sub__(self, other): return self._combine(other, -1)
    def __rsub__(self, other): return (-self)._combine(other, 1)
    def __neg__(self): return self * -1

    def __mul__(self, other):
        if isinstance(other, ParameterExpression):
            raise CircuitError("qsim only supports expressions that are linear in the parameters, such as 2*theta + 0.5.")
        k = float(other)
        return ParameterExpression({p: c * k for p, c in self._terms.items()}, self._const * k)

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ParameterExpression):
            raise CircuitError("qsim only supports expressions that are linear in the parameters, such as theta/2.")
        return self.__mul__(1.0 / float(other))

    def bind(self, values):
        """Replace parameters by numbers: values is a dict {Parameter: number}. Returns a float when nothing is left."""
        terms, const = {}, self._const
        for p, c in self._terms.items():
            if p in values:
                const += c * float(values[p])
            elif c != 0:
                terms[p] = c
        return const if not terms else ParameterExpression(terms, const)

    def __float__(self):
        free = self.parameters
        if free:
            names = ", ".join(sorted(str(p) for p in free))
            raise TypeError(f"Parameter expression with unbound parameters {{{names}}} is not numeric.")
        return float(self._const)

    def __str__(self):
        parts = []
        for p, c in sorted(self._terms.items(), key=lambda pc: _param_key(pc[0])):
            if c == 0:
                continue
            mag = abs(c)
            body = str(p) if mag == 1 else (f"{p}/{1 / mag:g}" if mag < 1 and float(1 / mag).is_integer() else f"{mag:g}*{p}")
            parts.append(("-" if c < 0 else "+", body))
        out = ""
        if self._const != 0:
            out = f"{self._const:g}"
        for sign, body in parts:
            if not out:
                out = ("-" if sign == "-" else "") + body
            else:
                out += f" {sign} {body}"
        return out or "0"

    def __repr__(self):
        return f"ParameterExpression({self})"


class Parameter(ParameterExpression):
    """A named, unbound circuit parameter, as in Qiskit: theta = Parameter("θ")."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = str(name)
        ParameterExpression.__init__(self, {self: 1.0}, 0.0)

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Parameter({self.name})"


class ParameterVectorElement(Parameter):
    __slots__ = ("vector", "index")

    def __init__(self, vector, index):
        self.vector, self.index = vector, index
        Parameter.__init__(self, f"{vector.name}[{index}]")

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"ParameterVectorElement({self.name})"


class ParameterVector:
    """A list of parameters named name[0], name[1], ...: theta = ParameterVector("θ", 4)."""

    def __init__(self, name, length=0):
        self.name = str(name)
        self._params = [ParameterVectorElement(self, i) for i in range(int(length))]

    @property
    def params(self):
        return list(self._params)

    def __getitem__(self, key):
        return self._params[key]

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def __repr__(self):
        return f"ParameterVector(name='{self.name}', length={len(self)})"


def _param_key(p):
    if isinstance(p, ParameterVectorElement):
        return (p.vector.name, p.index)
    return (p.name, -1)


def _angle(x):
    """A gate angle: a float, or a ParameterExpression that still holds parameters."""
    if isinstance(x, ParameterExpression):
        return x if x.parameters else float(x)
    return float(x)


def _fmt_param(x):
    return str(x) if isinstance(x, ParameterExpression) else f"{x:.3g}"


def _controlled(u, n_controls=1):
    """Matrix of a gate controlled on the first n_controls qubits (Qiskit order: controls first)."""
    k = n_controls + 1
    dim = 2 ** k
    m = np.eye(dim, dtype=complex)
    # In the gate's own little-endian ordering the controls are bits 0..n-1 and the target is bit n.
    ctrl_mask = (1 << n_controls) - 1
    for i in range(dim):
        for j in range(dim):
            if (i & ctrl_mask) == ctrl_mask and (j & ctrl_mask) == ctrl_mask:
                m[i, j] = u[i >> n_controls, j >> n_controls]
            elif (i & ctrl_mask) == ctrl_mask or (j & ctrl_mask) == ctrl_mask:
                m[i, j] = 0
    return m


_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _cswap_matrix():
    """Controlled swap on (control, t1, t2); in the gate's little-endian order the control is bit 0."""
    m = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        j = i
        if i & 1:
            b1, b2 = (i >> 1) & 1, (i >> 2) & 1
            j = (i & 1) | (b2 << 1) | (b1 << 2)
        m[j, i] = 1
    return m


_CSWAP = _cswap_matrix()


class Instruction:
    """One operation in a circuit: a name, the qubits and classical bits it acts on, and parameters."""

    __slots__ = ("name", "qubits", "clbits", "params")

    def __init__(self, name, qubits, clbits=(), params=()):
        self.name = name
        self.qubits = tuple(qubits)
        self.clbits = tuple(clbits)
        self.params = tuple(params)

    def matrix(self):
        n = self.name
        free = [p for x in self.params if isinstance(x, ParameterExpression) for p in x.parameters]
        if free:
            names = ", ".join(sorted({str(p) for p in free}))
            raise CircuitError(f"The circuit has parameters without values ({names}). "
                               "Give them values first with qc.assign_parameters(...).")
        if n in _FIXED:
            return _FIXED[n]
        if n in _PARAM:
            return _PARAM[n](self.params[0])
        if n == "cx":
            return _controlled(_FIXED["x"])
        if n == "cy":
            return _controlled(_FIXED["y"])
        if n == "cz":
            return _controlled(_FIXED["z"])
        if n == "ch":
            return _controlled(_FIXED["h"])
        if n == "cp":
            return _controlled(_p(self.params[0]))
        if n == "crz":
            return _controlled(_rz(self.params[0]))
        if n == "rzz":
            return _rzz(self.params[0])
        if n == "ccx":
            return _controlled(_FIXED["x"], 2)
        if n == "ccz":
            return _controlled(_FIXED["z"], 2)
        if n == "mcx":
            return _controlled(_FIXED["x"], len(self.qubits) - 1)
        if n == "cswap":
            return _CSWAP
        if n == "swap":
            return _SWAP
        raise CircuitError(f"'{n}' has no matrix")

    def __repr__(self):
        p = f", params={list(self.params)}" if self.params else ""
        c = f", clbits={list(self.clbits)}" if self.clbits else ""
        return f"Instruction({self.name!r}, qubits={list(self.qubits)}{c}{p})"


class IfElseOp(Instruction):
    """A classically controlled block (Qiskit's if_else): true_body runs when clbit == value, else false_body."""

    __slots__ = ("condition", "true_body", "false_body")

    def __init__(self, condition, true_body, false_body=None):
        qs = sorted({q for i in list(true_body) + list(false_body or []) for q in i.qubits})
        cs = sorted({condition[0]} | {c for i in list(true_body) + list(false_body or []) for c in i.clbits})
        super().__init__("if_else", qs, cs, ())
        self.condition = (int(condition[0]), int(condition[1]))
        self.true_body = list(true_body)
        self.false_body = list(false_body) if false_body else []

    def matrix(self):
        raise CircuitError("An if_test block depends on a measurement result, so it has no fixed matrix.")

    def remap(self, qmap, cmap):
        def m(insts):
            return [i.remap(qmap, cmap) if isinstance(i, IfElseOp) else
                    Instruction(i.name, [qmap[q] for q in i.qubits], [cmap[c] for c in i.clbits], i.params)
                    for i in insts]
        return IfElseOp((cmap[self.condition[0]], self.condition[1]), m(self.true_body), m(self.false_body))

    def __repr__(self):
        e = f", else={len(self.false_body)} ops" if self.false_body else ""
        return f"IfElseOp(c{self.condition[0]} == {self.condition[1]}: {len(self.true_body)} ops{e})"


class _ElseContext:
    def __init__(self, qc):
        self.qc = qc

    def __enter__(self):
        last = self.qc.data[-1] if self.qc.data else None
        if not isinstance(last, IfElseOp) or last.false_body:
            raise CircuitError("An else block must come straight after its if_test block.")
        self._start = len(self.qc.data)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        body = self.qc.data[self._start:]
        del self.qc.data[self._start:]
        op = self.qc.data.pop()
        self.qc.data.append(IfElseOp(op.condition, op.true_body, body))
        return False


class _IfContext:
    def __init__(self, qc, condition):
        self.qc = qc
        self.condition = condition

    def __enter__(self):
        self._start = len(self.qc.data)
        return _ElseContext(self.qc)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        body = self.qc.data[self._start:]
        del self.qc.data[self._start:]
        self.qc.data.append(IfElseOp(self.condition, body))
        return False


def _as_list(x):
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    return [int(v) for v in x]


# ---------------------------------------------------------------- the circuit
class QuantumCircuit:
    """A quantum circuit: QuantumCircuit(number_of_qubits, number_of_classical_bits)."""

    def __init__(self, num_qubits, num_clbits=0):
        if int(num_qubits) < 1:
            raise CircuitError("A circuit needs at least one qubit.")
        self.num_qubits = int(num_qubits)
        self._cregs = [("c", int(num_clbits))] if num_clbits else []
        self.data = []
        self.global_phase = 0.0

    # ------------------------------------------------------------ sizes and checks
    @property
    def num_clbits(self):
        return sum(size for _, size in self._cregs)

    @property
    def qubits(self):
        return list(range(self.num_qubits))

    @property
    def clbits(self):
        return list(range(self.num_clbits))

    def _check_qubits(self, qs):
        for q in qs:
            if not 0 <= q < self.num_qubits:
                raise CircuitError(f"Index {q} out of range for size {self.num_qubits}.")
        if len(set(qs)) != len(qs):
            raise CircuitError(f"duplicate qubit arguments {qs}")

    def _check_clbits(self, cs):
        for c in cs:
            if not 0 <= c < self.num_clbits:
                raise CircuitError(f"Index {c} out of range for size {self.num_clbits}.")

    def _add(self, name, qubits, params=()):
        self._check_qubits(list(qubits))
        self.data.append(Instruction(name, qubits, (), params))
        return self

    def _one(self, name, qubit, params=()):
        for q in _as_list(qubit):
            self._add(name, [q], params)
        return self

    # ------------------------------------------------------------ single-qubit gates
    def id(self, q): return self._one("id", q)
    def x(self, q): return self._one("x", q)
    def y(self, q): return self._one("y", q)
    def z(self, q): return self._one("z", q)
    def h(self, q): return self._one("h", q)
    def s(self, q): return self._one("s", q)
    def sdg(self, q): return self._one("sdg", q)
    def t(self, q): return self._one("t", q)
    def tdg(self, q): return self._one("tdg", q)
    def sx(self, q): return self._one("sx", q)
    def rx(self, theta, q): return self._one("rx", q, (_angle(theta),))
    def ry(self, theta, q): return self._one("ry", q, (_angle(theta),))
    def rz(self, phi, q): return self._one("rz", q, (_angle(phi),))
    def p(self, lam, q): return self._one("p", q, (_angle(lam),))

    # ------------------------------------------------------------ multi-qubit gates
    def _two(self, name, a, b, params=()):
        a, b = _as_list(a), _as_list(b)
        if len(a) == 1 and len(b) > 1:
            a = a * len(b)
        if len(b) == 1 and len(a) > 1:
            b = b * len(a)
        if len(a) != len(b):
            raise CircuitError("The control and target lists must have the same length.")
        for x, y in zip(a, b):
            self._add(name, [x, y], params)
        return self

    def cx(self, control, target): return self._two("cx", control, target)
    def cnot(self, control, target): return self.cx(control, target)
    def cy(self, control, target): return self._two("cy", control, target)
    def cz(self, control, target): return self._two("cz", control, target)
    def ch(self, control, target): return self._two("ch", control, target)
    def cp(self, lam, control, target): return self._two("cp", control, target, (_angle(lam),))
    def crz(self, phi, control, target): return self._two("crz", control, target, (_angle(phi),))
    def rzz(self, theta, qubit1, qubit2):
        """exp(-i theta/2 Z(x)Z) on two qubits, as in Qiskit; QAOA uses it for each edge of a graph."""
        return self._two("rzz", qubit1, qubit2, (_angle(theta),))
    def swap(self, a, b): return self._two("swap", a, b)
    def ccx(self, c1, c2, target): return self._add("ccx", [int(c1), int(c2), int(target)])
    def ccz(self, c1, c2, target): return self._add("ccz", [int(c1), int(c2), int(target)])
    def cswap(self, control_qubit, target_qubit1, target_qubit2):
        return self._add("cswap", [int(control_qubit), int(target_qubit1), int(target_qubit2)])
    def fredkin(self, control_qubit, target_qubit1, target_qubit2):
        return self.cswap(control_qubit, target_qubit1, target_qubit2)

    def mcx(self, control_qubits, target_qubit, ancilla_qubits=None, mode=None):
        """X on target_qubit when every qubit in control_qubits is 1 (Qiskit's mcx; no ancillas are needed here)."""
        controls = [int(q) for q in _as_list(control_qubits)]
        if not controls:
            raise CircuitError("mcx needs at least one control qubit.")
        return self._add("mcx", controls + [int(target_qubit)])

    # ------------------------------------------------------------ non-unitary operations
    def barrier(self, *qubits):
        qs = [q for x in qubits for q in _as_list(x)] or self.qubits
        self.data.append(Instruction("barrier", qs))
        return self

    def reset(self, qubit):
        for q in _as_list(qubit):
            self._check_qubits([q])
            self.data.append(Instruction("reset", [q]))
        return self

    def measure(self, qubit, clbit):
        qs, cs = _as_list(qubit), _as_list(clbit)
        if len(qs) != len(cs):
            raise CircuitError("The number of qubits and classical bits to measure must match.")
        for q, c in zip(qs, cs):
            self._check_qubits([q])
            self._check_clbits([c])
            self.data.append(Instruction("measure", [q], [c]))
        return self

    def measure_all(self):
        """Add a barrier and a new classical register 'meas', then measure every qubit into it."""
        start = self.num_clbits
        self._cregs.append(("meas", self.num_qubits))
        self.barrier()
        for q in range(self.num_qubits):
            self.data.append(Instruction("measure", [q], [start + q]))
        return self

    def if_test(self, condition):
        """Classical feedforward, as in Qiskit 2: `with qc.if_test((clbit, value)):` runs the block's
        gates only in shots where that classical bit was measured as value (0 or 1)."""
        if not (isinstance(condition, tuple) and len(condition) == 2):
            raise CircuitError("if_test needs a condition (clbit, value), for example qc.if_test((0, 1)).")
        clbit, value = condition
        if not isinstance(clbit, (int, np.integer)):
            raise CircuitError("In this course the condition is one classical bit: qc.if_test((clbit, value)).")
        self._check_clbits([int(clbit)])
        if int(value) not in (0, 1):
            raise CircuitError("A single classical bit can only be compared with 0 or 1.")
        return _IfContext(self, (int(clbit), int(value)))

    # ------------------------------------------------------------ parameters
    @property
    def parameters(self):
        """The circuit's unbound parameters, sorted by name (vector elements by index), as in Qiskit."""
        found = {}
        for inst in self.data:
            for x in inst.params:
                if isinstance(x, ParameterExpression):
                    for p in x.parameters:
                        found[id(p)] = p
        return sorted(found.values(), key=_param_key)

    @property
    def num_parameters(self):
        return len(self.parameters)

    def assign_parameters(self, parameters, inplace=False, strict=True):
        """Give the parameters values: a list in the order of qc.parameters, or a dict {Parameter: value}
        (a ParameterVector key takes a list). Returns a new circuit, or changes this one with inplace=True."""
        params = self.parameters
        if isinstance(parameters, dict):
            values = {}
            for k, v in parameters.items():
                if isinstance(k, ParameterVector):
                    v = list(np.ravel(v))
                    if len(v) != len(k):
                        raise ValueError(f"ParameterVector {k.name} has length {len(k)}, but {len(v)} values were given.")
                    values.update(zip(k.params, v))
                else:
                    values[k] = v
            missing = [k for k in values if not any(k is p for p in params)]
            if missing and strict:
                names = ", ".join(str(k) for k in missing)
                raise CircuitError(f"'Cannot bind parameters ({names}) not present in the circuit.'")
        else:
            vals = list(np.ravel(np.asarray(parameters, dtype=float)))
            if len(vals) != len(params):
                raise ValueError("Mismatching number of values and parameters. For partial binding please pass a "
                                 "mapping of {parameter: value} pairs.")
            values = dict(zip(params, vals))
        target = self if inplace else self.copy()
        new = []
        for inst in target.data:
            if any(isinstance(x, ParameterExpression) for x in inst.params):
                ps = tuple(x.bind(values) if isinstance(x, ParameterExpression) else x for x in inst.params)
                inst = Instruction(inst.name, inst.qubits, inst.clbits, ps)
            new.append(inst)
        target.data = new
        return None if inplace else target

    # ------------------------------------------------------------ circuit tools
    def append(self, inst):
        self.data.append(inst)
        return self

    def copy(self):
        new = self.copy_empty_like()
        new.data = list(self.data)
        return new

    def copy_empty_like(self):
        new = QuantumCircuit.__new__(QuantumCircuit)
        new.num_qubits = self.num_qubits
        new._cregs = list(self._cregs)
        new.data = []
        new.global_phase = self.global_phase
        return new

    def count_ops(self):
        return dict(Counter(i.name for i in self.data))

    def depth(self):
        level = [0] * self.num_qubits
        for i in self.data:
            if i.name == "barrier":
                continue
            d = max(level[q] for q in i.qubits) + 1
            for q in i.qubits:
                level[q] = d
        return max(level) if level else 0

    def _final_measure_indices(self):
        """Indices of measurements that are the last operation on their qubit (barriers aside)."""
        final, done = [], set()
        for idx in range(len(self.data) - 1, -1, -1):
            inst = self.data[idx]
            if inst.name == "barrier":
                continue
            if inst.name == "measure" and inst.qubits[0] not in done:
                final.append(idx)
                done.add(inst.qubits[0])
                continue
            done.update(inst.qubits)
            if len(done) == self.num_qubits:
                break
        return set(final)

    def remove_final_measurements(self, inplace=True):
        """Remove the last measurement on each qubit, and barriers that follow them."""
        final = self._final_measure_indices()
        keep = [inst for k, inst in enumerate(self.data) if k not in final]
        while keep and keep[-1].name == "barrier":
            keep.pop()
        target = self if inplace else self.copy()
        target.data = keep
        if not any(i.name == "measure" for i in keep):
            target._cregs = []
        return None if inplace else target

    def inverse(self):
        inv = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
        new = self.copy_empty_like()
        for i in reversed(self.data):
            if i.name in ("measure", "reset", "if_else"):
                raise CircuitError("A circuit with measurements, resets or if_test blocks has no inverse.")
            if i.name in _PARAM or i.name in ("cp", "crz", "rzz"):
                new.data.append(Instruction(i.name, i.qubits, (), (-i.params[0],)))
            elif i.name == "sx":
                new.data.append(Instruction("rx", i.qubits, (), (-math.pi / 2,)))
            else:
                new.data.append(Instruction(inv.get(i.name, i.name), i.qubits))
        new.global_phase = -self.global_phase
        return new

    def compose(self, other, qubits=None, clbits=None, front=False, inplace=False):
        """Add another circuit's instructions, as in Qiskit: other's qubit i acts on qubits[i] of this circuit.

        Returns a new circuit, or changes this one and returns None with inplace=True.
        """
        if not isinstance(other, QuantumCircuit):
            raise CircuitError("compose() needs a QuantumCircuit.")
        if other.num_qubits > self.num_qubits or other.num_clbits > self.num_clbits:
            raise CircuitError("Trying to compose with another QuantumCircuit which has more 'in' edges.")
        qmap = list(range(other.num_qubits)) if qubits is None else _as_list(qubits)
        if len(qmap) != other.num_qubits:
            raise CircuitError(f"Number of items in qubits parameter ({len(qmap)}) does not match number of "
                               f"qubits in the circuit ({other.num_qubits}).")
        if len(set(qmap)) != len(qmap):
            raise CircuitError("Duplicate qubits in the qubits parameter.")
        self._check_qubits(qmap)
        cmap = list(range(other.num_clbits)) if clbits is None else _as_list(clbits)
        if len(cmap) != other.num_clbits:
            raise CircuitError(f"Number of items in clbits parameter ({len(cmap)}) does not match number of "
                               f"clbits in the circuit ({other.num_clbits}).")
        if cmap:
            self._check_clbits(cmap)
        added = [i.remap(qmap, cmap) if isinstance(i, IfElseOp) else
                 Instruction(i.name, [qmap[q] for q in i.qubits], [cmap[c] for c in i.clbits], i.params)
                 for i in other.data]
        target = self if inplace else self.copy()
        target.data = added + list(target.data) if front else list(target.data) + added
        target.global_phase = target.global_phase + other.global_phase
        return None if inplace else target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return str(self.draw())

    def draw(self, output="text", **_):
        return _TextDrawing(self)


# ---------------------------------------------------------------- text drawing
class _TextDrawing:
    def __init__(self, qc):
        self.qc = qc

    def _repr_html_(self):
        import html
        return f'<pre style="line-height: 1.2; font-family: monospace;">{html.escape(str(self))}</pre>'

    def __repr__(self):
        return str(self)

    def __str__(self):
        qc = self.qc
        n = qc.num_qubits
        names = [f"q_{q}: " for q in range(n)]
        width = max(len(s) for s in names)
        rows = [s.rjust(width) for s in names]
        gaps = [" " * width for _ in range(n)]          # rows between qubit wires
        has_c = qc.num_clbits > 0
        crow = ("c: ".rjust(width) + "") if has_c else None
        if has_c:
            crow = f"c: {qc.num_clbits}/".rjust(width)

        def label(inst):
            if inst.name in _PARAM:
                return f"{inst.name.upper()}({_fmt_param(inst.params[0])})"
            if inst.name == "if_else":
                c, v = inst.condition
                return f"IF c{c}={v}" + (" / ELSE" if inst.false_body else "")
            return {"measure": "M", "reset": "|0>", "sdg": "Sdg", "tdg": "Tdg", "sx": "√X"}.get(
                inst.name, inst.name.upper())

        for inst in qc.data:
            qs = inst.qubits
            if inst.name == "barrier":
                col = ["░" if q in qs else "─" for q in range(n)]
                w = 1
            elif inst.name in ("cx", "cy", "cz", "ch", "cp", "crz", "ccx", "ccz", "mcx", "swap", "cswap", "rzz"):
                lo, hi = min(qs), max(qs)
                tgt = qs[-1]
                if inst.name == "swap":
                    marks = {qs[0]: "X", qs[1]: "X"}
                elif inst.name == "cswap":
                    marks = {qs[0]: "■", qs[1]: "X", qs[2]: "X"}
                elif inst.name in ("cz", "ccz"):
                    marks = {q: "■" for q in qs}
                elif inst.name == "rzz":
                    marks = {qs[0]: "■", qs[1]: f"ZZ({_fmt_param(inst.params[0])})"}
                else:
                    sym = {"cx": "X", "ccx": "X", "mcx": "X", "cy": "Y", "ch": "H",
                           "cp": f"P({_fmt_param(inst.params[0])})" if inst.params else "P",
                           "crz": f"RZ({_fmt_param(inst.params[0])})" if inst.params else "RZ"}[inst.name]
                    marks = {q: "■" for q in qs[:-1]}
                    marks[tgt] = "⊕" if inst.name in ("cx", "ccx", "mcx") else sym
                w = max(len(m) for m in marks.values())
                col = []
                for q in range(n):
                    if q in marks:
                        col.append(marks[q].center(w, "─"))
                    elif lo < q < hi:
                        col.append("┼".center(w, "─"))
                    else:
                        col.append("─" * w)
            else:
                lab = label(inst)
                w = len(lab) + 2
                col = [("[" + lab + "]") if q in qs else "─" * w for q in range(n)]
            for q in range(n):
                rows[q] += "─" + col[q] + "─"
            vert = set()
            if inst.name in ("cx", "cy", "cz", "ch", "cp", "crz", "ccx", "ccz", "mcx", "swap", "cswap", "rzz"):
                vert = set(range(min(qs), max(qs)))
            for q in range(n):
                mid = ("│".center(w)) if q in vert else " " * w
                gaps[q] += " " + mid + " "
            if has_c:
                if inst.name == "measure":
                    crow += "═" + f"{inst.clbits[0]}".center(w, "═") + "═"
                else:
                    crow += "═" * (w + 2)
        lines = []
        for q in range(n):
            lines.append(rows[q] + "─")
            if q < n - 1:
                lines.append(gaps[q].rstrip())
        if has_c:
            lines.append(crow + "═")
        return "\n".join(lines)


# ---------------------------------------------------------------- linear algebra core
def _apply_unitary_vec(psi, u, qubits, n):
    """Apply a k-qubit unitary to a state vector. Qubit q is tensor axis n-1-q."""
    k = len(qubits)
    t = psi.reshape([2] * n)
    axes = [n - 1 - q for q in qubits]
    # The gate's own index is little-endian over 'qubits': gate qubit j is bit j.
    ug = u.reshape([2] * (2 * k))
    # ug axes: output bits (k-1 .. 0), input bits (k-1 .. 0)
    in_axes = list(range(2 * k - 1, k - 1, -1))      # input bit j sits at axis 2k-1-j
    t = np.tensordot(ug, t, axes=(in_axes, axes))     # result: output bits k-1..0, then the rest
    # move output axes back into place
    out_positions = [n - 1 - qubits[j] for j in range(k - 1, -1, -1)]
    rest = [a for a in range(n) if a not in axes]
    order = out_positions + rest
    inv = np.argsort(order)
    return np.transpose(t, inv).reshape(-1)


def _apply_unitary_rho(rho, u, qubits, n):
    """rho -> U rho U^dagger, with U acting on the given qubits (qubits[0] is the lowest bit of U's index).

    Version 1.10.0 does this with two tensor contractions instead of one matrix-vector product per row and
    column; the result is the same to rounding (about 1e-16), and density-matrix runs are much faster.
    """
    k = len(qubits)
    t = np.asarray(rho, dtype=complex).reshape([2] * (2 * n))
    ut = np.asarray(u, dtype=complex).reshape([2] * (2 * k))     # axes: output bits, then input bits (highest first)
    row_axes = [n - 1 - q for q in reversed(qubits)]            # axis of qubit q in the row index
    col_axes = [2 * n - 1 - q for q in reversed(qubits)]        # and in the column index
    t = np.tensordot(ut, t, axes=(list(range(k, 2 * k)), row_axes))
    t = np.moveaxis(t, list(range(k)), row_axes)
    t = np.tensordot(t, ut.conj(), axes=(col_axes, list(range(k, 2 * k))))
    t = np.moveaxis(t, list(range(2 * n - k, 2 * n)), col_axes)
    return t.reshape(2 ** n, 2 ** n)


def _projector_diag(n, q, bit):
    idx = np.arange(2 ** n)
    return ((idx >> q) & 1) == bit


def _pauli_matrix(label):
    mats = {"I": _FIXED["id"], "X": _FIXED["x"], "Y": _FIXED["y"], "Z": _FIXED["z"]}
    m = np.array([[1]], dtype=complex)
    for ch in label.upper():          # leftmost letter is the highest qubit
        m = np.kron(m, mats[ch])
    return m


def _as_operator(op, n):
    if isinstance(op, SparsePauliOp):
        if op.num_qubits != n:
            raise ValueError(f"The operator acts on {op.num_qubits} qubits; the state has {n} qubits.")
        return op.to_matrix()
    if isinstance(op, Operator):
        op = op.data
    if isinstance(op, str):
        if len(op) != n:
            raise ValueError(f"The Pauli label {op!r} has {len(op)} letters; the state has {n} qubits.")
        return _pauli_matrix(op)
    m = np.asarray(op, dtype=complex)
    if m.shape != (2 ** n, 2 ** n):
        raise ValueError(f"The operator must be {2 ** n} x {2 ** n}.")
    return m


# ---------------------------------------------------------------- states
def _marginal(p, n, qargs, decimals):
    if qargs is not None:
        qargs = [int(q) for q in _as_list(qargs)]
        for q in qargs:
            if not 0 <= q < n:
                raise CircuitError(f"Index {q} out of range for size {n}.")
        if len(set(qargs)) != len(qargs):
            raise CircuitError("Duplicate qubits in qargs.")
        t = p.reshape([2] * n)                 # axis a holds qubit n - 1 - a
        axes = [n - 1 - q for q in qargs]
        keep = tuple(a for a in range(n) if a not in axes)
        t = t.sum(axis=keep) if keep else t
        order = sorted(axes)                   # remaining axes, in their old order
        perm = [order.index(n - 1 - q) for q in reversed(qargs)]
        p = np.transpose(t, perm).reshape(-1)
    if decimals is not None:
        p = np.round(p, decimals)
    return p


def _prob_dict(p, decimals):
    m = int(round(math.log2(len(p))))
    out = {}
    for i, x in enumerate(p):
        if x > 1e-12:
            out[format(i, f"0{m}b")] = float(x)
    return out


class Statevector:
    """A pure state. Statevector(list of amplitudes) or Statevector(circuit without measurements)."""

    def __init__(self, data):
        if isinstance(data, QuantumCircuit):
            data = _run_unitary(data)
        v = np.asarray(data, dtype=complex).reshape(-1)
        n = int(round(math.log2(len(v)))) if len(v) else 0
        if len(v) < 2 or 2 ** n != len(v):
            raise ValueError("A state vector needs 2, 4, 8, ... amplitudes.")
        self.data = v
        self.num_qubits = n

    @classmethod
    def from_label(cls, label):
        n = len(label)
        v = np.zeros(2 ** n, dtype=complex)
        v[int(label, 2)] = 1
        return cls(v)

    def probabilities(self, qargs=None, decimals=None):
        """Measurement probabilities, as in Qiskit. With qargs, only those qubits are measured:
        qargs[0] is the least significant bit of the result's index."""
        return _marginal(np.abs(self.data) ** 2, self.num_qubits, qargs, decimals)

    def probabilities_dict(self, qargs=None, decimals=None):
        """Probabilities keyed by result strings such as "011" (qargs[0], or qubit 0, on the right)."""
        return _prob_dict(self.probabilities(qargs, decimals), decimals)

    def evolve(self, other):
        if isinstance(other, QuantumCircuit):
            return Statevector(_run_unitary(other, self.data))
        return Statevector(np.asarray(other, dtype=complex) @ self.data)

    def expectation_value(self, op):
        m = _as_operator(op, self.num_qubits)
        return complex(np.vdot(self.data, m @ self.data))

    def inner(self, other):
        return complex(np.vdot(self.data, Statevector(other).data if not isinstance(other, Statevector) else other.data))

    def is_valid(self, atol=1e-8):
        return abs(np.linalg.norm(self.data) - 1) < atol

    def draw(self, output="text"):
        return str(self)

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        try:
            return abs(abs(np.vdot(self.data, Statevector(other).data)) - 1) < 1e-8 and \
                np.allclose(self.data, Statevector(other).data, atol=1e-8)
        except Exception:
            return False

    def __repr__(self):
        amps = ", ".join(_fmt_complex(a) for a in self.data)
        return f"Statevector([{amps}])"


class DensityMatrix:
    """A state that may be mixed. DensityMatrix(Statevector), DensityMatrix(matrix) or DensityMatrix(circuit)."""

    def __init__(self, data):
        if isinstance(data, QuantumCircuit):
            data = simulate_density_matrix(data).data
        if isinstance(data, Statevector):
            data = np.outer(data.data, data.data.conj())
        m = np.asarray(data, dtype=complex)
        if m.ndim == 1:
            m = np.outer(m, m.conj())
        n = int(round(math.log2(m.shape[0])))
        if m.shape != (2 ** n, 2 ** n):
            raise ValueError("A density matrix must be 2^n x 2^n.")
        self.data = m
        self.num_qubits = n

    def probabilities(self, qargs=None, decimals=None):
        """Measurement probabilities, as in Qiskit; qargs selects and orders the measured qubits."""
        return _marginal(np.real(np.diag(self.data)).clip(min=0), self.num_qubits, qargs, decimals)

    def probabilities_dict(self, qargs=None, decimals=None):
        return _prob_dict(self.probabilities(qargs, decimals), decimals)

    def evolve(self, other):
        if isinstance(other, QuantumCircuit):
            r = self.data
            for inst in other.data:
                if inst.name == "barrier":
                    continue
                if inst.name in ("measure", "reset"):
                    raise CircuitError("evolve() takes a circuit of gates only.")
                r = _apply_unitary_rho(r, inst.matrix(), list(inst.qubits), self.num_qubits)
            return DensityMatrix(r)
        u = np.asarray(other, dtype=complex)
        return DensityMatrix(u @ self.data @ u.conj().T)

    def expectation_value(self, op):
        m = _as_operator(op, self.num_qubits)
        return complex(np.trace(m @ self.data))

    def purity(self):
        return float(np.real(np.trace(self.data @ self.data)))

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def __repr__(self):
        return "DensityMatrix(\n" + np.array2string(np.round(self.data, 4), precision=4) + ")"


# ---------------------------------------------------------------- operators
_LABEL_1Q = {"I": "id", "X": "x", "Y": "y", "Z": "z", "H": "h", "S": "s", "T": "t"}


class Operator:
    """A matrix, as in qiskit.quantum_info.Operator.

    Operator(circuit without measurements) gives the circuit's unitary matrix, in Qiskit's qubit
    order. Operator(matrix) wraps a square matrix. The matrix is in .data.
    """

    def __init__(self, data):
        if isinstance(data, Operator):
            m = data.data.copy()
        elif isinstance(data, SparsePauliOp):
            m = data.to_matrix()
        elif isinstance(data, QuantumCircuit):
            n = data.num_qubits
            dim = 2 ** n
            cols = []
            for j in range(dim):
                e = np.zeros(dim, dtype=complex)
                e[j] = 1
                try:
                    cols.append(_run_unitary(data, e))
                except CircuitError:
                    raise CircuitError(
                        "Cannot make an Operator from a circuit with measurements or resets. "
                        "Remove them first, for example with qc.remove_final_measurements().") from None
            m = np.array(cols).T
        else:
            m = np.array(data, dtype=complex)
            if m.ndim != 2 or m.shape[0] != m.shape[1]:
                raise ValueError("An Operator needs a square matrix.")
        n = int(round(math.log2(m.shape[0]))) if m.shape[0] else 0
        if m.shape[0] < 2 or 2 ** n != m.shape[0]:
            raise ValueError("An Operator acts on qubits, so it must be 2 x 2, 4 x 4, 8 x 8, ...")
        self.data = m
        self.num_qubits = n

    @classmethod
    def from_label(cls, label):
        """Operator.from_label("X"), "HZ", ... Letters I, X, Y, Z, H, S, T; qubit 0 is on the right."""
        m = np.array([[1]], dtype=complex)
        for ch in label:
            if ch not in _LABEL_1Q:
                raise ValueError(f"Unknown label letter {ch!r}; use I, X, Y, Z, H, S or T.")
            m = np.kron(m, _FIXED[_LABEL_1Q[ch]])
        return cls(m)

    @property
    def dim(self):
        return (self.data.shape[1], self.data.shape[0])

    def adjoint(self):
        """The conjugate transpose, U-dagger: the inverse of a unitary."""
        return Operator(self.data.conj().T)

    def conjugate(self):
        return Operator(self.data.conj())

    def transpose(self):
        return Operator(self.data.T)

    def compose(self, other, front=False):
        """a.compose(b) applies a first, then b (the matrix b @ a), as in Qiskit."""
        b = Operator(other).data
        return Operator(self.data @ b if front else b @ self.data)

    def dot(self, other):
        """a.dot(b) is the matrix product a @ b (b is applied first)."""
        return Operator(self.data @ Operator(other).data)

    def power(self, k):
        return Operator(np.linalg.matrix_power(self.data, int(k)))

    def is_unitary(self, atol=1e-8):
        d = self.data
        return bool(np.allclose(d.conj().T @ d, np.eye(d.shape[0]), atol=atol))

    def equiv(self, other, atol=1e-8):
        """True if the two operators are equal up to a global phase."""
        try:
            b = Operator(other).data
        except (ValueError, TypeError):
            return False
        a = self.data
        if a.shape != b.shape:
            return False
        k = int(np.argmax(np.abs(b)))
        i, j = divmod(k, b.shape[1])
        if abs(b[i, j]) < atol or abs(a[i, j]) < atol:
            return bool(np.allclose(a, b, atol=atol))
        phase = a[i, j] / b[i, j]
        if abs(abs(phase) - 1) > 1e-6:
            return False
        return bool(np.allclose(a, phase * b, atol=atol))

    def __eq__(self, other):
        try:
            b = Operator(other).data
        except (ValueError, TypeError):
            return False
        return self.data.shape == b.shape and bool(np.allclose(self.data, b, rtol=1e-5, atol=1e-8))

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def __repr__(self):
        rows = ",\n          ".join("[" + ", ".join(_fmt_complex(x) for x in r) + "]" for r in self.data)
        dims = tuple([2] * self.num_qubits)
        return f"Operator([{rows}],\n         input_dims={dims}, output_dims={dims})"


class _PauliList(list):
    """The Pauli labels of a SparsePauliOp (a plain list of strings that prints like Qiskit's PauliList)."""

    def __repr__(self):
        return "PauliList([" + ", ".join(repr(x) for x in self) + "])"

    def __str__(self):
        return "[" + ", ".join(repr(x) for x in self) + "]"


class SparsePauliOp:
    """An operator written as a sum of Pauli strings with coefficients, as in qiskit.quantum_info.SparsePauliOp.

    SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)]) is ZZ + 0.5 XI. In a label, qubit 0 is the rightmost letter.
    """

    def __init__(self, data, coeffs=None):
        if isinstance(data, SparsePauliOp):
            labels, c = list(data.paulis), data.coeffs.copy()
        else:
            labels = [data] if isinstance(data, str) else [str(x) for x in data]
            c = None
        if not labels:
            raise ValueError("A SparsePauliOp needs at least one Pauli label.")
        for lab in labels:
            if not lab or any(ch not in "IXYZ" for ch in lab):
                raise ValueError(f"{lab!r} is not a Pauli label: use only the letters I, X, Y and Z.")
        if len({len(lab) for lab in labels}) != 1:
            raise ValueError("All Pauli labels must have the same number of letters (one per qubit).")
        if coeffs is not None:
            c = np.asarray(coeffs, dtype=complex).reshape(-1)
        elif c is None:
            c = np.ones(len(labels), dtype=complex)
        if len(c) != len(labels):
            raise ValueError(f"{len(labels)} Pauli labels but {len(c)} coefficients.")
        self._labels = labels
        self.coeffs = c

    @classmethod
    def from_list(cls, obj, num_qubits=None):
        obj = list(obj)
        if not obj:
            if num_qubits is None:
                raise ValueError("from_list needs at least one term, or num_qubits.")
            return cls(["I" * num_qubits], [0])
        return cls([lab for lab, _ in obj], [c for _, c in obj])

    @classmethod
    def from_sparse_list(cls, obj, num_qubits):
        """Terms (letters, qubits, coefficient): ("ZZ", [0, 2], 1.0) puts Z on qubits 0 and 2."""
        labels, coeffs = [], []
        for letters, qubits, c in obj:
            if len(letters) != len(qubits):
                raise ValueError(f"{letters!r} has {len(letters)} letters but {len(qubits)} qubit indices.")
            lab = ["I"] * num_qubits
            for ch, q in zip(letters, qubits):
                if not 0 <= q < num_qubits:
                    raise ValueError(f"Qubit index {q} is out of range for {num_qubits} qubits.")
                lab[num_qubits - 1 - q] = ch
            labels.append("".join(lab))
            coeffs.append(c)
        if not labels:
            return cls(["I" * num_qubits], [0])
        return cls(labels, coeffs)

    @property
    def num_qubits(self):
        return len(self._labels[0])

    @property
    def paulis(self):
        return _PauliList(self._labels)

    @property
    def size(self):
        return len(self._labels)

    def __len__(self):
        return len(self._labels)

    def to_list(self):
        return [(lab, complex(c)) for lab, c in zip(self._labels, self.coeffs)]

    def to_matrix(self, sparse=False):
        dim = 2 ** self.num_qubits
        m = np.zeros((dim, dim), dtype=complex)
        for lab, c in zip(self._labels, self.coeffs):
            m += c * _pauli_matrix(lab)
        return m

    def to_operator(self):
        return Operator(self.to_matrix())

    def simplify(self, atol=1e-8):
        """Add up repeated Pauli strings and drop terms whose coefficient is about zero."""
        acc = {}
        for lab, c in zip(self._labels, self.coeffs):
            acc[lab] = acc.get(lab, 0) + c
        keep = [(lab, c) for lab, c in acc.items() if abs(c) > atol]
        if not keep:
            return SparsePauliOp(["I" * self.num_qubits], [0])
        return SparsePauliOp([lab for lab, _ in keep], [c for _, c in keep])

    def _check(self, other):
        if not isinstance(other, SparsePauliOp):
            raise TypeError("Both operands must be SparsePauliOp.")
        if other.num_qubits != self.num_qubits:
            raise ValueError(f"Cannot add operators on {self.num_qubits} and {other.num_qubits} qubits.")

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return SparsePauliOp(self)
        self._check(other)
        return SparsePauliOp(self._labels + list(other.paulis), np.concatenate([self.coeffs, other.coeffs]))

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:     # so that sum([...]) works
            return SparsePauliOp(self)
        return self.__add__(other)

    def __neg__(self):
        return SparsePauliOp(self._labels, -self.coeffs)

    def __sub__(self, other):
        self._check(other)
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, SparsePauliOp):
            raise TypeError("Use a number to scale a SparsePauliOp; products of operators are not needed here.")
        return SparsePauliOp(self._labels, self.coeffs * complex(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return SparsePauliOp(self._labels, self.coeffs / complex(other))

    def equiv(self, other, atol=1e-8):
        """True if the two operators have the same matrix (the order of the terms does not matter)."""
        if not isinstance(other, SparsePauliOp) or other.num_qubits != self.num_qubits:
            return False
        return bool(np.allclose(self.to_matrix(), other.to_matrix(), atol=atol))

    def __eq__(self, other):
        """True if the labels and coefficients are the same, in the same order (as in Qiskit; see equiv)."""
        return (isinstance(other, SparsePauliOp) and self._labels == list(other.paulis)
                and np.allclose(self.coeffs, other.coeffs))

    __hash__ = None

    def __repr__(self):
        return (f"SparsePauliOp({self._labels!r},\n              coeffs="
                + np.array2string(self.coeffs, separator=", ") + ")")


# ---------------------------------------------------------------- the Estimator primitive
class _DataBin:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def keys(self):
        return list(self.__dict__)

    def __repr__(self):
        return "DataBin(" + ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items()) + ")"


class _PubResult:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

    def __repr__(self):
        return f"PubResult(data={self.data!r}, metadata={self.metadata!r})"


class _PrimitiveResult(list):
    def __init__(self, items, metadata=None):
        list.__init__(self, items)
        self.metadata = metadata or {"version": 2}

    def __repr__(self):
        return f"PrimitiveResult([{', '.join(repr(x) for x in self)}], metadata={self.metadata!r})"


class _PrimitiveJob:
    _count = 0

    def __init__(self, result):
        _PrimitiveJob._count += 1
        self._result = result
        self._id = f"qsim-job-{_PrimitiveJob._count}"

    def result(self):
        return self._result

    def job_id(self):
        return self._id

    def status(self):
        return "DONE"

    def done(self):
        return True


def _obs_array(obs):
    """Observables of a PUB as an object array (a single observable gives shape ())."""
    def one(o):
        if isinstance(o, SparsePauliOp):
            return o
        if isinstance(o, str):
            return SparsePauliOp(o)
        if isinstance(o, dict):
            return SparsePauliOp.from_list(list(o.items()))
        raise TypeError("An observable must be a SparsePauliOp or a Pauli label such as 'ZZ'.")

    def walk(o):
        if isinstance(o, (list, tuple)):
            return [walk(x) for x in o]
        return one(o)

    w = walk(obs)
    if not isinstance(w, list):
        a = np.empty((), dtype=object)
        a[()] = w
        return a
    shape, x = [], w
    while isinstance(x, list):
        shape.append(len(x))
        x = x[0] if x else None
    a = np.empty(tuple(shape), dtype=object)
    for idx in np.ndindex(*shape):
        v = w
        for i in idx:
            v = v[i]
        a[idx] = v
    return a


class StatevectorEstimator:
    """Exact expectation values, as qiskit.primitives.StatevectorEstimator.

    estimator.run([(circuit, observable, parameter_values)]).result()[0].data.evs gives <psi|H|psi>.
    observable can be a list (one value each); parameter_values can be one list of values or a 2-D array
    (one row per set). With precision > 0, normal noise of that size is added, as in Qiskit.
    """

    def __init__(self, *, default_precision=0.0, seed=None):
        self.default_precision = float(default_precision)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def run(self, pubs, *, precision=None):
        if isinstance(pubs, tuple) or isinstance(pubs, QuantumCircuit):
            raise ValueError("run() takes a list of PUBs: estimator.run([(circuit, observable, values)]).")
        out = []
        for pub in pubs:
            if not isinstance(pub, (tuple, list)) or len(pub) < 2:
                raise ValueError("Each PUB is a tuple (circuit, observables) or (circuit, observables, parameter_values).")
            circ, obs = pub[0], pub[1]
            vals = pub[2] if len(pub) > 2 else None
            prec = pub[3] if len(pub) > 3 and pub[3] is not None else (self.default_precision if precision is None else precision)
            if not isinstance(circ, QuantumCircuit):
                raise TypeError("The first item of a PUB must be a QuantumCircuit.")
            oa = _obs_array(obs)
            for o in oa.flat:
                if o.num_qubits != circ.num_qubits:
                    raise ValueError(f"The observable has {o.num_qubits} qubits but the circuit has {circ.num_qubits}.")
            npar = circ.num_parameters
            if vals is None:
                if npar:
                    raise ValueError(f"The circuit has {npar} parameters; give their values as the third item of the PUB.")
                va = np.zeros((0,))
                pshape = ()
            else:
                va = np.asarray(vals, dtype=float)
                if va.ndim == 0:
                    va = va.reshape(1)
                if va.shape[-1] != npar:
                    raise ValueError(f"The circuit has {npar} parameters, but each set of values has {va.shape[-1]}.")
                pshape = va.shape[:-1]
            try:
                shape = np.broadcast_shapes(oa.shape, pshape)
            except ValueError:
                raise ValueError(f"The observables (shape {oa.shape}) and parameter values (shape {pshape}) "
                                 "cannot be broadcast together.") from None
            mats = {}
            evs = np.zeros(shape)
            states = {}
            for idx in np.ndindex(*shape) if shape else [()]:
                oi = tuple(i if s > 1 else 0 for i, s in zip(idx[len(idx) - oa.ndim:], oa.shape)) if oa.ndim else ()
                pi = tuple(i if s > 1 else 0 for i, s in zip(idx[len(idx) - len(pshape):], pshape)) if pshape else ()
                if pi not in states:
                    bound = circ.assign_parameters(va[pi]) if npar else circ
                    states[pi] = _run_unitary(bound)
                o = oa[oi]
                if id(o) not in mats:
                    mats[id(o)] = o.to_matrix()
                psi = states[pi]
                evs[idx] = float(np.real(np.vdot(psi, mats[id(o)] @ psi)))
            if prec > 0:
                evs = evs + self._rng.normal(0.0, prec, size=evs.shape)
            stds = np.full(shape, float(prec))
            data = _DataBin(evs=evs if shape else np.asarray(evs), stds=stds if shape else np.asarray(stds))
            out.append(_PubResult(data, {"target_precision": float(prec), "circuit_metadata": {}}))
        return _PrimitiveJob(_PrimitiveResult(out))


def _fmt_complex(a):
    a = complex(a)
    re, im = round(a.real, 4), round(a.imag, 4)
    re = 0.0 if re == 0 else re
    im = 0.0 if im == 0 else im
    if im == 0:
        return f"{re:g}+0j"
    return f"{re:g}{im:+g}j"


def partial_trace(state, qargs):
    """The reduced state after tracing out the qubits in qargs, as qiskit.quantum_info.partial_trace.

    state is a Statevector or DensityMatrix (or its array); the result is a DensityMatrix on the
    qubits that remain, in their original order (qubit 0 still the least significant).
    """
    rho = np.asarray(state, dtype=complex)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    n = int(round(math.log2(rho.shape[0])))
    out = sorted({int(q) for q in _as_list(qargs)})
    for q in out:
        if not 0 <= q < n:
            raise CircuitError(f"Index {q} out of range for a state of {n} qubits.")
    keep = [q for q in range(n) if q not in out]
    t = rho.reshape([2] * (2 * n))          # axes: row bits n-1..0, then column bits n-1..0
    row = {q: n - 1 - q for q in range(n)}
    letters = "abcdefghijklmnopqrstuvwxyz"
    rl = [letters[i] for i in range(n)]
    cl = [letters[n + i] for i in range(n)]
    for q in out:
        cl[row[q]] = rl[row[q]]
    kept_rows = [rl[row[q]] for q in sorted(keep, reverse=True)]
    kept_cols = [cl[row[q]] for q in sorted(keep, reverse=True)]
    expr = "".join(rl) + "".join(cl) + "->" + "".join(kept_rows) + "".join(kept_cols)
    m = len(keep)
    red = np.einsum(expr, t).reshape(2 ** m, 2 ** m) if m else np.array([[np.trace(rho)]])
    return DensityMatrix(red)


def state_fidelity(a, b):
    """Fidelity between two states (pure or mixed): 1 means identical up to a global phase."""
    def as_rho(x):
        if isinstance(x, DensityMatrix):
            return x.data, False
        if isinstance(x, Statevector):
            return x.data, True
        arr = np.asarray(x, dtype=complex)
        return (arr, arr.ndim == 1)

    da, pa = as_rho(a)
    db, pb = as_rho(b)
    if pa and pb:
        return float(abs(np.vdot(da, db)) ** 2)
    if pa:
        return float(np.real(np.vdot(da, db @ da)))
    if pb:
        return float(np.real(np.vdot(db, da @ db)))
    # both mixed: (tr sqrt(sqrt(a) b sqrt(a)))^2
    w, v = np.linalg.eigh(da)
    sa = v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.conj().T
    ev = np.linalg.eigvalsh(sa @ db @ sa)
    return float(np.sum(np.sqrt(np.clip(ev, 0, None))) ** 2)


# ---------------------------------------------------------------- simulation
def _run_unitary(qc, initial=None):
    n = qc.num_qubits
    if initial is None:
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1
    else:
        psi = np.asarray(initial, dtype=complex).copy()
    for inst in qc.data:
        if inst.name == "barrier":
            continue
        if inst.name in ("measure", "reset", "if_else"):
            raise CircuitError(
                "Cannot make a Statevector from a circuit with measurements, resets or if_test blocks. "
                "Remove them first, for example with qc.remove_final_measurements().")
        psi = _apply_unitary_vec(psi, inst.matrix(), list(inst.qubits), n)
    return psi * np.exp(1j * qc.global_phase)


def simulate_density_matrix(qc, noise_model=None):
    """The state at the end of the circuit as a density matrix.

    Measurements are applied without looking at the result (the state becomes a mixture of the
    outcomes), and resets put the qubit back to |0>. This is what happens to the quantum state
    when a circuit measures in the middle. With a noise_model, its quantum errors are applied too
    (readout errors change only the recorded results, so they do not appear here).
    """
    if any(i.name == "if_else" for i in qc.data):
        nm = noise_model if noise_model is not None and not noise_model.is_ideal() else None
        return DensityMatrix(_branch_rho(qc, nm))
    if noise_model is not None and not noise_model.is_ideal():
        return DensityMatrix(_noisy_rho(qc, noise_model, qc.data))
    n = qc.num_qubits
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1
    for inst in qc.data:
        if inst.name == "barrier":
            continue
        q = inst.qubits[0] if inst.qubits else None
        if inst.name == "measure":
            keep = _projector_diag(n, q, 0)
            mask = np.equal.outer(keep, keep)
            rho = np.where(mask, rho, 0)
        elif inst.name == "reset":
            p0 = _projector_diag(n, q, 0)
            idx = np.arange(2 ** n)
            flip = idx ^ (1 << q)
            new = np.where(np.outer(p0, p0), rho, 0)
            moved = rho[np.ix_(flip, flip)]
            new = new + np.where(np.outer(p0, p0), moved, 0)
            rho = new
        else:
            rho = _apply_unitary_rho(rho, inst.matrix(), list(inst.qubits), n)
    return DensityMatrix(rho)


# ---------------------------------------------------------------- noise models (as in qiskit_aer.noise)
class QuantumError:
    """A noise channel on k qubits, stored as Kraus matrices (like qiskit_aer.noise.QuantumError).

    The channel maps rho to sum_i K_i rho K_i^dagger. Qubit 0 of the error is the rightmost
    tensor factor, as in Qiskit.
    """

    def __init__(self, kraus_ops):
        ops = [np.asarray(k, dtype=complex) for k in kraus_ops]
        if not ops:
            raise ValueError("A QuantumError needs at least one Kraus matrix.")
        dim = ops[0].shape[0]
        n = int(round(math.log2(dim)))
        if any(k.shape != (2 ** n, 2 ** n) for k in ops):
            raise ValueError("All Kraus matrices must be square and the same size, 2^k x 2^k.")
        total = sum(k.conj().T @ k for k in ops)
        if not np.allclose(total, np.eye(dim), atol=1e-8):
            raise ValueError("The Kraus matrices do not preserve probability (sum of K^dagger K is not the identity).")
        keep = [k for k in ops if np.linalg.norm(k) > 1e-15]
        self.kraus = keep or ops
        self.num_qubits = n

    def compose(self, other):
        """This error, then the other one, on the same qubits."""
        if other.num_qubits != self.num_qubits:
            raise ValueError("compose() needs two errors on the same number of qubits.")
        return QuantumError([b @ a for a in self.kraus for b in other.kraus])

    def tensor(self, other):
        """self on the higher qubits and other on the lower ones (self ⊗ other), as in Qiskit."""
        return QuantumError([np.kron(a, b) for a in self.kraus for b in other.kraus])

    def expand(self, other):
        """other ⊗ self."""
        return other.tensor(self)

    def to_superop_matrix(self):
        return sum(np.kron(k.conj(), k) for k in self.kraus)

    def ideal(self):
        d = self.kraus[0].shape[0]
        return np.allclose(self.to_superop_matrix(), np.eye(d * d), atol=1e-12)

    def __repr__(self):
        return f"QuantumError on {self.num_qubits} qubit(s) with {len(self.kraus)} Kraus matrices"


_PAULI_1 = {"I": _FIXED["id"], "X": _FIXED["x"], "Y": _FIXED["y"], "Z": _FIXED["z"]}


def pauli_error(noise_ops):
    """pauli_error([("X", 0.1), ("I", 0.9)]): apply each Pauli with its probability."""
    labels = [str(lab).upper() for lab, _ in noise_ops]
    probs = [float(p) for _, p in noise_ops]
    if any(p < -1e-12 for p in probs) or abs(sum(probs) - 1) > 1e-8:
        raise ValueError("The probabilities must be at least 0 and add up to 1.")
    n = len(labels[0])
    if any(len(lab) != n or set(lab) - set("IXYZ") for lab in labels):
        raise ValueError("Use Pauli labels of equal length made of I, X, Y and Z.")
    return QuantumError([math.sqrt(max(p, 0.0)) * _pauli_matrix(lab) for lab, p in zip(labels, probs)])


def coherent_unitary_error(unitary):
    """A coherent error: the same unitary every time, as qiskit_aer.noise.coherent_unitary_error.

    Example: coherent_unitary_error(RZ(0.05)) on an idle qubit is a small unwanted Z rotation each time
    it idles (a qubit frequency slightly off), the error that dynamical decoupling removes.
    """
    u = np.asarray(unitary, dtype=complex)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or not np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-8):
        raise ValueError("Input matrix is not unitary.")
    return QuantumError([u])


def depolarizing_error(param, num_qubits):
    """Depolarizing channel: rho -> (1 - param) rho + param * I / 2^n, as in qiskit_aer.noise."""
    n = int(num_qubits)
    lam = float(param)
    num_terms = 4 ** n
    if lam < 0 or lam > num_terms / (num_terms - 1) + 1e-12:
        raise ValueError(f"Depolarizing parameter must be between 0 and {num_terms}/{num_terms - 1}.")
    import itertools
    ops = []
    for letters in itertools.product("IXYZ", repeat=n):
        lab = "".join(letters)
        p = lam / num_terms + (1 - lam if lab == "I" * n else 0.0)
        ops.append((lab, p))
    err = pauli_error(ops)
    err._depolarizing = lam          # lets density-matrix runs use the short formula (version 1.10.0)
    return err


def amplitude_damping_error(param_amp, excited_state_population=0):
    """Energy loss: |1> decays to |0> with probability param_amp."""
    g = float(param_amp)
    if excited_state_population:
        raise ValueError("This course's simulator supports excited_state_population = 0 only.")
    if not 0 <= g <= 1:
        raise ValueError("param_amp must be between 0 and 1.")
    return QuantumError([np.array([[1, 0], [0, math.sqrt(1 - g)]]), np.array([[0, math.sqrt(g)], [0, 0]])])


def phase_damping_error(param_phase):
    """Loss of phase: the off-diagonal terms shrink by sqrt(1 - param_phase)."""
    lam = float(param_phase)
    if not 0 <= lam <= 1:
        raise ValueError("param_phase must be between 0 and 1.")
    return QuantumError([np.array([[1, 0], [0, math.sqrt(1 - lam)]]), np.array([[0, 0], [0, math.sqrt(lam)]])])


def thermal_relaxation_error(t1, t2, time, excited_state_population=0):
    """T1 and T2 relaxation during a time 'time' (same units as t1 and t2), as in qiskit_aer.noise.

    Populations relax as exp(-time/T1) and coherences as exp(-time/T2). Needs T2 <= 2 T1.
    """
    t1, t2, time = float(t1), float(t2), float(time)
    if excited_state_population:
        raise ValueError("This course's simulator supports excited_state_population = 0 only.")
    if t1 <= 0 or t2 <= 0 or time < 0:
        raise ValueError("T1 and T2 must be positive and the time at least 0.")
    if t2 - 2 * t1 > 1e-12 * t1:
        raise ValueError("Invalid T2: it must be at most 2 T1.")
    if time == 0:
        return QuantumError([np.eye(2)])
    gamma = 1 - math.exp(-time / t1)
    lam = 1 - math.exp(-2 * time / t2 + time / t1)
    return amplitude_damping_error(gamma).compose(phase_damping_error(min(max(lam, 0.0), 1.0)))


class ReadoutError:
    """Measurement error on one qubit: probabilities[i][j] = P(record j | the qubit was i)."""

    def __init__(self, probabilities):
        m = np.asarray(probabilities, dtype=float)
        if m.shape != (2, 2):
            raise ValueError("This course's simulator supports one-qubit readout errors: a 2 x 2 list.")
        if np.any(m < -1e-12) or not np.allclose(m.sum(axis=1), 1, atol=1e-8):
            raise ValueError("Each row of a ReadoutError must be probabilities that add up to 1.")
        self.probabilities = m
        self.number_of_qubits = 1

    def __repr__(self):
        return f"ReadoutError({self.probabilities.tolist()})"


class NoiseModel:
    """Which errors happen after which instructions, as in qiskit_aer.noise.NoiseModel.

    A quantum error added for an instruction is applied right after that instruction (for
    'measure', just before the measurement). A readout error changes the recorded bit.
    An error added for particular qubits replaces the all-qubit error on those qubits.
    """

    def __init__(self, basis_gates=None):
        self.basis_gates = list(basis_gates) if basis_gates else ["id", "rz", "sx", "cx"]
        self._all = {}
        self._local = {}
        self._ro_all = None
        self._ro_local = {}

    @staticmethod
    def _names(instructions):
        return [instructions] if isinstance(instructions, str) else [str(i) for i in instructions]

    def add_all_qubit_quantum_error(self, error, instructions):
        for name in self._names(instructions):
            self._all[name] = self._all[name].compose(error) if name in self._all else error

    def add_quantum_error(self, error, instructions, qubits):
        qs = tuple(int(q) for q in qubits)
        if len(qs) != error.num_qubits:
            raise ValueError(f"The error acts on {error.num_qubits} qubit(s) but {len(qs)} qubit(s) were given.")
        for name in self._names(instructions):
            key = (name, qs)
            self._local[key] = self._local[key].compose(error) if key in self._local else error

    def add_all_qubit_readout_error(self, error):
        if not isinstance(error, ReadoutError):
            error = ReadoutError(error)
        self._ro_all = error

    def add_readout_error(self, error, qubits):
        if not isinstance(error, ReadoutError):
            error = ReadoutError(error)
        qs = tuple(int(q) for q in qubits)
        if len(qs) != 1:
            raise ValueError("Give one qubit for a one-qubit readout error.")
        self._ro_local[qs[0]] = error

    @property
    def noise_instructions(self):
        names = set(self._all) | {k[0] for k in self._local}
        if self._ro_all is not None or self._ro_local:
            names.add("measure")
        return sorted(names)

    @property
    def noise_qubits(self):
        qs = {q for _, t in self._local for q in t} | set(self._ro_local)
        return sorted(qs)

    def is_ideal(self):
        return not (self._all or self._local or self._ro_all is not None or self._ro_local)

    def _error_for(self, inst):
        qs = tuple(inst.qubits)
        err = self._local.get((inst.name, qs), self._all.get(inst.name))
        if err is not None and err.num_qubits != len(qs):
            raise CircuitError(f"The noise model has a {err.num_qubits}-qubit error for '{inst.name}', "
                               f"which acts on {len(qs)} qubit(s).")
        return err

    def _readout_for(self, qubit):
        return self._ro_local.get(int(qubit), self._ro_all)

    def __repr__(self):
        lines = ["NoiseModel:", f"  Basis gates: {self.basis_gates}"]
        if self.is_ideal():
            return "NoiseModel: Ideal"
        lines.append(f"  Instructions with noise: {self.noise_instructions}")
        if self._all:
            lines.append(f"  All-qubits errors: {sorted(self._all)}")
        if self._local:
            lines.append("  Specific qubit errors: " + str(sorted((n, list(q)) for n, q in self._local)))
        if self._ro_all is not None:
            lines.append("  All-qubits readout error")
        if self._ro_local:
            lines.append(f"  Readout errors on qubits: {sorted(self._ro_local)}")
        return "\n".join(lines)


def _apply_kraus_rho(rho, error, qubits, n):
    lam = getattr(error, "_depolarizing", None)
    if lam is not None:
        return _depolarize_rho(rho, lam, qubits, n)
    return sum(_apply_unitary_rho(rho, k, qubits, n) for k in error.kraus)


def _depolarize_rho(rho, lam, qubits, n):
    """The depolarizing channel without its 4^k Kraus matrices: (1 - lam) rho + lam (I / 2^k) x (rho traced over the qubits)."""
    if lam == 0:
        return rho
    k = len(qubits)
    t = np.asarray(rho, dtype=complex).reshape([2] * (2 * n))
    row_axes = [n - 1 - q for q in qubits]
    col_axes = [2 * n - 1 - q for q in qubits]
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sub = list(letters[:2 * n])
    for ra, ca in zip(row_axes, col_axes):
        sub[ca] = sub[ra]
    keep = [i for i in range(2 * n) if i not in row_axes + col_axes]
    reduced = np.einsum("".join(sub) + "->" + "".join(sub[i] for i in keep), t)
    full = np.multiply.outer(reduced, np.eye(2 ** k).reshape([2] * (2 * k)))
    full = np.moveaxis(full, list(range(len(keep), 2 * n)), row_axes[::-1] + col_axes[::-1])
    return (1 - lam) * rho + lam * full.reshape(2 ** n, 2 ** n) / 2 ** k


def _noisy_rho(qc, noise_model, body):
    """Density matrix after the instructions in 'body', with the noise model's quantum errors."""
    n = qc.num_qubits
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1
    for inst in body:
        if inst.name == "barrier":
            continue
        err = noise_model._error_for(inst) if noise_model is not None else None
        if inst.name == "measure":
            if err is not None:
                rho = _apply_kraus_rho(rho, err, list(inst.qubits), n)
            q = inst.qubits[0]
            keep = _projector_diag(n, q, 0)
            rho = np.where(np.equal.outer(keep, keep), rho, 0)
            continue
        if inst.name == "reset":
            q = inst.qubits[0]
            p0 = _projector_diag(n, q, 0)
            idx = np.arange(2 ** n)
            flip = idx ^ (1 << q)
            new = np.where(np.outer(p0, p0), rho, 0)
            moved = rho[np.ix_(flip, flip)]
            rho = new + np.where(np.outer(p0, p0), moved, 0)
        else:
            rho = _apply_unitary_rho(rho, inst.matrix(), list(inst.qubits), n)
        if err is not None:
            rho = _apply_kraus_rho(rho, err, list(inst.qubits), n)
    return rho


def _branch_rho(qc, nm=None):
    """Exact final density matrix of a circuit with if_test blocks (the sum of _branch_states)."""
    final = _branch_states(qc, nm)
    total = sum(r for r, _ in final)
    return total / np.real(np.trace(total))


def _branch_states(qc, nm=None):
    """Branches (unnormalized density matrix, classical bits) at the end of a circuit.

    Each measurement splits the state into branches, one per recorded result (with readout errors
    if the noise model has them); an if_test block then acts on each branch according to that
    branch's classical bits. The result is the sum over branches, weighted by their probabilities.
    """
    n = qc.num_qubits
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1
    branches = [(rho, (0,) * qc.num_clbits)]

    def step(branches, insts):
        for inst in insts:
            if inst.name == "barrier":
                continue
            if inst.name == "if_else":
                out = []
                for r, bits in branches:
                    c, v = inst.condition
                    body = inst.true_body if bits[c] == v else inst.false_body
                    out.extend(step([(r, bits)], body))
                branches = out
                continue
            err = nm._error_for(inst) if nm is not None else None
            out = []
            for r, bits in branches:
                if inst.name == "measure":
                    q = inst.qubits[0]
                    if err is not None:
                        r = _apply_kraus_rho(r, err, [q], n)
                    ro = nm._readout_for(q) if nm is not None else None
                    for o in (0, 1):
                        keep = _projector_diag(n, q, o)
                        ro_ = np.where(np.outer(keep, keep), r, 0)
                        w = float(np.real(np.trace(ro_)))
                        if w < 1e-15:
                            continue
                        for rec in (0, 1):
                            pr = (ro.probabilities[o][rec] if ro is not None else float(rec == o))
                            if pr <= 0:
                                continue
                            nb = list(bits)
                            nb[inst.clbits[0]] = rec
                            out.append((ro_ * pr, tuple(nb)))
                    continue
                if inst.name == "reset":
                    q = inst.qubits[0]
                    p0 = _projector_diag(n, q, 0)
                    idx = np.arange(2 ** n)
                    flip = idx ^ (1 << q)
                    moved = r[np.ix_(flip, flip)]
                    r = np.where(np.outer(p0, p0), r, 0) + np.where(np.outer(p0, p0), moved, 0)
                else:
                    r = _apply_unitary_rho(r, inst.matrix(), list(inst.qubits), n)
                if err is not None:
                    r = _apply_kraus_rho(r, err, list(inst.qubits), n)
                out.append((r, bits))
            # merge branches with the same classical bits to keep the list short
            merged = {}
            for r, bits in out:
                merged[bits] = merged[bits] + r if bits in merged else r
            branches = list(merged.items())
            branches = [(r, b) for b, r in branches]
        return branches

    return step(branches, qc.data)


def _noisy_distribution(qc, nm):
    """Exact probabilities of the recorded results of a circuit whose measurements are all at the end.

    Returns (dist, meas): dist[o] is the probability of outcome o, where bit j of o is the recorded
    result of the measurement meas[j] = (qubit, clbit).
    """
    n = qc.num_qubits
    final = qc._final_measure_indices()
    body = [i for k, i in enumerate(qc.data) if k not in final]
    meas_insts = [qc.data[k] for k in sorted(final)]
    rho = _noisy_rho(qc, nm, body)
    for inst in meas_insts:
        err = nm._error_for(inst) if nm is not None else None
        if err is not None:
            rho = _apply_kraus_rho(rho, err, list(inst.qubits), n)
    probs = np.real(np.diag(rho)).clip(min=0)
    probs = probs / probs.sum()
    meas = [(inst.qubits[0], inst.clbits[0]) for inst in meas_insts]
    m = len(meas)
    idx = np.arange(2 ** n)
    code = np.zeros(2 ** n, dtype=int)
    for j, (q, _) in enumerate(meas):
        code |= ((idx >> q) & 1) << j
    dist = np.zeros(2 ** m)
    np.add.at(dist, code, probs)
    t = dist.reshape([2] * m) if m else dist
    for j, (q, _) in enumerate(meas):
        ro = nm._readout_for(q) if nm is not None else None
        if ro is None:
            continue
        ax = m - 1 - j
        t = np.moveaxis(np.tensordot(t, ro.probabilities, axes=([ax], [0])), -1, ax)
    dist = np.asarray(t).reshape(-1).clip(min=0)
    return dist / dist.sum(), meas


class _Result:
    def __init__(self, counts, shots, memory=None):
        self._counts = counts
        self.shots = shots
        self._memory = memory

    def get_counts(self, circuit=None):
        return dict(self._counts)

    def get_memory(self, circuit=None):
        if self._memory is None:
            raise CircuitError("Run with memory=True to keep each shot's result.")
        return list(self._memory)


class _Job:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def status(self):
        return "DONE"


def _format_key(bits, cregs):
    """Qiskit-style key: each register's bits with bit 0 on the right, registers joined by spaces,
    the most recently added register on the left."""
    parts, start = [], 0
    for _, size in cregs:
        reg = bits[start:start + size]
        parts.append("".join(str(b) for b in reversed(reg)))
        start += size
    return " ".join(reversed(parts))


class AerSimulator:
    """Runs circuits and returns measurement counts, like Qiskit Aer's AerSimulator."""

    def __init__(self, seed_simulator=None, method="automatic", noise_model=None, **_):
        self.seed_simulator = seed_simulator
        self.method = method
        self.noise_model = noise_model
        self.name = "qsim_simulator"

    def __repr__(self):
        return "AerSimulator('qsim_simulator'" + (", noise_model=<NoiseModel>)" if self.noise_model is not None else ")")

    def run(self, circuits, shots=1024, seed_simulator=None, memory=False, noise_model=None, **_):
        if isinstance(circuits, (list, tuple)):
            if len(circuits) != 1:
                raise CircuitError("Run one circuit at a time in this course.")
            circuits = circuits[0]
        qc = circuits
        if not isinstance(qc, QuantumCircuit):
            raise CircuitError("run() needs a QuantumCircuit.")
        if not any(i.name == "measure" for i in qc.data):
            raise CircuitError(
                "No counts for this circuit: it has no measurements. Add qc.measure(...) or qc.measure_all().")
        seed = seed_simulator if seed_simulator is not None else self.seed_simulator
        rng = np.random.default_rng(seed)
        shots = int(shots)
        nm = noise_model if noise_model is not None else self.noise_model
        if nm is not None and not nm.is_ideal():
            keys = self._run_noisy(qc, nm, rng, shots)
            counts = Counter(keys)
            return _Job(_Result(dict(sorted(counts.items())), shots, keys if memory else None))
        nc = qc.num_clbits
        final = qc._final_measure_indices()
        mid = any((i.name in ("measure", "reset") and k not in final) or i.name == "if_else"
                  for k, i in enumerate(qc.data))
        if not mid:
            body = [i for k, i in enumerate(qc.data) if k not in final]
            tmp = qc.copy_empty_like()
            tmp.data = body
            probs = np.abs(_run_unitary(tmp)) ** 2
            probs = probs / probs.sum()
            outcomes = rng.choice(len(probs), size=shots, p=probs)
            meas = [(qc.data[k].qubits[0], qc.data[k].clbits[0]) for k in sorted(final)]
            keys = []
            cache = {}
            for o in outcomes:
                if o not in cache:
                    bits = [0] * nc
                    for q, c in meas:
                        bits[c] = (int(o) >> q) & 1
                    cache[o] = _format_key(bits, qc._cregs)
                keys.append(cache[o])
        else:
            keys = [self._one_shot(qc, rng) for _ in range(shots)]
        counts = Counter(keys)
        ordered = dict(sorted(counts.items()))
        return _Job(_Result(ordered, shots, keys if memory else None))

    @staticmethod
    def _run_noisy(qc, nm, rng, shots):
        final = qc._final_measure_indices()
        mid = any((i.name in ("measure", "reset") and k not in final) or i.name == "if_else"
                  for k, i in enumerate(qc.data))
        if mid:
            return [AerSimulator._one_shot_noisy(qc, nm, rng) for _ in range(shots)]
        dist, meas = _noisy_distribution(qc, nm)
        outcomes = rng.choice(len(dist), size=shots, p=dist)
        cache, keys = {}, []
        for o in outcomes:
            if o not in cache:
                bits = [0] * qc.num_clbits
                for j, (q, c) in enumerate(meas):
                    bits[c] = (int(o) >> j) & 1
                cache[o] = _format_key(bits, qc._cregs)
            keys.append(cache[o])
        return keys

    @staticmethod
    def _one_shot_noisy(qc, nm, rng):
        n = qc.num_qubits
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1
        bits = [0] * qc.num_clbits

        def channel(psi, err, qubits):
            outs = [_apply_unitary_vec(psi, k, qubits, n) for k in err.kraus]
            ps = np.array([float(np.vdot(o, o).real) for o in outs])
            i = rng.choice(len(outs), p=ps / ps.sum())
            return outs[i] / np.linalg.norm(outs[i])

        def run(psi, insts):
            for inst in insts:
                if inst.name == "barrier":
                    continue
                if inst.name == "if_else":
                    c, v = inst.condition
                    psi = run(psi, inst.true_body if bits[c] == v else inst.false_body)
                    continue
                err = nm._error_for(inst)
                if inst.name in ("measure", "reset"):
                    q = inst.qubits[0]
                    if inst.name == "measure" and err is not None:
                        psi = channel(psi, err, [q])
                    one = _projector_diag(n, q, 1)
                    p1 = float(np.sum(np.abs(psi[one]) ** 2))
                    outcome = 1 if rng.random() < p1 else 0
                    keep = one if outcome == 1 else ~one
                    psi = np.where(keep, psi, 0)
                    psi = psi / np.linalg.norm(psi)
                    if inst.name == "measure":
                        ro = nm._readout_for(q)
                        rec = outcome
                        if ro is not None:
                            rec = 1 if rng.random() < ro.probabilities[outcome][1] else 0
                        bits[inst.clbits[0]] = rec
                    else:
                        if outcome == 1:
                            psi = _apply_unitary_vec(psi, _FIXED["x"], [q], n)
                        if err is not None:
                            psi = channel(psi, err, [q])
                else:
                    psi = _apply_unitary_vec(psi, inst.matrix(), list(inst.qubits), n)
                    if err is not None:
                        psi = channel(psi, err, list(inst.qubits))
            return psi

        run(psi, qc.data)
        return _format_key(bits, qc._cregs)

    @staticmethod
    def _one_shot(qc, rng):
        n = qc.num_qubits
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1
        bits = [0] * qc.num_clbits

        def run(psi, insts):
            for inst in insts:
                if inst.name == "barrier":
                    continue
                if inst.name == "if_else":
                    c, v = inst.condition
                    psi = run(psi, inst.true_body if bits[c] == v else inst.false_body)
                    continue
                if inst.name in ("measure", "reset"):
                    q = inst.qubits[0]
                    one = _projector_diag(n, q, 1)
                    p1 = float(np.sum(np.abs(psi[one]) ** 2))
                    outcome = 1 if rng.random() < p1 else 0
                    keep = one if outcome == 1 else ~one
                    psi = np.where(keep, psi, 0)
                    psi = psi / np.linalg.norm(psi)
                    if inst.name == "measure":
                        bits[inst.clbits[0]] = outcome
                    elif outcome == 1:
                        psi = _apply_unitary_vec(psi, _FIXED["x"], [q], n)
                else:
                    psi = _apply_unitary_vec(psi, inst.matrix(), list(inst.qubits), n)
            return psi

        run(psi, qc.data)
        return _format_key(bits, qc._cregs)


def transpile(circuits, backend=None, **_):
    """Kept for Qiskit compatibility. The simulator runs any circuit as written, so this returns it unchanged."""
    return circuits


def plot_histogram(counts, title=None, figsize=(6, 3.5)):
    """Bar chart of counts (or of a list of counts dictionaries)."""
    import matplotlib.pyplot as plt

    data = counts if isinstance(counts, (list, tuple)) else [counts]
    keys = sorted({k for d in data for k in d})
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / len(data)
    for j, d in enumerate(data):
        total = sum(d.values()) or 1
        xs = [i + (j - (len(data) - 1) / 2) * width for i in range(len(keys))]
        ys = [d.get(k, 0) for k in keys]
        bars = ax.bar(xs, ys, width=width * 0.95, color=["#99004C", "#6A626B", "#24313D"][j % 3])
        for b, y in zip(bars, ys):
            ax.annotate(str(y), (b.get_x() + b.get_width() / 2, y), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=0 if len(keys) <= 8 else 60)
    ax.set_ylabel("Counts")
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------- Bloch sphere plots
def _bloch_of(vec2):
    a, b = complex(vec2[0]), complex(vec2[1])
    n = abs(a) ** 2 + abs(b) ** 2
    if n == 0:
        raise ValueError("The zero vector has no Bloch vector.")
    ab = a.conjugate() * b / n
    return (2 * ab.real, 2 * ab.imag, (abs(a) ** 2 - abs(b) ** 2) / n)


def _draw_bloch(ax, points, title=None, elev=20.0, azim=25.0):
    """Draw a Bloch sphere on 2D axes, seen from a fixed angle, with one arrow per (x, y, z, colour).
    The view matches Qiskit's: +x points out of the page to the lower left, +y to the right, |0> up."""
    e, a = math.radians(elev), math.radians(azim)
    right = (-math.sin(a), math.cos(a), 0.0)
    up = (-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e))
    cam = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))

    def proj(x, y, z):
        p = (x, y, z)
        dot = lambda u: sum(pi * ui for pi, ui in zip(p, u))
        return (dot(right), dot(up)), dot(cam)

    t = np.linspace(0, 2 * np.pi, 241)
    ax.add_patch(__import__("matplotlib").patches.Circle((0, 0), 1, fill=True, fc="#F8F4F6", ec="#6A626B", lw=1.2))
    for ring in ([(math.cos(u), math.sin(u), 0.0) for u in t], [(math.sin(u), 0.0, math.cos(u)) for u in t],
                 [(0.0, math.sin(u), math.cos(u)) for u in t]):
        pts = [proj(*p) for p in ring]
        for i in range(len(pts) - 1):
            (p0, d0), (p1, _) = pts[i], pts[i + 1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#B9AEB4", lw=0.8, ls="-" if d0 >= 0 else ":")
    labels = [((0, 0, 1), "|0⟩"), ((0, 0, -1), "|1⟩"), ((1, 0, 0), "|+⟩"), ((-1, 0, 0), "|−⟩"),
              ((0, 1, 0), "|+i⟩"), ((0, -1, 0), "|−i⟩")]
    for (x, y, z), lab in labels:
        (p, _), (q, _) = proj(x, y, z), proj(1.22 * x, 1.22 * y, 1.22 * z)
        ax.plot([0, p[0]], [0, p[1]], color="#B9AEB4", lw=0.8)
        ax.text(q[0], q[1], lab, ha="center", va="center", fontsize=10, color="#24313D")
    for x, y, z, col in points:
        (p, _) = proj(x, y, z)
        ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.plot([p[0]], [p[1]], "o", color=col, ms=5)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)


def plot_bloch_vector(bloch, title="", ax=None, figsize=None, coord_type="cartesian", font_size=None):
    """Plot one Bloch vector [x, y, z], or [r, theta, phi] with coord_type="spherical", like Qiskit's function."""
    import matplotlib.pyplot as plt

    if coord_type == "spherical":
        r, th, ph = bloch
        bloch = [r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)]
    x, y, z = (float(v) for v in bloch)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (3.6, 3.6))
    _draw_bloch(ax, [(x, y, z, "#99004C")], title)
    if fig is not None:
        fig.tight_layout()
        plt.close(fig)
    return fig if fig is not None else ax.figure


def bloch_vectors(state):
    """The Bloch vector (x, y, z) of each qubit's reduced state, qubit 0 first."""
    sv = state if isinstance(state, (Statevector, DensityMatrix)) else Statevector(state)
    rho = DensityMatrix(sv).data if isinstance(sv, Statevector) else sv.data
    n = int(round(math.log2(rho.shape[0])))
    paulis = (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]))
    out = []
    for q in range(n):
        red = rho.reshape([2] * (2 * n))
        keep = n - 1 - q               # tensor axis of qubit q (qubit 0 is the last axis)
        for i in sorted((i for i in range(n) if i != keep), reverse=True):
            red = np.trace(red, axis1=i, axis2=i + red.ndim // 2)
        out.append(tuple(float(np.real(np.trace(red @ m))) for m in paulis))
    return out


def plot_bloch_multivector(state, title="", figsize=None, *, reverse_bits=False, **_):
    """One Bloch sphere per qubit, like Qiskit's function. Entangled qubits show a shorter arrow."""
    import matplotlib.pyplot as plt

    vecs = bloch_vectors(state)
    n = len(vecs)
    order = list(range(n))[::-1] if reverse_bits else list(range(n))
    fig, axes = plt.subplots(1, n, figsize=figsize or (3.4 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, q in zip(axes, order):
        _draw_bloch(ax, [(*vecs[q], "#99004C")], f"qubit {q}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.close(fig)
    return fig
