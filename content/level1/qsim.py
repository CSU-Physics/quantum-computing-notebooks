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
"""
import math
from collections import Counter

import numpy as np

__all__ = [
    "CircuitError", "QuantumCircuit", "Statevector", "DensityMatrix", "AerSimulator",
    "transpile", "state_fidelity", "plot_histogram", "simulate_density_matrix",
]
__version__ = "1.0.0"


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
        if n == "ccx":
            return _controlled(_FIXED["x"], 2)
        if n == "swap":
            return _SWAP
        raise CircuitError(f"'{n}' has no matrix")

    def __repr__(self):
        p = f", params={list(self.params)}" if self.params else ""
        c = f", clbits={list(self.clbits)}" if self.clbits else ""
        return f"Instruction({self.name!r}, qubits={list(self.qubits)}{c}{p})"


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
    def rx(self, theta, q): return self._one("rx", q, (float(theta),))
    def ry(self, theta, q): return self._one("ry", q, (float(theta),))
    def rz(self, phi, q): return self._one("rz", q, (float(phi),))
    def p(self, lam, q): return self._one("p", q, (float(lam),))

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
    def cp(self, lam, control, target): return self._two("cp", control, target, (float(lam),))
    def crz(self, phi, control, target): return self._two("crz", control, target, (float(phi),))
    def swap(self, a, b): return self._two("swap", a, b)
    def ccx(self, c1, c2, target): return self._add("ccx", [int(c1), int(c2), int(target)])

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
            if i.name in ("measure", "reset"):
                raise CircuitError("A circuit with measurements or resets has no inverse.")
            if i.name in _PARAM or i.name in ("cp", "crz"):
                new.data.append(Instruction(i.name, i.qubits, (), (-i.params[0],)))
            elif i.name == "sx":
                new.data.append(Instruction("rx", i.qubits, (), (-math.pi / 2,)))
            else:
                new.data.append(Instruction(inv.get(i.name, i.name), i.qubits))
        new.global_phase = -self.global_phase
        return new

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
                return f"{inst.name.upper()}({inst.params[0]:.3g})"
            return {"measure": "M", "reset": "|0>", "sdg": "Sdg", "tdg": "Tdg", "sx": "√X"}.get(
                inst.name, inst.name.upper())

        for inst in qc.data:
            qs = inst.qubits
            if inst.name == "barrier":
                col = ["░" if q in qs else "─" for q in range(n)]
                w = 1
            elif inst.name in ("cx", "cy", "cz", "ch", "cp", "crz", "ccx", "swap"):
                lo, hi = min(qs), max(qs)
                tgt = qs[-1]
                if inst.name == "swap":
                    marks = {qs[0]: "X", qs[1]: "X"}
                elif inst.name == "cz":
                    marks = {qs[0]: "■", qs[1]: "■"}
                else:
                    sym = {"cx": "X", "ccx": "X", "cy": "Y", "ch": "H",
                           "cp": f"P({inst.params[0]:.3g})" if inst.params else "P",
                           "crz": f"RZ({inst.params[0]:.3g})" if inst.params else "RZ"}[inst.name]
                    marks = {q: "■" for q in qs[:-1]}
                    marks[tgt] = "⊕" if inst.name in ("cx", "ccx") else sym
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
            if inst.name in ("cx", "cy", "cz", "ch", "cp", "crz", "ccx", "swap"):
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
    dim = 2 ** n
    # rho -> U rho U^dagger, done column by column then row by row.
    r = np.array([_apply_unitary_vec(rho[:, j], u, qubits, n) for j in range(dim)]).T
    r = np.array([_apply_unitary_vec(r[i, :].conj(), u, qubits, n).conj() for i in range(dim)])
    return r


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
    if isinstance(op, str):
        if len(op) != n:
            raise ValueError(f"The Pauli label {op!r} has {len(op)} letters; the state has {n} qubits.")
        return _pauli_matrix(op)
    m = np.asarray(op, dtype=complex)
    if m.shape != (2 ** n, 2 ** n):
        raise ValueError(f"The operator must be {2 ** n} x {2 ** n}.")
    return m


# ---------------------------------------------------------------- states
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

    def probabilities(self):
        return np.abs(self.data) ** 2

    def probabilities_dict(self, decimals=None):
        p = self.probabilities()
        out = {}
        for i, x in enumerate(p):
            if x > 1e-12:
                out[format(i, f"0{self.num_qubits}b")] = round(float(x), decimals) if decimals is not None else float(x)
        return out

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

    def probabilities(self):
        return np.real(np.diag(self.data)).clip(min=0)

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


def _fmt_complex(a):
    a = complex(a)
    re, im = round(a.real, 4), round(a.imag, 4)
    re = 0.0 if re == 0 else re
    im = 0.0 if im == 0 else im
    if im == 0:
        return f"{re:g}+0j"
    return f"{re:g}{im:+g}j"


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
        if inst.name in ("measure", "reset"):
            raise CircuitError(
                "Cannot make a Statevector from a circuit with measurements or resets. "
                "Remove them first, for example with qc.remove_final_measurements().")
        psi = _apply_unitary_vec(psi, inst.matrix(), list(inst.qubits), n)
    return psi * np.exp(1j * qc.global_phase)


def simulate_density_matrix(qc):
    """The state at the end of the circuit as a density matrix.

    Measurements are applied without looking at the result (the state becomes a mixture of the
    outcomes), and resets put the qubit back to |0>. This is what happens to the quantum state
    when a circuit measures in the middle.
    """
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

    def __init__(self, seed_simulator=None, method="automatic", **_):
        self.seed_simulator = seed_simulator
        self.method = method
        self.name = "qsim_simulator"

    def run(self, circuits, shots=1024, seed_simulator=None, memory=False, **_):
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
        nc = qc.num_clbits
        final = qc._final_measure_indices()
        mid = any(i.name in ("measure", "reset") and k not in final for k, i in enumerate(qc.data))
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
    def _one_shot(qc, rng):
        n = qc.num_qubits
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1
        bits = [0] * qc.num_clbits
        for inst in qc.data:
            if inst.name == "barrier":
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
