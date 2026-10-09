"""Tests for l3_m6_rep_check.py. Run from the repository root: python3 checks/test_l3_m6_rep_check.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content", "level3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsim import QuantumCircuit  # noqa: E402
import l3_m6_rep_check as c  # noqa: E402
from test_l3_m6_stats import run  # noqa: E402

RM, RE, RD = c.reference_memory_circuit, c.reference_detection_events, c.reference_decode


def mem(rounds, bit=0, order=((0, 3), (1, 3), (1, 4), (2, 4)), idle=True, reset=True, xgate=True, cmap=None, final_first=False):
    qc = QuantumCircuit(5, 2 * rounds + 3)
    if bit and xgate:
        qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    for r in range(rounds):
        if idle:
            for d in (0, 1, 2):
                qc.id(d)
        for a, b in order:
            qc.cx(a, b)
        s0, s1 = (2 * r, 2 * r + 1) if cmap is None else cmap(r)
        qc.measure(3, s0)
        qc.measure(4, s1)
        if reset:
            qc.reset(3)
            qc.reset(4)
    for d in range(3):
        qc.measure(d, 2 * rounds + d)
    return qc


def mem_lists(rounds, bit=0):
    qc = QuantumCircuit(5, 2 * rounds + 3)
    if bit == 1:
        qc.x(0)
    for t in (1, 2):
        qc.cx(0, t)
    for r in range(rounds):
        qc.id(0); qc.id(1); qc.id(2)
        for a, t in ((0, 3), (1, 3), (1, 4), (2, 4)):
            qc.cx(a, t)
        for q, cb in ((3, 2 * r), (4, 2 * r + 1)):
            qc.measure(q, cb)
        for q in (3, 4):
            qc.reset(q)
    for d, cb in zip(range(3), range(2 * rounds, 2 * rounds + 3)):
        qc.measure(d, cb)
    return qc


def mem_x_after(rounds, bit=0):
    qc = QuantumCircuit(5, 2 * rounds + 3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    if bit:
        for d in (0, 1, 2):
            qc.x(d)
    for r in range(rounds):
        for d in (0, 1, 2):
            qc.id(d)
        for a, b in ((0, 3), (1, 3), (1, 4), (2, 4)):
            qc.cx(a, b)
        qc.measure(3, 2 * r)
        qc.measure(4, 2 * r + 1)
        qc.reset(3)
        qc.reset(4)
    for d in range(3):
        qc.measure(d, 2 * rounds + d)
    return qc


def b(key, k):
    return int(key[::-1][k])


def ev_alt(key, rounds):
    bits = key[::-1]
    syn = [(int(bits[2 * r]), int(bits[2 * r + 1])) for r in range(rounds)]
    d0, d1, d2 = (int(bits[2 * rounds + j]) for j in range(3))
    syn = [(0, 0)] + syn + [(d0 ^ d1, d1 ^ d2)]
    return [((syn[i][0] + syn[i + 1][0]) % 2, (syn[i][1] + syn[i + 1][1]) % 2) for i in range(rounds + 1)]


def dec_alt(key, rounds):
    return int(key[:3].count("1") >= 2)


GOOD = [(RM, RE, RD), (mem_lists, ev_alt, dec_alt), (lambda r, bit=0: mem(r, bit), lambda k, r: [list(e) for e in RE(k, r)], RD)]
BAD = {
    # equivalent without noise, but the noise acts differently: the notebook fixes the order
    "different CNOT order": (lambda r, bit=0: mem(r, bit, order=((0, 3), (1, 4), (1, 3), (2, 4))), RE, RD),
    "X after encoding": (mem_x_after, RE, RD),
    "no idle gates": (lambda r, bit=0: mem(r, bit, idle=False), RE, RD),
    "no resets": (lambda r, bit=0: mem(r, bit, reset=False), RE, RD),
    "wrong parity checks": (lambda r, bit=0: mem(r, bit, order=((0, 3), (1, 3), (0, 4), (2, 4))), RE, RD),
    "ignores bit": (lambda r, bit=0: mem(r, bit, xgate=False), RE, RD),
    "syndrome bits swapped": (lambda r, bit=0: mem(r, bit, cmap=lambda k: (2 * k + 1, 2 * k)), RE, RD),
    "not written circuit": (lambda r, bit=0: (_ for _ in ()).throw(NotImplementedError()), RE, RD),
    "syndromes not changes": (RM, lambda k, r: [(b(k, 2 * i), b(k, 2 * i + 1)) for i in range(r)] +
                              [(b(k, 2 * r) ^ b(k, 2 * r + 1), b(k, 2 * r + 1) ^ b(k, 2 * r + 2))], RD),
    "no final event": (RM, lambda k, r: RE(k, r)[:-1], RD),
    "key read left to right": (RM, lambda k, r: RE(k[::-1], r), RD),
    "decode first data bit": (RM, RE, lambda k, r: b(k, 2 * r)),
    "decode from left of key wrong": (RM, RE, lambda k, r: int(k[-3:].count("1") >= 2)),
    "decode inverted": (RM, RE, lambda k, r: 1 - RD(k, r)),
}

if __name__ == "__main__":
    want = c.personal_value(3, 0.01, 0.02)
    assert want == 1516, want
    wrong = run("rep", c.check_l3_m6_rep, GOOD, BAD, (3, 0.01, 0.02), want,
                [(0, 0.01, 0.02), (6, 0.01, 0.02), (3, 0.04, 0.02), (3, 0.01, 0.001), (2.5, 0.01, 0.02), ("x", 1, 1)])
    sys.exit(1 if wrong else 0)
