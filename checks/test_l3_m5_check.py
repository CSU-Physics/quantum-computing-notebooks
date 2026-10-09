"""Tests for l3_m5_check.py: correct versions must pass with the same value, wrong versions must fail with a useful
message. Run from the repository root: python3 checks/test_l3_m5_check.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content", "level3"))
import l3_m5_check as c  # noqa: E402

R_S, R_G, R_M = c.reference_circuit_success, c.reference_max_two_qubit_gates, c.reference_summarize


def good_success_loop(e1, e2, e_ro, n1, n2, nm):
    p = 1.0
    for e, n in ((e1, n1), (e2, n2), (e_ro, nm)):
        for _ in range(n):
            p *= 1 - e
    return p


def good_gates_loop(e2, target=0.5):
    n = 0
    while (1 - e2) ** (n + 1) >= target:
        n += 1
    return n


def good_summarize_sorted(values, higher_is_better):
    v = sorted(values)
    m = len(v) // 2
    med = v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2
    return {"median": med, "best": v[-1] if higher_is_better else v[0], "worst": v[0] if higher_is_better else v[-1]}


GOOD = [(R_S, R_G, R_M), (good_success_loop, good_gates_loop, good_summarize_sorted),
        (lambda e1, e2, e_ro, n1, n2, nm: math.exp(n1 * math.log1p(-e1) + n2 * math.log1p(-e2) + nm * math.log1p(-e_ro)),
         lambda e2, target=0.5: int(math.floor(math.log(target) / math.log1p(-e2) + 1e-12)), R_M)]

BAD = {
    "first-order sum": (lambda e1, e2, e_ro, n1, n2, nm: 1 - e1 * n1 - e2 * n2 - e_ro * nm, R_G, R_M),
    "errors to the power": (lambda e1, e2, e_ro, n1, n2, nm: e1 ** n1 * e2 ** n2 * e_ro ** nm, R_G, R_M),
    "no measurements": (lambda e1, e2, e_ro, n1, n2, nm: (1 - e1) ** n1 * (1 - e2) ** n2, R_G, R_M),
    "products not powers": (lambda e1, e2, e_ro, n1, n2, nm: (1 - e1) * n1 * (1 - e2) * n2 * (1 - e_ro) * nm, R_G, R_M),
    "returns a string": (lambda *a: str(R_S(*a)), R_G, R_M),
    "ceil": (R_S, lambda e2, target=0.5: math.ceil(math.log(target) / math.log(1 - e2)), R_M),
    "round": (R_S, lambda e2, target=0.5: round(math.log(target) / math.log(1 - e2)), R_M),
    "log of e2": (R_S, lambda e2, target=0.5: int(math.log(target) / math.log(e2)), R_M),
    "approximate log": (R_S, lambda e2, target=0.5: int(math.log(target) / -e2), R_M),
    "float result": (R_S, lambda e2, target=0.5: math.log(target) / math.log(1 - e2), R_M),
    "no default target": (R_S, lambda e2, target: R_G(e2, target), R_M),
    "ignores target": (R_S, lambda e2, target=0.5: R_G(e2, 0.5), R_M),
    "mean not median": (R_S, R_G, lambda v, h: {"median": float(np.mean(v)), "best": max(v) if h else min(v), "worst": min(v) if h else max(v)}),
    "swapped best and worst": (R_S, R_G, lambda v, h: {"median": float(np.median(v)), "best": min(v) if h else max(v), "worst": max(v) if h else min(v)}),
    "upper middle": (R_S, R_G, lambda v, h: {"median": sorted(v)[len(v) // 2], "best": max(v) if h else min(v), "worst": min(v) if h else max(v)}),
    "returns a tuple": (R_S, R_G, lambda v, h: (float(np.median(v)), max(v), min(v))),
    "best always max": (R_S, R_G, lambda v, h: {"median": float(np.median(v)), "best": max(v), "worst": min(v)}),
    "not written": (R_S, R_G, lambda v, h: (_ for _ in ()).throw(NotImplementedError())),
}

if __name__ == "__main__":
    want = c.personal_value(500, 100, 20)
    assert want == 739, want
    wrong = 0
    for i, (s, g, m) in enumerate(GOOD):
        ok, msgs, v = c.check_l3_module5(s, g, m, 500, 100, 20)
        if not ok or v != want:
            wrong += 1
            print("GOOD", i, "FAILED", msgs)
    for name, (s, g, m) in BAD.items():
        ok, msgs, v = c.check_l3_module5(s, g, m, 500, 100, 20)
        if ok:
            wrong += 1
            print("BAD", name, "PASSED")
        else:
            print(f"bad {name:24s} -> {msgs[-1][:150]}")
    for args in [(50, 100, 20), (500, 10, 20), (500, 100, 60), (500.5, 100, 20), ("x", 1, 1)]:
        ok, msgs, v = c.check_l3_module5(R_S, R_G, R_M, *args)
        assert not ok, args
    print(f"{len(GOOD)} correct and {len(BAD)} wrong versions: {wrong} wrong verdicts; example value {want}")
