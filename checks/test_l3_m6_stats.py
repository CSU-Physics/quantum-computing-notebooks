"""Shared correct and wrong versions of the three statistics functions for the Level 3 Module 6 check tests."""
import math

import numpy as np

import l3_m6_common as cm

GOOD_STATS = [
    (cm.reference_fraction_sigma, cm.reference_expectation_sigma, cm.reference_combined_sigma),
    (lambda p, s: float(np.sqrt(p * (1 - p) / s)), lambda E, s: 2 * math.sqrt((1 + E) / 2 * (1 - E) / 2 / s),
     lambda c, sg: float(np.linalg.norm(np.multiply(c, sg)))),
]
F, E, C = GOOD_STATS[0]
BAD_STATS = {
    "fraction variance": ((lambda p, s: p * (1 - p) / s), E, C),
    "fraction sqrt then divide": ((lambda p, s: math.sqrt(p * (1 - p)) / s), E, C),
    "fraction with 1 - p^2": ((lambda p, s: math.sqrt((1 - p * p) / s)), E, C),
    "expectation as a fraction": (F, (lambda e, s: math.sqrt((1 + e) / 2 * (1 - e) / 2 / s)), C),
    "expectation without square": (F, (lambda e, s: math.sqrt((1 - e) / s)), C),
    "combined adds sigmas": (F, E, (lambda c, sg: sum(abs(a * b) for a, b in zip(c, sg)))),
    "combined ignores coefficients": (F, E, (lambda c, sg: math.sqrt(sum(b * b for b in sg)))),
    "combined unsquared coefficients": (F, E, (lambda c, sg: math.sqrt(sum(abs(a) * b * b for a, b in zip(c, sg))))),
    "stats not written": ((lambda p, s: (_ for _ in ()).throw(NotImplementedError())), E, C),
}


def run(name, check, good_rest, bad_rest, args, want, bad_args):
    """good_rest: list of tuples of the capstone functions; bad_rest: {name: tuple}. Returns the number of wrong verdicts."""
    wrong = 0
    for i, st in enumerate(GOOD_STATS):
        for j, rest in enumerate(good_rest):
            ok, msgs, v = check(*st, *rest, *args)
            if not ok or v != want:
                wrong += 1
                print("GOOD", i, j, "FAILED", v, msgs[-1:])
    for bname, st in BAD_STATS.items():
        ok, msgs, v = check(*st, *good_rest[0], *args)
        if ok:
            wrong += 1
            print("BAD", bname, "PASSED")
        else:
            print(f"bad {bname:34s} -> {msgs[-1][:140]}")
    for bname, rest in bad_rest.items():
        ok, msgs, v = check(*GOOD_STATS[0], *rest, *args)
        if ok:
            wrong += 1
            print("BAD", bname, "PASSED")
        else:
            print(f"bad {bname:34s} -> {msgs[-1][:140]}")
    for a in bad_args:
        ok, msgs, v = check(*GOOD_STATS[0], *good_rest[0], *a)
        if ok:
            wrong += 1
            print("BAD ARGS", a, "PASSED")
    print(f"{name}: {len(GOOD_STATS) * len(good_rest)} correct and {len(BAD_STATS) + len(bad_rest)} wrong versions, "
          f"{len(bad_args)} bad numbers: {wrong} wrong verdicts; example value {want}")
    return wrong
