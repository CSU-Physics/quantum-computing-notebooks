"""Auto-grader accuracy test for the Module 1 check cell: correct and wrong implementations."""
import numpy as np
from m1_check import check_module1


def p_ok(s): return np.abs(s) ** 2
def p_ok2(s): return (s * np.conj(s)).real
def n_ok(s): return s / np.linalg.norm(s)
def n_ok2(s): return s / np.sqrt(np.sum(np.abs(s) ** 2))
def sim_ok(s, shots, seed): return np.random.default_rng(seed).choice(2, size=shots, p=p_ok(s))
def sim_ok2(s, shots, seed): return (np.random.default_rng(seed).random(shots) < p_ok(s)[1]).astype(int)

def p_bad1(s): return np.abs(s)                 # forgot to square
def p_bad2(s): return s ** 2                    # squared the complex amplitude
def n_bad1(s): return s / np.sum(s)             # divided by the sum
def n_bad2(s):                                   # changes the input
    s /= np.linalg.norm(s)
    return s
def n_bad3(s): return s / np.linalg.norm(s) ** 2
def sim_bad1(s, shots, seed): return np.random.default_rng(0).choice(2, size=shots, p=[0.5, 0.5])
def sim_bad2(s, shots, seed): return np.random.default_rng().choice(2, size=shots, p=p_ok(s))   # not seeded
def sim_bad3(s, shots, seed): return {"0": shots // 2, "1": shots // 2}
def sim_bad4(s, shots, seed): return np.random.default_rng(seed).choice(2, size=shots, p=p_ok(s)[::-1])
def sim_bad5(s, shots, seed): return np.random.default_rng(seed).choice(2, size=shots // 2, p=p_ok(s))


def main():
    errors = 0
    ref = []
    for A, B, SEED in [(3, 4, 7), (1, 9, 123), (5, 2, 999)]:
        vals = set()
        for p in (p_ok, p_ok2):
            for n in (n_ok, n_ok2):
                for s in (sim_ok, sim_ok2):
                    ok, msgs, v = check_module1(p, n, s, A, B, SEED)
                    errors += not ok
                    vals.add(v)
        good = len(vals) == 1
        errors += not good
        ref.append((A, B, SEED, vals))
        print(f"  {'ok ' if good else 'BAD'} correct versions agree for A={A}, B={B}, SEED={SEED}: value {vals}")
    wrong = {
        "probabilities without squaring": (p_bad1, n_ok, sim_ok),
        "probabilities from state**2": (p_bad2, n_ok, sim_ok),
        "normalize divides by the sum": (p_ok, n_bad1, sim_ok),
        "normalize changes its input": (p_ok, n_bad2, sim_ok),
        "normalize divides by norm squared": (p_ok, n_bad3, sim_ok),
        "simulate ignores the state and seed": (p_ok, n_ok, sim_bad1),
        "simulate not seeded": (p_ok, n_ok, sim_bad2),
        "simulate returns counts, not shots": (p_ok, n_ok, sim_bad3),
        "simulate swaps 0 and 1": (p_ok, n_ok, sim_bad4),
        "simulate returns half the shots": (p_ok, n_ok, sim_bad5),
    }
    for name, (p, n, s) in wrong.items():
        ok, msgs, v = check_module1(p, n, s, 3, 4, 7)
        good = (not ok) and v is None
        errors += not good
        print(f"  {'ok ' if good else 'BAD'} reject {name}: {msgs[-1][:90]}")
    ok, msgs, v = check_module1(p_ok, n_ok, sim_ok, None, None, None)
    errors += ok or v is not None
    print(f"  {'ok ' if not ok else 'BAD'} no personal values yet: {msgs[-1][:70]}")
    print(f"\n{errors} wrong verdicts")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
