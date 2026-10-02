"""Check cell logic for the Module 1 lab (state vectors, normalization, measurement).

check_module1(probabilities, normalize, simulate, A, B, SEED) returns (passed, messages, verification_value).
The learner writes the three functions. The verification value is the number of 1s in 1,000 simulated
measurements of the learner's personal state (A|0> + iB|1>, normalized), drawn with
numpy.random.default_rng(SEED).choice(2, size=1000, p=...). It is printed only when every test passes.
"""
import numpy as np


def _close(x, y, tol=1e-9):
    x, y = np.asarray(x, dtype=complex), np.asarray(y, dtype=complex)
    return x.shape == y.shape and bool(np.allclose(x, y, atol=tol))


def check_module1(probabilities, normalize, simulate, A, B, SEED):
    msgs = []

    # Test 1: probabilities(state) gives |amplitude|^2 for every amplitude.
    cases = [
        ([0.6, 0.8j], [0.36, 0.64]),
        ([1, 0], [1, 0]),
        ([1 / np.sqrt(2), -1 / np.sqrt(2)], [0.5, 0.5]),
        ([0.5, 0.5j, -0.5, -0.5j], [0.25, 0.25, 0.25, 0.25]),
    ]
    for state, want in cases:
        try:
            got = probabilities(np.array(state, dtype=complex))
        except Exception as exc:  # noqa: BLE001
            return False, [f"probabilities() stopped with an error: {exc}"], None
        if not _close(np.real_if_close(got), want, 1e-9) or np.iscomplexobj(np.real_if_close(got)):
            return False, [f"probabilities({state}) gave {np.round(got, 4)}; it should be {want}. "
                           "Each probability is the size squared of its amplitude: np.abs(state)**2."], None
    msgs.append("Test 1, probabilities(): passed")

    # Test 2: normalize(state) divides by the length, keeps the direction and phases, and does not change the input.
    cases = [
        ([3, 4j], [0.6, 0.8j]),
        ([1, 1], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        ([2 - 2j, 0], [(1 - 1j) / np.sqrt(2), 0]),
        ([1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]),
    ]
    for state, want in cases:
        original = np.array(state, dtype=complex)
        arg = original.copy()
        try:
            got = normalize(arg)
        except Exception as exc:  # noqa: BLE001
            return False, [f"normalize() stopped with an error: {exc}"], None
        if not _close(got, want, 1e-9):
            return False, [f"normalize({state}) gave {np.round(got, 4)}; it should be {np.round(want, 4)}. "
                           "Divide the state by its length, np.linalg.norm(state)."], None
        if not _close(arg, original):
            return False, ["normalize() changed the array it was given. Return a new array instead, "
                           "for example: return state / np.linalg.norm(state)."], None
    msgs.append("Test 2, normalize(): passed")

    # Test 3: simulate(state, shots, seed) returns one 0 or 1 per shot, in the right proportion.
    state = np.array([0.6, 0.8j])
    try:
        out = np.asarray(simulate(state, 4000, 11))
    except Exception as exc:  # noqa: BLE001
        return False, [f"simulate() stopped with an error: {exc}"], None
    if out.shape != (4000,) or not np.isin(out, [0, 1]).all():
        return False, ["simulate(state, shots, seed) must return an array with one result, 0 or 1, for each shot. "
                       "Use: rng = np.random.default_rng(seed), then rng.choice(2, size=shots, p=probabilities(state))."], None
    frac = out.mean()
    if abs(frac - 0.64) > 0.03:     # about 4 standard deviations at 4,000 shots
        return False, [f"simulate() gave 1 in {frac:.1%} of 4,000 shots of [0.6, 0.8i]; it should be close to 64%."], None
    again = np.asarray(simulate(state, 4000, 11))
    if not np.array_equal(out, again):
        return False, ["simulate() gave different results for the same seed. Make the generator from the seed "
                       "inside the function: rng = np.random.default_rng(seed)."], None
    msgs.append("Test 3, simulate(): passed")

    # Your personal state, A|0> + iB|1>, normalized.
    if A is None or B is None or SEED is None:
        return False, msgs + ["Enter A, B and SEED from Canvas in Step 7, then run this cell again."], None
    a, b, seed = int(A), int(B), int(SEED)
    mine = normalize(np.array([a, 1j * b], dtype=complex))
    p1 = b * b / (a * a + b * b)
    if not _close(probabilities(mine), [1 - p1, p1], 1e-9):
        return False, msgs + ["Your personal state did not give the expected probabilities. Check A and B in Step 7."], None
    ones = int(np.random.default_rng(seed).choice(2, size=1000, p=[1 - p1, p1]).sum())
    msgs.append(f"Your state: P(0) = {1 - p1:.3f}, P(1) = {p1:.3f}; expected about {1000 * p1:.0f} ones in 1,000 shots.")
    return True, msgs, ones
