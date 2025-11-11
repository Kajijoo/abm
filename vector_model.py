import numpy as np

def build_grid(theta, width, height, rng):
    total_cells = width * height
    total_patches = 2 * total_cells

    total_h = int(round(total_patches * (theta / (1.0 + theta))))
    total_h = min(max(total_h, 0), total_patches)

    patches = np.zeros(total_patches, dtype=np.int8)
    patches[:total_h] = 1
    rng.shuffle(patches)

    pairs = patches.reshape(total_cells, 2)
    sums = pairs.sum(axis=1)
    patch_type_flat = np.where(sums == 2, 2, np.where(sums == 0, 0, 1))
    grid = patch_type_flat.reshape(height, width)
    return grid


def _choose_food(ptype, value_high, value_low, epsilon, rng):

    N = value_high.shape[0]
    eat_H = np.zeros(N, dtype=bool)

    hh_mask = (ptype == 2)
    eat_H[hh_mask] = True

    hl_mask = (ptype == 1)
    hl_idx = np.where(hl_mask)[0]
    if hl_idx.size > 0:
        vh = value_high[hl_idx]
        vl = value_low[hl_idx]
        pref_H = vh > vl
        tie = vh == vl
        if tie.any():
            eat_h_hl = np.copy(pref_H)
            eat_h_hl[tie] = (rng.random(tie.sum()) < 0.5)
        else:
            eat_h_hl = pref_H

        if epsilon > 0:
            flips = rng.random(hl_idx.size) < epsilon
            eat_h_hl[flips] = ~eat_h_hl[flips]

        eat_H[hl_idx] = eat_h_hl

    return eat_H


def _update_values(ptype, eat_H, value_high, value_low, lr, p_high, p_low, extinction_rate, delta):
    on_hl = (ptype == 1)
    not_hl = ~on_hl

    eatH_hl = on_hl & eat_H
    eatL_hl = on_hl & (~eat_H)
    eatH_not = not_hl & eat_H
    eatL_not = not_hl & (~eat_H)

    if eatH_hl.any():
        idx = np.where(eatH_hl)[0]
        vh = value_high[idx]
        vl = value_low[idx]
        value_high[idx] = vh + lr * ((vh ** delta) * (p_high - vh))
        value_low[idx]  = vl + lr * ((vl ** delta) * extinction_rate * (0.0 - vl))

    if eatL_hl.any():
        idx = np.where(eatL_hl)[0]
        vh = value_high[idx]
        vl = value_low[idx]
        value_low[idx]  = vl + lr * ((vl ** delta) * (p_low - vl))
        value_high[idx] = vh + lr * ((vh ** delta) * extinction_rate * (0.0 - vh))

    if eatH_not.any():
        idx = np.where(eatH_not)[0]
        vh = value_high[idx]
        value_high[idx] = vh + lr * (p_high - vh)

    if eatL_not.any():
        idx = np.where(eatL_not)[0]
        vl = value_low[idx]
        value_low[idx] = vl + lr * (p_low - vl)


def run_vectorized_learning(p_high=0.9, p_low=0.6, steps=100, theta=3.0, epsilon=0.05,
                            N=100, width=100, height=100, seed=123, learning_rate=0.3,
                            extinction_rate=1.0, delta=0.0):
    rng = np.random.default_rng(seed)

    x = np.zeros(N, dtype=np.int32)
    y = np.arange(N, dtype=np.int32) % height
    value_high = np.full(N, 0.001, dtype=np.float64)
    value_low  = np.full(N, 0.001, dtype=np.float64)

    grid = build_grid(theta=theta, width=width, height=height, rng=rng)

    for _ in range(steps):
        x = (x + 1) % width
        ptype = grid[y, x]
        eat_H = _choose_food(ptype, value_high, value_low, epsilon, rng)
        _update_values(ptype, eat_H, value_high, value_low,
                       learning_rate, p_high, p_low, extinction_rate, delta)

    return float(value_high.mean()), float(value_low.mean())


def run_vectorized_simulation(theta=1.5, epsilon=0.05, p_high=0.9, p_low=0.6,
                              steps=100, N=100, width=100, height=100, seed=0,
                              vhigh0=None, vlow0=None,
                              learning_rate=0.4, extinction_rate=1.0, delta=0.0):
    rng = np.random.default_rng(seed)

    x = np.zeros(N, dtype=np.int32)
    y = np.arange(N, dtype=np.int32) % height
    value_high = np.full(N, 0.001 if vhigh0 is None else vhigh0, dtype=np.float64)
    value_low  = np.full(N, 0.001 if vlow0 is None else vlow0, dtype=np.float64)
    foods_H = np.zeros(N, dtype=np.int32)
    foods_L = np.zeros(N, dtype=np.int32)

    grid = build_grid(theta=theta, width=width, height=height, rng=rng)

    for _ in range(steps):
        x = (x + 1) % width
        ptype = grid[y, x]
        eat_H = _choose_food(ptype, value_high, value_low, epsilon, rng)

        foods_H += eat_H.astype(np.int32)
        foods_L += (~eat_H).astype(np.int32)

        _update_values(ptype, eat_H, value_high, value_low,
                       learning_rate, p_high, p_low, extinction_rate, delta)

    mean_vh = float(value_high.mean())
    mean_vl = float(value_low.mean())
    delta_v = mean_vh - mean_vl

    denom = np.maximum(foods_H, 1)
    lh_ratio = float(np.nanmean(foods_L / denom))

    return {
        "Value_High": mean_vh,
        "Value_Low": mean_vl,
        "delta_V": delta_v,
        "LH_Ratio": lh_ratio
    }

if __name__ == "__main__":

# --- quick pre-learning phase ---
    vH0, vL0 = run_vectorized_learning(
        p_high=0.9,
        p_low=0.6,
        theta=4.0,
        epsilon=0.05,
        steps=100,
        N=100,
        width=100,
        height=100,
        seed=42
    )
    print(f"Pre-learning finished: V_H0={vH0:.4f}, V_L0={vL0:.4f}")

    # --- intervention phase ---
    result = run_vectorized_simulation(
        theta=0.25,
        epsilon=0.25,
        p_high=0.9,
        p_low=0.6,
        vhigh0=vH0,
        vlow0=vL0,
        steps=100,
        N=100,
        width=100,
        height=100,
        seed=42
    )
    print("Intervention result:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")