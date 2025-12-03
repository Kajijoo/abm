import numpy as np
from scipy.sparse import csr_matrix


# ==========================================================
# GRID INITIALISATION
# ==========================================================
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

    # 0L, 1HL, 2H
    patch_type_flat = np.where(
        sums == 2, 2, np.where(sums == 0, 0, 1)
    )

    return patch_type_flat.reshape(height, width)


# ==========================================================
# BA NETWORK GENERATION
# ==========================================================
def build_BA_adjacency(N, m, rng):
    seed_size = m + 1

    rows = []
    cols = []

    # fully connected seed
    for i in range(seed_size):
        for j in range(seed_size):
            if i != j:
                rows.append(i)
                cols.append(j)

    degrees = np.full(seed_size, seed_size - 1, dtype=np.int32)

    # add nodes with preferential attachment
    for i in range(seed_size, N):
        probs = degrees / degrees.sum()
        chosen = rng.choice(i, size=m, replace=False, p=probs)

        for t in chosen:
            rows.append(i)
            cols.append(t)
            rows.append(t)
            cols.append(i)

        degrees = np.append(degrees, m)
        degrees[chosen] += 1

    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(N, N))


# ==========================================================
# CHOICE FUNCTION
# ==========================================================
def choose_food(ptype, value_high, value_low, soc_high, soc_low, w_soc, epsilon, rng):
    """
    Hard-coded availability constraints:
        HH → Must eat High reward
        LL → Must eat Low reward
        HL → Compare Vh vs Vl (with/without social reinforcement)
    """

    N = value_high.shape[0]
    eat_H = np.zeros(N, dtype=bool)

    # HH → always high
    hh_mask = (ptype == 2)
    eat_H[hh_mask] = True

    # LL → always low (already zero)

    # HL → compare Qh & Ql
    hl_mask = (ptype == 1)
    if hl_mask.any():
        idx = np.where(hl_mask)[0]

        Vh = value_high[idx] + w_soc * soc_high[idx]
        Vl = value_low[idx] + w_soc * soc_low[idx]

        pref_H = (Vh > Vl)

        ties = (Vh == Vl)
        if ties.any():
            pref_H[ties] = rng.random(ties.sum()) < 0.5

        # epsilon exploration
        if epsilon > 0:
            flips = rng.random(idx.size) < epsilon
            pref_H[flips] = ~pref_H[flips]

        eat_H[idx] = pref_H

    return eat_H


# ==========================================================
# VALUE-UPDATING FUNCTION
# ==========================================================
def update_values(ptype, eat_H, value_high, value_low,
                  p_high, p_low, learning_rate, extinction_rate, delta):

    on_hl = (ptype == 1)
    not_hl = ~on_hl

    eatH_hl = on_hl & eat_H
    eatL_hl = on_hl & (~eat_H)
    eatH_not = not_hl & eat_H
    eatL_not = not_hl & (~eat_H)

    # HL patch updates
    if eatH_hl.any():
        idx = np.where(eatH_hl)[0]
        vh = value_high[idx]
        vl = value_low[idx]
        value_high[idx] = vh + learning_rate * ((vh ** delta) * (p_high - vh))
        value_low[idx] = vl + learning_rate * ((vl ** delta) *
                                               extinction_rate * (0.0 - vl))

    if eatL_hl.any():
        idx = np.where(eatL_hl)[0]
        vh = value_high[idx]
        vl = value_low[idx]
        value_low[idx] = vl + learning_rate * ((vl ** delta) * (p_low - vl))
        value_high[idx] = vh + learning_rate * ((vh ** delta) *
                                                extinction_rate * (0.0 - vh))

    # Non-HL patch updates (standard RW)
    if eatH_not.any():
        idx = np.where(eatH_not)[0]
        vh = value_high[idx]
        value_high[idx] = vh + learning_rate * (p_high - vh)

    if eatL_not.any():
        idx = np.where(eatL_not)[0]
        vl = value_low[idx]
        value_low[idx] = vl + learning_rate * (p_low - vl)


# ==========================================================
# SOCIAL REINFORCEMENT FUNCTION
# ==========================================================
def update_social(A, eat_H, soc_high, soc_low, learning_rate, deg_safe):
    prop_high = A.dot(eat_H) / deg_safe
    prop_low = 1.0 - prop_high

    same_choice = np.where(eat_H, prop_high, prop_low)

    r_social = 2.0 * same_choice - 1.0

    idxH = eat_H
    soc_high[idxH] += learning_rate * (r_social[idxH] - soc_high[idxH])

    idxL = ~eat_H
    soc_low[idxL] += learning_rate * (r_social[idxL] - soc_low[idxL])


# ==========================================================
# MAIN SIMULATION
# ==========================================================
def run_vectorized_simulation(theta=1.5, epsilon=0.05,
                              p_high=0.9, p_low=0.6,
                              steps=100, N=100, width=100, height=100,
                              seed=0, vhigh0=None, vlow0=None,
                              learning_rate=0.3, extinction_rate=1.0,
                              w_soc=0.5, delta=0.0,
                              record_history=False,
                              A_override=None,
                              soc_high0=None, soc_low0=None):

    rng = np.random.default_rng(seed)

    x = np.zeros(N, dtype=np.int32)
    y = np.arange(N, dtype=np.int32) % height

    # ---------- reward value init: handle scalar or vector ----------
    if vhigh0 is None:
        value_high = np.full(N, 0.001, dtype=np.float64)
    else:
        vhigh0_arr = np.array(vhigh0, dtype=np.float64)
        if vhigh0_arr.shape == ():  # scalar
            value_high = np.full(N, float(vhigh0_arr), dtype=np.float64)
        else:
            value_high = vhigh0_arr.copy()

    if vlow0 is None:
        value_low = np.full(N, 0.001, dtype=np.float64)
    else:
        vlow0_arr = np.array(vlow0, dtype=np.float64)
        if vlow0_arr.shape == ():
            value_low = np.full(N, float(vlow0_arr), dtype=np.float64)
        else:
            value_low = vlow0_arr.copy()

    # ---------- social value init: allow priors ----------
    if soc_high0 is not None:
        soc_high = np.array(soc_high0, dtype=np.float64).copy()
    else:
        soc_high = np.zeros(N, dtype=np.float64)

    if soc_low0 is not None:
        soc_low = np.array(soc_low0, dtype=np.float64).copy()
    else:
        soc_low = np.zeros(N, dtype=np.float64)

    foods_H = np.zeros(N, dtype=np.int32)
    foods_L = np.zeros(N, dtype=np.int32)

    # ---------- network: allow override / freezing ----------
    if A_override is not None:
        A = A_override
    else:
        A = build_BA_adjacency(N, m=1, rng=rng)

    deg = np.array(A.sum(axis=1)).flatten()
    deg_safe = np.maximum(deg, 1)

    if record_history:
        V_high_hist = np.zeros(steps)
        V_low_hist = np.zeros(steps)
        deltaV_hist = np.zeros(steps)

    grid = build_grid(theta=theta, width=width, height=height, rng=rng)

    # ------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------
    for t in range(steps):

        # move right each step
        x = (x + 1) % width
        ptype = grid[y, x]

        # food choice (hard-coded by patch type)
        eat_H = choose_food(
            ptype, value_high, value_low,
            soc_high, soc_low, w_soc, epsilon, rng
        )

        foods_H += eat_H.astype(np.int32)
        foods_L += (~eat_H).astype(np.int32)

        # reinforcement updates
        update_values(
            ptype, eat_H, value_high, value_low,
            p_high, p_low, learning_rate,
            extinction_rate, delta
        )

        # social update
        update_social(A, eat_H, soc_high, soc_low, learning_rate, deg_safe)

        # store history
        if record_history:
            V_high_hist[t] = value_high.mean()
            V_low_hist[t] = value_low.mean()
            deltaV_hist[t] = V_high_hist[t] - V_low_hist[t]

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------
    mean_vh = float(value_high.mean())
    mean_vl = float(value_low.mean())
    delta_v = mean_vh - mean_vl

    denom = np.maximum(foods_H, 1)
    lh_ratio = float(np.nanmean(foods_L / denom))

    result = {
        "Value_High": mean_vh,
        "Value_Low": mean_vl,
        "delta_V": delta_v,
        "LH_Ratio": lh_ratio,
    }

    if record_history:
        result["V_high_hist"] = V_high_hist
        result["V_low_hist"] = V_low_hist
        result["deltaV_hist"] = deltaV_hist

    result["value_high_vec"] = value_high.copy()
    result["value_low_vec"] = value_low.copy()
    result["soc_high_vec"] = soc_high.copy()
    result["soc_low_vec"] = soc_low.copy()

    return result, A
