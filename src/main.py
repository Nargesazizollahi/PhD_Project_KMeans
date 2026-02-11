import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 1: Load Data
# ============================================================
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "..", "data", "CC GENERAL.csv")

df = pd.read_csv(data_path)
print("✅ Original Shape:", df.shape)

# ============================================================
# STEP 2: Drop CUST_ID
# ============================================================
if "CUST_ID" in df.columns:
    df = df.drop(columns=["CUST_ID"])
print("✅ After dropping CUST_ID:", df.shape)

# ============================================================
# STEP 3: Impute missing values (numeric -> mean, categorical -> mode)
# ============================================================
missing_before = df.isnull().sum().sum()
print("🔸 Missing values before filling:", missing_before)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

# Numeric -> mean
for c in num_cols:
    df[c] = df[c].fillna(df[c].mean())

# Categorical -> mode
for c in cat_cols:
    mode_val = df[c].mode(dropna=True)
    df[c] = df[c].fillna(mode_val.iloc[0] if len(mode_val) else "UNKNOWN")

missing_after = df.isnull().sum().sum()
print("✅ Missing values after filling:", missing_after)

# ============================================================
# STEP 4: Prepare ranges for numeric part of Gower
# ============================================================
# range = max - min for each numeric feature (avoid division by zero)
if num_cols:
    num_min = df[num_cols].min()
    num_max = df[num_cols].max()
    num_range = (num_max - num_min).replace(0, 1.0)  # constant feature -> range=1
    num_range_arr = num_range.to_numpy(dtype=float)
else:
    num_range_arr = None

# Convert to numpy for speed
X_num = df[num_cols].to_numpy(dtype=float) if num_cols else None
X_cat = df[cat_cols].to_numpy(dtype=object) if cat_cols else None

# ============================================================
# STEP 5: Gower distance to centroids (mixed-type)
# ============================================================
def gower_distance_to_centroids(X_num, X_cat, cent_num, cent_cat, num_range_arr):
    """
    Compute Gower distance matrix between samples and centroids: shape (n_samples, k).

    Numeric part: mean(|x - c| / range)
    Categorical part: mean(x != c)
    Final: mean over all features (weighted by feature counts)
    """
    # Determine n, k
    n = X_num.shape[0] if X_num is not None else X_cat.shape[0]
    k = cent_num.shape[0] if cent_num is not None else cent_cat.shape[0]

    parts = []
    p_num = 0
    p_cat = 0

    # Numeric contribution
    if X_num is not None:
        p_num = X_num.shape[1]
        diff = np.abs(X_num[:, None, :] - cent_num[None, :, :]) / num_range_arr[None, None, :]
        parts.append(diff.mean(axis=2))  # (n,k)

    # Categorical contribution
    if X_cat is not None:
        p_cat = X_cat.shape[1]
        neq = (X_cat[:, None, :] != cent_cat[None, :, :]).astype(float)
        parts.append(neq.mean(axis=2))  # (n,k)

    # If only one type exists
    if len(parts) == 1:
        return parts[0]

    # Weighted mean over all features (strict Gower averaging)
    return (parts[0] * p_num + parts[1] * p_cat) / (p_num + p_cat)

def assign_clusters(X_num, X_cat, cent_num, cent_cat, num_range_arr):
    D = gower_distance_to_centroids(X_num, X_cat, cent_num, cent_cat, num_range_arr)
    return np.argmin(D, axis=1)

# ============================================================
# STEP 6: Update centroids (numeric -> mean, categorical -> mode)
# ============================================================
def update_centroids(df_full, clusters, k, num_cols, cat_cols, seed=None):
    new_num = []
    new_cat = []
    rng = np.random.default_rng(seed)

    for cid in range(k):
        sub = df_full.loc[clusters == cid]

        if sub.shape[0] == 0:
            # Empty cluster: choose a random row
            rnd = df_full.iloc[rng.integers(0, df_full.shape[0])]
            if num_cols:
                new_num.append(rnd[num_cols].to_numpy(dtype=float))
            if cat_cols:
                new_cat.append(rnd[cat_cols].to_numpy(dtype=object))
            continue

        if num_cols:
            new_num.append(sub[num_cols].mean().to_numpy(dtype=float))

        if cat_cols:
            modes = []
            for c in cat_cols:
                m = sub[c].mode(dropna=True)
                modes.append(m.iloc[0] if len(m) else "UNKNOWN")
            new_cat.append(np.array(modes, dtype=object))

    cent_num = np.array(new_num, dtype=float) if num_cols else None
    cent_cat = np.array(new_cat, dtype=object) if cat_cols else None
    return cent_num, cent_cat

# ============================================================
# STEP 7: Custom KMeans with Gower
# ============================================================
def kmeans_gower(df_full, k, num_cols, cat_cols, num_range_arr, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    n = df_full.shape[0]

    # Random init centroids from data rows
    init_idx = rng.choice(n, size=k, replace=False)
    init = df_full.iloc[init_idx]

    cent_num = init[num_cols].to_numpy(dtype=float) if num_cols else None
    cent_cat = init[cat_cols].to_numpy(dtype=object) if cat_cols else None

    X_num_local = df_full[num_cols].to_numpy(dtype=float) if num_cols else None
    X_cat_local = df_full[cat_cols].to_numpy(dtype=object) if cat_cols else None

    prev_clusters = None

    for it in range(max_iter):
        clusters = assign_clusters(X_num_local, X_cat_local, cent_num, cent_cat, num_range_arr)

        # Stop if assignments don't change
        if prev_clusters is not None and np.array_equal(clusters, prev_clusters):
            print(f"✅ Converged at iteration {it+1} (no cluster change)")
            break
        prev_clusters = clusters

        new_cent_num, new_cent_cat = update_centroids(
            df_full, clusters, k, num_cols, cat_cols, seed=seed + it + 1
        )

        # Stop if centroids are stable
        changed = False
        if num_cols and not np.allclose(cent_num, new_cent_num, atol=1e-8):
            changed = True
        if cat_cols and not np.array_equal(cent_cat, new_cent_cat):
            changed = True

        cent_num, cent_cat = new_cent_num, new_cent_cat

        if not changed:
            print(f"✅ Converged at iteration {it+1} (centroids stable)")
            break

    return clusters, cent_num, cent_cat

# ============================================================
# STEP 8: Required metric = sum of pairwise centroid distances
# ============================================================
def sum_pairwise_centroid_distances(cent_num, cent_cat, num_range_arr):
    """
    S(k) = sum_{a<b} D(c_a, c_b) where D is Gower distance.
    """
    # Treat centroids as "samples" and compute kxk distance matrix
    Xn = cent_num if cent_num is not None else None
    Xc = cent_cat if cent_cat is not None else None

    D = gower_distance_to_centroids(Xn, Xc, cent_num, cent_cat, num_range_arr)  # (k,k)
    return np.triu(D, k=1).sum()

# ============================================================
# STEP 9: Run for k=4..10 and save results
# ============================================================
results = []
for k in range(4, 11):
    print(f"\n🔹 Running KMeans-Gower for k={k} ...")
    clusters, cent_num, cent_cat = kmeans_gower(
        df_full=df,
        k=k,
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_range_arr=num_range_arr,
        max_iter=100,
        seed=42
    )

    s_cent = sum_pairwise_centroid_distances(cent_num, cent_cat, num_range_arr)
    results.append({"k": k, "sum_pairwise_centroid_distances": float(s_cent)})
    print(f"✅ k={k}  Sum of centroid distances (Gower) = {s_cent:.6f}")

results_df = pd.DataFrame(results)

report_path = os.path.join(base_path, "..", "report", "centroid_distance_results.csv")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
results_df.to_csv(report_path, index=False)

print("\n✅ Results saved to report/centroid_distance_results.csv")
print("\n📌 Final Results Table:")
print(results_df)
