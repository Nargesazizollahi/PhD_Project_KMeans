import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 1: Load Data
# ============================================================
import os

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "..", "data", "CC GENERAL.csv")

df = pd.read_csv(data_path)


print("✅ Original Shape:", df.shape)

# =========================
# STEP 2: Drop CUST_ID
# =========================
if "CUST_ID" in df.columns:
    df = df.drop(columns=["CUST_ID"])
print("✅ After dropping CUST_ID:", df.shape)

# =========================
# STEP 3: Impute missing (numeric->mean, categorical->mode)
# =========================
missing_before = df.isnull().sum().sum()
print("🔸 Missing values before filling:", missing_before)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    df[c] = df[c].fillna(df[c].mean())

for c in cat_cols:
    mode_val = df[c].mode(dropna=True)
    df[c] = df[c].fillna(mode_val.iloc[0] if len(mode_val) else "UNKNOWN")

missing_after = df.isnull().sum().sum()
print("✅ Missing values after filling:", missing_after)

# =========================
# STEP 4: Prepare ranges for numeric Gower
# =========================
# range = max - min for each numeric feature (avoid division by zero)
num_min = df[num_cols].min()
num_max = df[num_cols].max()
num_range = (num_max - num_min).replace(0, 1.0)  # if constant column -> range=1 to avoid /0

# Convert to numpy for speed
X_num = df[num_cols].to_numpy(dtype=float) if num_cols else None
X_cat = df[cat_cols].to_numpy(dtype=object) if cat_cols else None

<<<<<<< HEAD
# Save cleaned data
clean_data_path = os.path.join(base_path, "..", "data", "CC_GENERAL_clean_scaled.csv")
df_scaled.to_csv(clean_data_path, index=False)
print("✅ Saved cleaned & scaled dataset to: data/CC_GENERAL_clean_scaled.csv")

# ============================================================
# STEP 5: Define Gower Distance (Numeric Case)
# ============================================================
def gower_distance_numeric(x, y):
=======
# =========================
# STEP 5: Gower distance (mixed)
# =========================
def gower_distance_to_centroids(X_num, X_cat, cent_num, cent_cat, num_range_arr):
>>>>>>> fd2b5a0 (Final implementation: Custom Gower KMeans + centroid distance metric + plot)
    """
    Returns distance matrix shape: (n_samples, k) using Gower.
    - numeric part: mean(|x - c| / range)
    - categorical part: mean(x != c)
    Final distance = mean over all features (numeric + categorical).
    """
    n = X_num.shape[0] if X_num is not None else X_cat.shape[0]
    k = cent_num.shape[0] if cent_num is not None else cent_cat.shape[0]

    parts = []
    if X_num is not None:
        # (n,1,p) - (1,k,p) -> (n,k,p)
        diff = np.abs(X_num[:, None, :] - cent_num[None, :, :]) / num_range_arr[None, None, :]
        parts.append(diff.mean(axis=2))  # (n,k)

    if X_cat is not None:
        neq = (X_cat[:, None, :] != cent_cat[None, :, :]).astype(float)
        parts.append(neq.mean(axis=2))  # (n,k)

    if len(parts) == 1:
        return parts[0]
    # average numeric-part-distance and cat-part-distance weighted by number of features:
    # To be strictly "mean over all features", we weight by feature counts:
    p_num = X_num.shape[1] if X_num is not None else 0
    p_cat = X_cat.shape[1] if X_cat is not None else 0
    return (parts[0] * p_num + parts[1] * p_cat) / (p_num + p_cat)

<<<<<<< HEAD
def kmeans_gower(data, k, max_iter=100):
    """
    Custom K-Means algorithm using Gower distance (numeric case).
    """
    np.random.seed(42)
=======
def assign_clusters(X_num, X_cat, cent_num, cent_cat, num_range_arr):
    D = gower_distance_to_centroids(X_num, X_cat, cent_num, cent_cat, num_range_arr)
    return np.argmin(D, axis=1)
>>>>>>> fd2b5a0 (Final implementation: Custom Gower KMeans + centroid distance metric + plot)

def update_centroids(df_full, clusters, k, num_cols, cat_cols):
    new_num = []
    new_cat = []
    for cid in range(k):
        sub = df_full.loc[clusters == cid]
        if sub.shape[0] == 0:
            # empty cluster -> random row
            rnd = df_full.sample(1, random_state=None)
            if num_cols:
                new_num.append(rnd[num_cols].to_numpy(dtype=float)[0])
            if cat_cols:
                new_cat.append(rnd[cat_cols].to_numpy(dtype=object)[0])
            continue

        if num_cols:
            new_num.append(sub[num_cols].mean().to_numpy(dtype=float))
        if cat_cols:
            # mode for each categorical column
            modes = []
            for c in cat_cols:
                m = sub[c].mode(dropna=True)
                modes.append(m.iloc[0] if len(m) else "UNKNOWN")
            new_cat.append(np.array(modes, dtype=object))

    cent_num = np.array(new_num, dtype=float) if num_cols else None
    cent_cat = np.array(new_cat, dtype=object) if cat_cols else None
    return cent_num, cent_cat

def kmeans_gower(df_full, k, num_cols, cat_cols, num_range_series, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    n = df_full.shape[0]

    # init centroids from random rows
    idx = rng.choice(n, size=k, replace=False)
    init = df_full.iloc[idx]

    cent_num = init[num_cols].to_numpy(dtype=float) if num_cols else None
    cent_cat = init[cat_cols].to_numpy(dtype=object) if cat_cols else None

    X_num = df_full[num_cols].to_numpy(dtype=float) if num_cols else None
    X_cat = df_full[cat_cols].to_numpy(dtype=object) if cat_cols else None
    num_range_arr = num_range_series.to_numpy(dtype=float) if num_cols else None

    prev_clusters = None
    for it in range(max_iter):
        clusters = assign_clusters(X_num, X_cat, cent_num, cent_cat, num_range_arr)

        if prev_clusters is not None and np.array_equal(clusters, prev_clusters):
            print(f"✅ Converged at iteration {it+1} (no cluster change)")
            break
        prev_clusters = clusters

        new_cent_num, new_cent_cat = update_centroids(df_full, clusters, k, num_cols, cat_cols)

        # centroid change check (numeric)
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

# =========================
# STEP 6: required metric: sum of pairwise centroid distances
# =========================
def sum_pairwise_centroid_distances(cent_num, cent_cat, num_range_arr):
    # build full centroid matrix distance
    k = cent_num.shape[0] if cent_num is not None else cent_cat.shape[0]

<<<<<< HEAD
# ============================================================
# STEP 7 (FIXED): Sum of distances of points to their assigned centroids
# ============================================================
def within_cluster_distance_sum(data, clusters, centroids):
    """
    Compute sum of Gower distances from each point to its assigned centroid.
    (Gower-based inertia)
    """
    # distances for all points to all centroids: shape (n_samples, k)
    distances = np.mean(np.abs(data[:, None, :] - centroids[None, :, :]), axis=2)
    # pick distance to assigned centroid for each point and sum
    return distances[np.arange(len(data)), clusters].sum()

=======
    # compute D between centroids by reusing gower function (treat centroids as "samples")
    Xn = cent_num if cent_num is not None else None
    Xc = cent_cat if cent_cat is not None else None
>>>>>>> fd2b5a0 (Final implementation: Custom Gower KMeans + centroid distance metric + plot)

    D = gower_distance_to_centroids(Xn, Xc, cent_num, cent_cat, num_range_arr)  # (k,k)
    # sum upper triangle a<b
    return np.triu(D, k=1).sum()

# =========================
# STEP 7: run for k=4..10
# =========================
results = []
num_range_arr = num_range.to_numpy(dtype=float) if num_cols else None

for k in range(4, 11):
    print(f"\n🔹 Running KMeans-Gower for k={k} ...")
<<<<<<< HEAD
    clusters, centroids = kmeans_gower(data_np, k, max_iter=100)

    inertia_gower = within_cluster_distance_sum(data_np, clusters, centroids)

    results.append({"k": k, "within_cluster_distance_sum": inertia_gower})
    print(f"✅ k={k}  Within-cluster distance sum = {inertia_gower:.6f}")

results_df = pd.DataFrame(results)
report_path = os.path.join(base_path, "..", "report", "within_cluster_distance_results.csv")
=======
    clusters, cent_num, cent_cat = kmeans_gower(df, k, num_cols, cat_cols, num_range, max_iter=100, seed=42)

    s_cent = sum_pairwise_centroid_distances(cent_num, cent_cat, num_range_arr)
    results.append({"k": k, "sum_pairwise_centroid_distances": float(s_cent)})

    print(f"✅ k={k}  Sum of centroid distances (Gower) = {s_cent:.6f}")

results_df = pd.DataFrame(results)

report_path = os.path.join(base_path, "..", "report", "centroid_distance_results.csv")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
>>>>>>> fd2b5a0 (Final implementation: Custom Gower KMeans + centroid distance metric + plot)
results_df.to_csv(report_path, index=False)

print("\n✅ Results saved to report/within_cluster_distance_results.csv")
print("\n📌 Final Results Table:")
print(results_df)

