import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# STEP 1: Load Data
# ============================================================
import os

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "..", "data", "CC GENERAL.csv")

df = pd.read_csv(data_path)


print("✅ Original Shape:", df.shape)

# ============================================================
# STEP 2: Drop CUST_ID
# ============================================================
df.drop(columns=["CUST_ID"], inplace=True)

print("✅ After dropping CUST_ID:", df.shape)

# ============================================================
# STEP 3: Fill missing values (numeric -> mean)
# ============================================================
missing_before = df.isnull().sum().sum()
print("🔸 Missing values before filling:", missing_before)

for col in df.columns:
    df[col] = df[col].fillna(df[col].mean())

missing_after = df.isnull().sum().sum()
print("✅ Missing values after filling:", missing_after)

# ============================================================
# STEP 4: Normalize numeric features (Min-Max Scaling)
# ============================================================
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("✅ Data scaled successfully!")
print(df_scaled.head())

# Save cleaned data
clean_data_path = os.path.join(base_path, "..", "data", "CC_GENERAL_clean_scaled.csv")
df_scaled.to_csv(clean_data_path, index=False)
print("✅ Saved cleaned & scaled dataset to: data/CC_GENERAL_clean_scaled.csv")

# ============================================================
# STEP 5: Define Gower Distance (Numeric Case)
# ============================================================
def gower_distance_numeric(x, y):
    """
    Gower distance for numeric scaled data: mean absolute difference
    """
    return np.mean(np.abs(x - y))

# Test Gower distance
x0 = df_scaled.iloc[0].values
x1 = df_scaled.iloc[1].values
dist_test = gower_distance_numeric(x0, x1)
print("\n✅ Gower Distance Test (row0 vs row1):", dist_test)

# ============================================================
# STEP 6: Custom KMeans using Vectorized Gower Distance
# ============================================================
def assign_clusters(data, centroids):
    """
    Fast assignment using vectorized Gower distance (numeric case).
    distances shape = (n_samples, k)
    """
    distances = np.mean(np.abs(data[:, None, :] - centroids[None, :, :]), axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, clusters, k):
    """
    Update centroids by taking the mean of points in each cluster.
    """
    new_centroids = []
    for cluster_id in range(k):
        points = data[clusters == cluster_id]
        if len(points) == 0:
            # Empty cluster fix: choose random point
            new_centroids.append(data[np.random.randint(0, len(data))])
        else:
            new_centroids.append(points.mean(axis=0))
    return np.array(new_centroids)

def kmeans_gower(data, k, max_iter=100):
    """
    Custom K-Means algorithm using Gower distance (numeric case).
    """
    np.random.seed(42)

    # Random init centroids
    random_idx = np.random.choice(len(data), k, replace=False)
    centroids = data[random_idx]

    for iteration in range(max_iter):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        # Convergence check
        if np.allclose(centroids, new_centroids):
            print(f"✅ Converged at iteration {iteration+1}")
            break

        centroids = new_centroids

    return clusters, centroids

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


# ============================================================
# STEP 8: Run KMeans for k=4..10 and store results
# ============================================================
data_np = df_scaled.values
results = []

for k in range(4, 11):
    print(f"\n🔹 Running KMeans-Gower for k={k} ...")
    clusters, centroids = kmeans_gower(data_np, k, max_iter=100)

    inertia_gower = within_cluster_distance_sum(data_np, clusters, centroids)

    results.append({"k": k, "within_cluster_distance_sum": inertia_gower})
    print(f"✅ k={k}  Within-cluster distance sum = {inertia_gower:.6f}")

results_df = pd.DataFrame(results)
report_path = os.path.join(base_path, "..", "report", "within_cluster_distance_results.csv")
results_df.to_csv(report_path, index=False)

print("\n✅ Results saved to report/within_cluster_distance_results.csv")
print("\n📌 Final Results Table:")
print(results_df)

