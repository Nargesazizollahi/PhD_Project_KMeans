import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(base_path, "..", "report", "centroid_distance_results.csv")
df = pd.read_csv(csv_path)

plt.figure()
plt.plot(df["k"], df["sum_pairwise_centroid_distances"], marker="o")
plt.xlabel("k")
plt.ylabel("Sum of pairwise centroid distances (Gower)")
plt.title("Centroid separation vs k")
plt.grid(True)

out_path = os.path.join(base_path, "..", "report", "centroid_distance_plot.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")

print("✅ Saved plot to:", out_path)
