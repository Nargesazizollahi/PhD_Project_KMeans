import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("report/centroid_distance_results.csv")

print("✅ Results loaded successfully!")
print(df)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(df["k"], df["centroid_distance_sum"], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of centroid distances (Gower)")
plt.title("Centroid Distance Sum for k=4 to k=10 (Gower Distance)")
plt.grid(True)

# Save plot
plt.savefig("report/centroid_distance_plot.png", dpi=300)
print("✅ Plot saved to report/centroid_distance_plot.png")

plt.show()
