# PhD Project – Custom K-Means with Gower Distance

## 📌 Project Overview

This project implements a custom version of the K-Means clustering algorithm using **Gower distance** instead of Euclidean distance, as required in the PhD Machine Learning course project.

The dataset used is the *Credit Card Dataset (CC GENERAL.csv)*.

---

## 📂 Dataset Information

- Number of samples: 8950
- Original number of features: 18
- After removing ID column: 17 features
- Data type: Financial numerical features

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

1. The column `CUST_ID` was removed since it represents only an identifier.
2. Missing values were handled as follows:
   - Numeric features → replaced with the **mean** of the feature.
   - Categorical features (if any) → replaced with the **mode**.
3. A fixed random seed (42) was used to ensure reproducibility.

---

## 📏 Gower Distance

Since Euclidean distance is not suitable for mixed-type datasets, a custom implementation of **Gower distance** was developed.

For each feature:

- Numeric:

  d(i,j) = |x_i - x_j| / range

- Categorical:

  d(i,j) = 0 (if equal)  
  d(i,j) = 1 (if different)

The final Gower distance is computed as the average over all features.

---

## ⚙️ Custom K-Means Algorithm

The clustering algorithm was implemented from scratch with:

1. Random initialization of centroids (seed = 42)
2. Assignment step using Gower distance
3. Update step:
   - Numeric features → mean of cluster members
   - Categorical features → mode of cluster members
4. Maximum iterations: 100
5. Convergence when cluster assignments no longer change

---

## 📊 Required Evaluation Metric

For each k in {4, 5, 6, 7, 8, 9, 10}, the following metric was computed:

**Sum of pairwise Gower distances between all cluster centroids**

S(k) = Σ D(c_a, c_b) for a < b

This metric measures how separated the cluster centers are from each other.

---

## 📈 Final Results

| k  | Sum of Pairwise Centroid Distances |
|----|------------------------------------|
| 4  | 0.760745 |
| 5  | 1.193518 |
| 6  | 1.982623 |
| 7  | 2.960903 |
| 8  | 3.895007 |
| 9  | 5.078088 |
| 10 | 6.231211 |

The results show a monotonic increase as k increases, which is expected since increasing the number of clusters increases the number of centroid pairs.

The numerical results are stored in:

`report/centroid_distance_results.csv`

---

## 📈 Visualization

The following plot shows the trend of the sum of pairwise centroid distances for k = 4 to 10.

![Centroid Distance Plot](report/centroid_distance_plot.png)

---

## ▶️ How to Run

From the project root:

```bash
python src/main.py


## 🗂️ Project Structure

```text
PhD_Project_KMeans/
├── data/
│   └── CC GENERAL.csv
├── report/
│   ├── centroid_distance_results.csv
│   └── centroid_distance_plot.png
├── src/
│   ├── main.py
│   └── plot_centroid_distances.py
├── requirements.txt
├── README.md
└── .gitignore
