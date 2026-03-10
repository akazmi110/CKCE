# step9_compute_ami_ari.py
# Reads saved clustering results from multiple algorithms
# Computes AMI/ARI yearwise + summary + plots
# NOTE: All computations are filtered to the analytical window 2004-2024.
# Pre-2004 years are excluded due to sparse publication activity.
# 2025 is excluded due to incomplete indexing at time of data collection.

import csv
from pathlib import Path
from collections import defaultdict
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

# -----------------------------
# 1. CONFIG
# -----------------------------

CLUSTERS_DIR = Path(r"graphs using keyword processing\clusters")

ALGORITHMS = [
    "louvain",
    "leiden",
    "infomap",
    "label_propagation",
    "walktrap",
    "node2vec",
    "deepwalk",
]

# Analytical window — consistent with network construction and clustering steps
YEAR_START = 2004
YEAR_END   = 2024

OUT_DIR = CLUSTERS_DIR / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. LOAD CLUSTERS
# -----------------------------

def load_algorithm_clusters(algo_name):
    """
    Loads algo_clusters.csv into:
    year -> {keyword: cluster_id}

    Only years within [YEAR_START, YEAR_END] are loaded.
    """
    path = CLUSTERS_DIR / f"{algo_name}_clusters.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster file: {path}")

    year_map = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = int(row["year"])

            # Filter to analytical window
            if year < YEAR_START or year > YEAR_END:
                continue

            kw  = row["keyword"]
            cid = int(row["cluster_id"])
            year_map[year][kw] = cid

    print(f"  Loaded {algo_name}: {len(year_map)} years "
          f"({min(year_map)} - {max(year_map)})")
    return year_map


def intersect_keywords(map_a, map_b):
    return sorted(set(map_a.keys()) & set(map_b.keys()))


# -----------------------------
# 3. COMPUTE YEARWISE AMI / ARI
# -----------------------------

def compute_yearwise_pair_scores(year_clusters):
    """
    year_clusters: dict[algo -> dict[year -> dict[keyword->cid]]]

    Returns:
      ami_scores[year][(algo1,algo2)] = score
      ari_scores[year][(algo1,algo2)] = score
      years_sorted
    """
    # Only include years in analytical window
    all_years = sorted(
        set(itertools.chain.from_iterable(
            yc.keys() for yc in year_clusters.values()
        ))
    )
    all_years = [y for y in all_years if YEAR_START <= y <= YEAR_END]

    ami_scores = defaultdict(dict)
    ari_scores = defaultdict(dict)

    for year in all_years:
        for a1, a2 in itertools.combinations(ALGORITHMS, 2):
            if year not in year_clusters[a1] or year not in year_clusters[a2]:
                continue

            c1 = year_clusters[a1][year]
            c2 = year_clusters[a2][year]

            common_kws = intersect_keywords(c1, c2)
            if len(common_kws) < 2:
                ami = ari = np.nan
            else:
                labels1 = [c1[k] for k in common_kws]
                labels2 = [c2[k] for k in common_kws]
                ami = adjusted_mutual_info_score(labels1, labels2)
                ari = adjusted_rand_score(labels1, labels2)

            ami_scores[year][(a1, a2)] = ami
            ari_scores[year][(a1, a2)] = ari

    return ami_scores, ari_scores, all_years


# -----------------------------
# 4. SAVE CSVs
# -----------------------------

def save_yearwise_csv(scores, years, out_path, metric_name="AMI"):
    """
    scores[year][(a1,a2)] = value
    Output columns: year, algo1, algo2, metric
    """
    rows = []
    for y in years:
        for (a1, a2), val in scores[y].items():
            rows.append({
                "year":      y,
                "algo1":     a1,
                "algo2":     a2,
                metric_name: val
            })

    out_path = Path(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["year", "algo1", "algo2", metric_name]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved yearwise {metric_name} to: {out_path}")


def save_summary_csv(scores, years, out_path, metric_name="AMI"):
    """
    Mean±Std across years for each pair.
    Output columns: algo1, algo2, mean, std, num_years
    """
    pair_values = defaultdict(list)

    for y in years:
        for pair, val in scores[y].items():
            if not np.isnan(val):
                pair_values[pair].append(val)

    rows = []
    for (a1, a2), vals in pair_values.items():
        arr = np.array(vals)
        rows.append({
            "algo1":              a1,
            "algo2":              a2,
            f"{metric_name}_mean": float(arr.mean()) if len(arr) else np.nan,
            f"{metric_name}_std":  float(arr.std())  if len(arr) else np.nan,
            "num_years":           len(arr)
        })

    out_path = Path(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo1", "algo2",
                f"{metric_name}_mean",
                f"{metric_name}_std",
                "num_years"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary {metric_name} to: {out_path}")


# -----------------------------
# 5. PLOTTING
# -----------------------------

def plot_heatmap_for_year(scores_for_year, year, metric_name, out_dir):
    """
    Square matrix heatmap between algorithms for one year.
    """
    algos = ALGORITHMS
    mat   = np.full((len(algos), len(algos)), np.nan)

    for i in range(len(algos)):
        mat[i, i] = 1.0

    for (a1, a2), val in scores_for_year.items():
        i = algos.index(a1)
        j = algos.index(a2)
        mat[i, j] = val
        mat[j, i] = val

    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect="auto", vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(algos)), algos, rotation=45, ha="right")
    plt.yticks(range(len(algos)), algos)
    plt.title(f"{metric_name} Heatmap — Year {year} (2004–2024 window)")
    plt.tight_layout()

    out_path = Path(out_dir) / f"{metric_name.lower()}_heatmap_{year}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {out_path}")


def plot_mean_metric_over_time(scores, years, metric_name, out_dir):
    """
    For each algo pair, plot metric vs time (2004-2024 only).
    """
    pairs = list(itertools.combinations(ALGORITHMS, 2))

    plt.figure(figsize=(12, 5))
    for (a1, a2) in pairs:
        ys   = []
        vals = []
        for y in years:
            v = scores[y].get((a1, a2), np.nan)
            if not np.isnan(v):
                ys.append(y)
                vals.append(v)
        if ys:
            plt.plot(ys, vals, marker="o", markersize=3,
                     label=f"{a1}-{a2}")

    plt.xlabel("Year")
    plt.ylabel(metric_name)
    plt.xlim(YEAR_START - 0.5, YEAR_END + 0.5)
    plt.title(f"{metric_name} Over Time — All Algorithm Pairs (2004–2024)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    out_path = Path(out_dir) / f"{metric_name.lower()}_over_time.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved time plot: {out_path}")


# -----------------------------
# 6. MAIN
# -----------------------------

if __name__ == "__main__":
    print(f"Analytical window: {YEAR_START}–{YEAR_END}")
    print("=" * 50)

    # Load cluster maps (filtered to 2004-2024 inside loader)
    year_clusters = {}
    for algo in ALGORITHMS:
        print(f"Loading clusters for {algo}...")
        year_clusters[algo] = load_algorithm_clusters(algo)

    # Compute yearwise scores
    ami_scores, ari_scores, years_sorted = compute_yearwise_pair_scores(year_clusters)
    print(f"\nYears computed: {years_sorted}")

    # Save yearwise CSVs
    save_yearwise_csv(ami_scores, years_sorted,
                      OUT_DIR / "ami_yearwise.csv", "AMI")
    save_yearwise_csv(ari_scores, years_sorted,
                      OUT_DIR / "ari_yearwise.csv", "ARI")

    # Save summary CSVs (mean ± std across 2004-2024)
    save_summary_csv(ami_scores, years_sorted,
                     OUT_DIR / "ami_summary_2004_2024.csv", "AMI")
    save_summary_csv(ari_scores, years_sorted,
                     OUT_DIR / "ari_summary_2004_2024.csv", "ARI")

    # Heatmaps per year
    heatmap_dir = OUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for y in years_sorted:
        if ami_scores[y]:
            plot_heatmap_for_year(ami_scores[y], y, "AMI", heatmap_dir)
        if ari_scores[y]:
            plot_heatmap_for_year(ari_scores[y], y, "ARI", heatmap_dir)

    # Time series plots
    plot_mean_metric_over_time(ami_scores, years_sorted, "AMI", OUT_DIR)
    plot_mean_metric_over_time(ari_scores, years_sorted, "ARI", OUT_DIR)

    print("\n✅ AMI/ARI evaluation finished.")
    print(f"   Outputs saved to: {OUT_DIR}")