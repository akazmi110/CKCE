# leiden_clustering.py

import csv
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cdlib import algorithms


def load_year_graphs_from_edgelists(edges_dir):
    edges_dir = Path(edges_dir)
    year_graphs = {}

    for csv_path in edges_dir.glob("*_edgelist.csv"):
        year_str = csv_path.stem.split("_")[0]
        try:
            year = int(year_str)
        except ValueError:
            continue  # skip files that don't match the pattern
        if year<2004 or year>2024:
            continue

        G = nx.Graph()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = row["source"]
                v = row["target"]
                w = float(row.get("weight", 1.0))
                G.add_edge(u, v, weight=w)
        year_graphs[year] = G
        print(f"Loaded graph for year {year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return year_graphs


def run_leiden(G):
    """
    Run Leiden community detection via cdlib.

    Returns
    -------
    partition : dict[node, cluster_id]
    """
    res = algorithms.leiden(G, weights="weight")
    partition = {}
    for cid, comm in enumerate(res.communities):
        for node in comm:
            partition[node] = cid
    return partition


def compute_temporal_stability(cluster_csv_path, stability_out_path, jaccard_threshold=0.0):
    cluster_csv_path = Path(cluster_csv_path)
    stability_out_path = Path(stability_out_path)
    stability_out_path.parent.mkdir(parents=True, exist_ok=True)

    year_partitions = defaultdict(dict)
    with open(cluster_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            kw = row["keyword"]
            cid = int(row["cluster_id"])
            year_partitions[year][kw] = cid

    years = sorted(year_partitions.keys())
    rows = []

    for i in range(len(years) - 1):
        y = years[i]
        y_next = years[i + 1]

        part_y = year_partitions[y]
        part_next = year_partitions[y_next]

        clusters_y = defaultdict(set)
        clusters_next = defaultdict(set)

        for kw, cid in part_y.items():
            clusters_y[cid].add(kw)
        for kw, cid in part_next.items():
            clusters_next[cid].add(kw)

        best_jaccards = []

        for cid, nodes_y in clusters_y.items():
            best_j = 0.0
            for cid2, nodes_next in clusters_next.items():
                inter = len(nodes_y & nodes_next)
                if inter == 0:
                    continue
                union = len(nodes_y | nodes_next)
                j = inter / union
                if j > best_j:
                    best_j = j
            if best_j >= jaccard_threshold:
                best_jaccards.append(best_j)

        if best_jaccards:
            best_jaccards = np.array(best_jaccards)
            mean_j = float(best_jaccards.mean())
            median_j = float(np.median(best_jaccards))
            std_j = float(best_jaccards.std())
            matched_clusters = len(best_jaccards)
        else:
            mean_j = median_j = std_j = 0.0
            matched_clusters = 0

        rows.append({
            "year": y,
            "next_year": y_next,
            "num_clusters_year": len(clusters_y),
            "num_clusters_next": len(clusters_next),
            "matched_clusters": matched_clusters,
            "mean_best_jaccard": mean_j,
            "median_best_jaccard": median_j,
            "std_best_jaccard": std_j,
        })

    with open(stability_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "next_year",
                "num_clusters_year",
                "num_clusters_next",
                "matched_clusters",
                "mean_best_jaccard",
                "median_best_jaccard",
                "std_best_jaccard",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Temporal stability saved to: {stability_out_path}")


def plot_temporal_stability(stability_csv_path, algo_name, out_dir):
    stability_csv_path = Path(stability_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years, mean_js = [], []

    with open(stability_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            years.append(int(row["year"]))
            mean_js.append(float(row["mean_best_jaccard"]))

    if not years:
        print(f"No stability data to plot for {algo_name}.")
        return

    plt.figure()
    plt.plot(years, mean_js, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean Best Jaccard")
    plt.title(f"Temporal Stability (Mean Jaccard) - {algo_name}")
    plt.tight_layout()
    out_path = out_dir / f"{algo_name}_temporal_stability.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved stability plot to: {out_path}")


def plot_num_clusters_over_time(cluster_csv_path, algo_name, out_dir):
    cluster_csv_path = Path(cluster_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_clusters = defaultdict(set)
    with open(cluster_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            cid = int(row["cluster_id"])
            year_clusters[year].add(cid)

    years_sorted = sorted(year_clusters.keys())
    num_clusters = [len(year_clusters[y]) for y in years_sorted]

    plt.figure()
    plt.plot(years_sorted, num_clusters, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Number of Clusters")
    plt.title(f"Number of Clusters over Time - {algo_name}")
    plt.tight_layout()
    out_path = out_dir / f"{algo_name}_num_clusters_over_time.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved cluster-count plot to: {out_path}")


if __name__ == "__main__":
    edges_dir = r"graphs using keyword processing\co_occurance_graphs"
    clusters_dir =r"graphs using keyword processing\clusters"
    algo_name = "leiden"

    clusters_dir = Path(clusters_dir)
    clusters_dir.mkdir(parents=True, exist_ok=True)

    year_graphs = load_year_graphs_from_edgelists(edges_dir)

    cluster_csv_path = clusters_dir / f"{algo_name}_clusters.csv"
    with open(cluster_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "keyword", "cluster_id"])

        for year in sorted(year_graphs.keys()):
            G = year_graphs[year]
            print(f"Running Leiden for year {year}...")
            partition = run_leiden(G)
            for kw, cid in partition.items():
                writer.writerow([year, kw, cid])

    print(f"Cluster assignments saved to: {cluster_csv_path}")

    stability_out_path = clusters_dir / f"{algo_name}_temporal_stability.csv"
    compute_temporal_stability(cluster_csv_path, stability_out_path)

    plot_temporal_stability(stability_out_path, algo_name, clusters_dir)
    plot_num_clusters_over_time(cluster_csv_path, algo_name, clusters_dir)
