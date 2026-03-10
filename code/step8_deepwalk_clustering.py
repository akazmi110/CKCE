# step8_deepwalk_clustering.py  (pure deepwalk + silhouette-k)

import csv, random, math
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
                u, v = row["source"], row["target"]
                w = float(row.get("weight", 1.0))
                G.add_edge(u, v, weight=w)

        year_graphs[year] = G
        print(f"Loaded graph for year {year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return year_graphs


# ---------- DeepWalk random walks ----------

def deepwalk_walk(G, start, walk_length=40):
    walk = [start]
    for _ in range(walk_length - 1):
        cur = walk[-1]
        neigh = list(G.neighbors(cur))
        if not neigh:
            break
        walk.append(random.choice(neigh))
    return walk


def generate_deepwalk_walks(G, num_walks=80, walk_length=40, seed=42):
    random.seed(seed)
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for n in nodes:
            w = deepwalk_walk(G, n, walk_length=walk_length)
            walks.append([str(x) for x in w])
    return walks


# ---------- Silhouette-based K selection ----------

def choose_best_k_silhouette(embeddings, year, algo_name, out_dir,
                             k_min=2, k_max=50, seed=42):
    n = embeddings.shape[0]
    if n < 3:
        return max(1, n), []

    k_max = min(k_max, n - 1)
    k_candidates = list(range(k_min, k_max + 1))

    scores = []
    best_k = k_candidates[0]
    best_score = -1

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue

        s = silhouette_score(embeddings, labels, metric="euclidean")
        scores.append((k, s))
        if s > best_score:
            best_score = s
            best_k = k

    if scores:
        ks = [x[0] for x in scores]
        ss = [x[1] for x in scores]
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(ks, ss, marker="o")
        plt.xlabel("k (num clusters)")
        plt.ylabel("Silhouette score")
        plt.title(f"Silhouette vs k - {algo_name} ({year})")
        plt.tight_layout()
        out_path = out_dir / f"{algo_name}_silhouette_{year}.png"
        plt.savefig(out_path)
        plt.close()

    return best_k, scores


def run_deepwalk_clustering(G, year, dim=128, walk_length=40, num_walks=80,
                            seed=42, silhouette_dir=None):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}

    walks = generate_deepwalk_walks(G, num_walks=num_walks, walk_length=walk_length, seed=seed)

    w2v = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=10,
        min_count=0,
        sg=1,
        workers=4,
        epochs=5,
        seed=seed
    )

    emb = np.array([w2v.wv[str(n)] for n in nodes])

    best_k, _ = choose_best_k_silhouette(
        emb, year, "deepwalk", silhouette_dir, k_min=2, k_max=50, seed=seed
    )

    print(f"  DeepWalk ({year}): {len(nodes)} nodes, silhouette-best k={best_k}")

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = km.fit_predict(emb)

    return {node: int(cid) for node, cid in zip(nodes, labels)}


# ---------- Temporal stability + plotting ----------

def compute_temporal_stability(cluster_csv_path, stability_out_path, jaccard_threshold=0.0):
    year_partitions = defaultdict(dict)
    with open(cluster_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_partitions[int(row["year"])][row["keyword"]] = int(row["cluster_id"])

    years = sorted(year_partitions.keys())
    rows = []

    for i in range(len(years)-1):
        y, y_next = years[i], years[i+1]

        clusters_y = defaultdict(set)
        clusters_next = defaultdict(set)

        for kw, cid in year_partitions[y].items():
            clusters_y[cid].add(kw)
        for kw, cid in year_partitions[y_next].items():
            clusters_next[cid].add(kw)

        best_js = []
        for cid, set_y in clusters_y.items():
            best = 0.0
            for cid2, set_n in clusters_next.items():
                inter = len(set_y & set_n)
                if inter == 0:
                    continue
                best = max(best, inter / len(set_y | set_n))
            if best >= jaccard_threshold:
                best_js.append(best)

        if best_js:
            arr = np.array(best_js)
            mean_j, median_j, std_j = float(arr.mean()), float(np.median(arr)), float(arr.std())
            matched = len(arr)
        else:
            mean_j = median_j = std_j = 0.0
            matched = 0

        rows.append({
            "year": y,
            "next_year": y_next,
            "num_clusters_year": len(clusters_y),
            "num_clusters_next": len(clusters_next),
            "matched_clusters": matched,
            "mean_best_jaccard": mean_j,
            "median_best_jaccard": median_j,
            "std_best_jaccard": std_j
        })

    with open(stability_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Temporal stability saved: {stability_out_path}")


def plot_temporal_stability(stability_csv_path, algo_name, out_dir):
    years, mean_js = [], []
    with open(stability_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            years.append(int(row["year"]))
            mean_js.append(float(row["mean_best_jaccard"]))

    plt.figure()
    plt.plot(years, mean_js, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean Best Jaccard")
    plt.title(f"Temporal Stability - {algo_name}")
    plt.tight_layout()
    out_path = Path(out_dir) / f"{algo_name}_temporal_stability.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved stability plot: {out_path}")


def plot_num_clusters_over_time(cluster_csv_path, algo_name, out_dir):
    year_clusters = defaultdict(set)
    with open(cluster_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_clusters[int(row["year"])].add(int(row["cluster_id"]))

    ys = sorted(year_clusters.keys())
    ks = [len(year_clusters[y]) for y in ys]

    plt.figure()
    plt.plot(ys, ks, marker="o")
    plt.xlabel("Year")
    plt.ylabel("#Clusters")
    plt.title(f"Clusters per Year - {algo_name}")
    plt.tight_layout()
    out_path = Path(out_dir) / f"{algo_name}_num_clusters_over_time.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved cluster-count plot: {out_path}")


if __name__ == "__main__":
    edges_dir = r"graphs using keyword processing\co_occurance_graphs"
    clusters_dir =r"graphs using keyword processing\clusters"
    algo_name = "deepwalk"

    silhouette_dir = clusters_dir / "silhouette_curves"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    silhouette_dir.mkdir(parents=True, exist_ok=True)

    year_graphs = load_year_graphs_from_edgelists(edges_dir)

    cluster_csv_path = clusters_dir / f"{algo_name}_clusters.csv"
    with open(cluster_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "keyword", "cluster_id"])

        for year in sorted(year_graphs.keys()):
            print(f"Running DeepWalk for year {year}...")
            part = run_deepwalk_clustering(
                year_graphs[year],
                year=year,
                silhouette_dir=silhouette_dir
            )
            for kw, cid in part.items():
                writer.writerow([year, kw, cid])

    print(f"Clusters saved: {cluster_csv_path}")

    stability_out = clusters_dir / f"{algo_name}_temporal_stability.csv"
    compute_temporal_stability(cluster_csv_path, stability_out)

    plot_temporal_stability(stability_out, algo_name, clusters_dir)
    plot_num_clusters_over_time(cluster_csv_path, algo_name, clusters_dir)
