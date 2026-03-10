import json
import itertools
import csv
from collections import defaultdict
from pathlib import Path

import networkx as nx


# ---------- 1. Load JSONL ----------

def load_papers(jsonl_path):
    """Load JSONL file into a list of dicts."""
    papers = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


# ---------- 2. Build yearwise co-occurrence graphs ----------

def build_yearwise_cooccurrence_graphs(papers, min_cooccurrence=1, lowercase=True):
    """
    Build a co-occurrence graph for each year.
    
    Parameters
    ----------
    papers : list[dict]
        Each dict must have keys: 'year', 'keywords'.
    min_cooccurrence : int
        Only keep edges whose weight >= this value.
    lowercase : bool
        If True, convert keywords to lowercase for consistency.
    
    Returns
    -------
    dict[int, nx.Graph]
        Mapping from year -> NetworkX Graph.
        
        
    Example edge list:
    source,target,weight
    graph theory,network analysis,1
    graph theory,link prediction,1
    network analysis,link prediction,1
    """
    # year -> ( (kw1, kw2) -> count )
    year_pair_counts = defaultdict(lambda: defaultdict(int))

    for p in papers:
        year = p["year"]
        kws = p.get("keywords", []) or []

        # Normalize keywords
        if lowercase:
            kws = [k.strip().lower() for k in kws]
        else:
            kws = [k.strip() for k in kws]

        # Remove duplicates within a paper
        unique_kws = sorted(set(kws))

        # Generate all unordered pairs of keywords in this paper
        for kw1, kw2 in itertools.combinations(unique_kws, 2):
            pair = tuple(sorted((kw1, kw2)))
            year_pair_counts[year][pair] += 1

    # Now build a graph per year
    year_graphs = {}

    for year, pair_counts in year_pair_counts.items():
        G = nx.Graph()

        for (kw1, kw2), w in pair_counts.items():
            if w >= min_cooccurrence:
                # Add nodes (NetworkX will auto-add if not present)
                G.add_node(kw1)
                G.add_node(kw2)
                # Add weighted edge
                G.add_edge(kw1, kw2, weight=w)

        year_graphs[year] = G

    return year_graphs


# ---------- 3. Save yearwise edge lists to CSV ----------

def save_yearwise_edgelists(year_graphs, out_dir):
    """
    Save edge lists for each year's graph as CSV: {year}_edgelist.csv
    
    Columns: source, target, weight
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year, G in year_graphs.items():
        out_path = out_dir / f"{year}_edgelist.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("source,target,weight\n")
            for u, v, data in G.edges(data=True):
                w = data.get("weight", 1)
                f.write(f"{u},{v},{w}\n")
        print(f"Saved {year} graph edges to: {out_path}")


# ---------- 4. Save yearwise statistics to CSV ----------

def save_yearwise_statistics(papers, year_graphs, out_csv_path):
    """
    Save statistics (papers, nodes, edges per year) to a CSV file.
    
    CSV Columns:
        year, total_papers, nodes, edges
    """
    # Count how many papers per year
    papers_per_year = defaultdict(int)
    for p in papers:
        papers_per_year[p["year"]] += 1

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "total_papers", "nodes", "edges"])

        for year in sorted(year_graphs.keys()):
            G = year_graphs[year]
            writer.writerow([
                year,
                papers_per_year[year],
                G.number_of_nodes(),
                G.number_of_edges()
            ])

    print(f"Yearwise statistics saved to: {out_csv_path}")


# ---------- 5. Run everything ----------

if __name__ == "__main__":
    jsonl_path = r"graphs using keyword processing\keyword_processing\dblp_cleaned_multi_free.openai-gpt.jsonl"  
    out_dir = r"clustering paper\data\co_occurance_graphs"

    papers = load_papers(jsonl_path)
    print(f"Loaded {len(papers)} papers")

    year_graphs = build_yearwise_cooccurrence_graphs(papers, min_cooccurrence=1)

    # Print some quick stats to console
    for year in sorted(year_graphs.keys()):
        G = year_graphs[year]
        print(f"Year {year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save edge lists for inspection
    save_yearwise_edgelists(year_graphs, out_dir=out_dir)

    # Save yearwise statistics
    stats_csv_path = Path(out_dir) / "yearwise_stats.csv"
    save_yearwise_statistics(papers, year_graphs, out_csv_path=stats_csv_path)
