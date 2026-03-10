"""
Community Vitality Trajectory Analysis (CVTA)
==============================================
A framework for detecting and classifying the evolutionary momentum
of keyword communities in temporal co-occurrence networks.

Terminology (original, not derived from any prior framework):
  - Community Vitality Score (CVS)      : fraction of network co-occurrence weight within a community
  - Compound Momentum Rate (CMR)        : geometric-mean annual growth of CVS
  - Trajectory Zones:
      IGNITION   — low CVS, positive CMR  (small but accelerating)
      ASCENDANT  — high CVS, positive CMR (large and growing)
      RESIDUAL   — high CVS, negative CMR (large but decelerating)
      PERIPHERAL — low CVS, negative CMR  (small and contracting)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load edgelists and cluster assignments
# ─────────────────────────────────────────────────────────────────────────────

def load_edgelists(edgelist_dir, year_start=2004, year_end=2024):
    """
    Load yearly keyword co-occurrence edgelists from CSV files.
    Expected columns: source, target, weight  (or node1, node2, weight)
    Returns dict: {year: pd.DataFrame}
    """
    edgelists = {}
    for year in range(year_start, year_end + 1):
        path = os.path.join(edgelist_dir, f"{year}_edgelist.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            if c in ("source", "node1", "from", "keyword1", "src"):
                col_map[c] = "source"
            elif c in ("target", "node2", "to", "keyword2", "dst"):
                col_map[c] = "target"
            elif c in ("weight", "w", "count", "freq", "co_occurrence"):
                col_map[c] = "weight"
        df = df.rename(columns=col_map)
        if "weight" not in df.columns:
            df["weight"] = 1.0
        edgelists[year] = df[["source", "target", "weight"]]
    print(f"[load_edgelists] Loaded {len(edgelists)} yearly edgelists "
          f"({min(edgelists)} – {max(edgelists)})")
    return edgelists


def load_cluster_assignments(cluster_csv, year_start=2004, year_end=2024):
    """
    Load Leiden (or any algorithm) cluster assignments.
    Expected columns: year, keyword, cluster_id
    Returns dict: {year: {keyword: cluster_id}}
    """
    df = pd.read_csv(cluster_csv)
    df.columns = [c.lower().strip() for c in df.columns]
    assignments = {}
    for year in range(year_start, year_end + 1):
        sub = df[df["year"] == year]
        if sub.empty:
            continue
        assignments[year] = dict(zip(sub["keyword"], sub["cluster_id"]))
    print(f"[load_clusters] Loaded cluster assignments for "
          f"{len(assignments)} years.")
    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Compute Community Vitality Score (CVS)
#          CVS(C, t) = intra-community edge weight / total network edge weight
# ─────────────────────────────────────────────────────────────────────────────

def compute_cvs(edgelists, assignments, year_start=2004, year_end=2024):
    """
    For every (year, cluster_id) pair, compute the Community Vitality Score.

    CVS(C, t) = vol(C, t) / W(t)

    where vol(C, t)  = sum of weights for edges with BOTH endpoints in C
          W(t)       = sum of ALL edge weights in year t

    Returns: pd.DataFrame with columns [year, cluster_id, cvs, vol, total_w, n_nodes]
    """
    records = []
    for year in range(year_start, year_end + 1):
        if year not in edgelists or year not in assignments:
            continue
        edges = edgelists[year]
        node2cluster = assignments[year]

        total_w = edges["weight"].sum()
        if total_w == 0:
            continue

        community_vol = {}
        for _, row in edges.iterrows():
            c_src = node2cluster.get(row["source"])
            c_tgt = node2cluster.get(row["target"])
            if c_src is None or c_tgt is None:
                continue
            if c_src == c_tgt:
                community_vol[c_src] = community_vol.get(c_src, 0.0) + row["weight"]

        cluster_sizes = {}
        for kw, cid in node2cluster.items():
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        for cid, vol in community_vol.items():
            records.append({
                "year":       year,
                "cluster_id": cid,
                "cvs":        vol / total_w,
                "vol":        vol,
                "total_w":    total_w,
                "n_nodes":    cluster_sizes.get(cid, 0)
            })

    df = pd.DataFrame(records)
    print(f"[compute_cvs] Computed CVS for {len(df)} community-year pairs.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build Community Chains via Jaccard continuity
# ─────────────────────────────────────────────────────────────────────────────

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def build_community_chains(assignments, jaccard_threshold=0.0,
                           year_start=2004, year_end=2024):
    """
    Link communities across consecutive years by Jaccard keyword overlap.
    Returns list of chains, where each chain is a list of (year, cluster_id) tuples.

    jaccard_threshold: minimum J(C_t, C_{t+1}) to form a link (default 0.0 = any overlap)
    """
    years = sorted(y for y in range(year_start, year_end + 1) if y in assignments)

    kw_sets = {}
    for year in years:
        for kw, cid in assignments[year].items():
            key = (year, cid)
            kw_sets.setdefault(key, set()).add(kw)

    forward_links = {}
    for i in range(len(years) - 1):
        y_t, y_t1 = years[i], years[i + 1]
        clusters_t  = [cid for (y, cid) in kw_sets if y == y_t]
        clusters_t1 = [cid for (y, cid) in kw_sets if y == y_t1]
        for c_t in clusters_t:
            best_j, best_c = 0.0, None
            s_t = kw_sets.get((y_t, c_t), set())
            for c_t1 in clusters_t1:
                s_t1 = kw_sets.get((y_t1, c_t1), set())
                j = jaccard(s_t, s_t1)
                if j > best_j:
                    best_j, best_c = j, c_t1
            if best_j > jaccard_threshold and best_c is not None:
                forward_links[(y_t, c_t)] = (y_t1, best_c, best_j)

    visited = set()
    chains  = []
    for node in sorted(kw_sets.keys()):
        if node in visited:
            continue
        chain = [node]
        visited.add(node)
        cur = node
        while cur in forward_links:
            nxt_year, nxt_cid, _ = forward_links[cur]
            nxt = (nxt_year, nxt_cid)
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        chains.append(chain)

    print(f"[build_chains] Built {len(chains)} community chains "
          f"(threshold J>{jaccard_threshold}). "
          f"Chains ≥3 years: {sum(1 for c in chains if len(c) >= 3)}")
    return chains, kw_sets


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Compute Compound Momentum Rate (CMR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cmr(chains, cvs_df, min_chain_length=3):
    """
    For each chain (≥ min_chain_length years), compute:
      mean_cvs  — time-averaged CVS along the chain
      cmr       — compound momentum rate (geometric-mean annual growth of CVS)
      lifespan  — number of years active

    Returns: pd.DataFrame with columns [chain_id, mean_cvs, cmr, lifespan,
                                         first_year, last_year, node_list]
    """
    cvs_lookup = {(row.year, row.cluster_id): row.cvs
                  for row in cvs_df.itertuples()}

    records = []
    for chain_id, chain in enumerate(chains):
        if len(chain) < min_chain_length:
            continue

        cvs_vals = [cvs_lookup.get(node, 0.0) for node in chain]
        valid    = [(chain[i], v) for i, v in enumerate(cvs_vals) if v > 0]
        if len(valid) < min_chain_length:
            continue

        valid_cvs = [v for _, v in valid]
        mean_cvs  = np.mean(valid_cvs)

        ratios = []
        for i in range(len(valid_cvs) - 1):
            if valid_cvs[i] > 0:
                ratios.append(valid_cvs[i + 1] / valid_cvs[i])

        cmr = np.prod(ratios) ** (1.0 / len(ratios)) - 1.0 if ratios else -1.0

        years_active = [n[0] for n, _ in valid]
        records.append({
            "chain_id":   chain_id,
            "mean_cvs":   mean_cvs,
            "cmr":        cmr,
            "lifespan":   len(valid),
            "first_year": min(years_active),
            "last_year":  max(years_active),
            "node_list":  chain
        })

    df = pd.DataFrame(records)
    print(f"[compute_cmr] Computed CMR for {len(df)} chains "
          f"(min length {min_chain_length}).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Classify into Trajectory Zones
# ─────────────────────────────────────────────────────────────────────────────

ZONE_DESCRIPTIONS = {
    "IGNITION":   "Low activity, growing fast — nascent research frontier",
    "ASCENDANT":  "High activity, still growing — dominant expanding theme",
    "RESIDUAL":   "High activity, decelerating — established but plateauing",
    "PERIPHERAL": "Low activity, contracting — marginal or fading theme",
}


def classify_zones(chain_df, cvs_threshold=None, cmr_threshold=0.0):
    """
    Assign each chain to a Trajectory Zone based on its mean_cvs and cmr.

    cvs_threshold: horizontal split (default: median of mean_cvs)
    cmr_threshold: vertical split  (default: 0.0)
    """
    if cvs_threshold is None:
        cvs_threshold = chain_df["mean_cvs"].median()

    def zone(row):
        high_cvs = row["mean_cvs"] >= cvs_threshold
        pos_cmr  = row["cmr"] >= cmr_threshold
        if not high_cvs and pos_cmr:
            return "IGNITION"
        elif high_cvs and pos_cmr:
            return "ASCENDANT"
        elif high_cvs and not pos_cmr:
            return "RESIDUAL"
        else:
            return "PERIPHERAL"

    chain_df = chain_df.copy()
    chain_df["zone"]          = chain_df.apply(zone, axis=1)
    chain_df["cvs_threshold"] = cvs_threshold
    chain_df["cmr_threshold"] = cmr_threshold

    print("\n[classify_zones] Trajectory Zone distribution:")
    for z, grp in chain_df.groupby("zone"):
        print(f"  {z:12s}: {len(grp):4d} chains  — {ZONE_DESCRIPTIONS[z]}")
    return chain_df, cvs_threshold


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Attach top-keyword labels to each chain
# ─────────────────────────────────────────────────────────────────────────────

def attach_keyword_labels(chain_df, assignments, edgelists, top_n=3):
    """
    For each chain, find its keywords in the most recent year and rank by
    weighted degree. Attaches a 'label' column to chain_df.
    """
    labels = []
    for _, row in chain_df.iterrows():
        last_node    = row["node_list"][-1]
        year, cid    = last_node
        if year not in assignments or year not in edgelists:
            labels.append("")
            continue
        kws_in_cluster = {kw for kw, c in assignments[year].items() if c == cid}
        edges  = edgelists[year]
        degree = {kw: 0.0 for kw in kws_in_cluster}
        for _, er in edges.iterrows():
            s, t, w = er["source"], er["target"], er["weight"]
            if s in degree and t in degree:
                degree[s] += w
                degree[t] += w
        top_kws = sorted(degree, key=degree.get, reverse=True)[:top_n]
        labels.append(" | ".join(top_kws) if top_kws else "")
    chain_df        = chain_df.copy()
    chain_df["label"] = labels
    return chain_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Temporal Phase Analysis
# ─────────────────────────────────────────────────────────────────────────────

PHASES = {
    "Phase-I   [2004–2010]": (2004, 2010),
    "Phase-II  [2011–2017]": (2011, 2017),
    "Phase-III [2018–2024]": (2018, 2024),
}


def phase_analysis(chains, kw_sets, cvs_df, assignments, edgelists,
                   min_chain_length=3, output_dir="."):
    """
    For each temporal phase, run CVS → CMR → classify pipeline
    restricted to community-years within that phase.
    Saves per-phase chain CSVs and returns a summary DataFrame.
    """
    summary_records = []
    for phase_label, (y_start, y_end) in PHASES.items():
        cvs_phase = cvs_df[(cvs_df["year"] >= y_start) & (cvs_df["year"] <= y_end)]
        if cvs_phase.empty:
            continue

        phase_chains = []
        for chain in chains:
            sub = [(yr, cid) for yr, cid in chain if y_start <= yr <= y_end]
            if len(sub) >= min_chain_length:
                phase_chains.append(sub)

        if not phase_chains:
            continue

        chain_df = compute_cmr(
            [[(yr, cid) for yr, cid in c] for c in phase_chains],
            cvs_phase,
            min_chain_length=min_chain_length
        )
        if chain_df.empty:
            continue

        chain_df, _ = classify_zones(chain_df)
        chain_df    = attach_keyword_labels(chain_df, assignments, edgelists)

        safe_label = (phase_label.replace(" ", "_")
                                 .replace("[", "")
                                 .replace("]", "")
                                 .replace("–", "-"))
        chain_df.drop(columns=["node_list"]).to_csv(
            os.path.join(output_dir, f"chains_{safe_label}.csv"), index=False
        )

        zone_counts           = chain_df["zone"].value_counts().to_dict()
        zone_counts["phase"]  = phase_label
        zone_counts["n_chains"] = len(chain_df)
        summary_records.append(zone_counts)

    summary = pd.DataFrame(summary_records).fillna(0)
    cols    = ["phase", "n_chains", "IGNITION", "ASCENDANT", "RESIDUAL", "PERIPHERAL"]
    cols    = [c for c in cols if c in summary.columns]
    summary = summary[cols]
    print("\n[phase_analysis] Phase-wise zone summary:")
    print(summary.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Zone migration tracker
# ─────────────────────────────────────────────────────────────────────────────

def migration_report(chains, kw_sets, cvs_df, assignments, edgelists,
                     min_chain_length=3):
    """
    Detect chains that changed zone between consecutive temporal phases.
    Returns a DataFrame of transitions.
    """
    phase_keys       = list(PHASES.keys())
    phase_chain_zones = {}

    for phase_label, (y_start, y_end) in PHASES.items():
        cvs_phase    = cvs_df[(cvs_df["year"] >= y_start) & (cvs_df["year"] <= y_end)]
        phase_chains = []
        chain_ids    = []
        for i, chain in enumerate(chains):
            sub = [(yr, cid) for yr, cid in chain if y_start <= yr <= y_end]
            if len(sub) >= min_chain_length:
                phase_chains.append(sub)
                chain_ids.append(i)

        if not phase_chains:
            phase_chain_zones[phase_label] = {}
            continue

        chain_df = compute_cmr(
            [[(yr, cid) for yr, cid in c] for c in phase_chains],
            cvs_phase, min_chain_length=min_chain_length
        )
        if chain_df.empty:
            phase_chain_zones[phase_label] = {}
            continue

        chain_df, _ = classify_zones(chain_df)
        chain_df    = attach_keyword_labels(chain_df, assignments, edgelists)

        zone_map = {}
        for local_idx, orig_id in enumerate(chain_ids[:len(chain_df)]):
            row = chain_df.iloc[local_idx]
            zone_map[orig_id] = (row["zone"], row["label"])
        phase_chain_zones[phase_label] = zone_map

    transitions = []
    for i in range(len(phase_keys) - 1):
        p1, p2 = phase_keys[i], phase_keys[i + 1]
        zm1    = phase_chain_zones.get(p1, {})
        zm2    = phase_chain_zones.get(p2, {})
        for chain_id in zm1:
            if chain_id in zm2:
                z1, lbl1 = zm1[chain_id]
                z2, lbl2 = zm2[chain_id]
                if z1 != z2:
                    transitions.append({
                        "chain_id":   chain_id,
                        "from_phase": p1,
                        "to_phase":   p2,
                        "from_zone":  z1,
                        "to_zone":    z2,
                        "label":      lbl2 or lbl1
                    })

    df_transitions = pd.DataFrame(transitions)
    if not df_transitions.empty:
        print("\n[migration_report] Zone transitions across phases:")
        print(df_transitions.to_string(index=False))
    else:
        print("\n[migration_report] No zone transitions found "
              "(may need lower min_chain_length).")
    return df_transitions


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Full Pipeline Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_cvta_pipeline(
    edgelist_dir,
    cluster_csv,
    output_dir="cvta_output",
    dataset_label="Dataset",
    year_start=2004,
    year_end=2024,
    jaccard_threshold=0.0,
    min_chain_length=3,
):
    """
    Full Community Vitality Trajectory Analysis pipeline.
    Outputs: CVS CSV, chain summary CSV, phase CSVs, migration CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CVTA Pipeline — {dataset_label}")
    print(f"{'='*60}")

    # 1. Load data
    edgelists   = load_edgelists(edgelist_dir, year_start, year_end)
    assignments = load_cluster_assignments(cluster_csv, year_start, year_end)

    # 2. Community Vitality Score
    cvs_df = compute_cvs(edgelists, assignments, year_start, year_end)
    cvs_df.to_csv(os.path.join(output_dir, "cvs_all_years.csv"), index=False)

    # 3. Build chains
    chains, kw_sets = build_community_chains(
        assignments, jaccard_threshold, year_start, year_end
    )

    # 4. Compound Momentum Rate
    chain_df = compute_cmr(chains, cvs_df, min_chain_length)
    if chain_df.empty:
        print("[WARNING] No chains met min_chain_length. Reduce threshold.")
        return None, None, None

    # 5. Classify zones
    chain_df, cvs_threshold = classify_zones(chain_df)

    # 6. Keyword labels
    chain_df = attach_keyword_labels(chain_df, assignments, edgelists)

    # 7. Save chain summary
    out_csv = os.path.join(output_dir, "chain_summary.csv")
    chain_df.drop(columns=["node_list"]).to_csv(out_csv, index=False)
    print(f"[run_pipeline] Chain summary saved → {out_csv}")

    # 8. Phase-wise analysis
    phase_summary = phase_analysis(
        chains, kw_sets, cvs_df, assignments, edgelists,
        min_chain_length=min_chain_length,
        output_dir=output_dir
    )
    phase_summary.to_csv(
        os.path.join(output_dir, "phase_zone_summary.csv"), index=False
    )

    # 9. Migration report
    migrations = migration_report(
        chains, kw_sets, cvs_df, assignments, edgelists,
        min_chain_length=min_chain_length
    )
    if not migrations.empty:
        migrations.to_csv(
            os.path.join(output_dir, "zone_migrations.csv"), index=False
        )

    print(f"\n{'='*60}")
    print(f"  CVTA complete. All outputs in: {output_dir}/")
    print(f"{'='*60}\n")
    return chain_df, phase_summary, migrations


# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── ACM dataset ──────────────────────────────────────────────────────────
    run_cvta_pipeline(
        edgelist_dir      = "cooccurance_graph_acm",
        cluster_csv       = "acm_cluster/leiden_clusters.csv",
        output_dir        = "cvta_output/acm",
        dataset_label     = "ACM-CS-TKCN",
        year_start        = 2004,
        year_end          = 2024,
        jaccard_threshold = 0.0,
        min_chain_length  = 3,
    )

    # ── DBLP dataset ─────────────────────────────────────────────────────────
    run_cvta_pipeline(
        edgelist_dir      = "co_occurance_graphs_dlp",
        cluster_csv       = "dlp_clusters/leiden_clusters.csv",
        output_dir        = "cvta_output/dblp",
        dataset_label     = "DBLP-CS-TKCN",
        year_start        = 2004,
        year_end          = 2024,
        jaccard_threshold = 0.0,
        min_chain_length  = 3,
    )