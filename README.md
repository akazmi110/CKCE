# CKCE
This repository contains the code and processed datasets used in the study:

**"Modeling the Evolution and Vitality of Temporal Keyword Communities in Scientific Literature: A Comparative Clustering Study."**

The project analyzes the evolution of scientific research themes by constructing **temporal keyword co-occurrence networks** and applying multiple **community detection algorithms**. The detected communities are further analyzed longitudinally to understand their **growth, persistence, and vitality over time**.

The study focuses on cybersecurity research literature derived from two large bibliographic sources.

---

# Repository Structure
CKCE
│
├── code
│ ├── keyword_processing.py
│ ├── step1_co_occurrence_network_construction.py
│ ├── step2_louvain_clustering.py
│ ├── step3_label_propagation_clustering.py
│ ├── step4_infomap_clustering.py
│ ├── step5_leiden_clustering.py
│ ├── step6_walktrap_clustering.py
│ ├── step7_node2vec_clustering.py
│ ├── step8_deepwalk_clustering.py
│ ├── step9_compute_ami_ari.py
│ └── step10_community_vitality_tracker.py
│
└── data
├── ACM-CS-TKCN
├── DBLP-CS-TKCN
├── acm_cyber_security_regex.jsonl
└── dblp_cyber_security_regex.jsonl
├── dblp_cleaned_multi_free.openai-gpt.jsonl
└── acm_cyber_security_keywords.openai-gpt.jsonl

# Data Description

The repository includes processed datasets used to construct **Temporal Keyword Co-occurrence Networks (TKCN)**.

---
## Filtering Files

The following files contain cybersecurity-related publications extracted using domain-specific regular expressions:

- `acm_cyber_security_regex.jsonl`
- `dblp_cyber_security_regex.jsonl`

These filtered publications form the basis for constructing the temporal keyword networks.

## Preprocessed Files
dblp_cleaned_multi_free.openai-gpt.jsonl
acm_cyber_security_keywords.openai-gpt.jsonl


## ACM-CS-TKCN
Keyword co-occurrence networks derived from cybersecurity publications in the **ACM Digital Library**.

## DBLP-CS-TKCN
Keyword co-occurrence networks derived from cybersecurity publications indexed in **DBLP Computer Science Bibliography**.


---

# Experimental Pipeline

The repository implements a multi-stage pipeline for analyzing **temporal keyword communities**.

## 1. Keyword Processing
keyword_processing.py

Preprocesses and normalizes author-defined keywords extracted from the bibliographic datasets.

---

## 2. Temporal Keyword Co-occurrence Network Construction
step1_co_occurrence_network_construction.py

Constructs yearly keyword co-occurrence networks from the filtered publication datasets.

---

## 3. Community Detection Algorithms

Multiple clustering methods are applied to detect communities in the networks:

- Louvain
- Label Propagation
- Infomap
- Leiden
- Walktrap
- Node2Vec + k-means
- DeepWalk + k-means

Each algorithm is implemented in a separate script in the `code/` directory.

---

## 4. Cross-Algorithm Clustering Evaluation
step9_compute_ami_ari.py

Evaluates clustering similarity using:

- Adjusted Mutual Information (AMI)
- Adjusted Rand Index (ARI)

These metrics quantify agreement between community detection algorithms.

---

## 5. Community Vitality Tracking
step10_community_vitality_tracker.py

Tracks communities across consecutive years and analyzes their **temporal evolution and vitality trajectories**.

This step enables the identification of:

- emerging research themes
- expanding research areas
- stable thematic structures
- declining or peripheral topics

---

# Research Objective

The goal of this study is to model how **scientific research themes evolve over time** by analyzing temporal keyword networks.

The analysis focuses on:

- detecting keyword communities in yearly networks
- comparing multiple clustering algorithms
- measuring clustering agreement across methods
- tracking community evolution across time
- analyzing the vitality of research themes

---

# Requirements

The project is implemented in **Python**.

Main dependencies include:
networkx
numpy
pandas
scikit-learn
python-louvain
igraph
leidenalg
node2vec
gensim

Install dependencies using:
pip install -r requirements.txt

---

# Reproducibility

To reproduce the experiments:

1. Preprocess keywords
python keyword_processing.py

2. Construct temporal keyword co-occurrence networks
python step1_co_occurrence_network_construction.py

3. Run community detection algorithms
python step2_louvain_clustering.py
python step3_label_propagation_clustering.py
python step4_infomap_clustering.py
python step5_leiden_clustering.py
python step6_walktrap_clustering.py
python step7_node2vec_clustering.py
python step8_deepwalk_clustering.py

4. Evaluate clustering similarity
python step9_compute_ami_ari.py

5. Track community vitality trajectories
python step10_community_vitality_tracker.py


---

# Citation

If you use this repository in your research, please cite:
\url{https://github.com/akazmi110/CKCE}


---

# License

This repository is provided for **academic research and reproducibility purposes**.

---

# Author

Anab Batool Kazmi  
Computer Science Department  
National University of Modern Languages (NUML)  
National University of Computer and Emerging Sciences (NUCES-FAST)
Islamabad, Pakistan
