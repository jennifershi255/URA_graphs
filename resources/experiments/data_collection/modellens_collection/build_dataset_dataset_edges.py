"""
Build dataset-dataset edges from the collected ModelLens data.

  1. Text edges: cosine similarity between dataset_desp embeddings.
     dataset_desp was dropped from question_answering_top100_datasets_models_over40.csv
     in get_hf_dataset.py (remove_columns("dataset_desp")), so this script re-pulls
     it fresh from the HF corpus for just the dataset names you already kept --
     it does NOT re-run your whole filtering pipeline, just looks up descriptions.

  2. Interaction edges: cosine similarity between datasets' z-scored accuracy
     vectors (columns of the model x dataset matrix), restricted to models that
     evaluated both datasets.

"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_CSV = "../modellens_data/record_train.csv"
TOPK = 10
MIN_SHARED = 3 

train_df = pd.read_csv(TRAIN_CSV)
all_datasets = sorted(train_df["dataset"].unique())
dataset_idx = {d: i for i, d in enumerate(all_datasets)}

# Text edges

def build_text_edges():
 
    print("Re-pulling dataset_desp from luisrui/ModelLens-corpus-v2 "
          f"for {len(all_datasets)} dataset names...")
    hf = load_dataset("luisrui/ModelLens-corpus-v2", split="train")
    wanted = set(all_datasets)
    desp_lookup = {}

    for row in hf:
        d = row["dataset"]
        if d in wanted and d not in desp_lookup and row.get("dataset_desp"):
            desp_lookup[d] = row["dataset_desp"]
            if len(desp_lookup) == len(wanted):
                break
 
    missing = wanted - set(desp_lookup)
    if missing:
        print(f"Warning: no dataset_desp found for {len(missing)} datasets, "
              "falling back to the dataset name itself as text.")
 
    texts = [desp_lookup.get(d, d) for d in all_datasets]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    emb = vectorizer.fit_transform(texts)
    sim = cosine_similarity(emb)
 
    edges = []
    for i in range(len(all_datasets)):
        order = np.argsort(-sim[i])
        kept = 0
        for j in order:
            if j == i:
                continue
            edges.append((all_datasets[i], all_datasets[j], float(sim[i, j])))
            kept += 1
            if kept >= TOPK:
                break
    return pd.DataFrame(edges, columns=["dataset_a", "dataset_b", "weight"])
 
 
text_edges = build_text_edges()
text_edges.to_csv("../modellens_data/dataset_dataset_edges_text.csv", index=False)
print(f"Text edges: {len(text_edges)} rows "
      f"({text_edges['dataset_a'].nunique()} datasets covered)")

#Interaction edges
train_z = train_df.copy()
train_z["z_value"] = train_z.groupby("dataset")["eval_accuracy"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
)

all_models = sorted(train_z["model"].unique())
model_idx = {m: i for i, m in enumerate(all_models)}

mat = np.full((len(all_models), len(all_datasets)), np.nan)
for row in train_z.itertuples():
    mat[model_idx[row.model], dataset_idx[row.dataset]] = row.z_value

observed = ~np.isnan(mat)
filled = np.nan_to_num(mat, nan=0.0)

n_datasets = len(all_datasets)
inter_edges = []
for i in range(n_datasets):
    col_i = filled[:, i]
    obs_i = observed[:, i]
    shared_mask = obs_i[:, None] & observed  
    shared_counts = shared_mask.sum(axis=0)
    valid = np.where(shared_counts >= MIN_SHARED)[0]
    if len(valid) == 0:
        continue
    dots = (col_i[:, None] * filled[:, valid] * shared_mask[:, valid]).sum(axis=0)
    denom = (
        np.sqrt((col_i[:, None] ** 2 * shared_mask[:, valid]).sum(axis=0))
        * np.sqrt((filled[:, valid] ** 2 * shared_mask[:, valid]).sum(axis=0))
    )
    denom[denom == 0] = np.nan
    sims = dots / denom
    order = np.argsort(-np.nan_to_num(sims, nan=-1))
    kept = 0
    for k in order:
        j = valid[k]
        if j == i or np.isnan(sims[k]) or sims[k] <= 0:
            continue
        inter_edges.append(
            (all_datasets[i], all_datasets[j], float(sims[k]), int(shared_counts[k]))
        )
        kept += 1
        if kept >= TOPK:
            break

interaction_edges = pd.DataFrame(
    inter_edges, columns=["dataset_a", "dataset_b", "weight", "n_shared_models"]
)
interaction_edges.to_csv("../modellens_data/dataset_dataset_edges_interaction.csv", index=False)
print(f"Interaction edges: {len(interaction_edges)} rows "
      f"({interaction_edges['dataset_a'].nunique()} datasets covered, "
      f"min {MIN_SHARED} shared models required)")

print(
    "\nNote: interaction edges only exist between dataset pairs evaluated by "
    f">= {MIN_SHARED} shared models. Datasets with sparse model coverage will "
    "only get text edges -- merge both edge sets when assembling the graph."
)