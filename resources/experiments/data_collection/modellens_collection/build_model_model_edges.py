"""
Build model-model edges from the collected ModelLens triples.
Two edge sources:
- Metadata edges: cosine similarity over [family one-hot, log(size)] vectors,from model2family.json + model_profile.json
- Interaction edges: cosine similarity between models' z-scored accuracy vectors, restricted to datasets they share.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_CSV = "../modellens_data/record_train.csv"
MODEL2FAMILY = "../modellens_data/model2family.json"
MODEL_PROFILE = "../modellens_data/model_profile.json"
TOPK = 10        
MIN_SHARED = 3     

train_df = pd.read_csv(TRAIN_CSV)

def safe_size(s):
    if s is None:
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0
    
with open(MODEL2FAMILY) as f:
    model2family = json.load(f)
with open(MODEL_PROFILE) as f:
    model_profile = json.load(f)

all_models = sorted(train_df["model"].unique())
model_idx = {m: i for i, m in enumerate(all_models)}


def topk_edges(sim, n, names):
    edges = []
    for i in range(n):
        order = np.argsort(-sim[i])
        kept = 0
        for j in order:
            if j == i:
                continue
            if sim[i, j] <= 0:
                break
            edges.append((names[i], names[j], float(sim[i, j])))
            kept += 1
            if kept >= TOPK:
                break
    return pd.DataFrame(edges, columns=["model_a", "model_b", "weight"])


# Metadata edges: family + size
families = [model2family.get(m, "Unknown") for m in all_models]
sizes = [model_profile.get(m, {}).get("size") for m in all_models]
sizes = [np.log1p(safe_size(model_profile.get(m, {}).get("size"))) for m in all_models]
max_size = max(sizes) if max(sizes) > 0 else 1.0

ohe = OneHotEncoder(sparse_output=False)
fam_vec = ohe.fit_transform(np.array(families).reshape(-1, 1))
size_vec = (np.array(sizes) / max_size).reshape(-1, 1)
meta_feat = np.hstack([fam_vec, size_vec])

meta_sim = cosine_similarity(meta_feat)
metadata_edges = topk_edges(meta_sim, len(all_models), all_models)
metadata_edges.to_csv("../modellens_data/model_model_edges_metadata.csv", index=False)
print(f"Metadata edges: {len(metadata_edges)} rows "
      f"({metadata_edges['model_a'].nunique()} models covered)")

# Interaction edges: shared-dataset performance correlation (train only)
train_z = train_df.copy()
train_z["z_value"] = train_z.groupby("dataset")["eval_accuracy"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
)

all_datasets = sorted(train_z["dataset"].unique())
dataset_idx = {d: i for i, d in enumerate(all_datasets)}

mat = np.full((len(all_models), len(all_datasets)), np.nan)
for row in train_z.itertuples():
    mat[model_idx[row.model], dataset_idx[row.dataset]] = row.z_value

observed = ~np.isnan(mat)
filled = np.nan_to_num(mat, nan=0.0)

# cosine similarity computed only over commonly-observed columns
n_models = len(all_models)
inter_edges = []
norms = np.linalg.norm(filled, axis=1)
for i in range(n_models):
    shared_mask = observed[i] & observed  
    shared_counts = shared_mask.sum(axis=1)
    valid = np.where(shared_counts >= MIN_SHARED)[0]
    if len(valid) == 0:
        continue
    dots = (filled[i] * filled[valid] * shared_mask[valid]).sum(axis=1)
    denom = (
        np.sqrt((filled[i] ** 2 * shared_mask[valid]).sum(axis=1))
        * np.sqrt((filled[valid] ** 2 * shared_mask[valid]).sum(axis=1))
    )
    denom[denom == 0] = np.nan
    sims = dots / denom
    order = np.argsort(-np.nan_to_num(sims, nan=-1))
    kept = 0
    for k in order:
        j = valid[k]
        if j == i or np.isnan(sims[k]) or sims[k] <= 0:
            continue
        inter_edges.append((all_models[i], all_models[j], float(sims[k]), int(shared_counts[k])))
        kept += 1
        if kept >= TOPK:
            break

interaction_edges = pd.DataFrame(
    inter_edges, columns=["model_a", "model_b", "weight", "n_shared_datasets"]
)
interaction_edges.to_csv("../modellens_data/model_model_edges_interaction.csv", index=False)
print(f"Interaction edges: {len(interaction_edges)} rows "
      f"({interaction_edges['model_a'].nunique()} models covered, "
      f"min {MIN_SHARED} shared datasets required)")

print(
    "\nNote: interaction edges only exist between models with >= "
    f"{MIN_SHARED} shared evaluated datasets, so models with sparse "
    "coverage will only get metadata edges. That's expected -- merge both "
    "edge sets when you assemble the graph."
)