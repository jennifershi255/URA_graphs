import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("modellens_data/question_answering_top100_datasets_models_over40.csv")

# Keep only required columns
df = df[["model", "dataset", "value"]]

# Rename value column
df = df.rename(columns={"value": "eval_accuracy"})

# Convert percentage values (0-100) to 0-1 only for rows that are greater than 1
mask = df["eval_accuracy"] > 1
if mask.any():
    df.loc[mask, "eval_accuracy"] = df.loc[mask, "eval_accuracy"] / 100.0

# Optional: remove duplicate model-dataset pairs if needed
# (keep first evaluation if multiple metrics exist)
df = df.drop_duplicates(subset=["model", "dataset"])

# Get unique (dataset, model) pairs
pairs = df[["dataset", "model"]].drop_duplicates()

# Split pairs for known dataset setting
train_pairs, test_pairs = train_test_split(
    pairs,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Add split labels
train_pairs["split"] = "train"
test_pairs["split"] = "test"

pair_split = pd.concat([train_pairs, test_pairs])

# Merge split back
df = df.merge(
    pair_split,
    on=["dataset", "model"],
    how="inner"
)

# Create train/test sets
train_df = df[df["split"] == "train"].drop(columns=["split"])
test_df = df[df["split"] == "test"].drop(columns=["split"])

# Check results
print("Train:")
print(train_df.head())
print(f"Rows: {len(train_df)}")
print(f"Models: {train_df['model'].nunique()}")
print(f"Datasets: {train_df['dataset'].nunique()}")

print("\nTest:")
print(test_df.head())
print(f"Rows: {len(test_df)}")
print(f"Models: {test_df['model'].nunique()}")
print(f"Datasets: {test_df['dataset'].nunique()}")

# Save files
train_df.to_csv("modellens_data/record_train.csv", index=False)
test_df.to_csv("modellens_data/record_test.csv", index=False)