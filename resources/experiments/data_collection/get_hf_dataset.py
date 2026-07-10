from datasets import load_dataset
from collections import Counter

# Load dataset
dataset = load_dataset("luisrui/ModelLens-corpus-v2", split="train")

# Filter to Question Answering task
qa_dataset = dataset.filter(lambda x: x["task"] == "Question Answering" and x["metric"] in ["accuracy_norm", "exact_match", "accuracy", "f1"])

# Find top 100 most common datasets
dataset_counts = Counter(qa_dataset["dataset"])
top_100_datasets = {
    name for name, _ in dataset_counts.most_common(100)
}

# Keep only rows from top 100 datasets
qa_top100_dataset = qa_dataset.filter(
    lambda x: x["dataset"] in top_100_datasets
)

# Count rows per model after dataset filtering
model_counts = Counter(qa_top100_dataset["model"])

# Keep models with more than 40 rows
valid_models = {
    model for model, count in model_counts.items()
    if count > 40
}

# Keep only valid models
qa_filtered = qa_top100_dataset.filter(
    lambda x: x["model"] in valid_models
)

# Remove dataset_desp column
if "dataset_desp" in qa_filtered.column_names:
    qa_filtered = qa_filtered.remove_columns("dataset_desp")

# Print summary
print(f"Unique datasets kept: {len(set(qa_filtered['dataset']))}")
print(f"Unique models kept: {len(set(qa_filtered['model']))}")
print(f"Total rows: {len(qa_filtered)}")

# Save result
qa_filtered.to_csv("modellens_data/question_answering_top100_datasets_models_over40.csv", index=False)