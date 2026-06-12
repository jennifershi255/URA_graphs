import re
import yaml
import csv
from datasets import load_dataset
from huggingface_hub import HfApi, ModelCard
from collections import defaultdict, Counter
import time

FIXED_TIMESTAMP = "2026-01-01"
TARGET_RECORDS = 5000  # Target number of records (between 5000-10000)

def separate_tags_and_readme(card_content):
    tags, readme = None, None
    if card_content and card_content.startswith("---\n"):
        parts = card_content.split("---\n", 2)
        if len(parts) > 2:
            tags = parts[1]
            readme = parts[2]
        else:
            readme = parts[1]
    else:
        readme = card_content
    return tags, readme

# ── parse the YAML front matter ───────────────────────────────
def parse_yaml_tags(tags_str):
    result = {
        "base_model": None,
        "datasets": None,
        "license": None,
    }
    if not tags_str:
        return result
    try:
        data = yaml.safe_load(tags_str)
        if not isinstance(data, dict):
            return result
        result["base_model"] = data.get("base_model")
        result["license"]    = data.get("license")
        ds = data.get("datasets") or data.get("dataset")
        if isinstance(ds, list):
            result["datasets"] = ds
        elif isinstance(ds, str):
            result["datasets"] = [ds]
    except Exception:
        pass
    return result

# ── regex patterns for README body ────────────────────────────
ACCURACY_RE = re.compile(
    r"(?:eval_?)?accuracy[:\s=]+([0-9]+(?:\.[0-9]+)?)\s*%?",
    re.IGNORECASE
)
# CHANGED: was a single DATASET_RE match; now find ALL "fine-tuned on X dataset"
# mentions, plus also catch comma/and-separated lists like
# "fine-tuned on the SST-2, MNLI and QQP datasets"
DATASET_RE = re.compile(
    r"fine-?tuned (?:version of \[.*?\]\(.*?\) )?on (?:the\s+)?([^\n\.]+?) datasets?",
    re.IGNORECASE
)
LR_RE = re.compile(
    r"learning[_\s]rate[:\s]+([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)",
    re.IGNORECASE
)
BATCH_RE = re.compile(
    r"(?:train_batch_size|per_device_train_batch_size)[:\s]+([0-9]+)",
    re.IGNORECASE
)
EPOCHS_RE = re.compile(
    r"num_(?:train_)?epochs[:\s]+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE
)

# words to ignore as "dataset names" when splitting on , / and
_IGNORE_NAMES = {"none", "unknown", "an", "the", "an unknown", "a", "this", ""}

def _split_dataset_list(raw_name):
    """
    Take a raw matched string like "SST-2, MNLI and QQP" or "the GLUE"
    and split it into a clean list of individual dataset name strings.
    """
    # normalize " and " -> ","
    raw_name = re.sub(r"\s+and\s+", ",", raw_name, flags=re.IGNORECASE)
    parts = [p.strip(" .") for p in raw_name.split(",")]
    cleaned = []
    for p in parts:
        if p.lower() in _IGNORE_NAMES:
            continue
        if len(p) == 0:
            continue
        cleaned.append(p)
    return cleaned

def parse_readme_body(readme_str):
    result = {
        "eval_accuracy":   None,
        "datasets_readme": [],   # CHANGED: now a list instead of single value
        "learning_rate":   None,
        "batch_size":      None,
        "num_epochs":      None,
    }
    if not readme_str:
        return result

    acc = ACCURACY_RE.search(readme_str)
    if acc:
        val = float(acc.group(1))
        result["eval_accuracy"] = val / 100 if val > 1.5 else val

    # CHANGED: findall instead of search -> capture every "fine-tuned on X dataset(s)"
    # mention in the README, then split each match on commas/and
    for ds_match in DATASET_RE.findall(readme_str):
        for name in _split_dataset_list(ds_match):
            if name not in result["datasets_readme"]:
                result["datasets_readme"].append(name)

    lr = LR_RE.search(readme_str)
    if lr:
        result["learning_rate"] = float(lr.group(1))

    batch = BATCH_RE.search(readme_str)
    if batch:
        result["batch_size"] = int(batch.group(1))

    epochs = EPOCHS_RE.search(readme_str)
    if epochs:
        result["num_epochs"] = float(epochs.group(1))

    return result

# ── put it all together ────────────────────────────────────────
def parse_model_card(row):
    tags_str, readme_str = separate_tags_and_readme(row["card"])
    yaml_data   = parse_yaml_tags(tags_str)
    readme_data = parse_readme_body(readme_str)

    # CHANGED: combine YAML datasets list + README-detected datasets,
    # de-duplicated, instead of picking just one
    all_datasets = []
    if yaml_data["datasets"]:
        for d in yaml_data["datasets"]:
            if d and d not in all_datasets:
                all_datasets.append(d)
    for d in readme_data["datasets_readme"]:
        if d not in all_datasets:
            all_datasets.append(d)

    return {
        "model":             row["modelId"],
        "base_model":        yaml_data["base_model"],
        "finetuned_datasets": all_datasets,  # CHANGED: list, may be empty
        "eval_accuracy":     readme_data["eval_accuracy"],
        "task_type":         row["pipeline_tag"],
        "learning_rate":     readme_data["learning_rate"],
        "batch_size":        readme_data["batch_size"],
        "num_epochs":        readme_data["num_epochs"],
        "created_at":        row["createdAt"],
        "last_modified":     row["last_modified"],
        "license":           yaml_data["license"],
    }

# ── run it ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = load_dataset(
        "librarian-bots/model_cards_with_metadata",
        split="train",
        streaming=True
    )

    filtered = ds.filter(lambda row:
        row["pipeline_tag"] == "text-classification" and
        row["createdAt"] is not None and
        str(row["createdAt"]) < FIXED_TIMESTAMP
    )

    flat_rows = []  # CHANGED: one row per (model, dataset) pair
    author_counts = {}

    for i, row in enumerate(filtered.take(50000)):
        if i % 1000 == 0:
            print(f"  processed {i} cards, usable rows so far: {len(flat_rows)}")

        author = row["modelId"].split("/")[0]

        parsed = parse_model_card(row)
        if not parsed["finetuned_datasets"] or not parsed["eval_accuracy"]:
            continue

        author_counts[author] = author_counts.get(author, 0) + 1

        # CHANGED: emit one flat row per dataset this model was fine-tuned on
        for dataset_name in parsed["finetuned_datasets"]:
            flat_rows.append({
                "model":             parsed["model"],
                "base_model":        parsed["base_model"],
                "finetuned_dataset": dataset_name,
                "eval_accuracy":     parsed["eval_accuracy"],
                "task_type":         parsed["task_type"],
                "learning_rate":     parsed["learning_rate"],
                "batch_size":        parsed["batch_size"],
                "num_epochs":        parsed["num_epochs"],
                "created_at":        parsed["created_at"],
                "last_modified":     parsed["last_modified"],
                "license":           parsed["license"],
            })

    # save to CSV
    if flat_rows:
        keys = flat_rows[0].keys()
        with open("records.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)

    n_models = len({r["model"] for r in flat_rows})
    n_datasets = len({r["finetuned_dataset"] for r in flat_rows})
    print(f"Total (model, dataset) rows: {len(flat_rows)}")
    print(f"Unique models:   {n_models}")
    print(f"Unique datasets: {n_datasets}")
    print(f"Avg datasets per model: {len(flat_rows) / n_models:.2f}")
    print("Saved to records.csv")