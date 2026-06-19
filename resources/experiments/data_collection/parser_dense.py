import re
import yaml
import csv
from datasets import load_dataset
from huggingface_hub import HfApi, ModelCard
from collections import defaultdict, Counter
import time
import os
from dotenv import load_dotenv

DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env')
load_dotenv(DOTENV_PATH)

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
FIXED_TIMESTAMP = "2026-01-01"
TARGET_RECORDS = 5000

MODEL_TYPE_TO_BASE = {
    "distilbert": "distilbert-base-uncased",
    "bert":       "bert-base-uncased",
    "roberta":    "roberta-base",
    "albert":     "albert-base-v2",
    "xlnet":      "xlnet-base-cased",
    "deberta":    "microsoft/deberta-base",
    "deberta-v2": "microsoft/deberta-v2-xlarge",
    "electra":    "google/electra-base-discriminator",
    "camembert":  "camembert-base",
    "xlm-roberta":"xlm-roberta-base",
}

def infer_base_model(yaml_base_model, config):
    # Prefer explicit declaration in card YAML
    if yaml_base_model:
        return yaml_base_model
    # Fall back to model_type → canonical base
    if config and isinstance(config, dict):
        model_type = config.get("model_type")
        if model_type and model_type in MODEL_TYPE_TO_BASE:
            return MODEL_TYPE_TO_BASE[model_type]
    return None

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

def parse_yaml_tags(tags_str):
    result = {"base_model": None, "datasets": None, "license": None}
    if not tags_str:
        return result
    try:
        data = yaml.safe_load(tags_str)
        if not isinstance(data, dict):
            return result
        bm = data.get("base_model")
        # CHANGED: base_model can be a list (e.g. ["bert-base-uncased"]) — normalize to string
        if isinstance(bm, list):
            bm = bm[0] if bm else None
        result["base_model"] = bm
        result["license"] = data.get("license")
        ds = data.get("datasets") or data.get("dataset")
        if isinstance(ds, list):
            result["datasets"] = ds
        elif isinstance(ds, str):
            result["datasets"] = [ds]
    except Exception:
        pass
    return result

ACCURACY_RE = re.compile(
    r"(?:eval_?)?accuracy[:\s=]+([0-9]+(?:\.[0-9]+)?)\s*%?", re.IGNORECASE
)
DATASET_RE = re.compile(
    r"fine-?tuned (?:version of \[.*?\]\(.*?\) )?on (?:the\s+)?([^\n\.]+?) datasets?",
    re.IGNORECASE
)
LR_RE = re.compile(
    r"learning[_\s]rate[:\s]+([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", re.IGNORECASE
)
BATCH_RE = re.compile(
    r"(?:train_batch_size|per_device_train_batch_size)[:\s]+([0-9]+)", re.IGNORECASE
)
EPOCHS_RE = re.compile(
    r"num_(?:train_)?epochs[:\s]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)

_IGNORE_NAMES = {"none", "unknown", "an", "the", "an unknown", "a", "this", ""}

def _split_dataset_list(raw_name):
    raw_name = re.sub(r"\s+and\s+", ",", raw_name, flags=re.IGNORECASE)
    parts = [p.strip(" .") for p in raw_name.split(",")]
    return [p for p in parts if p.lower() not in _IGNORE_NAMES and len(p) > 0]

def parse_readme_body(readme_str):
    result = {
        "eval_accuracy": None,
        "datasets_readme": [],
        "learning_rate": None,
        "batch_size": None,
        "num_epochs": None,
    }
    if not readme_str:
        return result
    acc = ACCURACY_RE.search(readme_str)
    if acc:
        val = float(acc.group(1))
        result["eval_accuracy"] = val / 100 if val > 1.5 else val
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

def parse_model_card(row):
    tags_str, readme_str = separate_tags_and_readme(row["card"])
    yaml_data = parse_yaml_tags(tags_str)
    readme_data = parse_readme_body(readme_str)
    all_datasets = []
    if yaml_data["datasets"]:
        for d in yaml_data["datasets"]:
            if d and d not in all_datasets:
                all_datasets.append(d)
    for d in readme_data["datasets_readme"]:
        if d not in all_datasets:
            all_datasets.append(d)
    return {
        "model":              row["modelId"],
        "base_model":         yaml_data["base_model"],
        "finetuned_datasets": all_datasets,
        "eval_accuracy":      readme_data["eval_accuracy"],
        "task_type":          row["pipeline_tag"],
        "learning_rate":      readme_data["learning_rate"],
        "batch_size":         readme_data["batch_size"],
        "num_epochs":         readme_data["num_epochs"],
        "created_at":         row["createdAt"],
        "last_modified":      row["last_modified"],
        "license":            yaml_data["license"],
    }

def find_models_for_dataset(api, dataset_name, pipeline_tag="text-classification", max_models=500):
    try:
        dataset_name_clean = dataset_name.split(" [")[0].strip()
        models = api.list_models(
            filter=f"dataset:{dataset_name_clean}",
            sort="downloads",
            limit=max_models
        )
        return list(models)
    except Exception as e:
        print(f"  Error querying models for dataset '{dataset_name}': {e}")
        return []


def extract_model_info_from_api(api, model_id, retries=3):
    for attempt in range(retries):
        try:
            card = ModelCard.load(model_id)
            card_text = card.text if card else ""
            tags_str, readme_str = separate_tags_and_readme(card_text)
            yaml_data = parse_yaml_tags(tags_str)
            readme_data = parse_readme_body(readme_str)
            all_datasets = []
            if yaml_data["datasets"]:
                for d in yaml_data["datasets"]:
                    if d and d not in all_datasets:
                        all_datasets.append(d)
            for d in readme_data["datasets_readme"]:
                if d not in all_datasets:
                    all_datasets.append(d)
            model_info = api.model_info(model_id)
            return {
                "model":              model_id,
                "base_model": infer_base_model(yaml_data["base_model"], model_info.config),
                "finetuned_datasets": all_datasets,
                "eval_accuracy":      readme_data["eval_accuracy"],
                "task_type":          model_info.pipeline_tag if hasattr(model_info, "pipeline_tag") else None,
                "learning_rate":      readme_data["learning_rate"],
                "batch_size":         readme_data["batch_size"],
                "num_epochs":         readme_data["num_epochs"],
                "created_at":         model_info.created_at if hasattr(model_info, "created_at") else None,
                "last_modified":      model_info.lastModified if hasattr(model_info, "lastModified") else None,
                "license":            yaml_data["license"],
                "downloads":          model_info.downloads if hasattr(model_info, "downloads") else 0,
            }
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  Error extracting info for model '{model_id}': {e}")
                return None

# CHANGED: helper to build one flat CSV row, with row_type for lineage tracking
def _make_row(parsed, dataset_name, row_type="finetune"):
    return {
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
        "downloads":         parsed.get("downloads", None),
        "row_type":          row_type,
    }

def collect_by_dataset_approach(target_records=7500):
    global api

    print("=== DATASET-CENTRIC COLLECTION ===")
    print(f"Target: {target_records} records")

    print("\nStep 1: Identifying popular datasets...")
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
    dataset_counts = Counter()
    for i, row in enumerate(filtered.take(20000)):
        if i % 2000 == 0:
            print(f"  Sampled {i} models...")
        parsed = parse_model_card(row)
        for ds_name in parsed["finetuned_datasets"]:
            dataset_counts[ds_name] += 1

    JUNK_DATASETS = {"custom", "a custom", "the dataset", "my dataset", ""}
    top_datasets = [
        ds for ds, count in dataset_counts.most_common(50)
        if count >= 5
        and not ds.startswith("**")
        and ds.lower() not in JUNK_DATASETS
    ]
    top_datasets_raw = [
    ds for ds, count in dataset_counts.most_common(50)
    if count >= 5
    and not ds.startswith("**")
    and ds.lower() not in JUNK_DATASETS
    ]
    seen = set()
    top_datasets = []
    for ds in top_datasets_raw:
        clean = ds.split(" [")[0].strip()
        if clean not in seen:
            seen.add(clean)
            top_datasets.append(clean)
    top_datasets_set = set(top_datasets)

    print(f"\nFound {len(top_datasets)} popular datasets")
    print(f"Top 10: {top_datasets[:10]}")

    print("\nStep 2: Finding models for each dataset...")
    flat_rows = []
    dataset_model_counts = {}
    # CHANGED: track (base_model, dataset) pairs already emitted as lineage edges
    seen_lineage = set()

    for idx, dataset_name in enumerate(top_datasets):
        if len(flat_rows) >= target_records:
            print(f"Reached target of {target_records} records, stopping.")
            break

        print(f"\n[{idx+1}/{len(top_datasets)}] Processing dataset: {dataset_name}")
        models = find_models_for_dataset(api, dataset_name, max_models=500)
        print(f"  Found {len(models)} models")
        dataset_model_counts[dataset_name] = 0

        for model in models:
            if len(flat_rows) >= target_records:
                break

            model_id = model.modelId if hasattr(model, "modelId") else model.id
            pipeline_tag = model.pipeline_tag if hasattr(model, "pipeline_tag") else None
            if pipeline_tag != "text-classification":
                continue
            created_at = model.created_at if hasattr(model, "created_at") else None
            if created_at and str(created_at) >= FIXED_TIMESTAMP:
                continue

            parsed = extract_model_info_from_api(api, model_id)
            if not parsed:
                continue

            # Skip models missing required fields
            if parsed["base_model"] is None or parsed["eval_accuracy"] is None:
                continue

            # Primary edge: fine-tuned model → queried dataset
            flat_rows.append(_make_row(parsed, dataset_name, row_type="finetune"))
            dataset_model_counts[dataset_name] += 1

            # CHANGED: lineage edge — base_model → same dataset
            # Connects the parent model as its own node, building transfer chains
            base_model = parsed["base_model"]
            if base_model:
                lineage_key = (base_model, dataset_name)
                if lineage_key not in seen_lineage:
                    seen_lineage.add(lineage_key)
                    lineage_parsed = {**parsed, "model": base_model, "base_model": None, "downloads": None}
                    flat_rows.append(_make_row(lineage_parsed, dataset_name, row_type="lineage"))

            # Also add rows for any OTHER top datasets mentioned in the model card
            for other_ds in parsed["finetuned_datasets"]:
                if other_ds == dataset_name or other_ds not in top_datasets_set:
                    continue
                flat_rows.append(_make_row(parsed, other_ds, row_type="finetune"))

                # CHANGED: lineage edges for cross-dataset connections too
                if base_model:
                    lineage_key = (base_model, other_ds)
                    if lineage_key not in seen_lineage:
                        seen_lineage.add(lineage_key)
                        lineage_parsed = {**parsed, "model": base_model, "base_model": None, "downloads": None}
                        flat_rows.append(_make_row(lineage_parsed, other_ds, row_type="lineage"))

            if dataset_model_counts[dataset_name] % 20 == 0:
                print(f"    Collected {dataset_model_counts[dataset_name]} models, total rows: {len(flat_rows)}")

            time.sleep(0.4)

    print(f"\n=== COLLECTION COMPLETE ===")
    print(f"Total records: {len(flat_rows)}")
    print(f"Datasets with models: {len([c for c in dataset_model_counts.values() if c > 0])}")
    print(f"\nModels per dataset (top 10):")
    for ds, count in sorted(dataset_model_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ds}: {count} models")

    return flat_rows

if __name__ == "__main__":
    flat_rows = collect_by_dataset_approach(target_records=TARGET_RECORDS)

    if flat_rows:
        # Split into finetune vs lineage
        finetune_rows = [r for r in flat_rows if r["row_type"] == "finetune"]
        lineage_rows  = [r for r in flat_rows if r["row_type"] == "lineage"]

        # ── records.csv (TransferGraph input) ─────────────────────────
        # Only finetune rows, task_type renamed to match TransferGraph's convention
        TASK_TYPE_MAP = {
            "text-classification": "sequence_classification",
            "image-classification": "image_classification",
        }
        record_rows = []
        for r in finetune_rows:
            row = dict(r)
            row["task_type"] = TASK_TYPE_MAP.get(row["task_type"], row["task_type"])
            row.pop("row_type")   # TransferGraph doesn't need this column
            record_rows.append(row)

        records_path = "resources/experiments/sequence_classification/records.csv"
        os.makedirs(os.path.dirname(records_path), exist_ok=True)
        with open(records_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record_rows[0].keys())
            writer.writeheader()
            writer.writerows(record_rows)
        print(f"Saved {len(record_rows)} finetune rows to {records_path}")

        # ── records_dense.csv (full data including lineage, for graph construction later) ──
        with open("records_dense.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"Saved {len(flat_rows)} total rows to records_dense.csv")

        # ── stats ──────────────────────────────────────────────────────
        n_models   = len({r["model"] for r in flat_rows})
        n_datasets = len({r["finetuned_dataset"] for r in flat_rows})

        print(f"\n=== FINAL STATISTICS ===")
        print(f"Total rows:            {len(flat_rows)}")
        print(f"  finetune edges:      {len(finetune_rows)}")
        print(f"  lineage edges:       {len(lineage_rows)}")
        print(f"Unique models:         {n_models}")
        print(f"Unique datasets:       {n_datasets}")
        print(f"Avg rows per dataset:  {len(flat_rows) / n_datasets:.2f}")
        print(f"Avg rows per model:    {len(flat_rows) / n_models:.2f}")
        print(f"Unique base models:    {len({r['base_model'] for r in finetune_rows if r['base_model']})}")
    else:
        print("No records collected!")