import re
import yaml
import csv
import os
from datasets import load_dataset
from huggingface_hub import HfApi, ModelCard
from collections import defaultdict, Counter
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
FIXED_TIMESTAMP = "2026-01-01"
TARGET_RECORDS = 2000  # Target number of records (between 5000-10000)

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

ACCURACY_RE = re.compile(
    r"(?:eval_?)?accuracy[:\s=]+([0-9]+(?:\.[0-9]+)?)\s*%?",
    re.IGNORECASE
)
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

_IGNORE_NAMES = {"none", "unknown", "an", "the", "an unknown", "a", "this", ""}

def _split_dataset_list(raw_name):
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
        "datasets_readme": [],
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

def find_models_for_dataset(api, dataset_name, max_models=500):
    """
    Query HuggingFace Hub API to find models trained on a specific dataset.
    """
    try:
        models = api.list_models(
            trained_dataset=dataset_name,
            sort="downloads",
            limit=max_models
        )
        return list(models)
    except Exception as e:
        print(f"  Error querying models for dataset '{dataset_name}': {e}")
        return []

def extract_model_info_from_api(api, model_id):
    """
    Fetch and parse model card for a given model ID using the API.
    NOW INCLUDES base_model extraction!
    """
    try:
        card = ModelCard.load(model_id)
        card_text = card.text if card else ""
        
        # Parse it using existing functions
        tags_str, readme_str = separate_tags_and_readme(card_text)
        yaml_data = parse_yaml_tags(tags_str)
        readme_data = parse_readme_body(readme_str)
        
        # Combine datasets
        all_datasets = []
        if yaml_data["datasets"]:
            for d in yaml_data["datasets"]:
                if d and d not in all_datasets:
                    all_datasets.append(d)
        for d in readme_data["datasets_readme"]:
            if d not in all_datasets:
                all_datasets.append(d)
        
        # Get model info from API
        model_info = api.model_info(model_id)
        
        return {
            "model": model_id,
            "base_model": yaml_data["base_model"],  # NOW CAPTURED!
            "finetuned_datasets": all_datasets,
            "eval_accuracy": readme_data["eval_accuracy"],
            "task_type": model_info.pipeline_tag if hasattr(model_info, 'pipeline_tag') else None,
            "learning_rate": readme_data["learning_rate"],
            "batch_size": readme_data["batch_size"],
            "num_epochs": readme_data["num_epochs"],
            "created_at": model_info.created_at if hasattr(model_info, 'created_at') else None,
            "last_modified": model_info.lastModified if hasattr(model_info, 'lastModified') else None,
            "license": yaml_data["license"],
            "downloads": model_info.downloads if hasattr(model_info, 'downloads') else 0,
        }
    except Exception as e:
        print(f"  Error extracting info for model '{model_id}': {e}")
        return None

# ── HYBRID: Dataset-centric collection WITH base_model capture ────────
def collect_hybrid_approach(target_records=7500):
    """
    HYBRID APPROACH: Start with popular datasets (which we know works),
    then capture base_model information for each model found.
    This gives us: base_model → model → dataset chains!
    """
    global api
    
    print("=== HYBRID DATASET-CENTRIC WITH TRANSFER LEARNING ===")
    print(f"Target: {target_records} records")
    
    # Step 1: Identify popular datasets by sampling models first
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
        
        from parser_dense import parse_model_card
        parsed = parse_model_card(row)
        for ds_name in parsed["finetuned_datasets"]:
            dataset_counts[ds_name] += 1
    
    # Get top datasets by frequency
    top_datasets = [ds for ds, count in dataset_counts.most_common(50) if count >= 5]
    print(f"\nFound {len(top_datasets)} popular datasets")
    print(f"Top 10: {top_datasets[:10]}")
    
    # Step 2: For each dataset, find ALL models trained on it
    print("\nStep 2: Finding models for each dataset (WITH base_model extraction)...")
    flat_rows = []
    dataset_model_counts = {}
    models_with_base = 0
    models_without_base = 0
    
    for idx, dataset_name in enumerate(top_datasets):
        if len(flat_rows) >= target_records:
            print(f"\nReached target of {target_records} records, stopping.")
            break
            
        print(f"\n[{idx+1}/{len(top_datasets)}] Processing dataset: {dataset_name}")
        
        models = find_models_for_dataset(api, dataset_name, max_models=500)
        print(f"  Found {len(models)} models")
        
        dataset_model_counts[dataset_name] = 0
        
        for model in models:
            if len(flat_rows) >= target_records:
                break
                
            model_id = model.modelId if hasattr(model, 'modelId') else model.id
            
            # Filter for text-classification only
            pipeline_tag = model.pipeline_tag if hasattr(model, 'pipeline_tag') else None
            if pipeline_tag != "text-classification":
                continue
            
            # Skip if created after our cutoff date
            created_at = model.created_at if hasattr(model, 'created_at') else None
            if created_at and str(created_at) >= FIXED_TIMESTAMP:
                continue
            
            # Extract detailed info from model card (including base_model!)
            parsed = extract_model_info_from_api(api, model_id)
            if not parsed:
                continue
            
            # Track base_model presence
            if parsed["base_model"]:
                models_with_base += 1
                if models_with_base <= 5:  # Show first few
                    print(f"    ✓ {model_id[:50]}... → base: {parsed['base_model']}")
            else:
                models_without_base += 1
            
            # Add a row for this (model, dataset) pair with base_model info
            flat_rows.append({
                "model": parsed["model"],
                "base_model": parsed["base_model"],  # CAPTURED!
                "finetuned_dataset": dataset_name,
                "eval_accuracy": parsed["eval_accuracy"],
                "task_type": parsed["task_type"],
                "learning_rate": parsed["learning_rate"],
                "batch_size": parsed["batch_size"],
                "num_epochs": parsed["num_epochs"],
                "created_at": parsed["created_at"],
                "last_modified": parsed["last_modified"],
                "license": parsed["license"],
                "downloads": parsed["downloads"],
            })
            
            dataset_model_counts[dataset_name] += 1
            
            # Also add rows for any OTHER datasets mentioned in the model card
            for other_ds in parsed["finetuned_datasets"]:
                if other_ds != dataset_name and other_ds in top_datasets:
                    flat_rows.append({
                        "model": parsed["model"],
                        "base_model": parsed["base_model"],
                        "finetuned_dataset": other_ds,
                        "eval_accuracy": parsed["eval_accuracy"],
                        "task_type": parsed["task_type"],
                        "learning_rate": parsed["learning_rate"],
                        "batch_size": parsed["batch_size"],
                        "num_epochs": parsed["num_epochs"],
                        "created_at": parsed["created_at"],
                        "last_modified": parsed["last_modified"],
                        "license": parsed["license"],
                        "downloads": parsed["downloads"],
                    })
            
            if dataset_model_counts[dataset_name] % 50 == 0:
                print(f"    Collected {dataset_model_counts[dataset_name]} models, total rows: {len(flat_rows)}")
                print(f"    Models with base_model: {models_with_base}, without: {models_without_base}")
            
            # Rate limiting
            time.sleep(0.1)
    
    print(f"\n=== COLLECTION COMPLETE ===")
    print(f"Total records: {len(flat_rows)}")
    print(f"Models with base_model specified: {models_with_base}")
    print(f"Models without base_model: {models_without_base}")
    print(f"Base model coverage: {models_with_base / (models_with_base + models_without_base) * 100:.1f}%")
    print(f"Datasets with models: {len([c for c in dataset_model_counts.values() if c > 0])}")
    
    return flat_rows

# Import parse_model_card from parser_dense for sampling
def parse_model_card(row):
    tags_str, readme_str = separate_tags_and_readme(row["card"])
    yaml_data   = parse_yaml_tags(tags_str)
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
        "model":             row["modelId"],
        "base_model":        yaml_data["base_model"],
        "finetuned_datasets": all_datasets,
        "eval_accuracy":     readme_data["eval_accuracy"],
        "task_type":         row["pipeline_tag"],
        "learning_rate":     readme_data["learning_rate"],
        "batch_size":        readme_data["batch_size"],
        "num_epochs":        readme_data["num_epochs"],
        "created_at":        row["createdAt"],
        "last_modified":     row["last_modified"],
        "license":           yaml_data["license"],
    }

if __name__ == "__main__":
    # Run the hybrid approach
    flat_rows = collect_hybrid_approach(target_records=TARGET_RECORDS)
    
    # Save to CSV
    if flat_rows:
        keys = flat_rows[0].keys()
        with open("records_hybrid.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)
        
        n_models = len({r["model"] for r in flat_rows})
        n_base_models = len({r["base_model"] for r in flat_rows if r["base_model"]})
        n_datasets = len({r["finetuned_dataset"] for r in flat_rows})
        print(f"\n=== FINAL STATISTICS ===")
        print(f"Total (model, dataset) rows: {len(flat_rows)}")
        print(f"Unique base models:  {n_base_models}")
        print(f"Unique models:       {n_models}")
        print(f"Unique datasets:     {n_datasets}")
        print(f"Avg rows per dataset: {len(flat_rows) / n_datasets:.2f}")
        print(f"Avg rows per model: {len(flat_rows) / n_models:.2f}")
        print("Saved to records_hybrid.csv")
    else:
        print("No records collected!")
