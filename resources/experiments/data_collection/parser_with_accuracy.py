"""
parser_with_accuracy.py

Incrementally scrapes HuggingFace for text-classification models fine-tuned
on popular English datasets, writing results to a CSV and saving a checkpoint
after every round so the run can be resumed if interrupted.

Usage
-----
python parser_with_accuracy.py            # run with defaults
python parser_with_accuracy.py --help     # show all options

Key flags
---------
--target-rows     N   Stop after collecting N usable rows (default 500)
--datasets-per-round K  Datasets to scan each round (default 10)
--max-models-per-dataset  Max models fetched per dataset (default 150)
--min-models      Minimum fine-tuned model count to qualify a dataset (default 10)
--output          Path to output CSV (default records_from_datasets.csv)
--checkpoint      Path to checkpoint JSON (default checkpoint.json)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from tqdm.auto import tqdm

# ── constants ────────────────────────────────────────────────────────────────

FIXED_TIMESTAMP = "2026-01-01"

FIELDNAMES = [
    "model", "base_model", "finetuned_dataset", "eval_accuracy",
    "task_type", "learning_rate", "batch_size", "num_epochs",
    "created_at", "last_modified", "license",
]

# ── regex helpers ─────────────────────────────────────────────────────────────

ACCURACY_RE = re.compile(
    r"(?:eval_?)?accuracy[:\s=]+([0-9]+(?:\.[0-9]+)?)\s*%?", re.IGNORECASE
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

# ── parsing helpers ───────────────────────────────────────────────────────────

def get_readme_text(model_id: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def parse_readme_body(readme_str: str) -> dict:
    result = {"eval_accuracy": None, "learning_rate": None,
              "batch_size": None, "num_epochs": None}
    if not readme_str:
        return result

    acc = ACCURACY_RE.search(readme_str)
    if acc:
        val = float(acc.group(1))
        if 0 <= val <= 100:
            result["eval_accuracy"] = val / 100 if val > 1.0 else val

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


def parse_model(model_id: str, info, finetuned_dataset: str) -> dict:
    card_data = info.card_data.to_dict() if info.card_data else {}
    readme_text = get_readme_text(model_id)
    readme_data = parse_readme_body(readme_text)

    bm = card_data.get("base_model")
    base_model = bm[0] if isinstance(bm, list) and bm else (bm or "")

    return {
        "model":             model_id,
        "base_model":        base_model,
        "finetuned_dataset": finetuned_dataset,
        "eval_accuracy":     readme_data["eval_accuracy"],
        "task_type":         info.pipeline_tag or "",
        "learning_rate":     readme_data["learning_rate"],
        "batch_size":        readme_data["batch_size"],
        "num_epochs":        readme_data["num_epochs"],
        "created_at":        str(info.created_at) if info.created_at else "",
        "last_modified":     str(info.last_modified) if info.last_modified else "",
        "license":           card_data.get("license", "") or "",
    }

# ── dataset helpers ───────────────────────────────────────────────────────────

def get_candidate_datasets(api: HfApi, offset: int, limit: int) -> list:
    """
    Return a page of English text-classification datasets sorted by downloads.
    HF's list_datasets doesn't natively support offset, so we fetch offset+limit
    and slice. For large offsets this is wasteful but keeps the code simple.
    """
    datasets = api.list_datasets(
        filter=["task_categories:text-classification", "language:en"],
        sort="downloads",
        limit=offset + limit,
    )
    all_ds = list(datasets)
    return all_ds[offset:]


def count_finetuned_models(api: HfApi, dataset_id: str, cap: int = 50) -> int:
    try:
        models = api.list_models(filter=f"dataset:{dataset_id}", limit=cap)
        return len(list(models))
    except Exception as e:
        print(f"  [warn] failed for {dataset_id}: {e}")
        return 0

# ── checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "dataset_offset": 0,          # how many datasets we've already scanned
        "seen_datasets": [],          # dataset ids already processed
        "seen_models": [],            # model ids already written to CSV
        "usable_row_count": 0,        # rows written so far
        "round_number": 0,
    }


def save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)             # atomic on POSIX

# ── CSV helpers ───────────────────────────────────────────────────────────────

def open_csv(output_path: str, resume: bool):
    """Open CSV for appending (resume) or writing (fresh start)."""
    if resume and os.path.exists(output_path):
        f = open(output_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    else:
        f = open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
    return f, writer

# ── main loop ─────────────────────────────────────────────────────────────────

def run(args):
    api = HfApi()

    state = load_checkpoint(args.checkpoint)
    resuming = state["usable_row_count"] > 0

    if resuming:
        print(f"\n▶ Resuming from checkpoint: round {state['round_number']}, "
              f"{state['usable_row_count']} usable rows already collected.\n")
    else:
        print("\n▶ Starting fresh.\n")

    seen_datasets = set(state["seen_datasets"])
    seen_models   = set(state["seen_models"])

    csv_file, writer = open_csv(args.output, resume=resuming)

    try:
        while state["usable_row_count"] < args.target_rows:
            state["round_number"] += 1
            round_num = state["round_number"]
            print(f"\n{'='*60}")
            print(f"  ROUND {round_num}  |  usable rows so far: {state['usable_row_count']}/{args.target_rows}")
            print(f"{'='*60}")

            # ── Step 1: find datasets for this round ──────────────────────
            print(f"\n[Step 1] Fetching candidate datasets (offset={state['dataset_offset']}, "
                  f"looking for {args.datasets_per_round} new ones)…")

            round_datasets = []
            fetch_offset   = state["dataset_offset"]
            fetch_batch    = args.datasets_per_round * 3  # over-fetch to skip already-seen

            while len(round_datasets) < args.datasets_per_round:
                candidates = get_candidate_datasets(api, offset=fetch_offset,
                                                    limit=fetch_batch)
                if not candidates:
                    print("  No more datasets available on HuggingFace.")
                    break

                for ds in candidates:
                    if ds.id in seen_datasets:
                        continue
                    n = count_finetuned_models(api, ds.id, cap=args.min_models)
                    if n >= args.min_models:
                        round_datasets.append((ds.id, n))
                        seen_datasets.add(ds.id)
                        if len(round_datasets) >= args.datasets_per_round:
                            break

                fetch_offset += len(candidates)
                if len(candidates) < fetch_batch:
                    break   # exhausted HF

            state["dataset_offset"] = fetch_offset
            state["seen_datasets"]  = list(seen_datasets)

            if not round_datasets:
                print("  No new qualified datasets found — stopping early.")
                break

            print(f"  Qualified datasets this round: {len(round_datasets)}")
            for ds_id, n in round_datasets:
                print(f"    • {ds_id} ({n}+ models)")

            # ── Step 2: gather model details ──────────────────────────────
            print(f"\n[Step 2] Gathering model details…")
            round_new_rows = 0

            for ds_id, _ in tqdm(round_datasets, desc="  Datasets", unit="dataset"):
                try:
                    models = list(api.list_models(
                        filter=f"dataset:{ds_id}",
                        limit=args.max_models_per_dataset,
                        sort="downloads",
                        cardData=True,
                        full=True,
                    ))
                except Exception as e:
                    print(f"  [warn] list_models failed for {ds_id}: {e}")
                    continue

                for m in models:
                    model_id = m.id
                    if model_id in seen_models:
                        continue

                    parsed = parse_model(model_id, m, ds_id)
                    seen_models.add(model_id)

                    if parsed["eval_accuracy"] is not None and parsed["base_model"]:
                        writer.writerow(parsed)
                        csv_file.flush()
                        state["usable_row_count"] += 1
                        round_new_rows += 1

                    if state["usable_row_count"] >= args.target_rows:
                        break

                if state["usable_row_count"] >= args.target_rows:
                    break

            state["seen_models"] = list(seen_models)

            # ── checkpoint after each round ───────────────────────────────
            save_checkpoint(args.checkpoint, state)
            print(f"\n  ✓ Round {round_num} done. New usable rows: {round_new_rows}. "
                  f"Total: {state['usable_row_count']}/{args.target_rows}. "
                  f"Checkpoint saved.")

    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted! Saving checkpoint before exit…")
        state["seen_models"] = list(seen_models)
        save_checkpoint(args.checkpoint, state)
        print(f"   Checkpoint saved to {args.checkpoint}. "
              f"Re-run the script to resume.")
    finally:
        csv_file.close()

    print(f"\n✅ Done. {state['usable_row_count']} usable rows written to {args.output}.")
    if state["usable_row_count"] >= args.target_rows:
        print("   Target reached!")
    else:
        print("   Target not yet reached — re-run to continue.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-rows",              type=int, default=500,
                   help="Stop after this many usable rows (default: 500)")
    p.add_argument("--datasets-per-round",       type=int, default=10,
                   help="Datasets to scan per round (default: 10)")
    p.add_argument("--max-models-per-dataset",   type=int, default=150,
                   help="Max models fetched per dataset (default: 150)")
    p.add_argument("--min-models",               type=int, default=10,
                   help="Min fine-tuned models for a dataset to qualify (default: 10)")
    p.add_argument("--output",                   default="records_from_datasets.csv",
                   help="Output CSV path (default: records_from_datasets.csv)")
    p.add_argument("--checkpoint",               default="checkpoint.json",
                   help="Checkpoint JSON path (default: checkpoint.json)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())