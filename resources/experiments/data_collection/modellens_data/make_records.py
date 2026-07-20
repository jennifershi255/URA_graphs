"""
Reformat ModelLens QA triples into TransferGraph's records.csv format.
"""
import argparse
import pandas as pd

# Full TransferGraph records.csv column order (from head -3 of their file)
TG_COLUMNS = [
    "Unnamed: 0", "model", "finetuned_dataset", "model_name", "dataset_name",
    "task_type", "batch_size", "lr_scheduler_type", "learning_rate",
    "gradient_accumulation_steps", "num_train_epochs", "seed", "dataset_path",
    "train_runtime", "eval_accuracy", "push_to_hub", "hub_model_id",
    "push_to_hub_organization", "hub_token", "peft_method",
    "lora_attention_dimension", "lora_alpha", "lora_dropout", "lora_bias",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="record_train.csv",
                    help="ModelLens triples: columns model, dataset, eval_accuracy")
    ap.add_argument("--output", default="records.csv")
    ap.add_argument("--task_type", default="text-classification",
                    help="TransferGraph filters on this; keep it consistent with how you invoke run.py")
    args = ap.parse_args()

    src = pd.read_csv(args.input)
    assert {"model", "dataset", "eval_accuracy"}.issubset(src.columns), \
        f"input must have model,dataset,eval_accuracy; got {list(src.columns)}"

    out = pd.DataFrame(columns=TG_COLUMNS)
    out["model"] = src["model"]
    out["finetuned_dataset"] = src["dataset"]
    out["eval_accuracy"] = src["eval_accuracy"]
    out["task_type"] = args.task_type
    out["Unnamed: 0"] = range(len(src))

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"  unique models:   {out['model'].nunique()}")
    print(f"  unique datasets: {out['finetuned_dataset'].nunique()}")
    print(f"  eval_accuracy range: {out['eval_accuracy'].min():.3f} - {out['eval_accuracy'].max():.3f}")
    print("\nReminder: TransferGraph normalizes accuracy within each dataset and "
          "splits pos/neg by threshold. Check that each dataset has enough spread "
          "to produce BOTH positive and negative edges (it needs negatives to train).")


if __name__ == "__main__":
    main()