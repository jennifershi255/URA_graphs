import argparse
import json
import pandas as pd


def safe_size_to_params(size_b):
    """model_profile size is in billions of params; TG wants raw param count."""
    try:
        return float(size_b) * 1e9
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="record_train.csv")
    ap.add_argument("--model2family", default="model2family.json")
    ap.add_argument("--model_profile", default="model_profile.json")
    ap.add_argument("--output", default="model_config_dataset.csv")
    args = ap.parse_args()

    records = pd.read_csv(args.records)
    with open(args.model2family) as f:
        model2family = json.load(f)
    with open(args.model_profile) as f:
        model_profile = json.load(f)

    rows = []
    for i, r in enumerate(records.itertuples()):
        family = model2family.get(r.model, "Unknown")
        prof = model_profile.get(r.model, {})
        n_params = safe_size_to_params(prof.get("size"))
        rows.append({
            "": i,
            "model": r.model,
            # TG uses architectures/model_type as categorical model features;
            # family is the best proxy we have from ModelLens metadata.
            "architectures": family,
            "model_type": family,
            "number_of_parameters": n_params,
            "number_of_labels": None,
            "labels": None,
            "memory_consumption": (n_params * 4) if n_params else None,  # ~4 bytes/param, matches their scale
            "dataset": r.dataset,
            "accuracy": r.eval_accuracy,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"  unique models:   {out['model'].nunique()}")
    print(f"  families mapped:  {out['architectures'].nunique()} "
          f"({(out['architectures'] == 'Unknown').sum()} rows Unknown)")
    print(f"  params known for: {out['number_of_parameters'].notna().mean()*100:.1f}% of rows")


if __name__ == "__main__":
    main()