import argparse
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--train", default="record_train.csv")
ap.add_argument("--test", default="record_test.csv")
ap.add_argument("--dataset", required=True)
ap.add_argument("--k", type=int, default=10)
args = ap.parse_args()

pred = pd.read_csv(args.results)[["model", "score"]].dropna()
frames = []
for path in [args.train, args.test]:
    try:
        df = pd.read_csv(path)
        frames.append(df[["model", "dataset", "eval_accuracy"]])
    except Exception as e:
        print(f"(skipping {path}: {e})")
truth = pd.concat(frames, ignore_index=True)
truth = truth[truth["dataset"] == args.dataset]
if len(truth) == 0:
    print(f"ERROR: no ground-truth rows for '{args.dataset}'"); raise SystemExit
truth = truth.groupby("model", as_index=False)["eval_accuracy"].mean()
merged = pred.merge(truth, on="model", how="inner")
print(f"Dataset: {args.dataset}")
print(f"Predicted: {len(pred)} | truth: {len(truth)} | overlap: {len(merged)}")
if len(merged) < 3:
    print("Too few overlapping models."); print(merged); raise SystemExit
rho, rp = spearmanr(merged["score"], merged["eval_accuracy"])
tau, tp = kendalltau(merged["score"], merged["eval_accuracy"])
print(f"\nSpearman rho: {rho:.4f} (p={rp:.3g})")
print(f"Kendall  tau: {tau:.4f} (p={tp:.3g})")
k = min(args.k, len(merged))
tp_ = set(merged.sort_values("score", ascending=False).head(k)["model"])
tt_ = set(merged.sort_values("eval_accuracy", ascending=False).head(k)["model"])
print(f"Precision@{k}: {len(tp_ & tt_)/k:.3f}")
print(f"\nTop {k} predicted vs true accuracy:")
print(merged.sort_values("score", ascending=False).head(k).to_string(index=False))
