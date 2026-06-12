import pandas as pd

df = pd.read_csv("records_raw2.csv")

# THIS FILE CONVERTS THE RAW DATA INTO TRANSFERGRAPH'S FORMAT

def split_dataset(finetuned_dataset):
    if not isinstance(finetuned_dataset, str):
        return None, None

    if "[" in finetuned_dataset:
        path = finetuned_dataset.split("[")[0].strip()
        name = finetuned_dataset.split("[")[1].replace("]", "").strip()
        return path, name

    if "/" in finetuned_dataset:
        parts = finetuned_dataset.split("/")
        return finetuned_dataset, parts[-1]

    return finetuned_dataset, finetuned_dataset


df["dataset_path"], df["dataset_name"] = zip(
    *df["finetuned_dataset"].apply(split_dataset)
)

# ONE ROW PER (model, dataset) EDGE
tg_df = pd.DataFrame()

tg_df["model"] = df["model"]
tg_df["finetuned_dataset"] = df["finetuned_dataset"]

tg_df["model_name"] = df["model"]
tg_df["dataset_name"] = df["dataset_name"]

tg_df["task_type"] = "sequence_classification"
tg_df["batch_size"] = df["batch_size"]
tg_df["lr_scheduler_type"] = "linear"
tg_df["learning_rate"] = df["learning_rate"]
tg_df["gradient_accumulation_steps"] = 1
tg_df["num_train_epochs"] = df["num_epochs"]
tg_df["seed"] = 42
tg_df["dataset_path"] = df["dataset_path"]
tg_df["train_runtime"] = None
tg_df["eval_accuracy"] = df["eval_accuracy"]

tg_df["push_to_hub"] = None
tg_df["hub_model_id"] = None
tg_df["push_to_hub_organization"] = None
tg_df["hub_token"] = None

tg_df["peft_method"] = None
tg_df["lora_attention_dimension"] = None
tg_df["lora_alpha"] = None
tg_df["lora_dropout"] = None
tg_df["lora_bias"] = None

tg_df.to_csv("records_transfergraph.csv", index=True)

print(f"Total rows: {len(tg_df)}")
print(f"Unique models: {tg_df['model'].nunique()}")
print(f"Unique datasets: {tg_df['dataset_name'].nunique()}")

print("\nSaved records_transfergraph.csv")