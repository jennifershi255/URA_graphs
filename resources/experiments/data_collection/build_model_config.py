import pandas as pd
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv
import time

DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env')
load_dotenv(DOTENV_PATH)

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sequence_classification', 'records.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sequence_classification', 'model_config_dataset.csv')

MODEL_TYPE_TO_PARAMS = {
    "distilbert": 66_000_000,
    "bert":       110_000_000,
    "roberta":    125_000_000,
    "albert":     12_000_000,
    "xlnet":      117_000_000,
    "deberta":    139_000_000,
    "deberta-v2": 900_000_000,
    "electra":    14_000_000,
    "camembert":  125_000_000,
    "xlm-roberta":125_000_000,
}

def get_num_parameters(config):
    """Estimate parameter count from config if possible."""
    if not config or not isinstance(config, dict):
        return None
    # Common config fields for parameter estimation
    hidden = config.get("hidden_size", None)
    layers = config.get("num_hidden_layers", None)
    intermediate = config.get("intermediate_size", None)
    vocab = config.get("vocab_size", None)
    if hidden and layers and intermediate and vocab:
        # Rough estimate: embedding + transformer layers
        return vocab * hidden + layers * (4 * hidden * intermediate)
    return None

def fetch_model_config(model_id, retries=3):
    for attempt in range(retries):
        try:
            info = api.model_info(model_id)
            config = info.config or {}
            architectures = config.get("architectures", [])
            architecture = architectures[0] if architectures else None
            model_type = config.get("model_type", None)
            num_params = get_num_parameters(config) or MODEL_TYPE_TO_PARAMS.get(model_type)
            return {
                "model": model_id,
                "architectures": architecture,
                "model_type": model_type,
                "number_of_parameters": num_params,
                "number_of_labels": config.get("num_labels", None),
                "labels": None,  # skip for now
                "memory_consumption": num_params * 4 if num_params else None,  # rough bytes estimate
            }
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error fetching {model_id}: {e}")
                return None

def main():
    records = pd.read_csv(RECORDS_PATH)
    unique_models = records['model'].unique()
    print(f"Fetching config for {len(unique_models)} unique models...")

    rows = []
    for i, model_id in enumerate(unique_models):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(unique_models)}")
        result = fetch_model_config(model_id)
        if result:
            # Add dataset and accuracy from records
            model_records = records[records['model'] == model_id]
            result['dataset'] = model_records['finetuned_dataset'].iloc[0]
            result['accuracy'] = model_records['eval_accuracy'].iloc[0] if 'eval_accuracy' in model_records.columns else None
            rows.append(result)
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=True)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    main()