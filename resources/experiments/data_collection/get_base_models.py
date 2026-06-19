"""
Script to fetch base model information from Hugging Face for each model in records_dense.csv
"""

import pandas as pd
import os
from huggingface_hub import model_info
from typing import Optional
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_base_model_from_hf(model_id: str) -> Optional[str]:
    """
    Fetch base model information from Hugging Face model card.
    
    Args:
        model_id: Model identifier on Hugging Face (e.g., "username/model-name")
    
    Returns:
        Base model name if found, None otherwise
    """
    try:
        # Get model info from Hugging Face Hub
        info = model_info(repo_id=model_id)
        # print(info.get("card_data"))
        # print(info.get("tags"))
        # print(info, type(info))
        # print(info)

        # base_models is the promoted top-level field — most reliable
        parents = info.base_models or []
        parent = parents[0] if parents else None

        if not parent:
            # fallback: parse from model card content
            card_data = info.card_data or {}
            parent = card_data.get("base_model") or card_data.get("base_model_name")

        # fallback: parse tags if card_data is missing
        if not parent:
            for tag in (info.tags or []):
                if tag.startswith("base_model:"):
                    rest = tag[len("base_model:"):]
                    parts = rest.split(":", 1)
                    if len(parts) == 2 and parts[0] in ("finetune", "adapter", "quantized", "merge"):
                        relation, parent = parts
                    else:
                        parent = rest
        
        if isinstance(parent, list) and parent:
            parent = parent[0]
        
        if parent:
            print(f"{model_id} -> base_model: {parent}")
            return parent

        logger.info(f"{model_id} -> base_model not found")
        return None
        
    # except ModelNotFound:
    #     logger.warning(f"Model {model_id} not found on Hugging Face Hub")
    #     return None
    except Exception as e:
        logger.error(f"Error fetching base model for {model_id}: {e}")
        return None


def main():
    """Main function to process records_dense.csv and fetch base models."""
    
    # Define paths
    csv_path = "records_dense.csv"
    # csv_path = "records_dense_with_base_models.csv" 
    output_path = "records_dense_with_base_models.csv"
    
    if not os.path.exists(csv_path):
        logger.error(f"File {csv_path} not found")
        return
    
    # Load CSV
    logger.info(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check if base_model column exists and ensure it can hold strings
    if 'base_model' not in df.columns:
        df['base_model'] = pd.Series([None] * len(df), dtype=object)
    else:
        # If column exists but has a non-object dtype (e.g., float), convert it
        if df['base_model'].dtype != object:
            df['base_model'] = df['base_model'].astype(object)
    
    # Get unique models
    models = df['model'].unique()
    logger.info(f"Found {len(models)} unique models")
    
    # Fetch base models
    base_models_cache = {}
    for i, model_id in enumerate(models):
        logger.info(f"[{i+1}/{len(models)}] Processing {model_id}...")
        
        if pd.isna(model_id):
            continue
        
        model_id = str(model_id).strip()
        
        # Check cache
        if model_id in base_models_cache:
            base_model = base_models_cache[model_id]
        else:
            # Fetch from Hugging Face
            base_model = get_base_model_from_hf(model_id)
            base_models_cache[model_id] = base_model
        
        # Update dataframe
        if base_model:
            df.loc[df['model'] == model_id, 'base_model'] = base_model
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save updated CSV
    logger.info(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Print summary
    filled_count = df['base_model'].notna().sum()
    logger.info(f"Successfully fetched {filled_count}/{len(df)} base models")
    logger.info(f"Results saved to {output_path}")
    
    # Display samples
    logger.info("\nSample results:")
    sample = df[df['base_model'].notna()].head(10)[['model', 'base_model']]
    logger.info("\n" + str(sample))


if __name__ == "__main__":
    main()
