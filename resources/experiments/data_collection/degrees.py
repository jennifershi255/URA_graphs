import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# THIS FILE COMPARES THE NODE DEGREES BETWEEN OUR DATA AND TRANSFERGRAPH 
def load_and_clean(path, model_col, dataset_col):
    df = pd.read_csv(path)

    df = df[[model_col, dataset_col]].dropna()
    df = df.drop_duplicates()

    df.columns = ["model", "dataset"]

    return df


def graph_stats(df, name):
    n_models = df["model"].nunique()
    n_datasets = df["dataset"].nunique()
    n_edges = len(df)

    model_degree = df.groupby("model")["dataset"].nunique()
    dataset_degree = df.groupby("dataset")["model"].nunique()

    print(f"\n=== {name} ===")
    print(f"Models (nodes):   {n_models}")
    print(f"Datasets (nodes): {n_datasets}")
    print(f"Total nodes:      {n_models + n_datasets}")
    print(f"Edges:            {n_edges}")
    print(f"Avg degree (overall): {(2 * n_edges) / (n_models + n_datasets):.2f}")

    print(f"\nAvg datasets per model: {model_degree.mean():.2f}")
    print(f"Avg models per dataset: {dataset_degree.mean():.2f}")

    return model_degree, dataset_degree


def bucket_degrees(degrees):
    return {
        "0": (degrees == 0).sum(),
        "1-10": ((degrees >= 1) & (degrees <= 10)).sum(),
        "11-20": ((degrees >= 11) & (degrees <= 20)).sum(),
        "21-30": ((degrees >= 21) & (degrees <= 30)).sum(),
        "30+": (degrees > 30).sum(),
    }


def plot_degree_distributions(our_model_deg, our_dataset_deg, tg_model_deg, tg_dataset_deg):
    """
    Plot degree distributions comparing our graph with TransferGraph.
    Creates 2x2 subplots for model degrees and dataset degrees.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Degree Distribution Comparison: Our Dense Graph vs TransferGraph', 
                 fontsize=16, fontweight='bold')

    # Define buckets
    buckets = ["0", "1-10", "11-20", "21-30", "30+"]
    
    # --- Row 1: Model Degrees ---
    
    # Our Model Degrees (buckets)
    our_model_buckets = bucket_degrees(our_model_deg)
    axes[0, 0].bar(buckets, [our_model_buckets[b] for b in buckets], color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Our Graph: Model Degree Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Degree Range')
    axes[0, 0].set_ylabel('Number of Models')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bucket in enumerate(buckets):
        val = our_model_buckets[bucket]
        if val > 0:
            axes[0, 0].text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
    
    # TransferGraph Model Degrees (buckets)
    tg_model_buckets = bucket_degrees(tg_model_deg)
    axes[0, 1].bar(buckets, [tg_model_buckets[b] for b in buckets], color='coral', alpha=0.7)
    axes[0, 1].set_title('TransferGraph: Model Degree Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Degree Range')
    axes[0, 1].set_ylabel('Number of Models')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bucket in enumerate(buckets):
        val = tg_model_buckets[bucket]
        if val > 0:
            axes[0, 1].text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
    
    # --- Row 2: Dataset Degrees ---
    
    # Our Dataset Degrees (buckets)
    our_dataset_buckets = bucket_degrees(our_dataset_deg)
    axes[1, 0].bar(buckets, [our_dataset_buckets[b] for b in buckets], color='green', alpha=0.7)
    axes[1, 0].set_title('Our Graph: Dataset Degree Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Degree Range')
    axes[1, 0].set_ylabel('Number of Datasets')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bucket in enumerate(buckets):
        val = our_dataset_buckets[bucket]
        if val > 0:
            axes[1, 0].text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
    
    # TransferGraph Dataset Degrees (buckets)
    tg_dataset_buckets = bucket_degrees(tg_dataset_deg)
    axes[1, 1].bar(buckets, [tg_dataset_buckets[b] for b in buckets], color='purple', alpha=0.7)
    axes[1, 1].set_title('TransferGraph: Dataset Degree Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Degree Range')
    axes[1, 1].set_ylabel('Number of Datasets')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bucket in enumerate(buckets):
        val = tg_dataset_buckets[bucket]
        if val > 0:
            axes[1, 1].text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('degree_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'degree_distribution_comparison.png'")
    plt.show()


def plot_side_by_side_comparison(our_model_deg, our_dataset_deg, tg_model_deg, tg_dataset_deg):
    """
    Create side-by-side bar charts comparing bucket counts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Bucket Comparison: Our Dense Graph vs TransferGraph', 
                 fontsize=16, fontweight='bold')
    
    buckets = ["0", "1-10", "11-20", "21-30", "30+"]
    x = np.arange(len(buckets))
    width = 0.35
    
    # Model degrees comparison
    our_model_buckets = bucket_degrees(our_model_deg)
    tg_model_buckets = bucket_degrees(tg_model_deg)
    
    axes[0].bar(x - width/2, [our_model_buckets[b] for b in buckets], 
                width, label='Our Graph', color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, [tg_model_buckets[b] for b in buckets], 
                width, label='TransferGraph', color='coral', alpha=0.8)
    axes[0].set_xlabel('Degree Range')
    axes[0].set_ylabel('Number of Models')
    axes[0].set_title('Model Degree Distribution')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Dataset degrees comparison
    our_dataset_buckets = bucket_degrees(our_dataset_deg)
    tg_dataset_buckets = bucket_degrees(tg_dataset_deg)
    
    axes[1].bar(x - width/2, [our_dataset_buckets[b] for b in buckets], 
                width, label='Our Graph', color='green', alpha=0.8)
    axes[1].bar(x + width/2, [tg_dataset_buckets[b] for b in buckets], 
                width, label='TransferGraph', color='purple', alpha=0.8)
    axes[1].set_xlabel('Degree Range')
    axes[1].set_ylabel('Number of Datasets')
    axes[1].set_title('Dataset Degree Distribution')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(buckets)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('side_by_side_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'side_by_side_comparison.png'")
    plt.show()


if __name__ == "__main__":

    # OUR DENSE DATA
    our_df = load_and_clean(
        "records_dense.csv",
        model_col="model",
        dataset_col="finetuned_dataset"
    )
    our_model_deg, our_dataset_deg = graph_stats(our_df, "OUR DENSE GRAPH")

    # TRANSFERGRAPH
    tg_df = load_and_clean(
        "tg_records.csv",
        model_col="model_name",
        dataset_col="finetuned_dataset"
    )
    tg_model_deg, tg_dataset_deg = graph_stats(tg_df, "TRANSFERGRAPH")

    print("\n" + "="*60)
    print("DEGREE BUCKET COMPARISON")
    print("="*60)

    print("\n--- MODEL DEGREES ---")
    print(f"{'Bucket':<12} {'Our Graph':<15} {'TransferGraph':<15}")
    print("-" * 42)
    our_model_buckets = bucket_degrees(our_model_deg)
    tg_model_buckets = bucket_degrees(tg_model_deg)
    for bucket in ["0", "1-10", "11-20", "21-30", "30+"]:
        print(f"{bucket:<12} {our_model_buckets[bucket]:<15} {tg_model_buckets[bucket]:<15}")

    print("\n--- DATASET DEGREES ---")
    print(f"{'Bucket':<12} {'Our Graph':<15} {'TransferGraph':<15}")
    print("-" * 42)
    our_dataset_buckets = bucket_degrees(our_dataset_deg)
    tg_dataset_buckets = bucket_degrees(tg_dataset_deg)
    for bucket in ["0", "1-10", "11-20", "21-30", "30+"]:
        print(f"{bucket:<12} {our_dataset_buckets[bucket]:<15} {tg_dataset_buckets[bucket]:<15}")

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS...")
    print("="*60)
    
    plot_degree_distributions(our_model_deg, our_dataset_deg, tg_model_deg, tg_dataset_deg)
    plot_side_by_side_comparison(our_model_deg, our_dataset_deg, tg_model_deg, tg_dataset_deg)