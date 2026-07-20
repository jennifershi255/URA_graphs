#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /u401/j286shi/transfergraph/logs/embed_%j.out
#SBATCH -e /u401/j286shi/transfergraph/logs/embed_%j.err
#SBATCH --mail-user=j286shi@uwaterloo.ca
#SBATCH --mail-type=ALL

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/u401/j286shi/transfergraph/src:$PYTHONPATH
cd /u401/j286shi/transfergraph
export HF_TOKEN=$(grep HF_TOKEN /u401/j286shi/transfergraph/.env | cut -d= -f2)

declare -A DATASETS
DATASETS["stanfordnlp/imdb"]=""
DATASETS["dair-ai/emotion"]=""
DATASETS["emotion"]=""
DATASETS["go_emotions"]=""
DATASETS["google-research-datasets/go_emotions"]=""
DATASETS["sst2"]=""
DATASETS["google/boolq"]=""
DATASETS["nsmc"]=""
DATASETS["tweet_eval"]="sentiment"
DATASETS["financial_phrasebank"]="sentences_allagree"
DATASETS["AI-Secure/PolyGuard"]=""
DATASETS["ToxicityPrompts/PolyGuardMix"]=""
DATASETS["lmsys/toxic-chat"]=""
DATASETS["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"]=""
DATASETS["enguard/multi-lingual-prompt-moderation"]=""
DATASETS["visolex/vihsd"]=""
DATASETS["Intel/polite-guard"]=""
DATASETS["trl-internal-testing/tldr-preference-sft-trl-style"]=""
DATASETS["Tevatron/msmarco-passage"]=""
DATASETS["contemmcm/hate-speech-and-offensive-language"]=""
DATASETS["contemmcm/clickbait"]=""

for dataset_path in "${!DATASETS[@]}"; do
    dataset_name="${DATASETS[$dataset_path]}"
    echo "=== Embedding: $dataset_path (config: $dataset_name) ==="
    if [[ -z "$dataset_name" ]]; then
        python tools/1-preparation/embed_dataset.py \
            --task_type=sequence_classification \
            --model_name=EleutherAI/gpt-neo-125m \
            --embedding_method=domain_similarity \
            --batch_size=16 \
            --max_train_samples=5000 \
            --dataset_path=$dataset_path
    else
        python tools/1-preparation/embed_dataset.py \
            --task_type=sequence_classification \
            --model_name=EleutherAI/gpt-neo-125m \
            --embedding_method=domain_similarity \
            --batch_size=16 \
            --max_train_samples=5000 \
            --dataset_path=$dataset_path \
            --dataset_name=$dataset_name
    fi
    echo "=== Done: $dataset_path ==="
done

echo "All embeddings complete."
