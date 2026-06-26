#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /u401/j286shi/transfergraph/logs/tg_%j.out
#SBATCH -e /u401/j286shi/transfergraph/logs/tg_%j.err
#SBATCH --mail-user=j286shi@uwaterloo.ca
#SBATCH --mail-type=ALL

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/u401/j286shi/transfergraph/src:$PYTHONPATH
cd /u401/j286shi/transfergraph
export HF_TOKEN=$(grep HF_TOKEN /u401/j286shi/transfergraph/.env | cut -d= -f2)

python tools/2-transferability_estimation/run.py \
    --task_type=sequence_classification \
    --test_dataset=dair-ai/emotion \
    --dataset_reference_model=EleutherAI_gpt-neo-125m \
    --dataset_embed_method=domain_similarity \
    --gnn_method=SAGEConv_without_transfer \
    --contain_model_feature=False \
    --contain_dataset_feature=True