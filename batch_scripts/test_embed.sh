#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /u401/j286shi/transfergraph/logs/test_embed_%j.out
#SBATCH -e /u401/j286shi/transfergraph/logs/test_embed_%j.err


source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/u401/j286shi/transfergraph/src:$PYTHONPATH
cd /u401/j286shi/transfergraph
export HF_TOKEN=$(grep HF_TOKEN /u401/j286shi/transfergraph/.env | cut -d= -f2)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python tools/1-preparation/embed_dataset.py \
    --task_type=sequence_classification \
    --model_name=EleutherAI/gpt-neo-125m \
    --embedding_method=domain_similarity \
    --batch_size=16 \
    --dataset_path=stanfordnlp/imdb
