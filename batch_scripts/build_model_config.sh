#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH -o /u401/j286shi/transfergraph/logs/model_config_%j.out
#SBATCH -e /u401/j286shi/transfergraph/logs/model_config_%j.err
#SBATCH --mail-user=j286shi@uwaterloo.ca
#SBATCH --mail-type=ALL
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tg
export PYTHONNOUSERSITE=1
export HF_TOKEN=$(grep HF_TOKEN /u401/j286shi/transfergraph/.env | cut -d= -f2)
cd /u401/j286shi/transfergraph/resources/experiments/data_collection
python  build_model_config.py
