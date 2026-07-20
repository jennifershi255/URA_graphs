#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH -o /u401/j286shi/transfergraph/logs/collect_%j.out
#SBATCH -e /u401/j286shi/transfergraph/logs/collect_%j.err
#SBATCH --mail-user=j286shi@uwaterloo.ca
#SBATCH --mail-type=ALL

source /opt/anaconda3/etc/profile.d/conda.sh
conda deactivate

cd /u401/j286shi/transfergraph/resources/experiments/data_collection
/opt/anaconda3/bin/python3 parser_dense.py
