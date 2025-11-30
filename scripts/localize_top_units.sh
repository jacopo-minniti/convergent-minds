#!/bin/bash
#SBATCH --job-name=localization
#SBATCH --account=aip-rudner
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=/scratch/jacopo04/convergent-minds/logs/localization-%j.out
#SBATCH --error=/scratch/jacopo04/convergent-minds/logs/localization-%j.err
#SBATCH -D /scratch/jacopo04/convergent-minds

seed=42
num_units=128
embed_agg=last-token
model=gpt2

source .venv/bin/activate

python localization/extract_activations.py --model-name $model --seed $seed --embed-agg $embed_agg --overwrite
python localization/localize_fed10.py --model-name $model --seed $seed --embed-agg $embed_agg --num-units $num_units --overwrite
