#!/bin/bash
#SBATCH --job-name=hierarchical_alignment
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/jacopo04/convergent-minds/logs/hierarchical-%j.out
#SBATCH --error=/scratch/jacopo04/convergent-minds/logs/hierarchical-%j.err
#SBATCH -D /scratch/jacopo04/convergent-minds

module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6 cudnn
module load gcc opencv arrow mpi4py

export CUDA_LAUNCH_BLOCKING=1

source .venv/bin/activate

MODEL=gpt2
SAVE_PATH="data/scores/hierarchical_alignment/${MODEL}_untrained"

python scripts/score_hierarchical_alignment.py \
    --model ${MODEL} \
    --benchmark Pereira2018.243sentences-partialr2 \
    --save_path ${SAVE_PATH} \
    --num-units 256 \
    --localize \
    --untrained \
    --depths 1 2 3 4 6 8 12
