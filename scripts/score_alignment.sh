#!/bin/bash
#SBATCH --job-name=alignment
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=/scratch/jacopo04/convergent-minds/logs/alignment-%j.out
#SBATCH --error=/scratch/jacopo04/convergent-minds/logs/alignment-%j.err
#SBATCH -D /scratch/jacopo04/convergent-minds

module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6 cudnn
module load gcc opencv arrow mpi4py

export CUDA_LAUNCH_BLOCKING=1

source .venv/bin/activate

MODEL=locality_gpt2
DECAY_RATE=0.5

python score_alignment.py \
    --model ${MODEL} \
    --benchmark Pereira2018.243sentences-partialr2 \
    --save_path "data/scores/partialr2_localized_128/${MODEL}/untrained_pereira2018_linear_${DECAY_RATE}" \
    --num-units 128 \
    --decay-rate ${DECAY_RATE} \
    --localize \
    --untrained