#!/bin/bash
#SBATCH --job-name=hierarchical_alignment
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
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
SAVE_PATH="data/scores/hierarchical_alignment/${MODEL}_no_topic_cv"

python score_hierarchical_alignment.py \
    --model ${MODEL} \
    --benchmark Pereira2018.243sentences-partialr2 \
    --save_path ${SAVE_PATH} \
    --num-units 512 \
    --localize \
    --depths 1 2 3 4 5 6 7 8 9 10 11 12 \
    --no_topic_wise_cv
