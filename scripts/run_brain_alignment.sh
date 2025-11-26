#!/bin/bash
#SBATCH --job-name=alignment
#SBATCH --account=aip-rudner
#SBATCH --time=02:00:00
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

source .venv/bin/activate

model_name="locality_gpt"
benchmark_name="Pereira2018.243sentences-linear"
cuda=0

python main.py \
    --model "$model_name" \
    --benchmark "$benchmark_name" \
    --device "$cuda" \
    --untrained