#!/bin/bash
#SBATCH --job-name=calc_lexical
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/jacopo04/convergent-minds/logs/calc_lexical-%j.out
#SBATCH --error=/scratch/jacopo04/convergent-minds/logs/calc_lexical-%j.err
#SBATCH -D /scratch/jacopo04/convergent-minds

export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="/scratch/jacopo04/convergent-minds/"

module --force purge
module load StdEnv/2023 python/3.11.5
module load cuda/12.6 cudnn
module load gcc opencv arrow mpi4py

source .venv/bin/activate

# Example usage for Pereira benchmarks
python alignment/precompute_objective_features.py \
    --benchmark Pereira2018.243sentences-linear \
    --output_dir data

# python alignment/precompute_objective_features.py \
#     --benchmark Pereira2018.384sentences-linear \
#     --output_dir data
