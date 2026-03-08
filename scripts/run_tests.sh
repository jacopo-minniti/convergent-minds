#!/bin/bash
#SBATCH --job-name=run_tests
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/jacopo04/convergent-minds/logs/run_tests-%j.out
#SBATCH --error=/scratch/jacopo04/convergent-minds/logs/run_tests-%j.err
#SBATCH -D /scratch/jacopo04/convergent-minds

export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="/scratch/jacopo04/convergent-minds/"

module --force purge
module load StdEnv/2023 python/3.11.5
module load cuda/12.6 cudnn
module load gcc opencv arrow mpi4py

source .venv/bin/activate

# Run new tests
echo "Running partial-r2 tests..."
pytest tests/test_objective_features.py tests/test_partial_r2.py

# Run model tests
echo "Running model tests..."
pytest models/