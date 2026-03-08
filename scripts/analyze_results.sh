#!/bin/bash
#SBATCH --job-name=analyze_results
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/jacopo04/convergent-minds/logs/analyze_results-%j.out
#SBATCH --error=/scratch/jacopo04/convergent-minds/logs/analyze_results-%j.err
#SBATCH -D /scratch/jacopo04/convergent-minds

module --force purge
module load StdEnv/2023
module load python/3.11.5

source .venv/bin/activate

# Example: Run attention plotting
# python alignment/plot_attention.py --model locality_gpt --untrained

# Example: Run other analysis scripts
# python alignment/some_other_analysis.py ...
