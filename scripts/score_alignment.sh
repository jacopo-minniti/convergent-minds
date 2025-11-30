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

python main.py \
    --model locality_gpt \
    --benchmark Pereira2018.243sentences-linear \
    --save_path results/localized/locality_gpt2/v2_untrained_pereira2018_linear_0.3 \
    --num-units 128 \
    --decay-rate 0.3 \
    --localize \
    --untrained

# pytest models/locality_gpt/test_robustness.py
# python alignment/plot_attention.py --untrained --model locality_gpt


Nice! I want to include all the features you talked about UP UNTIL 3.1 (so no 3.2 and 3.3)!

Now, give me a compresnive, markdown inside a .md cell report of what a developer should be given to make all the changes in the base repo for our new idea of partial r^2. Remember, most of the things should remain unchanges, such as the ridge regression multi objective stuff, or the CV, or the way the pipeline should be started through the scripts/score_alignment.sh, but we should introduce all the changes we discussed (also the details, such as nornalizing X_obj and X_obj+X_LLM). The final delta r^2 is still called alignment score.

Importantly, for the object matrix, add a .sh script similar to the score_alignment.sh and in the same folder called calculate_lexical_statistics.sh which basically uses a simple corpus to then calculate the statistics for 3.1