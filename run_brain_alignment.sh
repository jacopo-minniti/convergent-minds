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

model_name="locality-gpt2"
benchmark_name="Pereira2018.243sentences-linear" # "Pereira2018.384sentences-cka"
seed=42
cuda=0

source .venv/bin/activate

python brain-alignment/score_brain_alignment.py --model-name $model_name \
    --benchmark-name $benchmark_name \
    --seed $seed \
    --cuda $cuda \
    --overwrite \
    --savepath dumps/scores_untrained_locality_-0.3_gpt2_Pereira2018.243_linear.pkl \
    --untrained