#!/bin/bash
#SBATCH --job-name=hierarchical_alignment
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
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

############################################
# Configurable variables
############################################

MODEL="gpt2"
NUM_UNITS=512

UNTRAINED=false            # true or false
USE_TOPIC_WISE_CV=false     # true or false
USE_SURPRISAL=false        # true or false

############################################
# Build tag strings for path
############################################

# Topic-wise tag: include ONLY if active
TOPIC_TAG=""
if [ "$USE_TOPIC_WISE_CV" = true ]; then
    TOPIC_TAG="topic_wise_cv"
fi

# Surprisal tag: include ONLY if active
SURP_TAG=""
if [ "$USE_SURPRISAL" = true ]; then
    SURP_TAG="with_surprisal"
fi

# Combine context tags, skip empty entries cleanly
CTX_TAG=$(printf "%s_%s" "$TOPIC_TAG" "$SURP_TAG" | sed 's/^_//;s/_$//;s/__/_/')

# Training tag: keep the trained / untrained convention
if [ "$UNTRAINED" = true ]; then
    TRAIN_TAG="untrained"
else
    TRAIN_TAG="trained"
fi

############################################
# Construct save path
############################################

BASE="data/scores/hierarchical_alignment"

# Add context tags only when non-empty
if [ -n "$CTX_TAG" ]; then
    BASE="${BASE}_${CTX_TAG}"
fi

# Add model folder
BASE="${BASE}/${MODEL}"

# Final path
SAVE_PATH="${BASE}/${TRAIN_TAG}_pereira2018"

mkdir -p "${SAVE_PATH}"
echo "Saving to: ${SAVE_PATH}"

############################################
# Run
############################################

python score_hierarchical_alignment.py \
    --model "${MODEL}" \
    --benchmark "Pereira2018.243sentences-partialr2" \
    --save_path "${SAVE_PATH}" \
    --num-units "${NUM_UNITS}" \
    $( [ "$UNTRAINED" = true ] && echo "--untrained" ) \
    --localize \
    --depths 1 2 3 4 5 6 7 8 9 10 11 12 \
    $( [ "$USE_TOPIC_WISE_CV" = false ] && echo "--no_topic_wise_cv" ) \
    $( [ "$USE_SURPRISAL" = true ] && echo "--use_surprisal" )
