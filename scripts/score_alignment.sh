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

############################################
# Configurable variables
############################################

MODEL="locality_gpt2"
DECAY_RATE="-0.3"
NUM_UNITS=256

UNTRAINED=true            # true or false
USE_TOPIC_WISE_CV=true   # true or false
USE_SURPRISAL=false      # true or false

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
# Decay-rate tag only for locality_gpt2
############################################

DECAY_TAG=""
if [ "$MODEL" = "locality_gpt2" ]; then
    DECAY_TAG="linear_${DECAY_RATE}"
fi

############################################
# Construct save path
############################################

BASE="data/scores/localized_${NUM_UNITS}"

# Add context tags only when non-empty
if [ -n "$CTX_TAG" ]; then
    BASE="${BASE}_${CTX_TAG}"
fi

# Add model folder
BASE="${BASE}/${MODEL}"

# Final path: attach trained/untrained and decay
if [ -n "$DECAY_TAG" ]; then
    SAVE_PATH="${BASE}/${TRAIN_TAG}_pereira2018_${DECAY_TAG}"
else
    SAVE_PATH="${BASE}/${TRAIN_TAG}_pereira2018"
fi

mkdir -p "${SAVE_PATH}"
echo "Saving to: ${SAVE_PATH}"


############################################
# Run
############################################

python score_alignment.py \
    --model "${MODEL}" \
    --benchmark "Pereira2018.243sentences-partialr2" \
    --save_path "${SAVE_PATH}" \
    --num-units "${NUM_UNITS}" \
    $( [ "$MODEL" = "locality_gpt2" ] && echo "--decay-rate ${DECAY_RATE}" ) \
    $( [ "$UNTRAINED" = true ] && echo "--untrained" ) \
    --localize \
    $( [ "$USE_TOPIC_WISE_CV" = false ] && echo "--no_topic_wise_cv" ) \
    $( [ "$USE_SURPRISAL" = true ] && echo "--use_surprisal" )
