#!/bin/bash

model_name="gpt2"
benchmark_name="Pereira2018.384sentences-cka"
seed=42
cuda=0

python brain-alignment/score_brain_alignment.py --model-name $model_name \
    --benchmark-name $benchmark_name \
    --seed $seed \
    --cuda $cuda \
    --overwrite

