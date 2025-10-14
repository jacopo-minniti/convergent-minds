#!/bin/bash

seed=42
num_units=4096
embed_agg=last-token
model=gpt2

python language-localization/extract_activations.py --model-name $model --seed $seed --embed-agg $embed_agg --overwrite
python language-localization/localize_fed10.py --model-name $model --seed $seed --embed-agg $embed_agg --num-units $num_units --overwrite
