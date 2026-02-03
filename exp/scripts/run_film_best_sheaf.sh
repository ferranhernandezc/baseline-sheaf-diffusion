#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --d=4 \
    --dataset=film \
    --dropout=0.5 \
    --early_stopping=100 \
    --epochs=1000 \
    --folds=10 \
    --hidden_channels=64 \
    --input_dropout=0.1 \
    --layers=1 \
    --lr=0.0002 \
    --model=DiagSheaf \
    --second_linear=True \
    --weight_decay=0.0000001 \
    --left_weights=True \
    --right_weights=True \
    --normalised=False \
    --deg_normalised=True \
    --stop_strategy=acc \
    --return_rayleigh_quotient=True \
    --save_best_model=True \
    --use_act=True