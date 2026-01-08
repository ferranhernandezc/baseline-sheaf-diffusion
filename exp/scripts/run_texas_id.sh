#!/bin/sh

python -m exp.run \
    --dataset=texas \
    --add_hp=False \
    --d=1 \
    --layers=5 \
    --hidden_channels=32 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.03 \
    --weight_decay=5e-3 \
    --input_dropout=0.0 \
    --dropout=0.7 \
    --use_act=True \
    --model=IdentitySheaf \
    --normalised=True \
    --sparse_learner=True \
    --save_best_model=True \
    --return_rayleigh_quotient=True \
    --entity="${ENTITY}"