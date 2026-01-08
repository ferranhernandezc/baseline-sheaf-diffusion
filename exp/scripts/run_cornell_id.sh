#!/bin/sh

python -m exp.run \
    --add_hp=True \
    --add_lp=False \
    --d=1 \
    --dataset=cornell \
    --dropout=0.9 \
    --early_stopping=200 \
    --epochs=500 \
    --folds=10 \
    --hidden_channels=64 \
    --input_dropout=0.0 \
    --layers=4 \
    --lr=0.02 \
    --model=IdentitySheaf \
    --sheaf_decay=0.00031764232712732976 \
    --weight_decay=0.0006914841722570725 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --sparse_learner=False \
    --return_rayleigh_quotient=True \
    --save_best_model=True \
    --entity="${ENTITY}"