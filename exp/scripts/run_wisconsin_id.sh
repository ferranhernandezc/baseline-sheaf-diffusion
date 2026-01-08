#!/bin/sh

python -m exp.run \
    --add_hp=True \
    --add_lp=False \
    --d=1 \
    --dataset=wisconsin \
    --dropout=0.8 \
    --early_stopping=200 \
    --epochs=500 \
    --folds=10 \
    --hidden_channels=96 \
    --input_dropout=0 \
    --layers=5 \
    --lr=0.02 \
    --model=IdentitySheaf \
    --orth=householder \
    --sheaf_act=tanh \
    --weight_decay=0.0006685729356079199 \
    --use_act=True \
    --normalised=True \
    --sparse_learner=False \
    --edge_weights=True \
    --return_rayleigh_quotient=True \
    --save_best_model=True \
    --entity="${ENTITY}"