#!/bin/bash

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=$((i % 4)) wandb agent "${ENTITY}"/sheaf/"$1" &
done