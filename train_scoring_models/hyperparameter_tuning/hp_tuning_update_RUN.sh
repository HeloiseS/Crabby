#!/bin/bash

LR_list=(0.1 0.2 0.3 0.4 0.5)
L2_list=(1 10)

for LR in "${LR_list[@]}"; do
    for L2 in "${L2_list[@]}"; do
       #echo $LR $L2 $TYPE
       python hp_tuning_update_RUN.py 'real' $LR $L2
       python hp_tuning_update_RUN.py 'gal' $LR $L2
    done
done


