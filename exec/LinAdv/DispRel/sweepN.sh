#!/bin/zsh

for FQ in $(seq -f "%03g" 1 128);
do
    sed "s/FQ/$FQ/g" disprel.init.ref > disprel.init
    bin/kfvm.ex disprel.init
done
