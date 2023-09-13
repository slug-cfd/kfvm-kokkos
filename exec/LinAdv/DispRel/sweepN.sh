#!/bin/zsh

for ELL in {5,10};
do
    sed "s/ELL/$ELL/g" disprel.init.ref > disprel.init.tmp
    for FQ in $(seq -f "%03g" 1 128);
    do
	sed "s/FQ/$FQ/g" disprel.init.tmp > disprel.init
	bin/kfvm.ex disprel.init
    done
    rm disprel.init.tmp
done
