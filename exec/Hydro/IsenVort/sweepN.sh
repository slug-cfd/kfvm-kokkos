#!/bin/zsh

for ELL in {2,5,10};
do
    sed "s/ELL/$ELL/g" isenvort.init.ref > isenvort.init.tmp
    for NG in {32,64,128,256};
    do
	sed "s/NG/$NG/g" isenvort.init.tmp > isenvort.init
	bin/kfvm.ex isenvort.init
    done
    rm isenvort.init.tmp
done
