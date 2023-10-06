#!/bin/zsh

for NG in {32,64,128,256};
do
    sed "s/NG/$NG/g" isenvort.init.ref > isenvort.init
    bin/kfvm.ex isenvort.init
done
