#!/bin/zsh

for NG in {32,64,128,256,512};
do
    sed "s/NG/$NG/g" magvort.init.ref > magvort.init
    bin/kfvm.ex magvort.init
done
