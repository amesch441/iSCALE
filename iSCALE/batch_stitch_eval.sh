#!/bin/bash
set -e

for factor in 1 2 4 8; do
    for dataname in \
        "xenium/" \
        "xenium/oos/" \
        ; do
        python stitch_eval.py $dataname $factor &
    done
    wait
done
wait
