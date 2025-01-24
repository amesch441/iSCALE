#!/bin/bash
set -e

# for factor in {9000..9004} 9100; do
# for factor in 1 2 4 8; do
    for dataname in \
        "xenium/rep1/" \
        "xenium/rep2/" \
        "xenium/oos/rep1/" \
        "xenium/oos/rep2/" \
        ; do

        for method in "" "xfuse/"; do
            pref=data/${method}${dataname}
            python evaluate_imputed.py $pref $factor &
            # python evaluate_fit.py $pref &
        done
        wait

        for metric in pearson rmse ssim; do
            python plot_cmp.py $dataname $factor $metric &
        done
        wait
    done
    wait
done
wait
