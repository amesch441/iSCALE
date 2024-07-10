#!/bin/bash
set -e

pref=$1  # e.g. data/multipat/all/
extra=$2  # e.g. data/markers/lymphoid-aggregate/genes.txt
n_genes=1000

mkdir -p "${pref}gene-lists"
cat "${pref}cnts.tsv" |
    head -n1 | cut -f2- | tr -s '\t' '\n' |
cat > "${pref}gene-lists/all.txt"
cat $extra |
    grep -x -f "${pref}gene-lists/all.txt" |
cat > "${pref}gene-lists/la.txt"
python select_genes.py "${pref}cnts.tsv" ${n_genes} "${prefix}gene-lists/top${n_genes}.txt"
cat "${prefix}gene-lists/top${n_genes}.txt" |
    grep -v -x -f "${pref}gene-lists/la.txt" |
    cat "${pref}gene-lists/la.txt" - |
cat > "${pref}gene-lists/la-top${n_genes}.txt"
echo "${pref}gene-lists/la-top${n_genes}.txt"
