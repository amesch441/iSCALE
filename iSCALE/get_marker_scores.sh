#!/bin/bash
set -e

prefix_data=$1  # e.g. data/la/her2st/G123/
prefix_marker="data/markers/lymphoid/"
struct_name_file="${prefix_marker}structures.txt"
taxonomy_file="${prefix_marker}structures.yml"

while read struct; do
    gene_name_file=${prefix_marker}${struct}/gene-names.txt
    python marker_score.py ${prefix_data} $gene_name_file ${prefix_data}markers/phenotype/raw/${struct}
done < $struct_name_file

cp $taxonomy_file ${prefix_data}markers/phenotype/structures.yml
