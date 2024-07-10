#!/bin/bash
set -e

prefix="data/dataSet1_mother/"  # e.g. data/dataSet1_mother/

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis
n_genes=1000  # number of most variable genes to impute

# download checkpoints available at: https://upenn.box.com/s/kmtmsa3lv8u92wuaw23uui40mgqiyxrc
# place checkpoints in the folder "checkpoints" 

# preprocess large-sized histology image
echo $pixel_size > ${prefix}pixel-size.txt
python rescale.py ${prefix} --image
python preprocess.py ${prefix} --image

# extract histology features
python extract_features.py ${prefix} --device=${device}
# # If you want to retun model, you need to delete the existing results:
# rm ${prefix}embeddings-hist-raw.pickle

# auto detect tissue mask
python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}mask-small.png
# If you have a user-defined tissue mask, put it at `${prefix}mask-raw.png` and comment out the line below

# # segment large-sized image by histology features
python cluster.py --mask=${prefix}mask-small.png --n-clusters=10 ${prefix}embeddings-hist.pickle ${prefix}clusters-hist/

## Concatenate k daughter captures with locations aligned to large-sized mother image
python stitch_sections.py ${prefix} data/dataSet1_daughters/d1/ data/dataSet1_daughters/d2/ data/dataSet1_daughters/d3/

# select most highly variable genes to predict
# If you have a user-defined list of genes, put it at `${prefix}gene-names.txt` and comment out the line below
python select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"

# predict super-resolution gene expression
# rescale coordinates and spot radius
python rescale.py ${prefix} --locs --radius

# visualize spot-level gene expression data
python plot_spots.py ${prefix}

# train gene expression prediction model and predict at super-resolution
python impute.py ${prefix} --epochs=400 --device=${device}  # train model from scratch
# # If you want to retrain model, you need to delete the existing model:
# rm -r ${prefix}states

# visualize imputed gene expression
python plot_imputed.py ${prefix}

# visualize imputed gene expression by cell-type (if marker-gene list is provided) 
python plot_imputed_byCellType.py ${prefix}

# segment image by gene features
python cluster.py --filter-size=8 --min-cluster-size=20 --n-clusters=10 --mask=${prefix}mask-small.png ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/
# # segment image without tissue mask
# python cluster.py --filter-size=8 --min-cluster-size=20 ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unmasked/
# # segment image without spatial smoothing
# python cluster.py --mask=${prefix}mask-small.png ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unsmoothed/
# python cluster.py ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unsmoothed/unmasked/

# differential analysis by clusters
python aggregate_imputed.py ${prefix}
python reorganize_imputed.py ${prefix}
python differential.py ${prefix}

# compute training performance, final RMSE for each gene in the model
python evaluate_fit.py ${prefix}

# cell type inference
python pixannot_percentileScore.py ${prefix} data/markers/celltype.tsv ${prefix}markers/celltype/ #store cell-type marker list in celltype.tsv



