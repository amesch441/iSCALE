#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=4:00:00
#SBATCH --mem-per-gpu=64G


set -e

prefix="Data/demo/gastricTumor/"  # e.g. Data/demo/


device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis
n_genes=100  # number of most variable genes to impute


#### preprocess histology image
#echo $pixel_size > ${prefix}pixel-size.txt
#python rescale.py ${prefix} --image --mask
#python preprocess.py ${prefix} --mask --image


#### Step 1: Use Alignment_scripts/AlignmentMethod.ipynb jupyter notebook to perform semi-automatic allignment of daughter captures to mother image. Place data in "AllignedToMother" folder

#### Combine data from n number of daughter captures
python stitch_locs_cnts_relativeToM.py ${prefix}/MotherImage/ ${prefix}DaughterCaptures/AllignedToMother/D1/ ${prefix}DaughterCaptures/AllignedToMother/D2/ ${prefix}DaughterCaptures/AllignedToMother/D3/ ${prefix}DaughterCaptures/AllignedToMother/D4/ ${prefix}DaughterCaptures/AllignedToMother/D5/  ## ... Dn

prefix="${prefix}MotherImage/"  

#### auto detect tissue mask
#### If you have a user-defined tissue mask, put it at `${prefix}mask-raw.png` and comment out the line below
#python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}mask-small.png


#### Next, use FilterH&E_scripts/FilterH&E_RBG_LB jupyter notebook to generate the necessary files to filter out super-pixels which are not likely to contain nuclei
## This will be added as a .py file in the pipeline shortly


#### select most highly variable genes to predict
#### If you have a user-defined list of genes, put it at `${prefix}gene-names.txt` and comment out the line below
python select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"

#### visualize spot-level gene expression data
python plot_spots.py ${prefix}
python plot_spots_integrated.py ${prefix} 100

#### extract histology features
python extract_features.py ${prefix} --device=${device}
python plot_embeddings.py ${prefix}embeddings-hist.pickle ${prefix} --mask=${prefix}mask-small.png  
#### # If you want to retun model, you need to delete the existing results:
# rm ${prefix}embeddings-hist-raw.pickle


#### segment image by histology features
#python cluster.py --mask=${prefix}mask-small.png --n-clusters=10 ${prefix}embeddings-hist.pickle ${prefix}clusters-hist/


#### predict super-resolution gene expression
#### rescale coordinates and spot radius
#python rescale.py ${prefix} --locs --radius

#### train gene expression prediction model and predict at super-resolution
python impute_integrated.py ${prefix} --epochs=1000 --device=${device}  --n-states=5  --dist=100 # train model from scratch
python refine_gene.py ${prefix} "conserve_index_linearBoundary_m1.pickle"  #input: prefix conserve_image_index

##### # If you want to retrain model, you need to delete the existing model:
# rm -r ${prefix}states

#### visualize imputed gene expression
python plot_imputed_iSCALE.py ${prefix}

#### merge imputed
python merge_imputed.py ${prefix} 1


#### segment image by gene features
python cluster_iSCALE.py --filter-size=8 --min-cluster-size=20 --n-clusters=20 --mask=${prefix}mask-small-refined.png --refinedImage=${prefix}filterRGB/conserve_index_linearBoundary1_image.pickle ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/


#### Evaluate performance
#python evaluate_fit.py ${prefix} ## training rmse and pearson
#python evaluate_imputed_iSCALE_plus.py ${prefix} 1
#python evaluate_imputed_iSCALE_bySuperPixel.py ${prefix} 1










