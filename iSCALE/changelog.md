# Changelog

## [0.5.1-beta] - 2023-02-24

### Changed

- Refactored pipeline running in demo
- Replaced turbo with tab30 for visualizing >20 labels

### Fixed

- Updated files that were missed in the previous merging

## [0.5.0-beta] - 2023-02-23

- Major
    - Rescale histology images to make the pixel size consistent
    - Apply tissue mask only to clustering and visualization
        - Remove mask shrinking
    - Replace square with circular spot masks
    - Standardize predicted gene expression with the spot mask size
    - New feature: superpixel-level classification (e.g. cell type inference)
    - New feature: user-defined feature (e.g. TLS) prediction

- Minor
    - Improve computation efficiency of weakly supervised learning
    - Bug fix on clustering with >10 clusters

## [0.4.1] - 2023-01-06

- Minor
    - Do not use parallele training by default, as it is unstable on some machines

## [0.4.0] - 2023-01-02

- Major
    - Improve predictive model
        - Use high-, low-, and rgb-level features
        - Remove sparsity in outcomes
        - Replace mean with median as the point estimate across states
        - Parallelize multi-state model training
        - Use minibatches more thoroughly to reduce memory usage
    - Improve clustering
        - Use high-level embeddings only for clustering
        - Smoothen gene embeddings before clustering
    - Load spot radius from data

- Minor
    - Use OpenCV for smoothing

## [0.3.0] - 2022-12-07

- Major
    - Select the most variable genes for imputation
    - Stablize imputation model by fitting multiple times with different random states
    - Simplify spots masks into rectangular for imputation
    - Add differential analysis
- Minor
    - Plot spot-level gene expression data
    - Reduce memory usage by decreasing batch sizes
    - Bug fix on histology-based clustering
    - Plot cluster masks

## [0.2.1] - 2022-11-09

- Minor
    - Improve boundary effects of low-level histology features
    - Improve CPU and GPU memory management
    - Turn off relabeling small connected components by default
    - Increase default number of epochs for imputation
    - Bug fix on setting device
    - Improve usage of clustering
    - Improve path names

## [0.2.0] - 2022-11-04

- Major
    - Use neural network for imputation and prediction
    - Use shifting windows to prevent patch effects in low-level features
    - Improve feature extraction memory efficiency
    - Add scripts for downloading demo data and model checkpoints
    - Reorganize directories
- Minor
    - Improve tissue mask computation
    - Use low-level histology features for sub-clustering
    - Bug fix on small connected component relabeling

## [0.1.3] - 2022-10-19

- Minor
    - Bug fix on out-of-bound adjacent indices for small component detection

## [0.1.2] - 2022-10-14

- Minor
    - Bug fix on setting device for imputation
    - Bug fix on coordinate rescaling

## [0.1.1] - 2022-10-12

- Minor
    - Added option of using CPUs for super-resolution imputation

## [0.1.0] - 2022-10-11

- Initial release
