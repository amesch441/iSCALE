import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils import load_pickle, save_tsv, load_mask,read_lines
from scipy.spatial.distance import jaccard
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler


# Function to standardize a vector between 0 and 1
def standardize_vector(vector):
    return (vector - np.nanmin(vector)) / (np.nanmax(vector) - np.nanmin(vector))

# Function to binarize data using the given threshold, retaining NaN values from the mask
def binarize_data_with_mask(vector, threshold, nan_mask):
    # Apply NaN mask from the true data and then binarize using the threshold
    binarized = np.where(nan_mask, np.nan, np.where(vector > threshold, 1, 0))
    return binarized

# Function to compute Dice coefficient
def compute_dice_coefficient(true_binarized, pred_binarized):
    intersection = np.nansum((true_binarized == 1) & (pred_binarized == 1))
    total_sum = np.nansum(true_binarized == 1) + np.nansum(pred_binarized == 1)
    dice_coeff = (2 * intersection) / total_sum if total_sum != 0 else np.nan
    return dice_coeff

# Function to compute Jaccard index
def compute_jaccard_index(true_binarized, pred_binarized):
    intersection = np.nansum((true_binarized == 1) & (pred_binarized == 1))
    union = np.nansum((true_binarized == 1) | (pred_binarized == 1))
    jaccard_index = intersection / union if union != 0 else np.nan
    return jaccard_index

# Function to perform Chi-square test
def compute_chi_square_test(true_binarized, pred_binarized):
    contingency_table = np.zeros((2, 2))
    
    # Fill contingency table: [ [True 0/0, True 1/0], [Pred 0/1, Pred 1/1] ]
    contingency_table[0, 0] = np.nansum((true_binarized == 0) & (pred_binarized == 0))
    contingency_table[0, 1] = np.nansum((true_binarized == 0) & (pred_binarized == 1))
    contingency_table[1, 0] = np.nansum((true_binarized == 1) & (pred_binarized == 0))
    contingency_table[1, 1] = np.nansum((true_binarized == 1) & (pred_binarized == 1))
    
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return chi2, p

# Function to save binarized expression plots using imshow for faster plotting
def save_expression_plot_imshow(binarized_data, gene_name, output_dir):
    # Custom color map: lighter gray for low expression, blue for high expression, and white for NaN
    cmap = ListedColormap(['lightgray', 'blue'])
    
    # Create a figure
    plt.figure(figsize=(19, 8))
    
    # Use imshow for faster plotting; aspect='auto' preserves the aspect ratio
    plt.imshow(binarized_data, cmap=cmap, origin='upper') # , interpolation='none'
    
    # Plot NaN values as white by adding a white patch to the colormap
    nan_mask = np.isnan(binarized_data)
    plt.imshow(nan_mask, cmap=ListedColormap(['white']), interpolation='none', alpha=0.5)
    
    plt.title(f'Binarized Expression for Gene: {gene_name}')
    plt.axis('off')  # Optional, remove axes for a cleaner image
    
    # Save the plot with reduced dpi for faster saving (you can adjust dpi as needed)
    plt.savefig(f'{output_dir}/{gene_name}_binarized_expression.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    import sys

    prefix = sys.argv[1]  # e.g. data/xenium/rep1/
    #thresh = int(sys.argv[2])  # threshold for binarization
    true_output_dir = sys.argv[2]  # Output directory for true plots
    pred_output_dir = sys.argv[3]  # Output directory for predicted plots
    metrics_output_file = sys.argv[4]  # Output file to save the metrics (e.g., CSV or TSV)

    infile_genes = f'{prefix}gene-names.txt'
    #gene_names_pred = read_lines(infile_genes)

    # Load predicted expression data (shape: [n_spots_y, n_spots_x, n_genes])
    ct_pred = load_pickle(f'{prefix}cnts-super-merged/factor0001.pickle')['x'].astype(np.float32)
    gene_names_pred = list(load_pickle(f'{prefix}cnts-super-merged/factor0001.pickle')['gene_names'])

    # Create directories for true and predicted output plots if they don't exist
    os.makedirs(true_output_dir, exist_ok=True)
    os.makedirs(pred_output_dir, exist_ok=True)

    scaler = MinMaxScaler()

    # Load the ground truth expression data (shape: [n_spots_y, n_spots_x, n_genes])
    truth = load_pickle(
            prefix+'cnts-truth-agg/radius0008-stride01-square/data.pickle')
    cnts, gene_names = truth['cnts'], truth['gene_names']
    gene_inPred = list(pd.DataFrame(truth['gene_names']).isin(gene_names_pred)[0])
    cnts = cnts[:,:,gene_inPred]
    gene_names = [item for item, flag in zip(gene_names, gene_inPred) if flag]

    cnts = cnts.astype(np.float32)


    # Reorder genes and cnts to match `gene_names_pred`
    indices = [gene_names.index(g) for g in gene_names_pred if g in gene_names]  # Map `gene_names_pred` order
    cnts = cnts[:, :, indices]  # Reorder cnts
    gene_names = [gene_names[i] for i in indices]  # Reorder gene names

  

    print(cnts.shape)
    print(len(gene_names))


    # Load mask
    tissue_mask = load_mask(prefix+'mask-small.png')
    tissue_mask_flat = tissue_mask.flatten()

    # Dynamically determine the number of rows and columns
    nrows, ncols = cnts.shape[0], cnts.shape[1]

    truth_final = cnts.reshape((nrows*ncols, cnts.shape[2]))
    pred_final = ct_pred.reshape((nrows*ncols, ct_pred.shape[2]))

    # Replace observations with NaN where the mask is 0
    truth_final[tissue_mask_flat == 0] = np.nan
    pred_final[tissue_mask_flat == 0] = np.nan

    true_exp_std = scaler.fit_transform(truth_final)
    pred_ex_std = scaler.fit_transform(pred_final)


    ## Dictionary to store the metrics for each gene
    metrics = {
        'Gene': [],
        'Dice_Coefficient': [],
        'Jaccard_Index': [],
        'Chi_Square': [],
        'p_value': []
    }

    ## Loop over each gene and process the true and predicted data
    for i, gene_name in enumerate(gene_names):

        ## Get the expression for the current gene and reshape it to [nrows, ncols]
        true_expression_std = true_exp_std[:,i].reshape(nrows, ncols)
        pred_expression_std = pred_ex_std[:,i].reshape(nrows, ncols)
        #true_expression_std = true_exp_std[..., i].reshape(nrows, ncols)
        #pred_expression_std = pred_ex_std[..., i].reshape(nrows, ncols)

        ## Standardize the true and predicted expression values between 0 and 1
        #true_expression_std = standardize_vector(true_expression)
        #pred_expression_std = standardize_vector(pred_expression)

        # Create a mask to identify where expression has NaN values (was masked)
        nan_mask = np.isnan(true_expression_std)

        ## Compute separate jth percentile thresholds for the true and predicted standardized expressions
        threshold_true = 0 #np.nanpercentile(true_expression_std, thresh)
        threshold_pred = 0.025 #np.nanpercentile(pred_expression_std, thresh)

        print('Threshold True Dist:')
        print(threshold_true)
        print('Threshold Pred Dist:')
        print(threshold_pred)



        # Binarize the true and predicted expression using their respective thresholds and apply the true NaN mask
        true_binarized = binarize_data_with_mask(true_expression_std, threshold_true, nan_mask)  # Do not flatten for imshow
        pred_binarized = binarize_data_with_mask(pred_expression_std, threshold_pred, nan_mask)  # Same NaN mask used for predicted data

        # Save binarized plots for both true and predicted data in separate folders using imshow
        save_expression_plot_imshow(true_binarized, gene_name, true_output_dir)
        save_expression_plot_imshow(pred_binarized, gene_name, pred_output_dir)

        # Compute Dice, Jaccard, and Chi-square test
        dice_coeff = compute_dice_coefficient(true_binarized, pred_binarized)
        jaccard_index = compute_jaccard_index(true_binarized, pred_binarized)
        chi2, p_value = compute_chi_square_test(true_binarized, pred_binarized)

        # Store the results for this gene
        metrics['Gene'].append(gene_name)
        metrics['Dice_Coefficient'].append(dice_coeff)
        metrics['Jaccard_Index'].append(jaccard_index)
        metrics['Chi_Square'].append(chi2)
        metrics['p_value'].append(p_value)

        # Progress tracking for large datasets
        if i % 100 == 0:
            print(f"Processed {i}/{len(gene_names)} genes...")

    # Convert metrics dictionary to DataFrame and save as CSV or TSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_output_file, index=True)

    print(f"Metrics saved to {metrics_output_file}")
    print("Binarization and plotting completed for all genes!")

if __name__ == '__main__':
    main()

