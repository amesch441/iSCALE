import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import load_image, load_tsv, read_lines, read_string
from utils import load_pickle, save_tsv

scaler = MinMaxScaler()

# Pearson correlation stabilized
def corr_pearson_stablized(x, y, epsilon=1e-6):
    x = standardize(x)
    y = standardize(y)
    x = x - x.mean()
    y = y - y.mean()
    x_std = (x**2).mean()**0.5
    y_std = (y**2).mean()**0.5
    corr = ((x * y).mean() + epsilon) / (x_std * y_std + epsilon)
    return corr

# Pearson correlation
def corr_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan  # Avoid division by zero
    return pearsonr(x, y)[0]

# Spearman correlation
def corr_spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan  # Return NaN if constant input is detected
    return spearmanr(x, y)[0]

# Uncentered correlation
def corr_uncentered(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan  # Return NaN if constant input is detected
    return np.mean(x * y) / (np.mean(x**2)**0.5 * np.mean(y**2)**0.5)

# RMSE
def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))

# Peak signal-to-noise ratio
def psnr(x, y):
    mse = np.mean((x - y)**2)
    if mse == 0:
        return np.inf  # Perfect match, infinite PSNR
    return 10 * np.log10(1 / mse)

# Standardization function
def standardize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)

# Metric calculation function that handles NaNs
def metric_fin(x, y, method='pearson'):
    mask = np.isfinite(x) & np.isfinite(y)  # Handle NaN values
    x, y = x[mask], y[mask]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    method_dict = {
        'pearson': corr_pearson,
        'pearson_stablized': corr_pearson_stablized,
        'spearman': corr_spearman,
        'uncentered': corr_uncentered,
        'psnr': psnr,
        'rmse': rmse
    }

    return method_dict[method](x, y)

def main():
    prefix = sys.argv[1]  # e.g. data/xenium/rep1/
    factor = int(sys.argv[2])  # e.g. 2

    infile_genes = f'{prefix}gene-names.txt'
    gene_names_pred = read_lines(infile_genes)

    # Load ground truth data
    truth = load_pickle(f'{prefix}cnts-truth-agg/radius0008-stride01-square/data.pickle')
    cnts, gene_names = truth['cnts'], truth['gene_names']
    gene_inPred = list(pd.DataFrame(truth['gene_names']).isin(gene_names_pred)[0])
    cnts = cnts[:,:,gene_inPred]
    gene_names = [item for item, flag in zip(gene_names, gene_inPred) if flag]

    cnts = cnts.astype(np.float32)
    print(cnts.shape)
    print(len(gene_names))

    # Load predicted data
    ct_pred = load_pickle(f'{prefix}cnts-super-merged/factor0001.pickle')['x'].astype(np.float32)

    # Reshape cnts and ct_pred to match spatial spots
    num_spots = cnts.shape[0] * cnts.shape[1]
    cnts_flat = cnts.reshape(num_spots, -1)  # (num_spots, num_genes)
    ct_pred_flat = ct_pred.reshape(num_spots, -1)  # Same shape

    eval_list = []

    # Loop through each row (spatial spot)
    for i in range(num_spots):
        if i % 10000 == 0:  # Add a progress indicator for large datasets
            print(f"Processing row {i}/{num_spots}")

        ct_row = cnts_flat[i, :]
        ct_pred_row = ct_pred_flat[i, :]

        # Skip if the entire row is NaN for either cnts or predictions
        if np.isnan(ct_row).all() or np.isnan(ct_pred_row).all():
            eval = {key: np.nan for key in ['pearson', 'rmse', 'pearson_stablized', 'spearman', 'uncentered', 'psnr']}
            eval_list.append(eval)
            continue

        # Fill NaNs with 0 and apply MinMax scaling
        ct_row_filled = np.nan_to_num(ct_row).reshape(-1, 1)
        ct_pred_row_filled = np.nan_to_num(ct_pred_row).reshape(-1, 1)
        ct_row_scaled = scaler.fit_transform(ct_row_filled).flatten()
        ct_pred_row_scaled = scaler.fit_transform(ct_pred_row_filled).flatten()

        # Compute metrics using metric_fin
        eval = {
            'pearson': metric_fin(ct_row_scaled, ct_pred_row_scaled, 'pearson'),
            'rmse': metric_fin(ct_row_scaled, ct_pred_row_scaled, 'rmse'),
            'pearson_stablized': metric_fin(ct_row_scaled, ct_pred_row_scaled, 'pearson_stablized'),
            'spearman': metric_fin(ct_row_scaled, ct_pred_row_scaled, 'spearman'),
            'uncentered': metric_fin(ct_row_scaled, ct_pred_row_scaled, 'uncentered'),
            'psnr': metric_fin(ct_row_scaled, ct_pred_row_scaled, 'psnr')
        }

        eval_list.append(eval)

    # Check if eval_list is empty
    if len(eval_list) == 0:
        print("No valid rows to evaluate.")
        return

    # Create DataFrame from eval_list
    df = pd.DataFrame(eval_list)

    # Assign metric names as column headers
    df.columns = ['Pearson', 'RMSE', 'Pearson_Stabilized', 'Spearman', 'Uncentered', 'PSNR']

    # Save the DataFrame to a .tsv file
    df.to_csv(f'{prefix}cnts-super-eval/superPixel{factor:04d}.tsv', sep='\t', index=False)

    print(f"DataFrame saved successfully with shape: {df.shape}")

if __name__ == '__main__':
    main()

