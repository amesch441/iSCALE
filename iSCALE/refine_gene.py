import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
import seaborn as sns
from visual import plot_labels, plot_label_masks

from utils import load_pickle, read_lines, save_tsv, load_image


import os
import pickle
import numpy as np

import os
import pickle
import numpy as np

def update_expression(gene_folder, output_folder, mask):
    """
    This function reads each gene's pickle file, applies a filter mask to update the expression,
    and saves the modified expression to a new folder.

    Parameters:
    - gene_folder (str): Path to the folder containing the original gene .pickle files.
    - output_folder (str): Path to the folder where the refined .pickle files will be saved.
    - mask (np.array): Boolean mask where True indicates locations to filter out and set expression to 0.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all .pickle files in the gene_folder
    for file_name in os.listdir(gene_folder):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(gene_folder, file_name)

            # Load the gene expression data
            with open(file_path, 'rb') as f:
                expression = pickle.load(f)

            mask = np.array(mask).reshape(expression.shape[0], expression.shape[1])

            # Check if the dimensions of the mask match the expression data
            #print(expression.shape)
            #print(mask.shape)
            if expression.shape != mask.shape:
                raise ValueError(f"Mask dimensions {mask.shape} do not match expression dimensions {expression.shape}")

            # Update the expression values: set to 0 where mask is True, but leave NaNs untouched
            expression = np.where(~mask & ~np.isnan(expression), 0, expression)

            # Save the updated expression to the output folder
            output_path = os.path.join(output_folder, file_name)
            with open(output_path, 'wb') as f:
                pickle.dump(expression, f)

            print(f"Updated expression for {file_name} saved to {output_path}")


def main(): 
    prefix = sys.argv[1]  
    index_image = sys.argv[2]  
    gene_folder = f'{prefix}cnts-super'
    output_folder = f'{prefix}cnts-super-refined'  # Folder where refined labels will be saved


    with open(f'{prefix}filterRGB/{index_image}', 'rb') as f:
        conserve_index_image = pickle.load(f)



    update_expression(gene_folder, output_folder, conserve_index_image)



if __name__ == '__main__':
    main()



