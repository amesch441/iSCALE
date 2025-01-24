import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
import seaborn as sns
from visual import plot_labels, plot_label_masks

from utils import load_pickle, read_lines, save_tsv


import os
import pickle
import numpy as np

def refine_labels(pickle_file, output_folder, conserve_index):
    """
    Update the labels by setting labels at False locations in conserve_index to NaN.
    
    Parameters:
    - pickle_file: str, path to the labels.pickle file.
    - output_folder: str, path to the folder where the refined labels.pickle will be saved.
    - conserve_index: list of bool, mask indicating which locations are valid (True) and which should be set to NaN (False).
    """
    
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the labels.pickle file
    with open(pickle_file, 'rb') as f:
        labels = pickle.load(f)
    
    # Update the labels: set values to NaN where conserve_index is False

    labels = np.array(labels)  # Ensure labels is a NumPy array and can handle NaN values
    labels[~np.array(conserve_index)] = -1  # Set False locations to NaN

    # Save the updated labels to the new folder
    output_path = os.path.join(output_folder, os.path.basename(pickle_file))
    with open(output_path, 'wb') as f:
        pickle.dump(labels, f)
        
    plot_labels(labels, output_folder+'/labels.png', white_background=True)
    plot_label_masks(labels, output_folder+'/masks/')

    print(f"Updated labels saved to {output_path}.")



def main(): 
    prefix = sys.argv[1]  
    index_image = sys.argv[2]  
    clusterFolder = sys.argv[3] 
    pickle_file = f'{prefix}{clusterFolder}/labels.pickle'
    output_folder = f'{prefix}clusters-gene-refined'  # Folder where refined labels will be saved

    with open(f'{prefix}filterRGB/{index_image}', 'rb') as f:
        conserve_index_image = pickle.load(f)

    output_folder = f'{prefix}{clusterFolder}-refined'  
    refine_labels(pickle_file, output_folder, conserve_index_image)



if __name__ == '__main__':
    main()



