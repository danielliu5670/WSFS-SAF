# Supervised Feature Selection on Sparse Autoencoder Representations for Financial Company Similarity

## Authors: Daniel Liu and Nina Mesyngier

This is the GitHub repository associated with our paper, "Supervised Feature Selection on Sparse Autoencoder Representations for Financial Company Similarity." Inside are all of the files necessary to recreate the tables and figures referenced in the manuscript. The `main.ipynb` file, in particular, contains all of the command-line arguments you will need to run, as well as several valuable annotations.

### Code Files

1. `collect_features.py`: This collects all of the necessary information from Molinari et al.'s HuggingFace repository.
2. `evaluate.py`: This performs the comparison between WSFS-SAF, Molinari et al., and the SIC code baseline on the correlation-at-k evaluation metric.
3. `evaluate_clustering.py`: This performs the comparison between WSFS-SAF, Molinari et al., and the SIC code baseline on the weighted MC(Gk) metric.
4. `extract_features.py`: This iterates through all provided top-K and alpha values to determine the ideal configuration under our approach.
5. `main.ipynb`: As previously stated, this is the Colab Notebook file that is the recommended "entry point," since it contains annotations and all the necessary file calls.
6. `ablation/ablation_pca_supervised.py`: This performs the second ablation test from the manuscript.
7. `ablation/ablation_robustness.py`: This performs the third and fourth ablation tests from the manuscript.
8. `ablation/ablation_unsupervised_sae.py`: This performs the first ablation test from the manuscript.
9. `graphs/norm_plot.py`: This creates the hexbin plot referenced in the manuscript.

### Contact

Daniel Liu
Email: dxl5670@psu.edu OR danielliu.datascience@gmail.com

Nina Mesyngier
Email: nvm5600@psu.edu
