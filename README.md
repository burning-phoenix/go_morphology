# Morphogenetic Spaces: Hierarchical Concept Discovery in Leela Zero

This project explores the internal representations of the **Leela Zero** Go engine using **Matryoshka Sparse Autoencoders (MSAE)**. By analyzing the activation space of the neural network, we aim to uncover hierarchical concepts and study the topology of the "game manifold" as stratagems evolve from simple to complex.

## Project Overview

The core hypothesis is that game engines like Leela Zero develop a structured, hierarchical representation of the game state. We use Sparse Autoencoders (SAEs) to disentangle these representations into interpretable features and apply **Matryoshka** training to enforce a nested structure, allowing us to analyze features at varying levels of granularity.

### Key Components
*   **Activation Extraction:** Capturing intermediate layer outputs from Leela Zero.
*   **MSAE Training:** Training Matryoshka Sparse Autoencoders to learn hierarchical features.
*   **Manifold Analysis:** Using PCA, UMAP, and TDA (Topological Data Analysis) to study the shape of the activation space.
*   **Bifurcation Detection:** Identifying critical moments in a game where the landscape of possibilities shifts dramatically.

## Installation

### Prerequisites
*   Python 3.10 or higher
*   Conda (recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd chaos_github
    ```

2.  **Create the Conda environment:**
    This project uses `conda` to manage dependencies. Create the environment using the provided `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    conda activate morpho
    ```

### Note on External Repositories
Some external repositories (which contained submodules like `SAELens` or `leela-zero-pytorch`) has been removed to keep this repository lightweight. However, these libraries are excellent references and tools if you wish to extend the project or modify the underlying architecture:

*   **[SAELens](https://github.com/decoderesearch/SAELens):** A comprehensive library for training and analyzing Sparse Autoencoders, particularly valuable for mechanistic interpretability research.
*   **[leela-zero-pytorch](https://github.com/yukw777/leela-zero-pytorch):** A pure PyTorch implementation of Leela Zero, useful if you need to retrain the game engine or modify its architecture directly.

## Project Structure

```
.
├── 01_setup_and_extraction.ipynb   # Data download, validation, and activation extraction
├── 02_train_msae.ipynb             # Training Matryoshka Sparse Autoencoders
├── 03_hierarchy_analysis.ipynb     # Analyzing feature hierarchy (nestedness, stability)
├── 04_pca_baseline.ipynb           # PCA baseline for dimensionality reduction analysis
├── 05_1_continuous_manifold.ipynb  # Manifold analysis using continuous methods
├── 05_2_attractor_basins_discrete.ipynb # Discrete attractor basin analysis
├── 06_bifurcation_detection.ipynb  # Detecting qualitative changes in game dynamics
├── 07_morphological_topology.ipynb # TDA and topological analysis
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Pip requirements (alternative)
└── src/                            # Source code package
    ├── analysis/                   # Analysis modules (hierarchy, topology, etc.)
    ├── data/                       # Data loading and processing
    ├── models/                     # PyTorch models (MSAE, LeelaZero wrappers)
    ├── training/                   # Training loops and utilities
    ├── utils/                      # Helper functions
    ├── unit_tests/                 # Unit tests
    └── visualization/              # Plotting and visualization tools
```

## Usage Workflow

The project is structured as a series of numbered notebooks that should be executed in order:

1.  **Data Extraction (`01_setup_and_extraction.ipynb`):**
    *   Downloads sample Leela Zero game data (or generates it).
    *   Extracts activations from specific residual blocks (e.g., blocks 5, 20, 35).
    *   Saves activations to disk for training.

2.  **Train MSAE (`02_train_msae.ipynb`):**
    *   Loads the extracted activations.
    *   Trains Matryoshka Sparse Autoencoders with different sparsity penalties (k-levels).
    *   Saves the trained autoencoder models.

3.  **Hierarchy Analysis (`03_hierarchy_analysis.ipynb`):**
    *   Investigates the relationship between features at different granularities.
    *   Computes metrics like nestedness and reconstruction R².

4.  **Manifold & Bifurcation Analysis (`05_*`, `06_*`):**
    *   Explores the geometry of the game state space.
    *   Identifies bifurcations (critical decision points) and attractor basins.

5.  **Topological Analysis (`07_morphological_topology.ipynb`):**
    *   Applies Persistent Homology to study the shape of feature clouds.

## Contributing

Contributions are welcome! If you find a bug or have an idea for a new analysis method, please open an issue or submit a pull request. I am still looking into a better modelling strategy to better understand the shape of the game strategy space, and I am conlficted between using a discrete vs a continuous approach, but will probably go with a hybrid approach.
