# PCO: Protein Crystallizability Prediction

This repository contains code, datasets, and models for predicting protein crystallizability from its amino acid sequence.

---

## Installation

### 1. Install NetSurfP-3.0
Download the standalone package of [NetSurfP-3.0](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/) and unzip it directly inside this repository.

### 2. Set up the conda environment

```bash
conda env create -f environment.yml
conda activate pco
```
### 3. Run predictions on your fasta file
```bash
python main.py -i input_file -o output_directory
```
---

## If you wish to train the classifier yourself, you can do so:

1. Download the .csv files from https://zenodo.org/records/17074480 and put them in `./data` directory
2. Run the subsequent cells in the `train.ipynb` notebook
