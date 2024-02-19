# 📣 Drug-Drug-Synergy model (DDoS) 💊

This repository contains the code for the DDoS model implementation with PyTorch. 


## Installation

* `git clone` the repo and `cd` into it.
* Optionally, create a python `virtualenv` and install inside the `pip` packages from the `requirements.txt` file. Note that `torch` and `CUDA` versions depend on the GPU/CUDA which you use locally. 

## Data Download

Download `Gene Expression` data (link: [1](https://www.cancerrxgene.org/gdsc1000//GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip)) and `Drug Combination` data (link: [2](https://drugcomb.fimm.fi/jing/summary_v_1_5.csv)). Place them into the `data/preprocessing` directory.


## Running 🏃

1. use `notebooks/DDoS_Preprocessing.ipynb` and `notebooks/DDoS_Dataset_Generation.ipynb` to generate DataTensors from the drug combination features (chemical structures and gene expression).
2. use `notebooks/DDoS-Hyperparam.ipynb` to find the best performing model hyperparameters.
3. use `notebooks/DDoS_Train_Eval.ipynb` to train / test on the best analyze the results.
