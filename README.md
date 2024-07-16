# ML-VEG

This repository contains the data and the scripts used for the manuscript "*Machine Learning Approach To Vertical Energy Gap in Redox Processes*". 

## Building conda environment

We suggest running the following command to create a conda environment called `VEGPred`: 
> conda env create -f environment.yml

## Repo details

### Features: 

* The feature list calculated using Hartree-Fock (HF) and semi-empirical (EMP1) methods for each system are in their respective folders. For example, 
	1. The HF features for QM cutoff 0.0 Å for benzene are in the `benzene/feature_list_0.0.dat` folder. 
	2. The EMP1 features for QM cutoff 7.5 Å for lumiflavin are in the  `lumiflavin/EMP1_feature_list_7.5.dat` folder.

### Workflow:

* The workflow is organized as follows:
	1. Add ML models with parameter list in the `models_hyperparam_opt.py` file.
	2. Run `models_hyperparam_opt.py > model_list.dat` to create list of best models for each system. 
	3. Select one ML model from `model_list.dat` and add it to the system dictionary in `opt_model_test.py` .
	4. Run `save_models_and_plot.py` to produce heatmaps, parity plots, and learning curves as presented in the manuscript. 