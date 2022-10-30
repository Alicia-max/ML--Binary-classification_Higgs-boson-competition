# Machine Learning Project 1

This repository contains the code and the report for the first project of CS-433 Machine Learning 2022 course at EPFL. The purpose is to recreate the discovery of the Higgs boson by implementing binary classification using machine learning methods. The data is obtained from the online competition platform (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

This README consists of general information about the used methods and the structure of the code. A more detailed documentation is found inside the functions.

## Team Members

This project belongs to the team of `ASA_GANG` with members:

- Alicia Milloz: [@Alicia-max](https://github.com/Alicia-max)
- Sevda Öğüt: [@ogutsevda](https://github.com/ogutsevda)
- Alexandre de Skowronski: [@alexdesko](https://github.com/alexdesko)

## Prerequisites

The provided Python code uses the libraries of numpy (as np) for computation and seaborn (as sns) and matplotlib.pyplot (as plt) for visualization.

The folders should be in the below structure:

    .
    ├── data
        ├── train.csv
        └── test.csv
    ├── project
        ├── cross_val.py
        ├── helpers.py
        ├── implementations.py
        ├── plot.py
        ├── preprocessing.py
        ├── run_multigroups.py
        ├── visu.ipynb
        └── visu_jet.ipynb
    ├── run.py
    └── README.md

In order to reproduce the code, one needs to run the `run.py` file. The prediction output will be written to `submission_test.csv` file with {-1/1} labels.


## Implementation

### Helper functions

`helpers.py`: Loads .csv train and test data and creates .csv submission file.

### Data Preprocessing 

`preprocessing.py`: Preprocesses train and test data by adding offset, standardization, polynomial expansion, adding cross-terms, adding cos() and sin() for angle related features, dividing the dataset into subgroups, removing uninformative features, filling missing values and outlier removal.

### Training

`implementations.py`: Implements 6 different machine learning methods taking into account their corresponding loss functions. Uses RMSE loss for `mean_squared_error_GD`, `mean_squared_error_SGD`, `least_squares`, and `ridge regression`. Uses log-loss for `logistic_regression` and `reg_logistic_regression`.

### Model Selection

`cross_val.py`: Implements K-fold cross-validation to make hyperparameter tuning for the implemented methods such as, but not limited to, learning rate in GD/SGD and degree of polynomial expansion.

`plot.py`: Visualizes the results of the cross-validation code.

### Notebook

`visu.ipynb`: Explores the data and the distribution of the features. Implements baseline runs, tunes parameters with cross-validation and outputs accuracy plots.

`visu_jet.ipynb`: Divides the data according to the categoraical `PRI_jet_num` variable. Implements the methods, tunes parameters with cross-validation and outputs accuracy plots.