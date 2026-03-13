
# 4f-electron-lanthanide-sac

Code and dataset for the study of **4f-electron regulated dual-periodic bonding affinity and strong linear relationships on lanthanide single-atom catalysts**.

This repository contains the dataset and machine-learning workflow used in the paper to analyze bonding affinity relationships and perform missing-value prediction based on leave-one-out (LOO) model evaluation.

---
# Repository Structure
4f-electron-lanthanide-sac
│
├── code
│ └── Code.py
│
├── data
│ └── dataset.csv
│
├── results
│
└── README.md
- **code/**  
  Contains the machine learning workflow used in the study.

- **data/**  
  Contains the dataset used for model training and analysis.

- **results/**  
  Optional folder for storing model evaluation results and processed datasets.

---

# Dataset

The dataset includes physicochemical descriptors and target variables associated with lanthanide single-atom catalysts.

Target variables include:

h, c, n, f, cl, s, co2, cho, co, cooh,
n2, n2h, no, noh, oh, ooh, o2, h2o

Missing values in these targets are predicted using a machine learning workflow with leave-one-out cross-validation.

---

# Machine Learning Workflow

The script:

Code.py

performs the following steps:

1. Reads the dataset from a CSV file
2. Separates input features and target variables
3. Evaluates multiple regression models using leave-one-out cross-validation (LOO)
4. Selects the model with the best predictive performance (highest R²)
5. Uses the selected model to predict missing values
6. Outputs an updated dataset and model evaluation results

Candidate models include:

- Ridge regression  
- Random Forest  
- XGBoost  
- Support Vector Regression (SVR)  
- Multi-layer Perceptron (MLP)  
- Histogram Gradient Boosting

---

# Usage

Run the script from the command line:
python loo_model_selection_imputation.py --input-file path/to/input.csv --output-file path/to/output.csv


Example:


python loo_model_selection_imputation.py --input-file data/dataset.csv --output-file results/imputed_dataset.csv


The script will also generate model evaluation summaries.

---

# Requirements

Recommended Python version:


Python >= 3.9


Main dependencies:


pandas
numpy
scikit-learn
xgboost


You can install them using:


pip install pandas numpy scikit-learn xgboost


---

# Citation

If you use this code or dataset in your research, please cite the associated paper:


4f-Electron-Regulated Dual-Periodic Bonding Affinity and Strong Linear Relationship on Lanthanide Single Atom Catalysts


---

# License

This project is released under the MIT License.
