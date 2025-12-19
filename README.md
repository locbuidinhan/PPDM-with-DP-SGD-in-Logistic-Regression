## 1. Overview
This project implements *Privacy-Preserving Data Mining (PPDM)* using
*Differential Privacy*, specifically *DP-SGD*, applied to *Logistic Regression*.

    ### Implemented Models
    * *Non-private Logistic Regression* (baseline)
    * *Differentially Private Logistic Regression* using *DP-SGD*

    ### Datasets Used
    * *EMNIST* (Digits: Even vs Odd)
    * *Breast Cancer Dataset* (medical data)
---

## 2. Project Structure
.
├── models/
│   ├── logistic_standard.py
│   └── logistic_dp_sgd.py
├── utils/
│   ├── helpers.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── preprocessing_emnist.py
│   └── preprocessing.py
├── data/
│   ├── EMNIST/
│   └── breast_cancer.csv
├── figures/
├── train_standard.py
├── train_dp.py
├── train_standard_breast.py
├── train_dp_breast.py
├── noise.py
├── noise_breast.py
├── config.py
├── config_breast.py
├── README.md
└── .gitignore


## 3. Requirements
    - Python version: >= 3.9
    - Install Dependencies: pip install numpy pandas scikit-learn matplotlib torch torchvision
    - Install Dataset as below (Section 4) instructions
---

## 4. Datasets
*EMNIST Dataset*
    - Downloaded automatically using: pip install torch torchvision
    - *Not included* in this repository
    - Automatically downloaded on first run
    - File location:
        data/EMNIST
    - Task:
        Binary classification: *Even vs Odd digits


*Breast Cancer Dataset*
    - Source: [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset]
    - File location:
        data/breast_cancer.csv 
    - Task:
        Binary classification: *Classify tumors into Malignant (cancerous) or Benign(non cancerous)
---

## 5. Running Experiments
    ```bash
    python train_standard.py            # EMNIST – Non-private Logistic Regression
    python train_dp.py                  # EMNIST – DP-SGD Logistic Regression
    python noise.py                     # EMNIST – Privacy–Utility Trade-off
    python train_standard_breast.py     # Breast Cancer – Non-private Logistic Regression
    python train_dp_breast.py           # Breast Cancer – DP-SGD Logistic Regression
    python noise_breast.py              # Breast Cancer – Privacy–Utility Trade-off
    ```

*Output*
    `figures/non_dp_learning_curve.png`
    `figures/dp_learning_curve.png`
    `figures/privacy_utility_tradeoff.png`
    `figures/non_dp_breast_learning_curve.png`
    `figures/dp_breast_learning_curve.png`
    `figures/privacy_utility_tradeoff_breast.png`
---


*Privacy-related hyperparameters are explicitly defined in:*
   `config.py`
   `config_breast.py`
---






