Lỗi giao diện của README **không phải do GitHub**, mà do **Markdown bị thụt đầu dòng sai** và **thiếu code block cho cây thư mục**.
Tôi đưa bạn **PHIÊN BẢN ĐÃ SỬA CHUẨN MARKDOWN**.
👉 **Copy toàn bộ, ghi đè README.md hiện tại**.

---

````markdown
## 1. Overview

This project implements *Privacy-Preserving Data Mining (PPDM)* using  
*Differential Privacy*, specifically *DP-SGD*, applied to *Logistic Regression*.

### Implemented Models
- *Non-private Logistic Regression* (baseline)
- *Differentially Private Logistic Regression* using *DP-SGD*

### Datasets Used
- *EMNIST* (Digits: Even vs Odd)
- *Breast Cancer Dataset* (medical data)

---

## 2. Project Structure

```text
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
````

---

## 3. Requirements

* Python version: **>= 3.9**
* Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib torch torchvision
```

---

## 4. Datasets

### EMNIST Dataset

* Automatically downloaded using `torchvision`
* *Not included* in this repository
* Download occurs automatically on first run
* Local location:

  ```
  data/EMNIST
  ```
* Task:
  Binary classification — *Even vs Odd digits*

---

### Breast Cancer Dataset

* Source:
  [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
* File location:

  ```
  data/breast_cancer.csv
  ```
* Task:
  Binary classification — *Malignant vs Benign tumors*

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

### Output Files

```
figures/non_dp_learning_curve.png
figures/dp_learning_curve.png
figures/privacy_utility_tradeoff.png
figures/non_dp_breast_learning_curve.png
figures/dp_breast_learning_curve.png
figures/privacy_utility_tradeoff_breast.png
```

---

## 6. Configuration

Privacy-related hyperparameters are explicitly defined in:

* `config.py` (EMNIST)
* `config_breast.py` (Breast Cancer)

---

