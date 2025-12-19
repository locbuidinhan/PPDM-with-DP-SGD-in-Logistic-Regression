from config_breast import (
    DP_LEARNING_RATE,
    DP_EPOCHS,
    DP_BATCH_SIZE,
    CLIP_NORM,
    NOISE_MULTIPLIER
)
from utils.preprocessing import load_and_preprocess
from utils.metrics import evaluate_model
from utils.plotting import plot_learning_curve
from models.logistic_dp_sgd import LogisticRegressionDPSGD
import os

os.makedirs("figures", exist_ok=True)

# Load Breast Cancer dataset
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

# Train DP-SGD Logistic Regression
model = LogisticRegressionDPSGD(
    lr=DP_LEARNING_RATE,
    epochs=DP_EPOCHS,
    batch_size=DP_BATCH_SIZE,
    clip_norm=CLIP_NORM,
    noise_multiplier=NOISE_MULTIPLIER
)

model.fit(X_train, y_train, X_test, y_test)

# Plot learning curve
plot_learning_curve(
    model.train_acc,
    model.test_acc,
    title="DP-SGD Logistic Regression (Breast Cancer)",
    save_path="figures/dp_breast_learning_curve.png"
)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("DP-SGD Logistic Regression (Breast Cancer)")
print(evaluate_model(y_test, y_pred, y_proba))
