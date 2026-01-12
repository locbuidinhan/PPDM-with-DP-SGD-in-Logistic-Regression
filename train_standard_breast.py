from config_breast import LEARNING_RATE, EPOCHS, BATCH_SIZE
from utils.preprocessing import load_and_preprocess
from utils.metrics import evaluate_model
from utils.plotting import plot_learning_curve
from models.logistic_standard import LogisticRegressionScratch
import os

os.makedirs("figures", exist_ok=True)

# Load Breast Cancer dataset
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

# Train non-DP Logistic Regression
model = LogisticRegressionScratch(
    lr=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.fit(X_train, y_train, X_test, y_test)

# Plot learning curve
plot_learning_curve(
    model.train_acc,
    model.test_acc,
    title="Non-DP Logistic Regression (Breast Cancer)",
    save_path="figures/non_dp_breast_learning_curve.png"
)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

results = evaluate_model(y_test, y_pred, y_proba)

with open("figures/result_for_breast_non_dp.txt", "w") as f:
    f.write("Results for Breast Cancer – Non-DP Logistic Regression\n")
    f.write("=" * 50 + "\n")
    f.write(str(results))

print(results)
