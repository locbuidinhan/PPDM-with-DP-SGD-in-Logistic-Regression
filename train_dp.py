from config import (
    DP_LEARNING_RATE,
    DP_EPOCHS,
    DP_BATCH_SIZE,
    CLIP_NORM,
    NOISE_MULTIPLIER,
    BINARY_MODE
)

from utils.preprocessing_emnist import load_and_preprocess_emnist
from utils.metrics import evaluate_model
from utils.plotting import plot_learning_curve
from models.logistic_dp_sgd import LogisticRegressionDPSGD

X_train, X_test, y_train, y_test = load_and_preprocess_emnist(
    binary_mode=BINARY_MODE
)

model = LogisticRegressionDPSGD(
    lr=DP_LEARNING_RATE,
    epochs=DP_EPOCHS,
    batch_size=DP_BATCH_SIZE,
    clip_norm=CLIP_NORM,
    noise_multiplier=NOISE_MULTIPLIER
)

model.fit(X_train, y_train, X_test, y_test)

plot_learning_curve(
    model.train_acc,
    model.test_acc,
    title="DP-SGD Logistic Regression",
    save_path="figures/dp_learning_curve.png"
)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(evaluate_model(y_test, y_pred, y_proba))
