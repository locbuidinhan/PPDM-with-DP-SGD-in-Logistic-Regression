from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, BINARY_MODE
from utils.preprocessing_emnist import load_and_preprocess_emnist
from utils.metrics import evaluate_model
from utils.plotting import plot_learning_curve
from models.logistic_standard import LogisticRegressionScratch

X_train, X_test, y_train, y_test = load_and_preprocess_emnist(
    binary_mode=BINARY_MODE
)

model = LogisticRegressionScratch(
    lr=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.fit(X_train, y_train, X_test, y_test)

plot_learning_curve(
    model.train_acc,
    model.test_acc,
    title="Non-DP Logistic Regression",
    save_path="figures/non_dp_learning_curve.png"
)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(evaluate_model(y_test, y_pred, y_proba))
