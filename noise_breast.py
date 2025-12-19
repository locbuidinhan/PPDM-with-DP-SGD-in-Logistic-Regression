import os
from models.logistic_dp_sgd import LogisticRegressionDPSGD
from utils.preprocessing import load_and_preprocess
from utils.plotting import plot_noise_tradeoff

noise_levels = [0.5, 1.0, 2.0]
results = []

X_train, X_test, y_train, y_test, _ = load_and_preprocess()

for sigma in noise_levels:
    model = LogisticRegressionDPSGD(
        lr=0.03,
        epochs=100,
        batch_size=32,      
        clip_norm=1.0,
        noise_multiplier=sigma
    )

    model.fit(X_train, y_train)

    acc = (model.predict(X_test) == y_test).mean()
    results.append(acc)

plot_noise_tradeoff(
    noise_levels,
    results,
    save_path="figures/privacy_utility_tradeoff_breast.png"
)

print("Noise levels:", noise_levels)
print("Test accuracy:", results)
