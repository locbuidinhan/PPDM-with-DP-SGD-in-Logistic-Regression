import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# =====================
# Path setup
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# =====================
# Model
# =====================
from models.logistic_dp_sgd import LogisticRegressionDPSGD

# =====================
# Datasets
# =====================
from utils.preprocessing_emnist import load_and_preprocess_emnist
from utils.preprocessing import load_and_preprocess

# =====================
# Configs
# =====================
from config import BINARY_MODE, DP_BATCH_SIZE
from config_breast import DP_BATCH_SIZE as DP_BATCH_SIZE_BREAST


# ============================================================
# Core function: compute + plot gradient norm CDF for 1 dataset
# ============================================================
def compute_and_plot_gradient_norms(
    X_train,
    y_train,
    batch_size,
    title,
    save_png,
    epochs=3,
    seed=42,
    xlim=None
):
    n_samples, n_features = X_train.shape

    # Initialize model (no training, no noise)
    model = LogisticRegressionDPSGD(
        epochs=epochs,
        batch_size=batch_size,
        clip_norm=1e6,        # effectively disable clipping
        noise_multiplier=0.0,
        seed=seed
    )

    model.w = np.zeros(n_features)
    model.b = 0.0

    np.random.seed(seed)
    grad_norms = []

    # Collect per-example gradient norms
    for _ in range(epochs):
        indices = np.random.permutation(n_samples)
        Xs, ys = X_train[indices], y_train[indices]

        for start in range(0, n_samples, batch_size):
            Xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]

            if len(yb) == 0:
                continue

            per_example_grads = model._per_example_grads(Xb, yb)
            norms = np.linalg.norm(per_example_grads, axis=1)
            grad_norms.extend(norms)

    grad_norms = np.asarray(grad_norms)

    # Percentiles
    p50, p70, p80, p90, p95 = np.percentile(
        grad_norms, [50, 70, 80, 90, 95]
    )

    # CDF
    sorted_norms = np.sort(grad_norms)
    cdf = np.arange(1, len(sorted_norms) + 1) / len(sorted_norms)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_norms, cdf, linewidth=2)

    plt.axvline(p50, color="green", linestyle="--", label=f"P50 = {p50:.2f}")
    plt.axvline(p70, color="black", linestyle="--", label=f"P70 = {p70:.2f}")
    plt.axvline(p80, color="purple", linestyle="--", label=f"P80 = {p80:.2f}")
    plt.axvline(p90, color="blue", linestyle="--", label=f"P90 = {p90:.2f}")
    plt.axvline(p95, color="red", linestyle="--", label=f"P95 = {p95:.2f}")

    if xlim is not None:
        plt.xlim(*xlim)

    plt.xlabel("L2 norm of per-example gradient")
    plt.ylabel("Cumulative probability")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_png, dpi=300)
    plt.show()

    # Console log
    print(f"[{title}]")
    print(f"P50: {p50:.4f}")
    print(f"P70: {p70:.4f}")
    print(f"P80: {p80:.4f}")
    print(f"P90: {p90:.4f}")
    print(f"P95: {p95:.4f}")
    print(f"Max: {grad_norms.max():.4f}\n")

    return grad_norms


# =========================================
# Run gradient diagnostics for all datasets
# =========================================
def log_dp_gradient_norms_all():
    # -------- EMNIST --------
    X_emnist, _, y_emnist, _ = load_and_preprocess_emnist(
        binary_mode=BINARY_MODE
    )

    compute_and_plot_gradient_norms(
        X_train=X_emnist,
        y_train=y_emnist,
        batch_size=DP_BATCH_SIZE,
        title="EMNIST – Per-example Gradient Norm CDF",
        save_png="figures/emnist_gradient_norm_cdf.png",
        epochs=3,
        xlim=(0, 60)
    )

    # -------- Breast Cancer --------
    X_breast, _, y_breast, _, _ = load_and_preprocess()

    compute_and_plot_gradient_norms(
        X_train=X_breast,
        y_train=y_breast,
        batch_size=DP_BATCH_SIZE_BREAST,
        title="Breast Cancer – Per-example Gradient Norm CDF",
        save_png="figures/breast_gradient_norm_cdf.png",
        epochs=3,
        xlim=(0, 5)
    )


# =====================
# Entry point
# =====================
if __name__ == "__main__":
    log_dp_gradient_norms_all()
