import matplotlib.pyplot as plt
from models.logistic_standard import LogisticRegressionScratch
from utils.preprocessing_emnist import load_and_preprocess_emnist
from config import BINARY_MODE, BATCH_SIZE


def plot_lr_loss_emnist(
    lrs=(0.01, 0.03, 0.05),
    epochs=100,
    batch_size=BATCH_SIZE,
    save_path="figures/emnist_lr_loss.png"
):
    # Load EMNIST dataset
    X_train, _, y_train, _ = load_and_preprocess_emnist(
        binary_mode=BINARY_MODE
    )

    plt.figure(figsize=(7, 5))

    for lr in lrs:
        print(f"Training non-DP EMNIST with lr = {lr}")

        model = LogisticRegressionScratch(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
        )

        model.fit(X_train, y_train)

        plt.plot(
            model.train_losses,
            label=f"lr = {lr}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Non-DP Logistic Regression\nLearning Rate Sensitivity (EMNIST)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    plot_lr_loss_emnist()
