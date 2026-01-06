import matplotlib.pyplot as plt
from models.logistic_standard import LogisticRegressionScratch
from utils.preprocessing import load_and_preprocess


def plot_lr_loss_breast(
    lrs=(0.01, 0.03, 0.05),
    epochs=100,
    batch_size=32,
    save_path="figures/breast_lr_loss.png"
):
    # Load Breast Cancer dataset
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()

    plt.figure(figsize=(7, 5))

    for lr in lrs:
        print(f"Training non-DP with lr = {lr}")

        model = LogisticRegressionScratch(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
        )

        model.fit(X_train, y_train, X_test, y_test)

        plt.plot(
            model.train_losses,
            label=f"lr = {lr}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Non-DP Logistic Regression\nLearning Rate Sensitivity (Breast Cancer)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    plot_lr_loss_breast()

