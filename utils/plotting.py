import matplotlib.pyplot as plt

def plot_learning_curve(train_acc, test_acc, title, save_path=None):
    plt.figure()
    plt.plot(train_acc, label="Train")
    plt.plot(test_acc, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_noise_tradeoff(noise, acc, save_path=None):
    plt.figure()
    plt.plot(noise, acc, marker="o")
    plt.xlabel("Noise Multiplier (σ)")
    plt.ylabel("Test Accuracy")
    plt.title("Privacy–Utility Trade-off")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

