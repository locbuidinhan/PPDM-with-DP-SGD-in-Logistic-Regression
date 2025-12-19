import numpy as np
from torchvision.datasets import EMNIST
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_emnist(binary_mode="even_odd"):
    # Load EMNIST digits
    train_set = EMNIST(
        root="./data",
        split="digits",
        train=True,
        download=True
    )
    test_set = EMNIST(
        root="./data",
        split="digits",
        train=False,
        download=True
    )

    X_train = train_set.data.numpy().astype(np.float32)
    y_train = train_set.targets.numpy().astype(int)
    X_test = test_set.data.numpy().astype(np.float32)
    y_test = test_set.targets.numpy().astype(int)

    # Binary labels
    if binary_mode == "even_odd":
        y_train = (y_train % 2 == 0).astype(int)
        y_test = (y_test % 2 == 0).astype(int)
    elif binary_mode == "zero_vs_rest":
        y_train = (y_train == 0).astype(int)
        y_test = (y_test == 0).astype(int)
    else:
        raise ValueError("Unsupported binary mode")

    # Flatten
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
