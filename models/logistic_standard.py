import numpy as np
from utils.helpers import sigmoid, compute_loss

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=100, batch_size=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.b = 0.0

        # Logging
        self.train_acc = []
        self.test_acc = []

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for _ in range(self.epochs):
            if self.batch_size is None:
                linear = X.dot(self.w) + self.b
                probs = sigmoid(linear)
                dw = (1/n_samples) * X.T.dot(probs - y)
                db = (1/n_samples) * (probs - y).sum()

                self.w -= self.lr * dw
                self.b -= self.lr * db
            else:
                idx = np.random.permutation(n_samples)
                for start in range(0, n_samples, self.batch_size):
                    batch = idx[start:start + self.batch_size]
                    Xb, yb = X[batch], y[batch]

                    linear = Xb.dot(self.w) + self.b
                    probs = sigmoid(linear)
                    dw = (1/len(yb)) * Xb.T.dot(probs - yb)
                    db = (1/len(yb)) * (probs - yb).sum()

                    self.w -= self.lr * dw
                    self.b -= self.lr * db

            # ---- Logging per epoch ----
            train_probs = self.predict_proba(X)
            train_acc = ((train_probs >= 0.5) == y).mean()
            self.train_acc.append(train_acc)

            if X_val is not None:
                val_probs = self.predict_proba(X_val)
                val_acc = ((val_probs >= 0.5) == y_val).mean()
                self.test_acc.append(val_acc)

        return self

    def predict_proba(self, X):
        return sigmoid(X.dot(self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    