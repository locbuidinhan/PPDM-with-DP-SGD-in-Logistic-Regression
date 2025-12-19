import numpy as np
from utils.helpers import sigmoid, compute_loss


class LogisticRegressionDPSGD:
    """
    DP mechanism:
    - Per-example gradients
    - L2 gradient clipping
    - Gaussian noise addition    """

    def __init__(
        self,
        lr=0.1,
        epochs=100,
        batch_size=32,
        clip_norm=1.0,
        noise_multiplier=1.0,
        seed=42,
        verbose=False
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.seed = seed
        self.verbose = verbose

        self.w = None
        self.b = 0.0

        # ---- Logging (for TIER 1 visualization) ----
        self.train_acc = []
        self.test_acc = []

    # -------------------------------------------------
    # 1. Per-example gradients
    # -------------------------------------------------
    def _per_example_grads(self, Xb, yb):
        """
        Compute per-example gradients.
        """
        linear = Xb.dot(self.w) + self.b          
        probs = sigmoid(linear)                   
        errors = probs - yb                       

        grad_w = errors[:, None] * Xb             
        grad_b = errors.reshape(-1, 1)            

        return np.hstack([grad_w, grad_b])        

    # -------------------------------------------------
    # 2. L2 clipping
    # -------------------------------------------------
    def _clip(self, per_example_grads):
        """
        Clip per-example gradients to L2 norm <= clip_norm
        """
        norms = np.linalg.norm(per_example_grads, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.clip_norm / (norms + 1e-12))
        return per_example_grads * scale

    # -------------------------------------------------
    # 3. Add Gaussian noise
    # -------------------------------------------------
    def _add_noise(self, avg_grad, batch_size):
        """
        Add Gaussian noise calibrated for DP
        """
        stddev = self.noise_multiplier * self.clip_norm / batch_size
        noise = np.random.normal(0.0, stddev, size=avg_grad.shape)
        return avg_grad + noise

    # -------------------------------------------------
    # 4. Training with DP-SGD
    # -------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train model using DP-SGD.
            X, y      : training data
            X_val, y_val : test data
        """
        np.random.seed(self.seed)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            Xs, ys = X[indices], y[indices]

            for start in range(0, n_samples, self.batch_size):
                Xb = Xs[start:start + self.batch_size]
                yb = ys[start:start + self.batch_size]

                if len(yb) == 0:
                    continue

                # 1) Per-example gradients
                per_ex_grads = self._per_example_grads(Xb, yb)

                # 2) Clip gradients
                clipped_grads = self._clip(per_ex_grads)

                # 3) Average
                avg_grad = clipped_grads.mean(axis=0)

                # 4) Add noise
                noisy_grad = self._add_noise(avg_grad, len(yb))

                # Split weight and bias gradients
                grad_w = noisy_grad[:-1]
                grad_b = noisy_grad[-1]

                # 5) Parameter update
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

            # ---- Logging per epoch (TIER 1) ----
            train_probs = self.predict_proba(X)
            train_acc = ((train_probs >= 0.5) == y).mean()
            self.train_acc.append(train_acc)

            if X_val is not None:
                val_probs = self.predict_proba(X_val)
                val_acc = ((val_probs >= 0.5) == y_val).mean()
                self.test_acc.append(val_acc)

            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(
                    f"[DP-SGD] Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Acc: {train_acc:.4f}"
                )

        return self

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict_proba(self, X):
        return sigmoid(X.dot(self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
