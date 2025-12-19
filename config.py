# =====================
# DATASET
# =====================
DATASET = "EMNIST_DIGITS"
BINARY_MODE = "even_odd"
RANDOM_STATE = 42


# =====================
# NON-DP
# =====================
LEARNING_RATE = 0.05
EPOCHS = 100
BATCH_SIZE = 128

# =====================
# DP-SGD 
# =====================
DP_LEARNING_RATE = 0.03
DP_EPOCHS = 100
DP_BATCH_SIZE = 128
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0
