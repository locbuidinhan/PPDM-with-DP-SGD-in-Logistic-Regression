# =====================
# DATASET
# =====================
DATASET = "EMNIST_DIGITS"
BINARY_MODE = "even_odd"
RANDOM_STATE = 42


# =====================
# NON-DP
# =====================
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 128

# =====================
# DP-SGD 
# =====================
DP_LEARNING_RATE = 0.005
DP_EPOCHS = 100
DP_BATCH_SIZE = 128
CLIP_NORM = 16.0
NOISE_MULTIPLIER = 0.8
