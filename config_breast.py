# =====================
# DATASET
# =====================
DATA_PATH = "data/breast_cancer.csv"
RANDOM_STATE = 42

# =====================
# NON-DP 
# =====================
LEARNING_RATE = 0.05
EPOCHS = 100
BATCH_SIZE = 32

# =====================
# DP-SGD 
# =====================
DP_LEARNING_RATE = 0.03
DP_EPOCHS = 100
DP_BATCH_SIZE = 32
CLIP_NORM = 4.0
NOISE_MULTIPLIER = 6.5
