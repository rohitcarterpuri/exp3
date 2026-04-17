# Configuration file for the project

# Data paths
DATA_PATH = 'data/housing.csv'
MODEL_SAVE_PATH = 'models/ann_model.h5'
SCALER_SAVE_PATH = 'models/scaler.pkl'

# Model parameters
INPUT_LAYER_NEURONS = 64
HIDDEN_LAYER_1_NEURONS = 32
HIDDEN_LAYER_2_NEURONS = 16
OUTPUT_LAYER_NEURONS = 1

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Early stopping
PATIENCE = 10
