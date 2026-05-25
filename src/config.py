import os
import tensorflow as tf
import numpy as np

RANDOM_SEED = 42
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Hyperparameters
WINDOW_SIZE = 24
HORIZON = 12
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
FEATURES = ['TravelTime', 'TrafficFlow', 'AvgSpeed']

# Paths
DATASET_PATH = '/content/drive/MyDrive/TA_AI_Dataset/'
ROUTE_FILES = [
    "from01F0147Sto01F0155S.csv",
    "from01F0339Sto01F0376S.csv",
    "from01F0532Sto01F0557S.csv"
]