SEED=42
"""Specify a seed for reproducability purposes, set SEED to None for random"""

TWITTER_DATASET_PATH = "./twitter-datasets/"
"""Relative path to the dataset folder"""
SUBMISSIONS_PATH = "./submissions/"
"""Relative path to the submissions folder"""

# Declaration of the file names for the datasets
TRAIN_POS_FILE = TWITTER_DATASET_PATH + "train_pos.txt"
"""Name of the file containing the training dataset of positive tweets"""
TRAIN_NEG_FILE = TWITTER_DATASET_PATH + "train_neg.txt"
"""Name of the file containing the training dataset of negative tweets"""
TEST_DATA_FILE = TWITTER_DATASET_PATH + "test_data.txt"
"""Name of the file containing the testing dataset of tweets"""
EMBEDDING_VOCAB_FILE = "vocab_cut.txt"
"""Name of the file for the cut vocabulary"""
EMBEDDING_FILE = "embeddings.npy"
"""Name of the file for the embeddings"""