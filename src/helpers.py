PATH = "../data/"
WHITE = "winequality-white.csv"
RED = "winequality-red.csv"
FIXED = "winequality-fixed.csv"
CATEGORIES = "winequality-fixed-categories.csv"

THRESHOLD = 5
LEARNING_RATE = 0.00001

def num_examples(_ds):
    return _ds.shape[0]

def num_features(_ds):
    return _ds.shape[1]