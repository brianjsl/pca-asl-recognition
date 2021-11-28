"""
File of global constants.
"""

from string import ascii_uppercase


# Data
DATA_LABELS = list(ascii_uppercase) + ["del", "space", "nothing"]
NUM_DATA_LABELS = len(DATA_LABELS)
LABELS_TO_INDEX = {label: idx for idx, label in enumerate(DATA_LABELS)}
NUM_IMAGES_PER_LABEL = 3000