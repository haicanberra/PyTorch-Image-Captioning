# # DEBUG PATHS
# TRAIN_PATH = "dataset\\debug\\train\\train2014"
# TRAIN_ANNO_PATH = "dataset\\debug\\train\\annotations\\captions_train2014.json"

# VAL_PATH = "dataset\\debug\\val\\val2014"
# VAL_ANNO_PATH = "dataset\\debug\\val\\annotations\\captions_val2014.json"

# TEST_PATH = "dataset\\debug\\test\\test2014"
# TEST_ANNO_PATH = "dataset\\debug\\test\\annotations"

# PATHS
TRAIN_PATH = "dataset\\nobug\\train\\train2014"
TRAIN_ANNO_PATH = "dataset\\nobug\\train\\annotations\\captions_train2014.json"

VAL_PATH = "dataset\\nobug\\val\\val2014"
VAL_ANNO_PATH = "dataset\\nobug\\val\\annotations\\captions_val2014.json"

TEST_PATH = "dataset\\nobug\\test\\test2014"
TEST_ANNO_PATH = "dataset\\nobug\\test\\annotations"

# VARIABLES
BATCH_SIZE = 64
MAX_CAPTION_LENGTH = 50
MIN_FREQUENCY = 5

LEARNING_RATE = 0.001
NUM_EPOCHS = 20
CHECKPOINT_INTERVAL = 20

EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 1
