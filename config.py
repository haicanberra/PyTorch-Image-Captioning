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

# MODEL FILES PATHS
MODEL_PATH = "models\\model.pth"
TRAIN_VOCAB_PATH = "models\\train_vocab.pth"

# VARIABLES
BATCH_SIZE = 32
MAX_CAPTION_LENGTH = 100
MIN_FREQUENCY = 5

LEARNING_RATE = 0.001
NUM_EPOCHS = 10
SAVE_CHECKPOINT = False
CHECKPOINT_INTERVAL = 100

EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
