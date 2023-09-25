import torch
from torchvision import transforms
from model import CNNLSTM
from config import *
from PIL import Image
import os

# Assertion
assert os.path.isfile(TRAIN_VOCAB_PATH)
assert os.path.isfile(MODEL_PATH)

# Load the vocab.
train_vocab = torch.load(TRAIN_VOCAB_PATH)

# Image preprocessing transform
image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# For loading model.
model = CNNLSTM(
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    vocab_size=len(train_vocab),
    num_layers=NUM_LAYERS,
)
model_file = torch.load(MODEL_PATH)
model.load_state_dict(model_file)
model.eval()

# Get all images in evaluate
images = []
file_names = []
for filename in os.listdir(EVAL_PATH):
    file_names.append(filename)
    f = os.path.join(EVAL_PATH, filename)
    # Checking if it is a file
    if os.path.isfile(f):
        test_image = Image.open(f).convert("RGB")
        test_image = image_transform(test_image).unsqueeze(0)
        images.append(test_image)

# Load an image for testing
# image_path = "evaluate\\1.jpg"
# test_image = Image.open(image_path).convert("RGB")
# test_image = image_transform(test_image).unsqueeze(0)


def decode_caption(indices):
    # Convert index to string, join and remove paddings.
    words = [train_vocab.lookup_token(index) for index in indices]
    return (
        " ".join(words).replace("<PAD>", "").replace("<START>", "").replace("<END>", "")
    )


# Generate a caption for the test image
counter = 0
with open('output.txt', 'w') as f:
    with torch.no_grad():
        for ti in images:
            caption_indices = model.caption(
                ti, train_vocab, max_caption_length=MAX_CAPTION_LENGTH
            )
            generated_caption = decode_caption(caption_indices)
            # Print the generated caption
            f.write(file_names[counter] + generated_caption + '\n')
            counter = counter + 1
