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

# Load an image for testing
image_path = "evaluate\\test.jpg"
test_image = Image.open(image_path).convert("RGB")
test_image = image_transform(test_image).unsqueeze(0)


def decode_caption(indices):
    # Convert index to string, join and remove paddings.
    words = [train_vocab.lookup_token(index) for index in indices]
    return (
        " ".join(words).replace("<PAD>", "").replace("<START>", "").replace("<END>", "")
    )


# Generate a caption for the test image
with torch.no_grad():
    caption_indices = model.caption(
        test_image, train_vocab, max_caption_length=MAX_CAPTION_LENGTH
    )
    generated_caption = decode_caption(caption_indices)

# Print the generated caption
print("Generated Caption:", generated_caption)
