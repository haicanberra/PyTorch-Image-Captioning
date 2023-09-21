import torch
from torchvision import transforms
from model import CNNLSTM
from preprocess import MSCOCO
from config import *
from PIL import Image

# Path to the checkpoint file
# MODIFY THIS THEN CHECK LINE 32
checkpoint_path = "image_captioning_model.pth"

# Initialize data preprocessing and tokenizer
# Comment if already saved vocab file outside.
# mscoco = MSCOCO()

# Save the vocab, comment all lines after 18.
# torch.save(mscoco.train_vocab, 'train_vocab.pth')

# Load the vocab, comment line 14.
train_vocab = torch.load('train_vocab.pth')

# Image preprocessing transform
image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Load the model from the checkpoint
model = CNNLSTM(
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    vocab_size=len(train_vocab),
    num_layers=NUM_LAYERS,
)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

# Load an image for testing
image_path = "evaluate\\test.jpg"
test_image = Image.open(image_path).convert("RGB")
test_image = image_transform(test_image).unsqueeze(0)
# print(test_image.shape)

def decode_caption(indices):
    # Convert index to string, join and remove paddings.
    words = [train_vocab.lookup_token(index) for index in indices]
    return " ".join(words).replace("<PAD>", "")

# Generate a caption for the test image
with torch.no_grad():
    caption_indices = model.caption(
        test_image, train_vocab, max_caption_length=MAX_CAPTION_LENGTH
    )
    generated_caption = decode_caption(caption_indices)

# Print the generated caption
print("Generated Caption:", generated_caption)
