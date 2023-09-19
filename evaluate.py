import torch
from torchvision import transforms
from model import CNNLSTM
from preprocess import MSCOCO
from config import *
from PIL import Image

# Path to the checkpoint file
checkpoint_path = "checkpoints/model_checkpoint_epoch20.pth"

# Initialize data preprocessing and tokenizer
mscoco = MSCOCO()

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
    vocab_size=len(mscoco.train_vocab),
    num_layers=NUM_LAYERS,
)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load an image for testing
image_path = "evaluate\\test.jpg"
test_image = Image.open(image_path).convert("RGB")
test_image = image_transform(test_image).unsqueeze(0)
print(test_image.shape)

# Generate a caption for the test image
with torch.no_grad():
    caption_indices = model.caption(
        test_image, mscoco.train_vocab, max_caption_length=MAX_CAPTION_LENGTH
    )
    generated_caption = mscoco.decode_caption(caption_indices)

# Print the generated caption
print("Generated Caption:", generated_caption)
