import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from config import *
import matplotlib.pyplot as plt
import numpy as np

# Data Processing: The process method loads training and
# validation datasets using the CocoCaptions dataset class from
# torchvision (Returns a list of image-captions pairs).
# Vocabularies are built for both datasets using the tokenized
# captions (List of list of words). A special token "<UNK>" is
# included in the vocabularies.

# Data Transformation: Two image transformations: load_transform
# for loading images as Tensors and transform for preprocessing
# images before inputting them to the InceptionV3 model. The
# transform step resizes the images to (299, 299) pixels and
# normalizes according to Inception V3 requirements.

# Data Loader: Data loaders for both the training and validation
# datasets. collate_fn method to preprocess batches of data,
# including image transformation and caption processing, mostly
# to make the caption length the same by padding.

# Caption Processing: The process_caption method tokenizes
# captions, adds start and end tokens ("<START>" and "<END>"),
# pads them to a consistent length, and converts tokens to
# indices based on the training vocabulary. Unknown words are
# replaced with the "<UNK>" token.

# Decoding Captions: The decode_caption method converts a list
# of indices back into a human-readable caption string, removing
# padding tokens.

# Test Section: In the test section (executed only if the script
# is run as the main program), it creates an instance of the
# MSCOCO class, prints batch information, displays an image from
# the batch, and prints the corresponding caption.


class MSCOCO:
    def __init__(self):
        # Initialize data paths.
        self.train_path = TRAIN_PATH
        self.train_anno_path = TRAIN_ANNO_PATH
        self.val_path = VAL_PATH
        self.val_anno_path = VAL_ANNO_PATH

        # Variables.
        self.batch_size = BATCH_SIZE
        self.max_caption_length = MAX_CAPTION_LENGTH

        # Frequency to exclude less common words.
        self.min_freq = MIN_FREQUENCY

        # Load image transform.
        self.load_transform = transforms.ToTensor()
        # Transform image for input for InceptionV3
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299), antialias=True),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Tokenizer.
        self.tokenizer = get_tokenizer("basic_english")

        # Process the data.
        self.process()

    def process(self):
        # Load training and validation data.
        # See https://pytorch.org/vision/stable/generated/torchvision.datasets.CocoCaptions.html.
        # Each element in dataset is image-captions pair.
        # Captions is a list of strings (multiple captions per image).
        self.train_dataset = CocoCaptions(
            root=self.train_path,
            annFile=self.train_anno_path,
            transform=self.load_transform,
        )
        self.val_dataset = CocoCaptions(
            root=self.val_path,
            annFile=self.val_anno_path,
            transform=self.load_transform,
        )

        # Create vocabularies for both dataset above.
        # A list contain a list contain words, for the function to work.
        # Only take 1 caption from list of captions.
        # See https://stackoverflow.com/questions/73177807/unable-to-build-vocab-for-a-torchtext-text-classification.
        self.train_vocab = build_vocab_from_iterator(
            [self.tokenizer(caption[0]) for _, caption in self.train_dataset],
            specials=["<UNK>", "<START>", "<END>", "<PAD>"],
        )
        self.val_vocab = build_vocab_from_iterator(
            [self.tokenizer(caption[0]) for _, caption in self.val_dataset],
            specials=["<UNK>", "<START>", "<END>", "<PAD>"],
        )

        # Dataloader for both dataset.
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # Asterisk to unpack the image-caption pair into individual.
        images, captions = zip(*batch)

        # Process the image.
        images = [self.transform(image) for image in images]

        # Process the caption (for making same size etc.)
        captions = [self.process_caption(caption) for caption in captions]

        # Put back to previous format.
        return torch.stack(images), torch.stack(captions)

    def process_caption(self, caption):
        # Tokenize the caption.
        tokens = self.tokenizer(caption[0].lower())

        # Add start/end of sequence tokens.
        tokens = ["<START>"] + tokens + ["<END>"]

        # Pad so the length are consistant.
        tokens += ["<PAD>"] * (self.max_caption_length - len(tokens))

        # Get indices from training vocab. If not exist then index of '<UNK>' is used.
        indices = [
            self.train_vocab.get_stoi().get(
                token, self.train_vocab.lookup_indices(["<UNK>"])[0]
            )
            for token in tokens
        ]

        return torch.tensor(indices)

    def decode_caption(self, indices):
        # Convert index to string, join and remove paddings.
        words = [self.train_vocab.lookup_token(index) for index in indices]
        return " ".join(words).replace("<PAD>", "")


# Test
if __name__ == "__main__":
    mscoco = MSCOCO()

    for images, captions in mscoco.train_loader:
        print("Batch of training images shape:", images.shape)
        print("Individual image shape:", images[0].shape)
        plt.imshow((images[0].permute(1, 2, 0).numpy()))
        plt.show()
        print("Corresponding caption:", mscoco.decode_caption(captions[0]))
        print(
            "Vocab:", [mscoco.train_vocab.lookup_token(index) for index in captions[0]]
        )
        print("Vocab len:", len(mscoco.train_vocab))
