import torch
import torch.nn as nn
import torchvision.models as models
import os

# CNN (Convolutional Neural Network): a CNN model based on the
# Inception V3 architecture. The last fully connected layer of
# Inception V3 is replaced with a linear layer to map the output
# dimension to embed_size. Only fine-tune of the last layer while
# keeping the rest of the pre-trained Inception V3 layers frozen.

# LSTM (Long Short-Term Memory): an LSTM language model. Takes
# image features from CNN and captions as input and generates
# predictions for the next word in the caption sequence. Uses an
# embedding layer to convert word indices into embeddings and an
# LSTM layer for sequential modeling. The final linear layer is
# used to predict the next word.

# CNNLSTM: Combines the CNN and LSTM models for end-to-end image
# captioning. It takes images and captions as input, extracts
# image features using the CNN model, and generates captions
# using the LSTM model.

# Input: [1, channels, height, width] or [1, 3, 299, 299].
class CNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNN, self).__init__()

        # Default to not train, using pretrained CNN model.
        self.train_CNN = train_CNN

        # Using inception_v3.
        # Get last layer according to
        # https://pytorch.org/vision/master/_modules/torchvision/modelinception.html#inception_v3.
        # No idea why setting it false when not initializing works.
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True
            )
        self.inception.aux_logits = False

        # Replace with a linear layer, map dimension to embed_size.
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        # Only update the params for the last layer.
        self.finetune()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # images: [batch_size, 3, 299, 299] at minimum (Inceptionv3 requirements).
        output = self.inception(images)
        output = self.relu(output)
        output = self.dropout(output)

        # output: [batch_size, embed_size].
        return output

    def finetune(self):
        # Do not update pretrained layers' params.
        for param in self.inception.parameters():
            param.requires_grad = False
        # Update only last fc layer's param.
        for param in self.inception.fc.parameters():
            param.requires_grad = True


# Input: [batch_size, sequence_length, input_size]
class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(LSTM, self).__init__()

        self.embed_size = embed_size
        self.num_layers = num_layers

        # Dimension of hidden state h.
        self.hidden_size = hidden_size

        # Exclude the <UNK> token.
        self.vocab_size = vocab_size

        # Convert word indices into size embed_size.
        # https://saturncloud.io/blog/what-are-embeddings-in-pytorch-and-how-to-use-them/#how-do-embeddings-work
        self.embedding = nn.Embedding(self.vocab_size, embed_size)

        # Main LSTM.
        self.lstm = nn.LSTM(
            self.embed_size, self.hidden_size, self.num_layers, batch_first=True
        )

        # Get probability distribution from LSTM's hidden state for the next word.
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        # self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: [batch_size, embed_size] from CNN.
        # captions: [batch_size, max_caption_length].

        # Pass the caption through embedding layer.
        # Note: captions here is from previously unrolled layer of lstm.
        captions = captions[:, :-1]
        embeddings = self.embedding(captions)

        # Concatenate image features with word embedding, provide context.
        # Unsqueeze on dim 1 due to batch_first = True, else dim 0.
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        output, _ = self.lstm(embeddings)
        output = self.linear(output)

        return output


class CNNLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNLSTM, self).__init__()
        self.CNN = CNN(embed_size)
        self.LSTM = LSTM(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.CNN(images)
        output = self.LSTM(features, captions)
        return output

    def caption(self, image, vocabulary, max_caption_length):
        result = []

        # Not training so no_grad.
        with torch.no_grad():
            # Initialize first hidden state for LSTM.
            # TODO: what about having cnn's output as initial hidden state?
            h_n = None

            # 1st dim is batch_size, unsqueeze = batch size of 1.
            # output1: [embed_size], unsqueeze -> [1, embed_size]
            output1 = self.CNN(image).unsqueeze(0)

            # Iterate until finish captioning.
            for _ in range(max_caption_length):
                # Pass the image features as LSTM's input with previous hidden states.
                # Use .lstm to bypass the must have captions parameter of forward().
                output2, h_n = self.LSTM.lstm(output1, h_n)

                # Squeeze as self.linear = nn.Linear(hidden_size, vocab_size) i.e
                # output2: [1, sequence_length, embed_size], squeeze -> [sequence_length, embed_size]
                output2 = self.LSTM.linear(output2.squeeze(0))

                # Get prediction by picking word with highest probability.
                predicted = output2.argmax(1)
                result.append(predicted.item())

                # Use the predicted word as input for next unroll.
                output1 = self.LSTM.embedding(predicted).unsqueeze(0)

                # Stop if end token.
                if vocabulary.get_itos()[predicted.item()] == "<END>":
                    break

        return result
