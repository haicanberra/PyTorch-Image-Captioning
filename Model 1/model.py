import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN = False):
        super(EncoderCNN, self).__init__()

        # Default to not train, using pretrained CNN model.
        self.train_CNN = train_CNN

        # Using inception_v3, not training so do not need aux branch.
        # Get last layer according to https://pytorch.org/vision/master/_modules/torchvision/models/inception.html#inception_v3.
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)

        # Replace with a linear layer, map dimension to embed_size.
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        # Only update the params for the last layer.
        self.finetune()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        output = self.inception(images)
        output = self.relu(output)
        output = self.dropout(output)

        return output

    def finetune(self):
        # Do not update pretrained layers' params.
        for param in self.inception.parameters():
            param.requires_grad = False
        # Update only last fc layer's param.
        for param in self.inception.fc.parameters():
            param.requires_grad = True

        