import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoCaptions
import matplotlib.pyplot as plt

from model import CNNLSTM
from preprocess import MSCOCO
from config import *

if __name__ == "__main__":
    # Lets go GPU
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data and model
    mscoco = MSCOCO()
    model = CNNLSTM(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(mscoco.train_vocab),
        num_layers=NUM_LAYERS,
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(
        ignore_index=mscoco.train_vocab.lookup_indices(["<PAD>"])[0]
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Move model and criterion to GPU
    model.to(device)
    criterion.to(device)

    # Data loaders
    train_loader = mscoco.train_loader
    val_loader = mscoco.val_loader

    # Checkpoint directory
    if SAVE_CHECKPOINT:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    # Print to notify
    print("Training starts...")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            # Move images and captions to GPU
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model.forward(images, captions)

            # Compute loss
            loss = criterion(
                outputs.view(-1, len(mscoco.train_vocab)), captions.view(-1)
            )
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate average training loss for this epoch
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop (similar to training loop)
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for i, (val_images, val_captions) in enumerate(val_loader):
                # Move images and captions to GPU
                val_images = val_images.to(device)
                val_captions = val_captions.to(device)

                # Forward pass
                val_outputs = model.forward(val_images, val_captions)

                # Compute validation loss
                val_loss = criterion(
                    val_outputs.view(-1, len(mscoco.train_vocab)), val_captions.view(-1)
                )
                total_val_loss += val_loss.item()

        # Calculate average validation loss for this epoch
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print training and validation loss for this epoch
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss} - Validation Loss: {avg_val_loss}"
        )

        # Save checkpoint
        if SAVE_CHECKPOINT:
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"model_checkpoint_epoch{epoch+1}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                    checkpoint_path,
                )

    # Save the final trained model
    torch.save(model.state_dict(), "models\\model.pth")

    # Plot training and validation loss
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Epochs")
    plt.savefig("output\\train_val_loss_plot.png")  # Save the plot as an image
    plt.show()  # Show the plot
