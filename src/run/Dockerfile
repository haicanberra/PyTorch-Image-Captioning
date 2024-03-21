FROM python:3.11

WORKDIR /app

# Copy entire project directory
COPY . .
RUN pip install torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/cpu

# Download model file
RUN curl -L https://github.com/haicanberra/PyTorch-Image-Captioning/releases/download/v1.0.0/model.pth -o model/model.pth

# Volume for user-provided images
VOLUME ["image"]

# Entrypoint (command to run)
ENTRYPOINT ["python", "caption.py"]  # Rename evaluate.py to main.py
