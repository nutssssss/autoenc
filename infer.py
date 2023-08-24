from train import AE
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = AE(input_shape=784)

# Load the saved state dictionary
model.load_state_dict(torch.load("autoencoder_model.pth"))

# Make sure to set the model in evaluation mode if you're only using it for inference
model.eval()
transform = T.Compose([T.Resize((28, 28)), T.Grayscale(), T.ToTensor()])

# Load and preprocess the image from file
image_path = "/home/student/210905380/autoenc/sample_image.jpg"  # Replace with the actual image path
input_image = Image.open(image_path)
test_input = transform(input_image).view(1, -1)  # Reshape to match model's input shape

with torch.no_grad():
    reconstructed_output = model(test_input)
    print(type(reconstructed_output))

    # Reshape the reconstructed_output to match the original image shape
    reconstructed_image = reconstructed_output.view(28, 28).cpu().numpy()

    # Display the original and reconstructed images side by side
    plt.subplot(1, 2, 1)
    plt.imshow(test_input.view(28, 28).cpu().numpy().reshape(28, 28), cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')

    plt.show()
