from variational_train import VAE  # Assuming you have the VAE defined in train_variational.py
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Instantiate the VAE model with the same input_shape and latent_dim
model = VAE(input_shape=784, latent_dim=20)

# Load the saved state dictionary for the VAE
model.load_state_dict(torch.load("variational_autoencoder_model.pth"))

# Set the model in evaluation mode
model.eval()

transform = T.Compose([T.Resize((28, 28)), T.Grayscale(), T.ToTensor()])

image_path = "/home/student/210905380/autoenc/sample_image.jpg"  # Replace with the actual image path
input_image = Image.open(image_path)
test_input = transform(input_image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    reconstructed_output, _, _ = model(test_input)  # We ignore the mean and log variance here

    # Reshape the reconstructed_output to match the original image shape
    reconstructed_image = reconstructed_output.view(28, 28).cpu().numpy()

    # Display the original and reconstructed images side by side
    plt.subplot(1, 2, 1)
    plt.imshow(test_input.view(28, 28).cpu().numpy().squeeze(), cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')

    plt.show()
