import torch
from variational_train import VAE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
import torchvision
import umap

# Load the trained model and set it to evaluation mode
model = VAE(input_shape=784, latent_dim=20)
model.load_state_dict(torch.load("variational_autoencoder_model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load your dataset (e.g., MNIST)
# ...

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

# Encode the dataset to obtain latent space coordinates
# Encode the dataset to obtain latent space coordinates
latent_coordinates = []
labels = []

with torch.no_grad():
    for batch_features, batch_labels in train_loader:  # Also load labels
        batch_features = batch_features.to(device)
        _, latent_mean, _ = model(batch_features)  # Extract latent mean from the model output
        latent_coordinates.extend(latent_mean.cpu().numpy())
        labels.extend(batch_labels.numpy())  # Store the labels

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=2)  # You can adjust the number of components
reduced_latent = reducer.fit_transform(latent_coordinates)

# Create a scatter plot of the reduced latent space
plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], c=labels, cmap='viridis')  # Use 'viridis' colormap for better color distinction
plt.title("Latent Space Visualization using UMAP")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.colorbar(label="Labels")
plt.show()
