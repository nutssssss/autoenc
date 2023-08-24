import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Encoder layers (CNN architecture)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Corrected input channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_mean = nn.Linear(64 * 7 * 7, kwargs["latent_dim"])
        self.fc_logvar = nn.Linear(64 * 7 * 7, kwargs["latent_dim"])

        # Decoder layers
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs["latent_dim"], out_features=128
        )
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=1 * 28 * 28)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, features):
        # Encoder
        activation = self.encoder(features)
        activation = activation.view(activation.size(0), -1)  # Flatten the features
        
        mean = self.fc_mean(activation)
        logvar = self.fc_logvar(activation)
        z = self.reparameterize(mean, logvar)

        # Decoder
        activation = self.decoder_hidden_layer(z)
        
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        
        reconstructed = torch.sigmoid(activation)
        reconstructed = reconstructed.view(-1, 1, 28, 28)  # Reshape to match input size
        
        return reconstructed, mean, logvar

def kl_divergence_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

if __name__ == "__main__":
    # ... (Load data, optimizer, etc.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define latent dimension
    latent_dim = 20  # You can adjust this value
    
    # Create a VAE model
    model = VAE(input_shape=784, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    ## dataset loader mnist 

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

    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.to(device)  # Move to device
            
            # Reset the gradients back to zero
            optimizer.zero_grad()
            
            # Compute reconstructions and latent variables
            reconstructions, mean, logvar = model(batch_features)
            
            # Compute reconstruction loss and KL divergence loss
            reconstruction_loss = criterion(reconstructions, batch_features)
            divergence_loss = kl_divergence_loss(mean, logvar)
            
            # Compute the total loss as a combination of the two losses
            loss = reconstruction_loss + divergence_loss
            
            # Backpropagation
            loss.backward()
            
            # Perform parameter update based on current gradients
            optimizer.step()
            
            # Accumulate losses for reporting
            total_loss += loss.item()
            recon_loss += reconstruction_loss.item()
            kl_loss += divergence_loss.item()
        
        # Compute the average epoch losses
        avg_total_loss = total_loss / len(train_loader)
        avg_recon_loss = recon_loss / len(train_loader)
        avg_kl_loss = kl_loss / len(train_loader)
        
        # Display the epoch losses
        print(f"Epoch: [{epoch+1}/{epochs}], Total Loss: {avg_total_loss:.6f}, Reconstruction Loss: {avg_recon_loss:.6f}, KL Divergence Loss: {avg_kl_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), "variational_autoencoder_model.pth")
    print("Model saved.")