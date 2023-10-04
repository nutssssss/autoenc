import torch
import torch.nn as nn
from torch import optim
import torchvision

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


# def calculate_accuracy(model, data_loader, device):
#     num_samples = len(data_loader.dataset)
#     correct = 0

#     with torch.no_grad():
#         for batch_features, _ in data_loader:
#             batch_features = batch_features.view(-1, 784).to(device)
#             outputs = model(batch_features)
#             mse = nn.MSELoss(reduction='none')(outputs, batch_features)
#             mse = mse.sum(dim=1)  # Sum the reconstruction errors along the features
#             correct += (mse < 0.5).sum().item()  # Count correct reconstructions (you can adjust the threshold)

#     accuracy = correct / num_samples
#     return accuracy
if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a model from `AE` autoencoder class
    # Load it to the specified device, either GPU or CPU
    model = AE(input_shape=784).to(device)

    # Create an optimizer object (Adam optimizer with learning rate 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Mean squared error loss
    criterion = nn.MSELoss()

    # MNIST dataset loader
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
        loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, 784).to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(train_loader)
        print("Epoch : {}/{}, Loss = {:.6f}".format(epoch + 1, epochs, loss))

    # Calculate the reconstruction loss as a percentage
    max_possible_loss = 784.0  # The maximum possible loss (if all pixel values are 1)
    loss = loss*100
    reconstruction_loss_percentage = (loss / max_possible_loss) * 100.0

    print("Reconstruction accuracy: {:.2f}%".format(100 - reconstruction_loss_percentage))

    torch.save(model.state_dict(), "autoencoder_model.pth")
    print("Model saved.")
