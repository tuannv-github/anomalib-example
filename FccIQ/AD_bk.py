import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 150, 7]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [batch, 128, 75, 4]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [batch, 256, 38, 2]
            nn.ReLU(),
            nn.Flatten(),  # [batch, 256 * 38 * 2]
            nn.Linear(256 * 38 * 2, latent_dim)  # [batch, latent_dim]
        )
        
        # Decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * 38 * 2),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)), # [batch, 128, 75, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)), # [batch, 64, 150, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)), # [batch, 2, 300, 14]
            nn.Sigmoid()
        )
        
        # Validate shapes
        self._validate_shapes()

    def _validate_shapes(self):
        test_input = torch.randn(1, 2, 300, 14)
        self.eval()
        with torch.no_grad():
            test_output = self.forward(test_input)
        if test_output.shape != test_input.shape:
            raise ValueError(f"Output shape {test_output.shape} does not match input shape {test_input.shape}")
        print(f"Shape validation passed: Input {test_input.shape}, Output {test_output.shape}")

    def forward(self, x):
        latent = self.encoder(x)
        x_reconstructed = self.decoder_linear(latent)
        x_reconstructed = x_reconstructed.view(-1, 256, 38, 2)
        x_reconstructed = self.decoder(x_reconstructed)
        return x_reconstructed

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_autoencoder(model, train_loader, val_loader=None, epochs=50, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.apply(weights_init)
    criterion = lambda x, y: 0.5 * nn.MSELoss()(x, y) + 0.5 * nn.BCELoss()(x, y)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for data in train_loader:
            inputs = data[0] if isinstance(data, (list, tuple)) else data
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            if epoch % 5 == 0:
                output_mean = outputs.mean().item()
                output_std = outputs.std().item()
                print(f"Epoch {epoch+1}, Batch Mean Output: {output_mean:.4f}, Std: {output_std:.4f}")
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        epoch_val_loss = 0.0
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    inputs = data[0] if isinstance(data, (list, tuple)) else data
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    running_val_loss += loss.item() * inputs.size(0)
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.6f}" + 
              (f" | Val Loss: {epoch_val_loss:.6f}" if val_loader else ""))
    
    return train_losses, val_losses

# Example usage
if __name__ == "__main__":
    try:
        # Replace with your actual train_loader and val_loader
        from torch.utils.data import TensorDataset, DataLoader
        dummy_data = torch.randn(100, 2, 300, 14)
        train_dataset = TensorDataset(dummy_data, dummy_data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = ConvAutoencoder(latent_dim=128)
        train_losses, val_losses = train_autoencoder(
            model, train_loader, epochs=50, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Visualize reconstructions
        # visualize_reconstructions(model, train_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Error: {str(e)}")