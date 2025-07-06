import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, input_channels=2, latent_dim=1024, img_height=300, img_width=14):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width
        self.input_dim = input_channels * img_height * img_width  # 2 * 300 * 14 = 8400

        # FC Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 4096),
            nn.Tanh(),
            nn.Linear(4096, 2048),
            nn.Tanh(),
            nn.Linear(2048, latent_dim),  # [batch_size, 1024]
            nn.Tanh()
        )

        # FC Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.Tanh(),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Linear(4096, self.input_dim),  # [batch_size, 8400]
            # nn.Tanh()  # Output in [-1, 1]
        )

        # Verify output shape
        test_input = torch.zeros(1, input_channels, img_height, img_width)
        recon = self.forward(test_input)
        expected_shape = (input_channels, img_height, img_width)
        if recon.shape[1:] != expected_shape:
            raise ValueError(
                f"Output shape {recon.shape[1:]} does not match expected {expected_shape}"
            )
        print(f"Output shape verified: {recon.shape[1:]}")
        self.print_parameter_count()

    def count_parameters(self):
        """Count the number of parameters in the model in millions."""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params / 1_000_000
    
    def print_parameter_count(self):
        """Print the number of parameters in millions."""
        param_count = self.count_parameters()
        print(f"Number of parameters: {param_count:.2f}M")

    def encode(self, x):
        return self.encoder(x)  # [batch_size, 1024]

    def decode(self, z):
        recon_flat = self.decoder(z)  # [batch_size, 8400]
        recon_x = recon_flat.view(-1, 2, self.img_height, self.img_width)  # [batch_size, 2, 300, 14]
        return recon_x

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x

    def loss_function(self, recon_x, x):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        return recon_loss

if __name__ == "__main__":
    ae = AE()
    ae.eval()
    with torch.no_grad():
        recon = ae(torch.randn(1, 2, 300, 14))
        print(f"recon.shape: {recon.shape}")