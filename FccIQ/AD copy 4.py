from reportlab.lib.pagesizes import landscape
import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, input_channels=2, latent_dim=2**10, img_height=300, img_width=14):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width

        # Calculate feature map size after convolutions
        def calc_feature_size(h, w):
            # First conv: kernel=(5,3), stride=(2,1), padding=(2,1)
            h = (h - 5 + 2 * 2) // 2 + 1  # Height after first conv
            w = (w - 3 + 2 * 1) // 1 + 1  # Width after first conv
            # Second conv: kernel=(5,3), stride=(2,1), padding=(2,1)
            h = (h - 5 + 2 * 2) // 2 + 1  # Height after second conv
            w = (w - 3 + 2 * 1) // 1 + 1  # Width after second conv
            return h, w

        feat_h, feat_w = calc_feature_size(img_height, img_width)
        self.feature_dim = 32 * feat_h * feat_w  # 32 filters in last conv layer

        # Debug feature map
        print(f"Calculated feature map: [32, {feat_h}, {feat_w}], feature_dim={self.feature_dim}")

        # Validate feature map size
        if feat_h <= 0 or feat_w <= 0:
            raise ValueError(f"Invalid feature map size: {feat_h}x{feat_w}")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Flatten()
        )
        # Variational bottleneck layers
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_var = nn.Linear(self.feature_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.feature_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, feat_h, feat_w)),
            nn.ConvTranspose2d(
                32, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(1, 0)
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                16, input_channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(1, 0)
            ),
            nn.Tanh(),

            nn.Conv2d(input_channels, input_channels, kernel_size=15, stride=1, padding=7, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Tanh(),
        )

        # Verify output size
        test_input = torch.zeros(1, input_channels, img_height, img_width)
        recon, mu, log_var, mean = self.forward(test_input)
        expected_shape = (input_channels, img_height, img_width)
        if recon.shape[1:] != expected_shape:
            raise ValueError(
                f"Output shape {recon.shape[1:]} does not match expected {expected_shape}"
            )
        print(f"Output shape verified: {recon.shape[1:]}")

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mean = torch.mean(x, dim=2, keepdim=True)
        x = x - mean

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        
        recon_x = recon_x + mean
        return recon_x, mu, log_var, mean

    def loss_function(self, recon_x, x, mu, log_var, kld_weight=1.0):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + kld_weight * kld_loss
        
        return total_loss, recon_loss, kld_loss

if __name__ == "__main__":
    ae = AE()
    ae.eval()
    with torch.no_grad():
        recon, mu, log_var, mean = ae(torch.randn(1, 2, 300, 14))
        print(f"recon.shape: {recon.shape}")
        print(f"mu.shape: {mu.shape}")
        print(f"log_var.shape: {log_var.shape}")
        print(f"mean.shape: {mean.shape}")