import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_channels=2, latent_dim=16, img_height=300, img_width=14):
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
            # Third conv: kernel=(5,3), stride=(2,1), padding=(2,1)
            h = (h - 5 + 2 * 2) // 2 + 1  # Height after third conv
            w = (w - 3 + 2 * 1) // 1 + 1  # Width after third conv
            return h, w

        feat_h, feat_w = calc_feature_size(img_height, img_width)
        self.feature_dim = 64 * feat_h * feat_w  # 64 filters in last conv layer

        # Debug feature map
        print(f"Calculated feature map: [64, {feat_h}, {feat_w}], feature_dim={self.feature_dim}")

        # Validate feature map size
        if feat_h <= 0 or feat_w <= 0:
            raise ValueError(f"Invalid feature map size: {feat_h}x{feat_w}")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 2**12),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(2**12, latent_dim)
        self.fc_logvar = nn.Linear(2**12, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.feature_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, feat_h, feat_w)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(0, 0)
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(1, 0)
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                16, input_channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(1, 0)
            ),
            nn.Tanh()
        )

        # Verify output size
        test_input = torch.zeros(1, input_channels, img_height, img_width)
        recon, _, _ = self.forward(test_input)
        expected_shape = (input_channels, img_height, img_width)
        if recon.shape[1:] != expected_shape:
            raise ValueError(
                f"Output shape {recon.shape[1:]} does not match expected {expected_shape}"
            )
        print(f"Output shape verified: {recon.shape[1:]}")

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

if __name__ == "__main__":
    ae = AE(input_channels=2, latent_dim=1024, img_height=300, img_width=14)
    ae.eval()
    with torch.no_grad():
        recon, mu, logvar = ae(torch.randn(1, 2, 300, 14))
        print(recon.shape, mu.shape, logvar.shape)