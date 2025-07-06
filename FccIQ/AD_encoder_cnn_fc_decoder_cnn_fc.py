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
        self.latent_dim_per_encoder = latent_dim // 2  # 512

        # Calculate feature map size for CNN encoder and decoder
        def calc_feature_size(h, w):
            # First conv: kernel=(5,3), stride=(2,1), padding=(2,1)
            h = (h - 5 + 2 * 2) // 2 + 1  # 150
            w = (w - 3 + 2 * 1) // 1 + 1  # 14
            # Second conv: kernel=(5,3), stride=(2,1), padding=(2,1)
            h = (h - 5 + 2 * 2) // 2 + 1  # 75
            w = (w - 3 + 2 * 1) // 1 + 1  # 14
            return h, w

        feat_h, feat_w = calc_feature_size(img_height, img_width)  # [75, 14]
        self.feature_dim_cnn = 32 * feat_h * feat_w  # 32 * 75 * 14 = 33600

        # CNN Encoder (Variational)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Flatten()  # [batch_size, 33600]
        )
        self.cnn_fc_mu = nn.Linear(self.feature_dim_cnn, self.latent_dim_per_encoder)
        self.cnn_fc_var = nn.Linear(self.feature_dim_cnn, self.latent_dim_per_encoder)

        # FC Encoder (Non-variational)
        self.fc_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, self.latent_dim_per_encoder),
            nn.Tanh()  # [batch_size, 512]
        )

        # Decoder: CNN Branch
        self.decoder_cnn_input = nn.Linear(latent_dim, self.feature_dim_cnn)
        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(1, (32, feat_h, feat_w)),
            nn.ConvTranspose2d(
                32, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(1, 0)
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                16, input_channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), output_padding=(1, 0)
            ),
            nn.Tanh(),
            nn.Conv2d(
                input_channels, input_channels, kernel_size=15, stride=1, padding=7,
                groups=input_channels, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Decoder: FC Branch
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.Tanh(),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Linear(4096, self.input_dim),  # [batch_size, 8400]
            nn.Tanh()  # Output in [-1, 1]
        )

        # Fusion layer for decoder outputs
        self.decoder_fusion = nn.Linear(2 * self.input_dim, self.input_dim)  # Fuse to [batch_size, 8400]

        # Verify output shape
        test_input = torch.zeros(1, input_channels, img_height, img_width)
        recon, mu, log_var, z = self.forward(test_input)
        expected_shape = (input_channels, img_height, img_width)
        if recon.shape[1:] != expected_shape:
            raise ValueError(
                f"Output shape {recon.shape[1:]} does not match expected {expected_shape}"
            )
        print(f"Output shape verified: {recon.shape[1:]}")

    def encode(self, x):
        # CNN encoder (variational)
        cnn_features = self.cnn_encoder(x)  # [batch_size, 33600]
        mu = self.cnn_fc_mu(cnn_features)  # [batch_size, 512]
        log_var = self.cnn_fc_var(cnn_features)  # [batch_size, 512]

        # FC encoder (non-variational)
        z_fc = self.fc_encoder(x)  # [batch_size, 512]

        return mu, log_var, z_fc

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # CNN branch
        cnn_h = self.decoder_cnn_input(z)
        cnn_out = self.decoder_cnn(cnn_h)  # [batch_size, 2, 300, 14]
        cnn_out_flat = cnn_out.view(cnn_out.size(0), -1)  # [batch_size, 8400]

        # FC branch
        fc_out = self.decoder_fc(z)  # [batch_size, 8400]

        # Fuse decoder outputs
        fused_out = torch.cat([cnn_out_flat, fc_out], dim=1)  # [batch_size, 16800]
        fused_out = self.decoder_fusion(fused_out)  # [batch_size, 8400]
        recon_x = fused_out.view(-1, 2, self.img_height, self.img_width)  # [batch_size, 2, 300, 14]
        return recon_x

    def forward(self, x):
        # Encode
        mu, log_var, z_fc = self.encode(x)
        z_cnn = self.reparameterize(mu, log_var)  # [batch_size, 512]
        z = torch.cat([z_cnn, z_fc], dim=1)  # [batch_size, 1024]
        
        # Decode
        recon_x = self.decode(z)
        return recon_x, mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var, kld_weight=0.1):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kld_weight * kld_loss
        return total_loss, recon_loss, kld_loss

if __name__ == "__main__":
    vae = AE()
    vae.eval()
    with torch.no_grad():
        recon, mu, log_var, z = vae(torch.randn(1, 2, 300, 14))
        print(f"recon.shape: {recon.shape}")
        print(f"mu.shape: {mu.shape}")
        print(f"log_var.shape: {log_var.shape}")
        print(f"z.shape: {z.shape}")