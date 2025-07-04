import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        input_dim = 2 * 300 * 14  # 8400

        # Encoder: 8400 → 4200 → 512
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # 8400 → 4200
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1024)
        )

        # Decoder: 512 → 4200 → 8400
        self.decoder = nn.Sequential(
            nn.Linear(1024, input_dim // 2),  # 512 → 4200
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),  # 4200 → 8400
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten safely
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.reshape(x.size(0), 2, 300, 14)  # Reshape safely
        return decoded




if __name__ == "__main__":
    ae = AE()
    ae.eval()
    with torch.no_grad():
        recon= ae(torch.randn(1, 2, 300, 14))
        print(recon.shape)