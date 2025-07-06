import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

IQ_NORMALIZATION_FACTOR = 3.5

def VAE_loss(recon, data, mu, logvar, beta=1.0):
    # Reconstruction loss (e.g., MSE)
    recon_loss = torch.nn.functional.mse_loss(recon, data, reduction='sum')
    # print("recon_loss: ", recon_loss)
    # KL-divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div

# def VAE_loss(recon, data, mu, logvar, beta=1.0):
#     """
#     Computes the VAE loss with a cyclic phase loss for I and Q channel inputs.
    
#     Args:
#         recon (torch.Tensor): Reconstructed data, shape [batch_size, 2, subcarriers, symbols].
#                              The 2 channels represent I (real) and Q (imaginary) components.
#         data (torch.Tensor): Original input data, same shape as recon.
#         mu (torch.Tensor): Mean of the latent distribution, shape [batch_size, latent_dim].
#         logvar (torch.Tensor): Log-variance of the latent distribution, shape [batch_size, latent_dim].
#         beta (float, optional): Weight for the KL-divergence term. Defaults to 1.0.
    
#     Returns:
#         torch.Tensor: Total VAE loss (reconstruction loss + beta * KL-divergence).
#     """
#     # Convert I and Q channels to complex tensors
#     recon_complex = torch.complex(recon[:, 0], recon[:, 1])  # Shape: [batch_size, subcarriers, symbols]
#     data_complex = torch.complex(data[:, 0], data[:, 1])    # Shape: [batch_size, subcarriers, symbols]
    
#     # Reconstruction loss: MSE for amplitude, cyclic loss for phase
#     amplitude_loss = torch.nn.functional.mse_loss(
#         torch.abs(recon_complex), torch.abs(data_complex), reduction='sum'
#     )
    
#     # Cyclic phase loss: 1 - cos(phase_diff)
#     phase_diff = torch.angle(recon_complex) - torch.angle(data_complex)
#     phase_loss = (1 - torch.cos(phase_diff)).sum()

#     recon_loss = amplitude_loss
    
#     # mse_loss = torch.nn.functional.mse_loss(recon, data, reduction='sum')
#     # print("mse_loss: ", mse_loss)

#     # KL-divergence
#     kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
#     # Total loss
#     return recon_loss + beta * kl_div

# Generate synthetic WGN images (normal data)
def generate_wgn_images(num_samples, height=28, width=28, channels=1):
    images = np.random.normal(loc=0.5, scale=0.1, size=(num_samples, channels, height, width))
    images = np.clip(images, 0, 1)
    return torch.tensor(images, dtype=torch.float32)

# Generate anomalous images (WGN + bright straight line)
def generate_anomalous_images(num_samples, height=28, width=28, channels=1):
    images = np.random.normal(loc=0.5, scale=0.1, size=(num_samples, channels, height, width))
    for i in range(num_samples):
        # Add a horizontal straight line with high brightness (e.g., value 0.9)
        line_position = np.random.randint(height // 4, 3 * height // 4)  # Random row for variety
        for c in range(channels):
            images[i, c, line_position, :] = 0.9  # Set entire row to high brightness
    images = np.clip(images, 0, 1)
    return torch.tensor(images, dtype=torch.float32)

# Training function
def train_VAE(model, data_loader, epochs=20, learning_rate=1e-4, beta=1.0, device=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = model(data)
            loss = VAE_loss(recon_batch, data, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.6f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader.dataset):.6f}")

def AE_loss(recon, data):
    return nn.functional.mse_loss(recon, data, reduction='sum')

def train_AE(model, data_loader, epochs=20, learning_rate=1e-4, device=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            # print(data.shape)
            optimizer.zero_grad()
            recon_batch = model(data)
            # print(recon_batch.shape)
            loss = AE_loss(recon_batch, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.6f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader.dataset):.6f}")

# Anomaly detection
def detect_anomalies(model, data_loader, threshold, device=None):
    model.eval()
    anomalies = []
    recon_errors = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon, _, _ = model(data)
            recon_error = nn.functional.mse_loss(recon, data, reduction='none').mean(dim=(1,2,3))
            recon_errors.extend(recon_error.cpu().numpy())
            anomalies.extend((recon_error > threshold).cpu().numpy())
    return np.array(anomalies), np.array(recon_errors)

# Generate anomaly map
def generate_anomaly_map(original, reconstructed):
    anomaly_map = nn.functional.mse_loss(reconstructed, original, reduction='none')
    # print("anomaly_map.shape before sum", anomaly_map.shape)
    anomaly_map = torch.sum(anomaly_map, dim=(1), keepdim=True)
    # anomaly_map = torch.sqrt(anomaly_map)
    # print("anomaly_map.shape after sum", anomaly_map.shape)

    return anomaly_map

# Visualize anomaly maps
def visualize_anomaly_maps(model, test_data, num_samples=5, device=None):
    model.eval()
    with torch.no_grad():
        sample_data = test_data[:num_samples].to(device)
        recon, _, _ = model(sample_data)
        anomaly_maps = generate_anomaly_map(sample_data, recon)
        recon_errors = nn.functional.mse_loss(recon, sample_data, reduction='none').mean(dim=(1,2,3)).cpu().numpy()

        plt.figure(figsize=(15, 4))
        for i in range(num_samples):
            # Original
            plt.subplot(3, num_samples, i + 1)
            plt.imshow(sample_data[i].cpu().squeeze(), cmap='gray')
            plt.title("Anomalous")
            plt.axis('off')
            # Reconstructed
            plt.subplot(3, num_samples, i + 1 + num_samples)
            plt.imshow(recon[i].cpu().squeeze(), cmap='gray')
            plt.title(f"Error: {recon_errors[i]:.4f}")
            plt.axis('off')
            # Anomaly Map
            plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
            plt.imshow(anomaly_maps[i].cpu().squeeze(), cmap='gray')
            plt.title("Anomaly Map")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

import torch
import torch.nn as nn
import torch.optim as optim

def train_autoencoder(model, train_loader, val_loader=None, epochs=10, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the autoencoder model using provided DataLoader(s).
    
    Args:
        model: The autoencoder model (e.g., ConvAutoencoder).
        train_loader: PyTorch DataLoader for training data.
        val_loader: Optional PyTorch DataLoader for validation data (default: None).
        epochs: Number of training epochs (default: 10).
        lr: Learning rate for the optimizer (default: 1e-3).
        device: Device to train on ('cuda' or 'cpu', default: auto-detected).
    
    Returns:
        train_losses: List of average training losses per epoch.
        val_losses: List of average validation losses per epoch (if val_loader provided).
    """
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for data in train_loader:
            # Handle input (data can be input or (input, target))
            inputs = data[0] if isinstance(data, (list, tuple)) else data
            inputs = inputs.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
        
        # Average training loss
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
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
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.6f}" + 
              (f" | Val Loss: {epoch_val_loss:.6f}" if val_loader else ""))
    
    return train_losses, val_losses

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Create a text element for the PDF
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def gen_report(filename, anomaly_maps, anomaly_bins, verdicts, ground_truths):
    elements = []
    pdf_filename = f"./{filename}.pdf"
    pdf = SimpleDocTemplate(pdf_filename, pagesize=letter, title="Anomaly Maps")

    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, verdicts)
    precision = precision_score(ground_truths, verdicts, average='binary')
    recall = recall_score(ground_truths, verdicts, average='binary')
    f1 = f1_score(ground_truths, verdicts, average='binary')

    # Calculate confusion matrix with explicit labels to avoid warning
    cm = confusion_matrix(ground_truths, verdicts, labels=[0, 1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:")
    print(cm)

    # Add text information to the PDF
    text_content = f"""
    Overall Results:
    - Accuracy: {accuracy:.4f}
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1 Score: {f1:.4f}
    - Confusion Matrix:
      True Negatives: {cm[0,0]}
      False Positives: {cm[0,1]}
      False Negatives: {cm[1,0]}
      True Positives: {cm[1,1]}
    """

    styles = getSampleStyleSheet()
    text_paragraph = Paragraph(text_content, styles['Normal'])
    elements.append(text_paragraph)

    for i in range(len(anomaly_maps)):
        anomaly_map = anomaly_maps[i]
        anomaly_bin = anomaly_bins[i]
        verdict = verdicts[i]
        ground_truth = ground_truths[i]

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.sum(anomaly_map**2, axis=0), cmap='viridis')
        plt.colorbar()
        plt.title('Anomaly Map')

        plt.subplot(1, 2, 2)
        plt.imshow(np.sum(anomaly_bin**2, axis=0), cmap='viridis')
        plt.colorbar()
        plt.title(f'Anomaly Binary Masked')
        
        plt.suptitle(f'verdict: {verdict} ground_truth: {ground_truth}')

        # Save plot to BytesIO buffer
        bio = BytesIO()
        plt.savefig(bio, format='png', bbox_inches='tight')
        bio.seek(0)

        # Add to PDF
        plot_image = Image(bio, width=300, height=300)
        elements.append(plot_image)

        # Clean up
        # bio.close()
        plt.close()
    #             break
    #         break
    #     break
    # break

    print("Building PDF")
    pdf.build(elements)
    print("PDF built")

def plot_iq_recon(plt, iq, recon, anomaly_map, mse, anomaly_score):
    # loss = torch.nn.functional.mse_loss(recon, iq, reduction='sum')

    plt.figure(figsize=(30, 30))
    plt.subplot(3, 3, 1)
    to_be_plotted = np.sum(iq**2, axis=0)
    im = plt.imshow(to_be_plotted, cmap='viridis')
    plt.colorbar(im, ticks=[np.min(to_be_plotted), *np.linspace(np.min(to_be_plotted), np.max(to_be_plotted), 10)[1:-1], np.max(to_be_plotted)])
    plt.title(f'IQ, min: {np.min(to_be_plotted):.6f}, max: {np.max(to_be_plotted):.6f}')

    plt.subplot(3, 3, 2)
    to_be_plotted = iq[0, :, :]
    im = plt.imshow(to_be_plotted, cmap='viridis')
    plt.colorbar(im, ticks=[np.min(to_be_plotted), *np.linspace(np.min(to_be_plotted), np.max(to_be_plotted), 10)[1:-1], np.max(to_be_plotted)])
    plt.title(f'IQ / I, min: {np.min(to_be_plotted):.6f}, max: {np.max(to_be_plotted):.6f}')

    plt.subplot(3, 3, 3)
    to_be_plotted = iq[1, :, :]
    im = plt.imshow(to_be_plotted, cmap='viridis')
    plt.colorbar(im, ticks=[np.min(to_be_plotted), *np.linspace(np.min(to_be_plotted), np.max(to_be_plotted), 10)[1:-1], np.max(to_be_plotted)])
    plt.title(f'IQ / Q, min: {np.min(to_be_plotted):.6f}, max: {np.max(to_be_plotted):.6f}')

    plt.subplot(3, 3, 4)
    to_be_plotted = np.sum(recon**2, axis=0)
    im = plt.imshow(to_be_plotted, cmap='viridis')
    plt.colorbar(im, ticks=[np.min(to_be_plotted), *np.linspace(np.min(to_be_plotted), np.max(to_be_plotted), 10)[1:-1], np.max(to_be_plotted)])
    plt.title(f'Recon, min: {np.min(to_be_plotted):.6f}, max: {np.max(to_be_plotted):.6f}')

    plt.subplot(3, 3, 5)
    to_be_plotted = recon[0, :, :]
    im = plt.imshow(to_be_plotted, cmap='viridis')
    plt.colorbar(im, ticks=[np.min(to_be_plotted), *np.linspace(np.min(to_be_plotted), np.max(to_be_plotted), 10)[1:-1], np.max(to_be_plotted)])
    plt.title(f'Recon / I, min: {np.min(to_be_plotted):.6f}, max: {np.max(to_be_plotted):.6f}')

    plt.subplot(3, 3, 6)
    to_be_plotted = recon[1, :, :]
    im = plt.imshow(to_be_plotted, cmap='viridis')
    plt.colorbar(im, ticks=[np.min(to_be_plotted), *np.linspace(np.min(to_be_plotted), np.max(to_be_plotted), 10)[1:-1], np.max(to_be_plotted)])
    plt.title(f'Recon / Q, min: {np.min(to_be_plotted):.6f}, max: {np.max(to_be_plotted):.6f}')

    plt.subplot(3, 3, 7)
    to_be_plotted = anomaly_map
    im = plt.imshow(to_be_plotted, cmap='viridis')
    plt.colorbar(im, ticks=[np.min(to_be_plotted), *np.linspace(np.min(to_be_plotted), np.max(to_be_plotted), 10)[1:-1], np.max(to_be_plotted)])
    plt.title(f'Anomaly Map, min: {np.min(to_be_plotted):.6f}, max: {np.max(to_be_plotted):.6f}')

    plt.subplot(3, 3, 8)
    xlim = max(np.max(anomaly_map), 1)
    plt.xlim(0, xlim)
    plt.hist(anomaly_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black', range=(0, xlim))
    # plt.xscale('log')
    plt.title(f'Anomaly Map Histogram\nMean: {np.mean(anomaly_map):.6f}, Std: {np.std(anomaly_map):.6f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.ylim(1, 300*14*2)
    # print(anomaly_map.shape)

    plt.text(0.5, 0.3, f'RMSE: {np.sqrt(mse):.6f}\nAnomaly Score: {anomaly_score:.6f}', 
            horizontalalignment='left', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.subplot(3, 3, 9)
    xlim = max(np.max(np.sum(iq**2, axis=0)), 1)
    plt.xlim(0, xlim)
    plt.hist(np.sum(iq**2, axis=0).flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black', range=(0, xlim))
    # plt.xscale('log')
    plt.title(f'IQ Histogram\nMean: {np.mean(iq):.6f}, Std: {np.std(iq):.6f}')
    plt.xlabel('IQ Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.ylim(1, 300*14*2)
    
    return plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_autoencoder(
    model,
    dataloader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = None,
    save_path: str = None
):
    """
    Train a CNN autoencoder without progress bar.

    Args:
        model: The autoencoder model (must be on same device as inputs)
        dataloader: PyTorch DataLoader yielding batches of inputs
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to run on (e.g., torch.device('cuda'))
        save_path: Optional path to save trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.6f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

import scipy.io
import numpy as np
import torch
from matplotlib import pyplot as plt

def test_AE_no_plot(iq_data, ae, device):
    plt.ioff()  # Turn off interactive mode to prevent plots from showing
    
    # Convert to tensor and add batch dimension
    test_tensor = torch.tensor(iq_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # print(f"Tensor shape: {test_tensor.shape}")

    # Move to device
    test_tensor = test_tensor.to(device)

    # Encode the test sample
    with torch.no_grad():
        recon = ae(test_tensor)  # Get all outputs from AE
        # z = ae.reparameterize(mu, logvar)
        # recon = ae(test_tensor)

    # Calculate reconstruction error
    import torch.nn.functional as F
    mse_loss = F.mse_loss(recon, test_tensor)
    print(f"Reconstruction MSE: {np.sqrt(mse_loss.item()):.6f}")
    loss = torch.nn.functional.mse_loss(recon, test_tensor, reduction='sum')
    print("loss: ", loss)

    print(f"Original shape: {test_tensor.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    # print(f"Latent representation shape: {z.shape}")

    print(f"Recon min: {recon.min().item():.6f}")
    print(f"Recon max: {recon.max().item():.6f}")
    # print(z)

    # # Save test_tensor and recon to text files
    # np.savetxt('test_tensor.txt', test_tensor.cpu().numpy().flatten(), fmt='%.6f')
    # np.savetxt('recon.txt', recon.cpu().numpy().flatten(), fmt='%.6f')
    
    # # Also save the shapes for reference
    # with open('shapes.txt', 'w') as f:
    #     f.write(f"test_tensor shape: {test_tensor.shape}\n")
    #     f.write(f"recon shape: {recon.shape}\n")
    #     f.write(f"test_tensor flattened size: {test_tensor.cpu().numpy().flatten().shape}\n")
    #     f.write(f"recon flattened size: {recon.cpu().numpy().flatten().shape}\n")
    # # Visualize the original vs reconstructed
    # plt.figure(figsize=(15, 5))

    # Get the data for consistent colorbar scaling
    original_data = test_tensor.cpu().numpy()[0]
    recon_data = recon.cpu().numpy()[0]
    anomaly_map = generate_anomaly_map(test_tensor, recon)
    anomaly_map = anomaly_map.detach().cpu().numpy()[0].squeeze()
    print(f'anomaly_map.shape: {anomaly_map.shape}')

    # Find global min and max for consistent colorbar
    vmin = min(original_data.min(), recon_data.min())
    vmax = max(original_data.max(), recon_data.max())

    plot_iq_recon(plt, original_data, recon_data, anomaly_map, mse_loss.item(), np.percentile(anomaly_map, 95))
    
    # plt.show()
    return plt

def test_AE(path, ae, device):
    test_data = scipy.io.loadmat(path)
    # Extract the IQ data
    iq_data = test_data['noiseInterferenceGrid_IQ']  # Assuming the key is 'data', adjust if different
    # print(f"Test data shape: {iq_data.shape}")

    iq_data = iq_data.transpose(2, 0, 1)  # Convert from (H,W,C) to (C,H,W)
    iq_data = iq_data[:2, :, :].astype(np.float32)  # Convert to float32
    # iq_data = (iq_data - np.min(iq_data)) / (np.max(iq_data) - np.min(iq_data))
    # iq_data = iq_data / np.max(np.abs(iq_data))
    # iq_data = iq_data / IQ_NORMALIZATION_FACTOR
    # iq_data = np.clip(iq_data, -1, 1)
    # print(f"Normalized data shape: {iq_data.shape}")

    plt = test_AE_no_plot(iq_data, ae, device)
    plt.show()

def gen_report_db(datasets, ae, device, file_path, num_samples=-1):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Image
    from io import BytesIO

    pdf = SimpleDocTemplate(file_path, pagesize=letter)
    elements = []

    print(f"datasets.shape: {datasets.shape}")
    for index, iq_data in enumerate(datasets):
        print(f"Processing {index}/{len(datasets)}")
        plt = test_AE_no_plot(iq_data, ae, device)
        bio = BytesIO()
        plt.savefig(bio, format='png', bbox_inches='tight')
        bio.seek(0)
        plot_image = Image(bio, width=300, height=300)
        elements.append(plot_image)
        if num_samples >= 0 and index > num_samples:
            break

    print("Building PDF")
    pdf.build(elements)
    print("PDF built at: ", file_path)
