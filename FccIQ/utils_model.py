import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image
from io import BytesIO
from utils import *

def gen_model_report(model, train_loader, device, pdf_filename="anomaly_maps.pdf"):
    # Initialize PDF
    pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []

    for batch_idx, iqs in enumerate(train_loader):
        print(f"Processing batch {batch_idx}/{len(train_loader)}")
        print(iqs.shape)
        iqs = iqs.to(device)
        
        with torch.no_grad():
            recons, mu, logvar = model(iqs)
            anomaly_maps = generate_anomaly_map(iqs, recons)
            
            print("Saving to file...")
            iqs = iqs.cpu().numpy()
            for i in range(iqs.shape[0]):
                iq = iqs[i]
                recon = recons[i].cpu().numpy()
                anomaly_map = anomaly_maps[i].cpu().numpy().squeeze()
                # print("iq.shape: ", iq.shape)
                # print("recon.shape: ", recon.shape)
                # print("anomaly_map.shape: ", anomaly_map.shape)
                
                # Create figure
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 3, 1)
                plt.imshow(np.sum(iq**2, axis=0), cmap='viridis')
                plt.colorbar()
                plt.title('IQ')
                
                plt.subplot(1, 3, 2)
                plt.imshow(np.sum(recon**2, axis=0), cmap='viridis')
                plt.colorbar()
                plt.title('Recon')
                
                plt.subplot(1, 3, 3)
                plt.imshow(anomaly_map, cmap='gray')
                plt.colorbar()
                plt.title('Anomaly Map')
                
                # Save plot to BytesIO buffer
                bio = BytesIO()
                plt.savefig(bio, format='png', bbox_inches='tight')
                bio.seek(0)
                
                # Add to PDF
                plot_image = Image(bio, width=300, height=300)
                elements.append(plot_image)
                
                plt.close()
                # break
            break
        break

    print("Building PDF")
    pdf.build(elements)
    print(f"Saved anomaly plots to ./{pdf_filename}")
