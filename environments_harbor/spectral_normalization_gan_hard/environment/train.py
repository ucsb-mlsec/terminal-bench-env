import torch
import torch.nn as nn
import torch.optim as optim
from models import Discriminator

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize discriminator
    discriminator = Discriminator().to(device)
    
    # Optimizer
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training setup complete")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())}")
    
    # Training loop would go here
    # Example forward pass to verify discriminator works
    batch_size = 16
    sample_input = torch.randn(batch_size, 3, 64, 64).to(device)
    
    with torch.no_grad():
        output = discriminator(sample_input)
        print(f"Sample output shape: {output.shape}")
    
    print("Discriminator is ready for training")

if __name__ == '__main__':
    main()