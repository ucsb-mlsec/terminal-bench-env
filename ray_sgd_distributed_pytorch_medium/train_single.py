import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint

# Set seed for reproducibility
torch.manual_seed(42)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_func(config):
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    X = torch.randn(1000, 10)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1
    
    dataset = TensorDataset(X, y)
    
    # Prepare distributed dataloader
    from ray.train.torch import prepare_data_loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader = prepare_data_loader(dataloader)
    
    # Create model
    model = SimpleModel()
    
    # Prepare model for distributed training
    from ray.train.torch import prepare_model
    model = prepare_model(model)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Report metrics to Ray
        train.report({"loss": avg_loss, "epoch": epoch + 1})
    
    # Save checkpoint
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    checkpoint_dict = {
        "model_state_dict": model_state,
        "epoch": epochs
    }
    
    # Report final checkpoint
    checkpoint = Checkpoint.from_dict(checkpoint_dict)
    train.report({"loss": avg_loss, "epoch": epochs}, checkpoint=checkpoint)
    
    print("Training completed")

if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Configure distributed training with 2 workers
    scaling_config = ScalingConfig(num_workers=2, use_gpu=False)
    
    # Create trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config
    )
    
    # Run training
    result = trainer.fit()
    
    # Save the final checkpoint
    if result.checkpoint:
        checkpoint_dict = result.checkpoint.to_dict()
        torch.save(checkpoint_dict, "/workspace/model_checkpoint.pt")
        print("Checkpoint saved to /workspace/model_checkpoint.pt")
    
    # Write status file
    with open("/workspace/status.txt", "w") as f:
        f.write("workers=2\n")
        f.write("epochs=10\n")
        f.write("completed=yes\n")
    
    print("Status file created")
    
    # Shutdown Ray
    ray.shutdown()