#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from data_utils import DataLoader

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    return rank, world_size

def load_config():
    with open('/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    rank, world_size = setup_distributed()
    
    print(f"[Rank {rank}] Process group initialized successfully")
    
    config = load_config()
    
    model = SimpleNet(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_classes=config['model']['num_classes']
    )
    
    ddp_model = DDP(model)
    
    if rank == 0:
        dist.barrier()
        print(f"[Rank {rank}] Model wrapped with DDP")
    
    train_loader = DataLoader(
        data_path=config['data']['path'],
        batch_size=config['training']['batch_size'],
        rank=rank,
        world_size=world_size
    )
    
    optimizer = torch.optim.SGD(
        ddp_model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=0.9
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"[Rank {rank}] Starting training loop")
    
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(num_epochs):
        ddp_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = ddp_model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"[Rank {rank}] Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
    
    print(f"[Rank {rank}] Training completed")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()