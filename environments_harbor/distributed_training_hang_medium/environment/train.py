#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import yaml
import argparse
import os
import sys


class SimpleModel(nn.Module):
    """Simple neural network for testing DDP"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class SimpleDataset(Dataset):
    """Simple synthetic dataset for testing"""
    def __init__(self, size=1000, input_size=784, num_classes=10):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def init_process(rank, world_size, config):
    """Initialize the distributed environment and run training"""
    try:
        # Set environment variables for this process
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Get configuration parameters
        backend = config.get('backend', 'gloo')
        init_method = config.get('init_method', 'env://')
        master_addr = config.get('master_addr', 'localhost')
        master_port = config.get('master_port', '29500')
        
        # Set master address and port
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        print(f"Process {rank}: Initializing with backend={backend}, init_method={init_method}")
        print(f"Process {rank}: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        
        # Initialize the process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        
        print(f"Process {rank} initialized successfully")
        
        # Create model and move to CPU
        model = SimpleModel()
        
        # Wrap model with DDP
        ddp_model = DDP(model)
        
        # Create dataset and dataloader
        dataset = SimpleDataset(size=1000)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 32),
            sampler=sampler,
            num_workers=0
        )
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        num_epochs = config.get('num_epochs', 3)
        print(f"Starting training on rank {rank}")
        
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            ddp_model.train()
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0 and rank == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Rank {rank}, Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print(f"Process {rank} completed training")
        
        # Cleanup
        dist.destroy_process_group()
        print(f"Process {rank} cleaned up")
        
    except Exception as e:
        print(f"Process {rank} encountered error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
        print(f"Configuration: {config}")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Get world size from config
    world_size = config.get('world_size', 4)
    
    print(f"Starting distributed training with {world_size} processes")
    
    # Spawn processes
    mp.spawn(
        init_process,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
    
    print("All processes completed successfully")


if __name__ == '__main__':
    main()
```

Now let me examine the configuration file and other components to identify the issue:

<tool_calls>
<tool_call>
{"name": "execute_bash", "arguments": {"command": "cd /workspace/training && ls -la"}}
</tool_call>
<tool_call>
{"name": "execute_bash", "arguments": {"command": "cd /workspace/training && cat config.yaml"}}
</tool_call>
<tool_call>
{"name": "execute_bash", "arguments": {"command": "cd /workspace/training && cat start_training.sh"}}
</tool_call>
<tool_call>
{"name": "execute_bash", "arguments": {"command": "cd /workspace/training && cat requirements.txt"}}
</tool_call>
</tool_calls>