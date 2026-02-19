#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Import existing modules
import sys
sys.path.insert(0, '/workspace/legacy_project')
from simple_model import SimpleMLP
from synthetic_data import generate_synthetic_data


class LitSimpleMLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        
        # Use the existing model from simple_model.py
        self.model = SimpleMLP(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main():
    # Load configuration
    config_path = '/workspace/legacy_project/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    pl.seed_everything(seed)
    
    # Extract hyperparameters
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_size = config['output_size']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    train_size = config.get('train_size', 1000)
    val_size = config.get('val_size', 200)
    
    # Generate synthetic data using existing function
    X_train, y_train = generate_synthetic_data(train_size, input_size, output_size, seed=seed)
    X_val, y_val = generate_synthetic_data(val_size, input_size, output_size, seed=seed + 1)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize Lightning model
    model = LitSimpleMLP(input_size, hidden_size, output_size, learning_rate)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='/workspace/checkpoints',
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    # Setup logger
    logger = CSVLogger(save_dir='/workspace', name='lightning_logs')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='cpu',
        devices=1,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Extract final metrics
    final_train_loss = None
    final_val_loss = None
    
    # Get metrics from trainer's logged metrics
    if hasattr(trainer.callback_metrics, 'items'):
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            final_train_loss = metrics['train_loss'].item()
        if 'val_loss' in metrics:
            final_val_loss = metrics['val_loss'].item()
    
    # Fallback: manually compute final losses if needed
    if final_train_loss is None or final_val_loss is None:
        model.eval()
        with torch.no_grad():
            # Compute training loss
            train_losses = []
            for batch in train_loader:
                x, y = batch
                y_hat = model(x)
                loss = model.criterion(y_hat, y)
                train_losses.append(loss.item())
            final_train_loss = sum(train_losses) / len(train_losses)
            
            # Compute validation loss
            val_losses = []
            for batch in val_loader:
                x, y = batch
                y_hat = model(x)
                loss = model.criterion(y_hat, y)
                val_losses.append(loss.item())
            final_val_loss = sum(val_losses) / len(val_losses)
    
    # Check if checkpoint was saved
    checkpoint_dir = '/workspace/checkpoints'
    checkpoint_saved = 'no'
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if len(checkpoint_files) > 0:
            checkpoint_saved = 'yes'
    
    # Write verification file
    with open('/workspace/verification.txt', 'w') as f:
        f.write(f'final_train_loss={final_train_loss:.4f}\n')
        f.write(f'final_val_loss={final_val_loss:.4f}\n')
        f.write(f'checkpoint_saved={checkpoint_saved}\n')
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Checkpoint Saved: {checkpoint_saved}")
    print("="*60)


if __name__ == '__main__':
    main()