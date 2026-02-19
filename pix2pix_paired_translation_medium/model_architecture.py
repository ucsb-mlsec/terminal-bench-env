#!/usr/bin/env python3

import torch
import torch.nn as nn
import yaml
import sys


class Generator(nn.Module):
    """U-Net style generator for Pix2Pix"""
    
    def __init__(self, num_layers, num_filters, input_channels, output_channels, image_size):
        super(Generator, self).__init__()
        
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not isinstance(num_filters, int) or num_filters <= 0:
            raise ValueError(f"num_filters must be a positive integer, got {num_filters}")
        if not isinstance(input_channels, int) or input_channels <= 0:
            raise ValueError(f"input_channels must be a positive integer, got {input_channels}")
        if not isinstance(output_channels, int) or output_channels <= 0:
            raise ValueError(f"output_channels must be a positive integer, got {output_channels}")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError(f"image_size must be a positive integer, got {image_size}")
        
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.image_size = image_size
        
        # Encoder (downsampling)
        self.encoder_layers = nn.ModuleList()
        in_channels = input_channels
        out_channels = num_filters
        
        for i in range(num_layers):
            if i == 0:
                # First layer without batch norm
                self.encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            else:
                self.encoder_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)
        
        # Decoder (upsampling)
        self.decoder_layers = nn.ModuleList()
        in_channels = self.encoder_layers[-1][0].out_channels
        
        for i in range(num_layers):
            if i == 0:
                out_channels = in_channels // 2
            else:
                out_channels = in_channels // 2
            
            if i == num_layers - 1:
                # Last layer outputs to output_channels
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels * 2 if i > 0 else in_channels, 
                                     output_channels if i == num_layers - 1 else out_channels,
                                     kernel_size=4, stride=2, padding=1),
                    nn.Tanh() if i == num_layers - 1 else nn.ReLU(inplace=True)
                ))
            else:
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels * 2, out_channels,
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
            in_channels = out_channels
    
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_outputs.append(x)
        
        # Decoder with skip connections
        x = encoder_outputs[-1]
        for i, layer in enumerate(self.decoder_layers):
            if i > 0:
                # Skip connection
                skip = encoder_outputs[-(i + 1)]
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        return x


class Discriminator(nn.Module):
    """PatchGAN discriminator for Pix2Pix"""
    
    def __init__(self, num_layers, num_filters, input_channels):
        super(Discriminator, self).__init__()
        
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not isinstance(num_filters, int) or num_filters <= 0:
            raise ValueError(f"num_filters must be a positive integer, got {num_filters}")
        if not isinstance(input_channels, int) or input_channels <= 0:
            raise ValueError(f"input_channels must be a positive integer, got {input_channels}")
        
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.input_channels = input_channels
        
        # Build discriminator layers
        layers = []
        in_channels = input_channels
        out_channels = num_filters
        
        for i in range(num_layers):
            if i == 0:
                # First layer without batch norm
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)
        
        # Final layer
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Pix2PixModel(nn.Module):
    """Main Pix2Pix model combining generator and discriminator"""
    
    def __init__(self, config):
        super(Pix2PixModel, self).__init__()
        
        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary")
        
        if 'model' not in config:
            raise ValueError("config must contain 'model' key")
        
        if 'generator' not in config['model']:
            raise ValueError("config['model'] must contain 'generator' key")
        
        if 'discriminator' not in config['model']:
            raise ValueError("config['model'] must contain 'discriminator' key")
        
        if 'training' not in config:
            raise ValueError("config must contain 'training' key")
        
        # Validate generator config
        gen_config = config['model']['generator']
        required_gen_keys = ['num_layers', 'num_filters', 'input_channels', 'output_channels', 'image_size']
        
        for key in required_gen_keys:
            if key not in gen_config:
                raise ValueError(f"generator config missing required key: '{key}'")
        
        # Validate discriminator config
        disc_config = config['model']['discriminator']
        required_disc_keys = ['num_layers', 'num_filters', 'input_channels']
        
        for key in required_disc_keys:
            if key not in disc_config:
                raise ValueError(f"discriminator config missing required key: '{key}'")
        
        # Validate training config
        training_config = config['training']
        
        if 'learning_rate' not in training_config:
            raise ValueError("training config missing required key: 'learning_rate'")
        
        if 'batch_size' not in training_config:
            raise ValueError("training config missing required key: 'batch_size'")
        
        # Validate learning_rate
        lr = training_config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"learning_rate must be a positive number, got {lr}")
        
        # Validate batch_size
        bs = training_config['batch_size']
        if not isinstance(bs, int) or bs <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {bs}")
        
        # Store config
        self.config = config
        self.learning_rate = lr
        self.batch_size = bs
        
        # Initialize generator
        try:
            self.generator = Generator(
                num_layers=gen_config['num_layers'],
                num_filters=gen_config['num_filters'],
                input_channels=gen_config['input_channels'],
                output_channels=gen_config['output_channels'],
                image_size=gen_config['image_size']
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize generator: {str(e)}")
        
        # Initialize discriminator
        try:
            self.discriminator = Discriminator(
                num_layers=disc_config['num_layers'],
                num_filters=disc_config['num_filters'],
                input_channels=disc_config['input_channels']
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize discriminator: {str(e)}")
    
    def forward(self, x):
        return self.generator(x)


if __name__ == "__main__":
    try:
        # Load configuration
        with open('config/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model instance
        model = Pix2PixModel(config)
        
        print("Model initialized successfully!")
        
        # Write success report
        with open('/tmp/config_fix_report.txt', 'w') as f:
            f.write("INITIALIZATION_STATUS: SUCCESS\n")
            f.write("ISSUES_FOUND: 0\n")
            f.write("PRIMARY_FIX: No fixes needed, configuration was correct\n")
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
        with open('/tmp/config_fix_report.txt', 'w') as f:
            f.write("INITIALIZATION_STATUS: FAILED\n")
            f.write("ISSUES_FOUND: 1\n")
            f.write("PRIMARY_FIX: Configuration file not found\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        with open('/tmp/config_fix_report.txt', 'w') as f:
            f.write("INITIALIZATION_STATUS: FAILED\n")
            f.write("ISSUES_FOUND: 1\n")
            f.write(f"PRIMARY_FIX: {str(e)[:100]}\n")
        sys.exit(1)