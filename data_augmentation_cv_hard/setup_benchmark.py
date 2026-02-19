#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_benchmark_data():
    """Create synthetic benchmark images"""
    os.makedirs('/workspace/benchmark_data', exist_ok=True)
    
    # Create 120 synthetic images
    for i in range(120):
        # Random size between 256x256 and 512x512
        size = random.randint(256, 512)
        
        # Create synthetic image with random patterns
        img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        
        # Add some structure (colored rectangles)
        img = Image.fromarray(img_array, 'RGB')
        draw = ImageDraw.Draw(img)
        
        # Draw some random rectangles for visual variety
        for _ in range(5):
            x1 = random.randint(0, size - 50)
            y1 = random.randint(0, size - 50)
            x2 = x1 + random.randint(30, 100)
            y2 = y1 + random.randint(30, 100)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Save as JPEG
        img.save(f'/workspace/benchmark_data/image_{i:04d}.jpg', 'JPEG')
    
    print(f"Created 120 benchmark images in /workspace/benchmark_data/")

def create_current_pipeline():
    """Create the intentionally slow pipeline implementation"""
    
    pipeline_code = '''#!/usr/bin/env python3

import os
import time
import random
import numpy as np
from PIL import Image, ImageEnhance
import glob

def augment_image(img_path, output_path, seed_offset):
    """Inefficiently augment a single image with repeated conversions"""
    random.seed(42 + seed_offset)
    np.random.seed(42 + seed_offset)
    
    # Load image (PIL)
    img = Image.open(img_path).convert('RGB')
    
    # Random rotation - convert to numpy and back
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    
    # Horizontal flip - inefficient check
    if random.random() < 0.5:
        # Convert to numpy for flip, then back to PIL
        img_np = np.array(img)
        img_np = np.fliplr(img_np)
        img = Image.fromarray(img_np)
    
    # Brightness adjustment - inefficient
    brightness_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    
    # Convert to numpy and back (inefficient)
    img_np = np.array(img)
    img = Image.fromarray(img_np)
    
    # Contrast adjustment - inefficient
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Convert to numpy and back again (inefficient)
    img_np = np.array(img)
    img = Image.fromarray(img_np)
    
    # Random crop - inefficient calculation
    width, height = img.size
    crop_scale = random.uniform(0.85, 1.0)
    new_width = int(width * crop_scale)
    new_height = int(height * crop_scale)
    
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    
    img = img.crop((left, top, left + new_width, top + new_height))
    
    # Resize to 224x224
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Normalize - convert to numpy inefficiently
    img_np = np.array(img, dtype=np.float32)
    img_np = img_np / 255.0
    
    # Apply normalization one channel at a time (inefficient)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for c in range(3):
        img_np[:, :, c] = (img_np[:, :, c] - mean[c]) / std[c]
    
    # Convert back to uint8 for saving (denormalize)
    img_np = img_np * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Save
    result_img = Image.fromarray(img_np)
    result_img.save(output_path, 'JPEG')

def main():
    """Process all benchmark images sequentially (slow)"""
    input_dir = '/workspace/benchmark_data'
    output_dir = '/workspace/output_current'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    
    print(f"Processing {len(image_files)} images with current pipeline...")
    
    start_time = time.time()
    
    # Process images one by one (no parallelization)
    for idx, img_path in enumerate(image_files):
        output_path = os.path.join(output_dir, f'output_{idx:04d}.jpg')
        augment_image(img_path, output_path, idx)
        
        # Add some artificial delay to simulate real augmentation overhead
        time.sleep(0.01)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Current pipeline completed in {elapsed:.3f} seconds")
    print(f"Processed {len(image_files)} images")
    print(f"Average time per image: {elapsed/len(image_files):.3f} seconds")

if __name__ == '__main__':
    main()
'''
    
    with open('/workspace/current_pipeline.py', 'w') as f:
        f.write(pipeline_code)
    
    os.chmod('/workspace/current_pipeline.py', 0o755)
    print("Created current_pipeline.py with slow implementation")

def main():
    print("Setting up benchmark environment...")
    create_benchmark_data()
    create_current_pipeline()
    print("Benchmark setup complete")

if __name__ == '__main__':
    main()