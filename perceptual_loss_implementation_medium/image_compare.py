#!/usr/bin/env python3

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

def compute_similarity(image1, image2):
    """
    Compute perceptual similarity between two images.
    Currently uses a BROKEN naive pixel comparison approach.
    Returns a float between 0.0 (completely different) and 1.0 (identical).
    """
    # Convert images to numpy arrays
    img1_array = np.array(image1).astype(np.float32)
    img2_array = np.array(image2).astype(np.float32)
    
    # Ensure images are the same size
    if img1_array.shape != img2_array.shape:
        image2_resized = image2.resize(image1.size)
        img2_array = np.array(image2_resized).astype(np.float32)
    
    # BROKEN: Use perceptual similarity with VGG features instead of raw pixels
    # Load pre-trained VGG16 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    
    # Preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image1.mode != 'RGB':
        image1 = image1.convert('RGB')
    if image2.mode != 'RGB':
        image2 = image2.convert('RGB')
    
    # Preprocess images
    img1_tensor = preprocess(image1).unsqueeze(0).to(device)
    img2_tensor = preprocess(image2).unsqueeze(0).to(device)
    
    # Extract features from multiple layers
    features1 = []
    features2 = []
    
    x1 = img1_tensor
    x2 = img2_tensor
    
    # Extract features from specific layers
    layer_indices = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
    
    with torch.no_grad():
        for i, layer in enumerate(vgg):
            x1 = layer(x1)
            x2 = layer(x2)
            if i in layer_indices:
                features1.append(x1)
                features2.append(x2)
    
    # Compute cosine similarity for each layer
    similarities = []
    for f1, f2 in zip(features1, features2):
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(f1_flat, f2_flat)
        similarities.append(cos_sim.item())
    
    # Average similarity across layers
    avg_similarity = np.mean(similarities)
    
    # Convert from [-1, 1] to [0, 1]
    similarity_score = (avg_similarity + 1.0) / 2.0
    
    # Ensure score is in valid range
    similarity_score = np.clip(similarity_score, 0.0, 1.0)
    
    return float(similarity_score)


def create_test_images():
    """
    Create three pairs of synthetic test images for validation.
    Returns a list of tuples: [(img1_a, img1_b), (img2_a, img2_b), (img3_a, img3_b)]
    """
    size = 128
    
    # Pair 1: Two nearly identical images (same checkerboard with tiny noise)
    img1_a = Image.new('RGB', (size, size))
    pixels1_a = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create checkerboard pattern
    square_size = 16
    for i in range(size):
        for j in range(size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                pixels1_a[i, j] = [255, 255, 255]
            else:
                pixels1_a[i, j] = [0, 0, 0]
    
    img1_a = Image.fromarray(pixels1_a)
    
    # Nearly identical with tiny noise
    pixels1_b = pixels1_a.copy()
    noise = np.random.randint(-3, 4, pixels1_b.shape, dtype=np.int16)
    pixels1_b = np.clip(pixels1_b.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img1_b = Image.fromarray(pixels1_b)
    
    # Pair 2: Structurally similar with color variation (same shapes, different colors)
    img2_a = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw2_a = ImageDraw.Draw(img2_a)
    
    # Draw circles in a grid pattern
    circle_size = 20
    for i in range(3):
        for j in range(3):
            x = 20 + i * 40
            y = 20 + j * 40
            draw2_a.ellipse([x, y, x + circle_size, y + circle_size], fill=(0, 0, 255))
    
    img2_b = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw2_b = ImageDraw.Draw(img2_b)
    
    # Same circles but different color
    for i in range(3):
        for j in range(3):
            x = 20 + i * 40
            y = 20 + j * 40
            draw2_b.ellipse([x, y, x + circle_size, y + circle_size], fill=(255, 0, 0))
    
    # Pair 3: Completely different images (horizontal stripes vs random noise)
    img3_a = Image.new('RGB', (size, size))
    pixels3_a = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Horizontal stripes
    stripe_height = 16
    for i in range(size):
        if (i // stripe_height) % 2 == 0:
            pixels3_a[i, :] = [255, 0, 0]
        else:
            pixels3_a[i, :] = [0, 255, 0]
    
    img3_a = Image.fromarray(pixels3_a)
    
    # Random noise
    pixels3_b = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img3_b = Image.fromarray(pixels3_b)
    
    return [(img1_a, img1_b), (img2_a, img2_b), (img3_a, img3_b)]


def run_tests():
    """
    Run validation tests on the similarity metric.
    Returns tuple of three scores.
    """
    print("Running image similarity tests...\n")
    
    # Get test image pairs
    test_pairs = create_test_images()
    
    # Test 1: Nearly identical images
    score1 = compute_similarity(test_pairs[0][0], test_pairs[0][1])
    print(f"Test 1 (nearly identical): {score1:.4f}")
    if score1 > 0.90:
        print("  PASS (score > 0.90)")
    else:
        print("  FAIL (score should be > 0.90)")
    
    # Test 2: Structurally similar with color variation
    score2 = compute_similarity(test_pairs[1][0], test_pairs[1][1])
    print(f"\nTest 2 (color variation): {score2:.4f}")
    if score2 > 0.70:
        print("  PASS (score > 0.70)")
    else:
        print("  FAIL (score should be > 0.70)")
    
    # Test 3: Completely different images
    score3 = compute_similarity(test_pairs[2][0], test_pairs[2][1])
    print(f"\nTest 3 (different): {score3:.4f}")
    if score3 < 0.30:
        print("  PASS (score < 0.30)")
    else:
        print("  FAIL (score should be < 0.30)")
    
    print("\n" + "="*50)
    all_pass = (score1 > 0.90) and (score2 > 0.70) and (score3 < 0.30)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*50)
    
    return (score1, score2, score3)


if __name__ == "__main__":
    scores = run_tests()
    
    # Save results to solution.txt
    with open('/workspace/solution.txt', 'w') as f:
        f.write(f"test_1_score: {scores[0]:.2f}\n")
        f.write(f"test_2_score: {scores[1]:.2f}\n")
        f.write(f"test_3_score: {scores[2]:.2f}\n")
    
    print("\nResults saved to /workspace/solution.txt")