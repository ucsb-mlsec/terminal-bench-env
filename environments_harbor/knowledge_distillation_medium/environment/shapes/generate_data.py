#!/usr/bin/env python3

import numpy as np

def create_circle(size=32):
    """Create a circle shape."""
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 3
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist <= radius:
                img[i, j] = 255
    return img

def create_square(size=32):
    """Create a square shape."""
    img = np.zeros((size, size), dtype=np.float32)
    margin = size // 4
    img[margin:size-margin, margin:size-margin] = 255
    return img

def create_triangle(size=32):
    """Create a triangle shape."""
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            # Equilateral triangle pointing up
            # Bottom edge at row size-margin
            # Top vertex at row margin
            margin = size // 6
            base_row = size - margin
            top_row = margin
            
            if i >= top_row and i <= base_row:
                # Calculate triangle boundaries
                height = base_row - top_row
                current_height = i - top_row
                half_width = (current_height / height) * (size // 2 - margin)
                
                left_bound = center - half_width
                right_bound = center + half_width
                
                if j >= left_bound and j <= right_bound:
                    img[i, j] = 255
    return img

def create_star(size=32):
    """Create a 5-pointed star shape."""
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    outer_radius = size // 3
    inner_radius = size // 6
    
    # Create 5-pointed star
    angles = np.linspace(0, 2*np.pi, 11, endpoint=True)
    
    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            angle = np.arctan2(y, x)
            dist = np.sqrt(x**2 + y**2)
            
            # Determine if point is inside star
            # Check which sector the angle is in
            angle_normalized = (angle + np.pi) % (2*np.pi)
            sector = int(angle_normalized / (2*np.pi / 10))
            
            # Alternating outer and inner points
            if sector % 2 == 0:
                max_dist = outer_radius
            else:
                max_dist = inner_radius
            
            # Interpolate between radii
            sector_angle = (sector % 10) * (2*np.pi / 10)
            next_sector_angle = ((sector + 1) % 10) * (2*np.pi / 10)
            
            if sector % 2 == 0:
                current_radius = outer_radius
                next_radius = inner_radius
            else:
                current_radius = inner_radius
                next_radius = outer_radius
            
            angle_in_sector = angle_normalized - sector_angle
            sector_width = 2*np.pi / 10
            t = angle_in_sector / sector_width
            
            interpolated_radius = current_radius * (1 - t) + next_radius * t
            
            if dist <= interpolated_radius:
                img[i, j] = 255
    
    return img

def create_hexagon(size=32):
    """Create a hexagon shape."""
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 3
    
    # Hexagon vertices
    angles = np.linspace(0, 2*np.pi, 7, endpoint=True)
    vertices = [(center + radius * np.cos(a), center + radius * np.sin(a)) for a in angles]
    
    for i in range(size):
        for j in range(size):
            # Check if point is inside hexagon using cross product method
            inside = True
            for k in range(6):
                x1, y1 = vertices[k]
                x2, y2 = vertices[k+1]
                
                # Cross product to determine side
                cross = (x2 - x1) * (i - y1) - (y2 - y1) * (j - x1)
                if cross < 0:
                    inside = False
                    break
            
            if inside:
                img[i, j] = 255
    
    return img

def generate_shape(shape_class, size=32):
    """Generate a shape image based on class."""
    if shape_class == 0:
        return create_circle(size)
    elif shape_class == 1:
        return create_square(size)
    elif shape_class == 2:
        return create_triangle(size)
    elif shape_class == 3:
        return create_star(size)
    elif shape_class == 4:
        return create_hexagon(size)
    else:
        raise ValueError(f"Unknown shape class: {shape_class}")

def generate_teacher_predictions(true_label, num_classes=5, confidence=0.85):
    """Generate teacher predictions with controlled noise."""
    preds = np.zeros(num_classes, dtype=np.float32)
    preds[true_label] = confidence
    
    # Distribute remaining probability
    remaining = 1.0 - confidence
    other_classes = [i for i in range(num_classes) if i != true_label]
    
    # Random distribution with some structure
    random_weights = np.random.random(len(other_classes))
    random_weights = random_weights / random_weights.sum() * remaining
    
    for idx, class_idx in enumerate(other_classes):
        preds[class_idx] = random_weights[idx]
    
    # Add small noise
    noise = np.random.normal(0, 0.01, num_classes)
    preds = preds + noise
    preds = np.maximum(preds, 0)  # Ensure non-negative
    preds = preds / preds.sum()  # Renormalize
    
    return preds

def generate_data():
    """Generate training and test data."""
    print("Generating training data...")
    
    # Training data
    num_train = 5000
    num_test = 1000
    num_classes = 5
    img_size = 32
    
    # Generate balanced training data
    samples_per_class = num_train // num_classes
    train_images = []
    train_labels = []
    train_teacher_preds = []
    
    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            img = generate_shape(class_idx, img_size)
            train_images.append(img)
            train_labels.append(class_idx)
            teacher_pred = generate_teacher_predictions(class_idx, num_classes)
            train_teacher_preds.append(teacher_pred)
    
    # Convert to numpy arrays
    train_images = np.array(train_images, dtype=np.float32) / 255.0
    train_labels = np.array(train_labels, dtype=np.int64)
    train_teacher_preds = np.array(train_teacher_preds, dtype=np.float32)
    
    # Shuffle training data
    indices = np.random.permutation(len(train_images))
    train_images = train_images[indices]
    train_labels = train_labels[indices]
    train_teacher_preds = train_teacher_preds[indices]
    
    print("Generating test data...")
    
    # Generate balanced test data
    samples_per_class_test = num_test // num_classes
    test_images = []
    test_labels = []
    test_teacher_preds = []
    
    for class_idx in range(num_classes):
        for _ in range(samples_per_class_test):
            img = generate_shape(class_idx, img_size)
            test_images.append(img)
            test_labels.append(class_idx)
            teacher_pred = generate_teacher_predictions(class_idx, num_classes)
            test_teacher_preds.append(teacher_pred)
    
    # Convert to numpy arrays
    test_images = np.array(test_images, dtype=np.float32) / 255.0
    test_labels = np.array(test_labels, dtype=np.int64)
    test_teacher_preds = np.array(test_teacher_preds, dtype=np.float32)
    
    # Shuffle test data
    indices = np.random.permutation(len(test_images))
    test_images = test_images[indices]
    test_labels = test_labels[indices]
    test_teacher_preds = test_teacher_preds[indices]
    
    # Save files
    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    np.save('train_teacher_preds.npy', train_teacher_preds)
    np.save('test_images.npy', test_images)
    np.save('test_labels.npy', test_labels)
    np.save('test_teacher_preds.npy', test_teacher_preds)
    
    print("Data generation complete!")
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Image shape: {train_images.shape[1:]}")
    print(f"Number of classes: {num_classes}")

if __name__ == "__main__":
    np.random.seed(42)
    generate_data()