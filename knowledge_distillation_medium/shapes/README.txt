Shape Recognition Data Generator

This script generates synthetic shape images and simulated teacher model predictions for knowledge distillation experiments. The data is designed to help train compact student models that learn from both hard labels and soft probability distributions from a teacher model.

INSTRUCTIONS

To run the generator:
  python generate_data.py

Expected runtime: approximately 30-60 seconds

Output files created in /workspace/shapes/:
  - train_images.npy: Training image data
  - train_labels.npy: Training hard labels
  - train_teacher_preds.npy: Teacher model probability distributions for training
  - test_images.npy: Test image data
  - test_labels.npy: Test hard labels
  - test_teacher_preds.npy: Teacher model probability distributions for testing

DATA SPECIFICATIONS

Training samples: 5000 images
Test samples: 1000 images
Image dimensions: 32x32 pixels (grayscale)
Number of classes: 5
  - Class 0: Circle
  - Class 1: Square
  - Class 2: Triangle
  - Class 3: Star
  - Class 4: Hexagon

Teacher predictions: Soft probability distributions over all 5 classes, summing to 1.0 for each sample

USAGE NOTES

- All data generation is deterministic based on a fixed random seed for reproducibility
- Image pixel values are normalized to the [0, 1] range
- Teacher predictions include controlled noise to simulate realistic outputs from a trained model
- The teacher model is highly confident on most samples but shows some uncertainty to provide rich learning signals
- Shapes are generated using mathematical formulas to ensure consistent and deterministic patterns
- The generated data is suitable for testing knowledge distillation techniques on resource-constrained models