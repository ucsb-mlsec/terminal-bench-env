AutoAugment Policy Configuration Format

This document describes the expected structure for AutoAugment policy configuration files
used in the image preprocessing pipeline.

Structure Overview:
An AutoAugment policy consists of a list of sub-policies. Each sub-policy defines a sequence
of transformations that can be applied to training images to augment the dataset.

Sub-Policy Requirements:
Each sub-policy must contain exactly 2 operations. Sub-policies with fewer or more than 2
operations are considered invalid and will be rejected during validation.

Operation Parameters:
Every operation must specify three components:
1. Transformation type - the name of the image transformation (e.g., "ShearX", "Rotate")
2. Probability - a float value between 0.0 and 1.0 indicating application likelihood
3. Magnitude - an integer typically in the range 0-10 controlling transformation intensity

Valid Transformation Types:
Common transformations include: ShearX, ShearY, TranslateX, TranslateY, Rotate, Contrast,
Brightness, Sharpness, Color, Invert, Equalize, Solarize, Posterize, AutoContrast.

Validation Rules:
- Probability values must be in the range [0.0, 1.0]. Values outside this range are invalid.
- Magnitude values should typically be 0-10, though some transformations may accept slightly
  different ranges depending on implementation.
- The policy must contain at least one valid sub-policy to be considered usable.