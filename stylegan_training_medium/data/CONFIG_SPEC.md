# StyleGAN2 Configuration Specification

This document defines the configuration parameters required for StyleGAN2 training and their valid constraints.

## Required Parameters

All four parameters listed below are **required** in the configuration file.

### 1. resolution

The output image resolution for generated images.

**Type:** Integer

**Constraints:**
- Must be a power of 2
- Valid values: 64, 128, 256, 512, 1024
- Minimum: 64
- Maximum: 1024

**Recommended values:**
- 256: Fast training, good quality for many applications
- 512: High quality, balanced training time
- 1024: Highest quality, slower training

**Examples:**
- Valid: 64, 128, 256, 512, 1024
- Invalid: 100, 255, 300, 768, 2048

### 2. batch_size

Number of images processed in each training batch.

**Type:** Positive Integer

**Constraints:**
- Must be a positive integer
- Valid range: 1 to 64
- Must be chosen to fit within GPU memory constraints
- Resolution-dependent limitations:
  - When `resolution >= 512`: batch_size should not exceed 16
  - When `resolution <= 256`: batch_size can be up to 32
  - For `resolution = 1024`: typical values are 4-8

**Recommended values:**
- For resolution 1024: 4-8
- For resolution 512: 8-16
- For resolution 256: 16-32
- For resolution 128 or less: 32-64

**Examples:**
- Valid: 1, 4, 8, 16, 32
- Invalid: 0, -1, 128 (exceeds maximum)

### 3. learning_rate

The learning rate for the optimizer (both generator and discriminator).

**Type:** Positive Float

**Constraints:**
- Must be a positive float
- Valid range: 0.0001 to 0.01
- Recommended range: 0.001 to 0.003
- Must be specified in decimal notation (not scientific notation)

**Recommended values:**
- Standard: 0.002
- Conservative (more stable): 0.001
- Aggressive (faster convergence): 0.003

**Examples:**
- Valid: 0.001, 0.002, 0.0025, 0.003
- Invalid: 0.00005 (too low), 0.05 (too high), -0.001 (negative)

### 4. gradient_penalty

Weight for the R1 gradient penalty regularization term.

**Type:** Positive Float

**Constraints:**
- Must be a positive float
- Valid range: 0.1 to 50.0
- Standard recommended value: 10.0

**Recommended values:**
- Standard: 10.0 (works well for most cases)
- Lower values (5.0-9.0): Less regularization
- Higher values (11.0-20.0): More regularization

**Examples:**
- Valid: 5.0, 10.0, 15.0, 20.0
- Invalid: 0.0 (too low), 100.0 (exceeds maximum), -10.0 (negative)

## Configuration File Format

The configuration must be saved as a valid JSON file with the following structure:

```json
{
  "resolution": <integer>,
  "batch_size": <integer>,
  "learning_rate": <float>,
  "gradient_penalty": <float>
}
```

## Valid Configuration Examples

### Example 1: High Quality (1024x1024)
```json
{
  "resolution": 1024,
  "batch_size": 4,
  "learning_rate": 0.002,
  "gradient_penalty": 10.0
}
```

### Example 2: Balanced (512x512)
```json
{
  "resolution": 512,
  "batch_size": 16,
  "learning_rate": 0.002,
  "gradient_penalty": 10.0
}
```

### Example 3: Fast Training (256x256)
```json
{
  "resolution": 256,
  "batch_size": 32,
  "learning_rate": 0.0025,
  "gradient_penalty": 10.0
}
```

### Example 4: Quick Testing (128x128)
```json
{
  "resolution": 128,
  "batch_size": 32,
  "learning_rate": 0.003,
  "gradient_penalty": 10.0
}
```

## Constraint Summary Table

| Parameter | Type | Min | Max | Recommended | Special Constraints |
|-----------|------|-----|-----|-------------|-------------------|
| resolution | int | 64 | 1024 | 512-1024 | Must be power of 2 |
| batch_size | int | 1 | 64 | 4-32 | ≤16 if resolution≥512, ≤32 if resolution≤256 |
| learning_rate | float | 0.0001 | 0.01 | 0.001-0.003 | Decimal notation |
| gradient_penalty | float | 0.1 | 50.0 | 10.0 | Standard is 10.0 |

## Validation Rules

Before starting training, the configuration is validated against these rules:

1. All four parameters must be present
2. `resolution` must be a power of 2 within [64, 1024]
3. `batch_size` must be in range [1, 64]
4. If `resolution >= 512`, then `batch_size <= 16`
5. If `resolution <= 256`, then `batch_size <= 32`
6. `learning_rate` must be in range [0.0001, 0.01]
7. `gradient_penalty` must be in range [0.1, 50.0]
8. File must be valid JSON format

Any configuration that fails these validation rules will be rejected before training begins.