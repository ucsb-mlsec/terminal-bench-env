# Dask Distributed Worker Resource Configuration Guide

## Introduction

Dask distributed workers can advertise custom resources to the scheduler, enabling intelligent task placement based on hardware capabilities. This is essential for heterogeneous clusters where workers have different hardware profiles (GPUs, memory capacities, specialized processors, etc.).

When tasks are submitted with resource requirements, the Dask scheduler uses these advertised resources to determine which workers are capable of executing those tasks.

## YAML Configuration Structure

Custom resources must be declared in the worker configuration YAML file under the `resources` key at the top level of the configuration.

### Correct Key Structure

```yaml
resources:
  resource_name_1: numeric_value
  resource_name_2: numeric_value
```

The `resources` key should be at the root level of the YAML configuration, not nested under other keys like `distributed`, `worker`, or `scheduler`.

## Resource Format Specifications

### Data Types
- **Resource names**: Strings (keys in the YAML dictionary)
- **Resource values**: Numeric values (integers or floats)
- **DO NOT** use strings for values (e.g., "2" instead of 2)

### Naming Conventions
- Use lowercase with underscores for multi-word resource names
- Common examples: `gpu`, `memory_gb`, `cpu_cores`, `tpu`, `fpga`
- Avoid spaces or hyphens in resource names

## Valid Configuration Examples

### Example 1: Worker with GPU and Memory Resources

```yaml
resources:
  gpu: 2
  memory_gb: 64
```

This declares a worker with 2 GPUs and 64 GB of memory available for task allocation.

### Example 2: Worker with CPU-only Configuration

```yaml
resources:
  gpu: 0
  memory_gb: 128
  cpu_cores: 32
```

This declares a worker with no GPUs, 128 GB of memory, and 32 CPU cores. Note that even if a resource is zero, it can still be explicitly declared.

### Example 3: Worker with Custom Resources

```yaml
resources:
  gpu: 4
  memory_gb: 256
  tpu: 8
  nvme_storage_tb: 2
```

This shows how custom resource types can be defined for specialized hardware.

### Example 4: Complete Worker Configuration

```yaml
distributed:
  version: 2
  
resources:
  gpu: 1
  memory_gb: 32
  
worker:
  memory:
    target: 0.6
    spill: 0.7
    pause: 0.8
    terminate: 0.95
```

This shows resources in the context of a complete worker configuration file.

## Common Mistakes to Avoid

### ❌ Incorrect: Wrong Nesting
```yaml
distributed:
  worker:
    resources:
      gpu: 2
```

### ❌ Incorrect: String Values
```yaml
resources:
  gpu: "2"
  memory_gb: "64GB"
```

### ❌ Incorrect: Wrong Key Name
```yaml
custom_resources:
  gpu: 2
```

### ❌ Incorrect: Hyphenated Names
```yaml
resources:
  memory-gb: 64
```

### ✅ Correct Format
```yaml
resources:
  gpu: 2
  memory_gb: 64
```

## GPU and Memory_GB Resources

For the specific use case of declaring GPU and memory resources:

- **gpu**: Represents the number of GPU devices available on the worker
  - Type: Integer
  - Common values: 0, 1, 2, 4, 8
  
- **memory_gb**: Represents the amount of memory in gigabytes available for resource-constrained tasks
  - Type: Integer or Float
  - Common values: 8, 16, 32, 64, 128, 256
  - Note: This is separate from Dask's automatic memory management and is used for explicit resource allocation

## Validation

To verify your resource configuration is correct:

1. Ensure `resources` is at the root level of the YAML
2. All resource values are numeric (not strings)
3. Resource names use underscores, not hyphens or spaces
4. The YAML syntax is valid (proper indentation, colons, no tabs)

## Additional Notes

- Resources are arbitrary and can represent any quantifiable hardware capability
- The scheduler tracks resource availability across all workers
- Tasks can request resources using `client.submit(func, resources={'gpu': 1})`
- If a task requests resources that no worker advertises, it will remain pending indefinitely