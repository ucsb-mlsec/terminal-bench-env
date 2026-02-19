#!/usr/bin/env python3

import os
import sys
from lr_scheduler import get_learning_rate

def main():
    output_file = "/workspace/lr_schedule.txt"
    num_epochs = 100
    
    learning_rates = []
    
    print("Starting training simulation...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        lr = get_learning_rate(epoch)
        learning_rates.append(lr)
        
        if epoch < 5 or epoch % 10 == 0 or epoch >= 95:
            print(f"Epoch {epoch:3d}: Learning Rate = {lr:.8f}")
    
    print("=" * 50)
    print(f"Training complete. Saving learning rates to {output_file}")
    
    # Write learning rates to file
    with open(output_file, 'w') as f:
        for lr in learning_rates:
            f.write(f"{lr}\n")
    
    print(f"Successfully saved {len(learning_rates)} learning rate values")
    
    # Verify the output
    print("\nVerifying schedule requirements:")
    print(f"  Epoch 0 LR: {learning_rates[0]:.8f} (should be < 0.0002)")
    print(f"  Epoch 10 LR: {learning_rates[10]:.8f} (should be 0.001)")
    print(f"  Epoch 60 LR: {learning_rates[60]:.8f} (should be 0.001)")
    print(f"  Epoch 99 LR: {learning_rates[99]:.8f} (should be < 0.0005)")
    
    # Validation checks
    success = True
    if learning_rates[0] >= 0.0002:
        print("  ✗ FAILED: Epoch 0 LR should be < 0.0002")
        success = False
    else:
        print("  ✓ PASSED: Epoch 0 LR < 0.0002")
    
    if abs(learning_rates[10] - 0.001) > 1e-6:
        print("  ✗ FAILED: Epoch 10 LR should be 0.001")
        success = False
    else:
        print("  ✓ PASSED: Epoch 10 LR = 0.001")
    
    steady_phase_ok = all(abs(learning_rates[i] - 0.001) < 1e-6 for i in range(10, 61))
    if not steady_phase_ok:
        print("  ✗ FAILED: LR should remain 0.001 from epoch 10 to 60")
        success = False
    else:
        print("  ✓ PASSED: LR = 0.001 from epoch 10 to 60")
    
    if learning_rates[99] >= 0.0005:
        print("  ✗ FAILED: Epoch 99 LR should be < 0.0005")
        success = False
    else:
        print("  ✓ PASSED: Epoch 99 LR < 0.0005")
    
    if success:
        print("\n✓ All requirements met!")
        return 0
    else:
        print("\n✗ Some requirements failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())