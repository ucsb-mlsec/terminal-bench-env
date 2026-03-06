#!/usr/bin/env python3

import json
import sys
import os
from cmdstanpy import CmdStanModel

def main():
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Load data from JSON file
        print("Loading data from data.json...")
        with open('data.json', 'r') as f:
            data = json.load(f)
        print(f"Data loaded successfully: {list(data.keys())}")
        
        # Compile the Stan model
        print("\nCompiling Stan model...")
        model = CmdStanModel(stan_file='model.stan')
        print("Model compiled successfully")
        
        # Run the model with specified parameters
        print("\nRunning MCMC sampling...")
        fit = model.sample(
            data=data,
            iter_warmup=500,
            iter_sampling=500,
            chains=4,
            seed=12345,
            show_console=True
        )
        print("Sampling completed")
        
        # Get diagnostics
        print("\nExtracting diagnostics...")
        
        # Get divergences
        divergences = 0
        try:
            # Check for divergences in the fit
            diagnostic_output = fit.diagnose()
            # Parse divergences from diagnostic output
            if diagnostic_output is not None:
                for line in str(diagnostic_output).split('\n'):
                    if 'divergent transitions' in line.lower():
                        # Extract number from line
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                divergences = int(part)
                                break
        except Exception as e:
            print(f"Warning: Could not extract divergences via diagnose: {e}")
            # Alternative method
            try:
                divergences = fit.num_divergences()
            except:
                divergences = sum(fit.method_variables()['divergent__'])
        
        print(f"Divergent transitions: {divergences}")
        
        # Get maximum R-hat
        summary = fit.summary()
        rhat_max = summary['R_hat'].max()
        print(f"Maximum R-hat: {rhat_max}")
        
        # Print full summary for reference
        print("\nParameter Summary:")
        print(summary)
        
        # Determine status based on criteria
        if divergences < 10 and rhat_max < 1.10:
            status = "fixed"
            print("\n✓ Model diagnostics look good!")
        else:
            status = "failed"
            print("\n✗ Model has numerical issues")
        
        # Write results to solution.txt
        solution_path = '/workspace/solution.txt'
        print(f"\nWriting results to {solution_path}...")
        with open(solution_path, 'w') as f:
            f.write(f"status: {status}\n")
            f.write(f"divergences: {divergences}\n")
            f.write(f"rhat_max: {rhat_max:.2f}\n")
        
        print("Results written successfully")
        print(f"\nFinal Status: {status}")
        
        return 0
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Write failed status
        try:
            with open('/workspace/solution.txt', 'w') as f:
                f.write("status: failed\n")
                f.write("divergences: 9999\n")
                f.write("rhat_max: 99.99\n")
        except:
            pass
        
        return 1

if __name__ == "__main__":
    sys.exit(main())