#!/usr/bin/env python3

import sys
import os
import pandera as pa

# Add workspace to path
sys.path.insert(0, '/workspace')

from pipeline.stage1_ingestion import process_stage1
from pipeline.stage2_enrichment import process_stage2
from pipeline.stage3_aggregation import process_stage3


def main():
    """Orchestrate the complete pipeline execution."""
    print("Starting pipeline...")
    
    stages_completed = 0
    
    # Stage 1: Ingestion
    try:
        print("\n" + "="*50)
        print("Running Stage 1: Ingestion...")
        print("="*50)
        process_stage1()
        print("✓ Stage 1 completed successfully!")
        stages_completed += 1
    except pa.errors.SchemaError as e:
        print(f"✗ Stage 1 failed with schema validation error:")
        print(f"  {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Stage 1 failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        sys.exit(1)
    
    # Stage 2: Enrichment
    try:
        print("\n" + "="*50)
        print("Running Stage 2: Enrichment...")
        print("="*50)
        process_stage2()
        print("✓ Stage 2 completed successfully!")
        stages_completed += 1
    except pa.errors.SchemaError as e:
        print(f"✗ Stage 2 failed with schema validation error:")
        print(f"  {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Stage 2 failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        sys.exit(1)
    
    # Stage 3: Aggregation
    try:
        print("\n" + "="*50)
        print("Running Stage 3: Aggregation...")
        print("="*50)
        process_stage3()
        print("✓ Stage 3 completed successfully!")
        stages_completed += 1
    except pa.errors.SchemaError as e:
        print(f"✗ Stage 3 failed with schema validation error:")
        print(f"  {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Stage 3 failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        sys.exit(1)
    
    # All stages completed
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print(f"All {stages_completed} stages passed validation.")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())