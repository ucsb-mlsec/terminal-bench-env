#!/usr/bin/env python3

import yaml
import os

def load_config(config_path):
    """Load attribute configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data_file(filepath):
    """Load numeric data from text file, one value per line."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Try to parse as float, fallback to string
                try:
                    data.append(float(line))
                except ValueError:
                    data.append(line)
    return data

def simulate_rendering(config, vertex_data, instance_data, num_instances, vertices_per_instance):
    """
    Simulate instanced rendering with vertex attributes.
    
    Divisor logic:
    - divisor=0: Attribute advances for each vertex (per-vertex attribute)
    - divisor=1: Attribute advances for each instance (per-instance attribute)
    """
    
    print("=" * 80)
    print("INSTANCED RENDERING SIMULATION")
    print("=" * 80)
    print(f"\nRendering {num_instances} instances, each with {vertices_per_instance} vertices\n")
    
    # Print configuration
    print("-" * 80)
    print("ATTRIBUTE CONFIGURATION:")
    print("-" * 80)
    for attr in config['attributes']:
        print(f"  {attr['name']}:")
        print(f"    Divisor: {attr['divisor']} ({'PER-VERTEX' if attr['divisor'] == 0 else 'PER-INSTANCE'})")
        print(f"    Buffer: {attr['buffer']}")
        print(f"    Components: {attr.get('components', 1)}")
    print()
    
    # Initialize attribute pointers (indices into data buffers)
    attribute_pointers = {}
    for attr in config['attributes']:
        attribute_pointers[attr['name']] = 0
    
    print("-" * 80)
    print("RENDERING PROCESS:")
    print("-" * 80)
    
    # Simulate rendering each instance
    for instance_id in range(num_instances):
        print(f"\n{'=' * 40}")
        print(f"INSTANCE {instance_id}")
        print(f"{'=' * 40}")
        
        # For each vertex in this instance
        for vertex_id in range(vertices_per_instance):
            print(f"\n  Vertex {vertex_id}:")
            
            # Process each attribute
            for attr in config['attributes']:
                attr_name = attr['name']
                divisor = attr['divisor']
                buffer_name = attr['buffer']
                components = attr.get('components', 1)
                
                # Select data source
                if buffer_name == 'vertex':
                    data_source = vertex_data
                else:
                    data_source = instance_data
                
                # Get current pointer for this attribute
                pointer = attribute_pointers[attr_name]
                
                # Read attribute values
                values = []
                for i in range(components):
                    if pointer + i < len(data_source):
                        values.append(data_source[pointer + i])
                    else:
                        values.append(0.0)
                
                # Format output
                if components == 1:
                    value_str = f"{values[0]}"
                else:
                    value_str = f"({', '.join(str(v) for v in values)})"
                
                advancement = "CONSTANT" if divisor == 1 else "ADVANCING"
                print(f"    {attr_name:20s} = {value_str:30s} [pointer={pointer}, {advancement}]")
            
            # After processing this vertex, advance attributes with divisor=0
            for attr in config['attributes']:
                if attr['divisor'] == 0:
                    # Advance per-vertex attributes
                    components = attr.get('components', 1)
                    attribute_pointers[attr['name']] += components
        
        # After completing this instance, advance attributes with divisor=1
        for attr in config['attributes']:
            if attr['divisor'] == 1:
                # Advance per-instance attributes
                components = attr.get('components', 1)
                attribute_pointers[attr['name']] += components
        
        print(f"\n  → Instance {instance_id} complete. Per-instance attributes advanced.")
    
    print("\n" + "=" * 80)
    print("RENDERING ANALYSIS:")
    print("=" * 80)
    
    # Reset pointers for analysis
    for attr in config['attributes']:
        attribute_pointers[attr['name']] = 0
    
    # Analyze each attribute's behavior
    for attr in config['attributes']:
        attr_name = attr['name']
        divisor = attr['divisor']
        buffer_name = attr['buffer']
        components = attr.get('components', 1)
        
        if buffer_name == 'vertex':
            data_source = vertex_data
        else:
            data_source = instance_data
        
        print(f"\n{attr_name}:")
        print(f"  Configured divisor: {divisor}")
        print(f"  Buffer source: {buffer_name}")
        
        # Collect values used for each instance
        instance_values = []
        pointer = 0
        
        for instance_id in range(num_instances):
            vertex_values = []
            for vertex_id in range(vertices_per_instance):
                values = []
                for i in range(components):
                    if pointer + i < len(data_source):
                        values.append(data_source[pointer + i])
                    else:
                        values.append(0.0)
                vertex_values.append(tuple(values) if components > 1 else values[0])
                
                if divisor == 0:
                    pointer += components
            
            instance_values.append(vertex_values)
            if divisor == 1:
                pointer += components
        
        # Check for anomalies
        if buffer_name == 'vertex' and divisor == 1:
            print(f"  ⚠️  WARNING: Vertex buffer with divisor=1 (per-instance)")
            print(f"      All instances share the SAME vertex data!")
            print(f"      Expected: divisor=0 for geometry attributes")
        
        if buffer_name == 'instance' and divisor == 0:
            print(f"  ⚠️  WARNING: Instance buffer with divisor=0 (per-vertex)")
            print(f"      Instance attributes advance per-vertex instead of per-instance!")
            print(f"      Expected: divisor=1 for per-instance attributes")
        
        # Show value distribution
        all_unique = len(set(str(v) for inst in instance_values for v in inst))
        print(f"  Unique values used: {all_unique}")
        
        # Check if all instances got same values
        if divisor == 1 and buffer_name == 'instance':
            first_instance = str(instance_values[0])
            all_same = all(str(inst) == first_instance for inst in instance_values)
            if not all_same:
                print(f"  ✓ Instances have different values (correct for per-instance)")
            else:
                print(f"  ℹ️  All instances have identical values")
        
        if divisor == 0 and buffer_name == 'vertex':
            # Check if vertices within each instance share geometry
            first_instance = instance_values[0]
            if len(instance_values) > 1:
                all_instances_same_geometry = all(inst == first_instance for inst in instance_values)
                if all_instances_same_geometry:
                    print(f"  ✓ All instances share same geometry (correct for per-vertex)")
                else:
                    print(f"  ℹ️  Instances have different geometry patterns")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load configuration
    config_path = os.path.join(script_dir, 'attribute_config.yaml')
    config = load_config(config_path)
    
    # Load data files
    vertex_data_path = os.path.join(script_dir, 'vertex_data.txt')
    instance_data_path = os.path.join(script_dir, 'instance_data.txt')
    
    vertex_data = load_data_file(vertex_data_path)
    instance_data = load_data_file(instance_data_path)
    
    print(f"Loaded {len(vertex_data)} vertex data values")
    print(f"Loaded {len(instance_data)} instance data values")
    print()
    
    # Run simulation
    num_instances = 3
    vertices_per_instance = 3  # Triangle
    
    simulate_rendering(config, vertex_data, instance_data, num_instances, vertices_per_instance)

if __name__ == '__main__':
    main()