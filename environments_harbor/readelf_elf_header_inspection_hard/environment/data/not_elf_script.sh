#!/usr/bin/env python3

import os
import struct
import json
from pathlib import Path

# ELF magic number
ELF_MAGIC = b'\x7fELF'

# ELF class constants
ELFCLASS32 = 1
ELFCLASS64 = 2

# ELF machine types (common ones)
MACHINE_TYPES = {
    0x00: "No machine",
    0x02: "SPARC",
    0x03: "x86",
    0x08: "MIPS",
    0x14: "PowerPC",
    0x28: "ARM",
    0x2A: "SuperH",
    0x32: "IA-64",
    0x3E: "x86-64",
    0xB7: "AArch64",
    0xF3: "RISC-V"
}

def read_elf_header(filepath):
    """Read and parse ELF header from file"""
    try:
        with open(filepath, 'rb') as f:
            # Read first 64 bytes (enough for ELF header)
            header = f.read(64)
            
            if len(header) < 52:  # Minimum ELF32 header size
                return None
            
            # Check ELF magic
            if header[0:4] != ELF_MAGIC:
                return None
            
            # Parse ELF identification
            ei_class = header[4]
            ei_data = header[5]  # Endianness
            
            if ei_class not in [ELFCLASS32, ELFCLASS64]:
                return {'corrupted': True, 'reason': 'invalid_class'}
            
            if ei_data not in [1, 2]:  # 1=little-endian, 2=big-endian
                return {'corrupted': True, 'reason': 'invalid_endianness'}
            
            is_64bit = (ei_class == ELFCLASS64)
            is_little_endian = (ei_data == 1)
            endian_prefix = '<' if is_little_endian else '>'
            
            # Parse machine type and entry point
            if is_64bit:
                if len(header) < 64:
                    return {'corrupted': True, 'reason': 'truncated_header'}
                # e_machine at offset 18, e_entry at offset 24
                e_machine = struct.unpack(f'{endian_prefix}H', header[18:20])[0]
                e_entry = struct.unpack(f'{endian_prefix}Q', header[24:32])[0]
            else:
                # e_machine at offset 18, e_entry at offset 24
                e_machine = struct.unpack(f'{endian_prefix}H', header[18:20])[0]
                e_entry = struct.unpack(f'{endian_prefix}I', header[24:28])[0]
            
            return {
                'is_elf': True,
                'bits': 64 if is_64bit else 32,
                'machine': e_machine,
                'entry_point': e_entry,
                'corrupted': False
            }
            
    except (IOError, OSError, struct.error) as e:
        return None

def get_architecture_name(machine_code):
    """Convert machine type code to human-readable name"""
    return MACHINE_TYPES.get(machine_code, "unknown")

def determine_status(elf_info):
    """Determine if binary is VALID, SUSPICIOUS, or CORRUPTED"""
    if elf_info.get('corrupted'):
        return 'CORRUPTED'
    
    machine = elf_info['machine']
    entry_point = elf_info['entry_point']
    bits = elf_info['bits']
    
    # Check for clearly invalid values
    if machine not in MACHINE_TYPES and machine > 0xFFFF:
        return 'CORRUPTED'
    
    # Check for suspicious entry points
    if entry_point == 0:
        return 'SUSPICIOUS'
    
    # Very low entry points are suspicious for userspace binaries
    if entry_point < 0x1000 and entry_point != 0:
        return 'SUSPICIOUS'
    
    # Check for unreasonably high entry points (potential corruption)
    if bits == 32 and entry_point > 0xFFFFFFFF:
        return 'CORRUPTED'
    
    if bits == 64 and entry_point > 0x7FFFFFFFFFFF:  # Reasonable user-space limit
        return 'SUSPICIOUS'
    
    # Unknown architecture is suspicious
    if machine not in MACHINE_TYPES:
        return 'SUSPICIOUS'
    
    return 'VALID'

def analyze_binary(filepath):
    """Analyze a single binary file"""
    elf_info = read_elf_header(filepath)
    
    if elf_info is None:
        # Not an ELF file
        return {
            'path': str(filepath),
            'status': 'NOT_ELF',
            'architecture': 'N/A',
            'bits': 0,
            'entry_point': 'N/A'
        }
    
    if elf_info.get('corrupted'):
        return {
            'path': str(filepath),
            'status': 'CORRUPTED',
            'architecture': 'unknown',
            'bits': 0,
            'entry_point': '0x0'
        }
    
    status = determine_status(elf_info)
    arch_name = get_architecture_name(elf_info['machine'])
    
    return {
        'path': str(filepath),
        'status': status,
        'architecture': arch_name,
        'bits': elf_info['bits'],
        'entry_point': f"0x{elf_info['entry_point']:x}"
    }

def scan_directory(directory):
    """Scan directory and all subdirectories for files"""
    binaries = []
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                # Skip symlinks and special files
                if os.path.islink(filepath):
                    continue
                if not os.path.isfile(filepath):
                    continue
                
                result = analyze_binary(filepath)
                binaries.append(result)
            except Exception as e:
                # Handle any unexpected errors
                binaries.append({
                    'path': str(filepath),
                    'status': 'NOT_ELF',
                    'architecture': 'N/A',
                    'bits': 0,
                    'entry_point': 'N/A'
                })
    
    return binaries

def generate_summary(binaries):
    """Generate summary statistics"""
    summary = {
        'total': len(binaries),
        'valid': 0,
        'suspicious': 0,
        'corrupted': 0,
        'not_elf': 0
    }
    
    for binary in binaries:
        status = binary['status'].lower()
        if status == 'valid':
            summary['valid'] += 1
        elif status == 'suspicious':
            summary['suspicious'] += 1
        elif status == 'corrupted':
            summary['corrupted'] += 1
        elif status == 'not_elf':
            summary['not_elf'] += 1
    
    return summary

def main():
    target_directory = '/opt/production/bin/'
    output_file = '/tmp/elf_analysis_report.json'
    
    # Scan all binaries
    binaries = scan_directory(target_directory)
    
    # Sort by path
    binaries.sort(key=lambda x: x['path'])
    
    # Generate summary
    summary = generate_summary(binaries)
    
    # Create report
    report = {
        'binaries': binaries,
        'summary': summary
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis complete. Report saved to {output_file}")
    print(f"Total files analyzed: {summary['total']}")
    print(f"Valid: {summary['valid']}, Suspicious: {summary['suspicious']}, "
          f"Corrupted: {summary['corrupted']}, Not ELF: {summary['not_elf']}")

if __name__ == '__main__':
    main()