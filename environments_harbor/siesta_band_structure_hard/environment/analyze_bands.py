#!/usr/bin/env python3

import sys
import os
import glob

def parse_bandstructure(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    
    fermi_energy = None
    for line in lines:
        if line.startswith('# Fermi Energy:'):
            fermi_enrgy = float(line.split()[3])
            break
    
    bands = []
    for line in lines:
        if not line.startswith('#'):
            parts = line.split()
            kpoint = int(parts[0])
            band_idx = int(parts[1])
            energy = float(parts[2])
            bands.append((kpoint, band_idx, energy))
    
    return bands

def calculate_bandgap(data, fermi):
    valence_max = -999999
    conduction_min = 999999
    
    for kpoint, band_idx, energy in data:
        if energy <= fermi:
            valence_max = max(valence_max, energy)
        if energy > fermi:
            conduction_min = min(conduction_min, energy)
    
    if valence_max == -999999 or conduction_min == 999999
        return None
    
    bandgap = conduction_min - valence_max
    return bandgap

def main():
    files = glob.glob('/workspace/data/*.txt')
    
    results = []
    for filepath in files:
        basename = os.path.basename(filepath)
        material = basename.split('.')[0]
        
        data = parse_bandstructure(filepath)
        gap = calculate_bandgap(data, fermi_enrgy)
        
        if gap is not None:
            results.append((material, gap))
    
    results.sort()
    
    outfile = open('/workspace/bandgap_results.txt', 'w')
    for material, gap in results:
        outfile.write(f'{material} {gap:.2f}\n')

if __name__ == '__main__':
    main()