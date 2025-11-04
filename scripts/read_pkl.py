#!/usr/bin/env python3
"""
Simple script to read and print the contents of a pickle (.pkl) file.

Usage:
    python read_pkl.py <path_to_pkl_file>

Example:
    python read_pkl.py multiple_runs/run_00/convergence_results.pkl
"""

import pickle
import sys
import os
from pprint import pprint


def read_pkl_file(file_path):
    """Read and return the contents of a pickle file."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return None


def print_data_info(data):
    """Print information about the loaded data."""
    print(f"Data type: {type(data)}")
    print(f"Data: ")
    
    if isinstance(data, dict):
        print("Dictionary contents:")
        pprint(data, indent=2, width=100)
    elif isinstance(data, (list, tuple)):
        print(f"Sequence with {len(data)} items:")
        for i, item in enumerate(data):
            print(f"  [{i}]: {item}")
    else:
        print(data)


def main():
    if len(sys.argv) != 2:
        print("Usage: python read_pkl.py <path_to_pkl_file>")
        print("Example: python read_pkl.py multiple_runs/run_00/convergence_results.pkl")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    print(f"Reading pickle file: {pkl_file}")
    print("-" * 50)
    
    data = read_pkl_file(pkl_file)
    if data is not None:
        print_data_info(data)


if __name__ == "__main__":
    main()