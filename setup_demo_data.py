#!/usr/bin/env python3
"""
setup_demo_data.py
Copies validation dataset files to the dashboard data directory for demo.
Run this once before launching the dashboard.

Usage:
    python3 setup_demo_data.py --src /path/to/validation/files
"""

import argparse
import shutil
from pathlib import Path

REQUIRED_FILES = [
    'step1_idb_to_cl_validation_full.csv',
    'step2_validation_with_claude.xlsx',
    'step3_fr_lookup.csv',
]

def setup(src_dir: str):
    src = Path(src_dir)
    dst = Path(__file__).parent / 'data'
    dst.mkdir(exist_ok=True)

    print(f"Copying validation files to {dst}...")
    for fname in REQUIRED_FILES:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} not found in {src}")

    print(f"\nData directory ready: {dst}")
    print("Run the dashboard with: streamlit run app.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='.', help='Source directory containing validation files')
    args = parser.parse_args()
    setup(args.src)
