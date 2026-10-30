"""
Check if there's a sys.path issue causing wrong rotasim to be imported
"""
import sys
import os

print("="*60)
print("SYS.PATH ANALYSIS")
print("="*60)

print(f"\nCurrent working directory: {os.getcwd()}")
print(f"\nScript location: {__file__}")

print("\nsys.path entries:")
for i, path in enumerate(sys.path):
    print(f"  [{i}] {path}")
    # Check if rotasim exists at this path
    rotasim_at_path = os.path.join(path, 'rotasim')
    if os.path.exists(rotasim_at_path):
        is_file = os.path.isfile(rotasim_at_path)
        is_dir = os.path.isdir(rotasim_at_path)
        has_init = os.path.exists(os.path.join(rotasim_at_path, '__init__.py')) if is_dir else False
        print(f"       -> 'rotasim' found here! (file={is_file}, dir={is_dir}, has_init={has_init})")

print("\n" + "="*60)
print("IMPORT TEST")
print("="*60)

import rotasim
print(f"\nrotasim imported from: {rotasim.__file__}")
print(f"rotasim.__path__: {rotasim.__path__ if hasattr(rotasim, '__path__') else 'N/A'}")

# Check what's in the directory
rotasim_dir = os.path.dirname(rotasim.__file__)
print(f"\nContents of rotasim directory:")
files = os.listdir(rotasim_dir)
for f in sorted(files)[:20]:  # Show first 20 files
    print(f"  - {f}")

print("\n" + "="*60)
print("THE PROBLEM")
print("="*60)

# The issue: when running from calibration/, sys.path[0] is the calibration directory
# and the PARENT directory (rotasim/) might be interfering

calibration_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim/rotasim/calibration'
parent_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim/rotasim'

if calibration_path in sys.path:
    print(f"✓ Calibration dir is in sys.path at index: {sys.path.index(calibration_path)}")

if parent_path in sys.path:
    print(f"✓ Parent rotasim dir is in sys.path at index: {sys.path.index(parent_path)}")
    print("\n⚠️  THIS IS THE PROBLEM!")
    print("The parent 'rotasim' directory is in sys.path BEFORE the installed package.")
    print("Python finds rotasim/calibration/ first and treats 'rotasim' as a namespace package.")
