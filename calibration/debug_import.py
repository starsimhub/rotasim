"""
Debug script to identify rotasim import issue in PyCharm
Run this directly in PyCharm to see what's happening
"""
import sys
print("="*60)
print("PYTHON ENVIRONMENT INFO")
print("="*60)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"\nWorking directory: {sys.path[0]}")
print(f"\nsys.path entries:")
for i, p in enumerate(sys.path[:10]):
    print(f"  [{i}] {p}")

print("\n" + "="*60)
print("ATTEMPTING TO IMPORT ROTASIM")
print("="*60)

# Step 1: Try basic import
try:
    import rotasim
    print("✓ Step 1: import rotasim - SUCCESS")
    print(f"  Module location: {rotasim.__file__}")
except Exception as e:
    print(f"✗ Step 1: import rotasim - FAILED")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Check module contents
print("\n" + "="*60)
print("CHECKING MODULE CONTENTS")
print("="*60)
attrs = [a for a in dir(rotasim) if not a.startswith('_')]
print(f"Number of attributes: {len(attrs)}")
print(f"\nAll attributes:")
for attr in sorted(attrs):
    print(f"  - {attr}")

# Step 3: Check for Sim specifically
print("\n" + "="*60)
print("CHECKING FOR 'Sim' ATTRIBUTE")
print("="*60)
has_sim = hasattr(rotasim, 'Sim')
print(f"hasattr(rotasim, 'Sim'): {has_sim}")

if has_sim:
    print(f"✓ Sim is available: {rotasim.Sim}")
    print(f"  Sim module: {rotasim.Sim.__module__}")
else:
    print("✗ Sim is NOT available")
    print("\nLet's check if rotasim.rotasim module exists:")
    try:
        from rotasim import rotasim as rotasim_module
        print(f"  ✓ rotasim.rotasim module exists: {rotasim_module.__file__}")
        print(f"  Has Sim: {hasattr(rotasim_module, 'Sim')}")
        if hasattr(rotasim_module, 'Sim'):
            print(f"  Sim class: {rotasim_module.Sim}")
    except Exception as e:
        print(f"  ✗ Error accessing rotasim.rotasim: {e}")

# Step 4: Try the import pattern from calibrate_uk.py
print("\n" + "="*60)
print("TESTING calibrate_uk.py IMPORT PATTERN")
print("="*60)
try:
    import rotasim as rs
    print("✓ import rotasim as rs - SUCCESS")
    print(f"Has Sim: {hasattr(rs, 'Sim')}")

    if hasattr(rs, 'Sim'):
        print("✓ Creating test sim...")
        sim = rs.Sim(n_agents=100, scenario='single')
        print("✓ SUCCESS: Sim creation worked!")
    else:
        print("✗ FAILED: rs.Sim not found")

except AttributeError as e:
    print(f"✗ AttributeError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
