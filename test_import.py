"""
Test script to diagnose rotasim import issues
"""
import sys
print("Python executable:", sys.executable)
print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\n" + "="*60)
print("Attempting to import rotasim...")
print("="*60)

try:
    import rotasim as rs
    print("✓ Successfully imported rotasim")
    print(f"Module location: {rs.__file__}")
    print(f"Has Sim attribute: {hasattr(rs, 'Sim')}")

    if hasattr(rs, 'Sim'):
        print(f"Sim class: {rs.Sim}")
        print(f"Sim location: {rs.Sim.__module__}")
    else:
        print("\n✗ ERROR: rotasim module does not have 'Sim' attribute")
        print("\nAvailable attributes:")
        attrs = [a for a in dir(rs) if not a.startswith('_')]
        for attr in sorted(attrs):
            print(f"  - {attr}")

except Exception as e:
    print(f"\n✗ ERROR during import: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing calibrate_uk.py import pattern...")
print("="*60)

# Simulate what calibrate_uk.py does
try:
    import rotasim as rs
    sim = rs.Sim(n_agents=100)
    print("✓ Successfully created rs.Sim instance")
except AttributeError as e:
    print(f"✗ AttributeError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()
