"""
Test each import from rotasim/__init__.py to find which one is failing
"""
import sys
import traceback

print("="*60)
print("TESTING EACH IMPORT FROM rotasim/__init__.py")
print("="*60)

# Test each import in order
imports = [
    ("version", "from rotasim.version import __version__, __versiondate__"),
    ("utils", "from rotasim.utils import *"),
    ("rotavirus", "from rotasim.rotavirus import *"),
    ("immunity", "from rotasim.immunity import *"),
    ("reassortment", "from rotasim.reassortment import *"),
    ("interventions", "from rotasim.interventions import *"),
    ("analyzers", "from rotasim.analyzers import *"),
    ("aging", "from rotasim.aging import *"),
    ("age_networks", "from rotasim.age_networks import *"),
    ("rotasim", "from rotasim.rotasim import *"),
]

failed_import = None

for name, import_statement in imports:
    try:
        exec(import_statement)
        print(f"✓ {name:<20} SUCCESS")
    except Exception as e:
        print(f"✗ {name:<20} FAILED: {e}")
        print(f"\nFull traceback for {name}:")
        traceback.print_exc()
        failed_import = name
        break

if failed_import:
    print("\n" + "="*60)
    print(f"FOUND THE PROBLEM: {failed_import} import is failing!")
    print("="*60)
else:
    print("\n" + "="*60)
    print("All imports succeeded individually")
    print("The issue may be with __init__.py execution context")
    print("="*60)
