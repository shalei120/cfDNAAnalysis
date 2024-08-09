import sys
import os
import importlib

original_syspath = sys.path.copy()

def track_syspath_changes(module_name):
    print(f"Original sys.path:")
    for path in original_syspath:
        print(f"  {path}")
    
    print(f"\nImporting {module_name}...")
    try:
        module = importlib.import_module(module_name)
        print(f"Successfully imported {module_name}")
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return

    print("\nNew paths added to sys.path:")
    for path in sys.path:
        if path not in original_syspath:
            print(f"  + {path}")

    print("\nAnalyzing the imported module:")
    if hasattr(module, '__file__'):
        print(f"Module file: {module.__file__}")
        module_dir = os.path.dirname(module.__file__)
        print(f"Module directory: {module_dir}")
        if module_dir in sys.path:
            print(f"Note: The module's directory has been added to sys.path")
    else:
        print("This module doesn't have a __file__ attribute (it might be a built-in module)")

    if hasattr(module, '__path__'):
        print("This is a package. Package paths:")
        for path in module.__path__:
            print(f"  {path}")

    print("\nChecking for sys.path modifications in the module:")
    with open(module.__file__, 'r') as f:
        content = f.read()
        if 'sys.path' in content:
            print("Warning: The module contains code that might modify sys.path")
        else:
            print("The module doesn't seem to directly modify sys.path")

if __name__ == "__main__":
    module_to_import = input("Enter the name of the module to import: ")
    track_syspath_changes(module_to_import)

# 保存为 syspath_tracker.py 并运行 python syspath_tracker.py
