import sys
import importlib
import os

original_syspath = sys.path.copy()
syspath_changed = False
change_location = None

def trace_lines(frame, event, arg):
    global syspath_changed, change_location
    if event != 'line':
        return trace_lines
    
    current_syspath = sys.path.copy()
    if current_syspath != original_syspath and not syspath_changed:
        syspath_changed = True
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        change_location = (filename, lineno)
        
        # 停止跟踪
        sys.settrace(None)
    
    return trace_lines

def import_and_track(module_name):
    global syspath_changed, change_location
    
    print(f"Original sys.path:")
    for path in original_syspath:
        print(f"  {path}")
    
    print(f"\nImporting {module_name}...")
    
    # 设置跟踪
    sys.settrace(trace_lines)
    
    try:
        module = importlib.import_module(module_name)
        print(f"Successfully imported {module_name}")
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return
    finally:
        # 确保跟踪被停止
        sys.settrace(None)
    
    if syspath_changed:
        print(f"\nsys.path was changed during import.")
        print(f"Change occurred in file: {change_location[0]}")
        print(f"At line number: {change_location[1]}")
        
        # 显示变化的行
        with open(change_location[0], 'r') as file:
            lines = file.readlines()
            print("\nThe line that caused the change:")
            print(lines[change_location[1] - 1].strip())
        
        print("\nNew paths added to sys.path:")
        for path in sys.path:
            if path not in original_syspath:
                print(f"  + {path}")
    else:
        print("\nsys.path was not changed during import.")

if __name__ == "__main__":
    module_to_import = input("Enter the name of the module to import: ")
    import_and_track(module_to_import)
