import sys
import os

def print_sys_path():
    print("Current sys.path:")
    for index, path in enumerate(sys.path):
        print(f"{index}: {path}")

def remove_path(index):
    if 0 <= index < len(sys.path):
        removed_path = sys.path.pop(index)
        print(f"Removed path: {removed_path}")
    else:
        print(f"Invalid index: {index}")

def add_path(path):
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added path: {path}")
    else:
        print(f"Invalid path or path already exists: {path}")

def analyze_sys_path():
    print("\nAnalyzing sys.path:")
    for path in sys.path:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
        elif path == '':
            print(f"Warning: Empty string in sys.path")

if __name__ == "__main__":
    print_sys_path()
    analyze_sys_path()

    while True:
        print("\nOptions:")
        print("1: Remove a path")
        print("2: Add a path")
        print("3: Print current sys.path")
        print("4: Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            index = int(input("Enter the index of the path to remove: "))
            remove_path(index)
        elif choice == '2':
            path = input("Enter the path to add: ")
            add_path(path)
        elif choice == '3':
            print_sys_path()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

    print("\nFinal sys.path:")
    print_sys_path()

# 保存这个脚本为 syspath_cleanup.py 并运行 python syspath_cleanup.py
