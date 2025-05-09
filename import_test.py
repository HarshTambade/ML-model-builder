try:
    from utils.helpers import load_css
    print("Import successful!")
except ImportError as e:
    print(f"Import error: {e}")
    
    # Try with a sys.path modification
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))
    print("Modified Python path, trying again...")
    
    try:
        from utils.helpers import load_css
        print("Import successful after path modification!")
    except ImportError as e:
        print(f"Still failing: {e}")
        
        # Look for the helpers module
        import glob
        print("\nLooking for utils/helpers.py:")
        for path in glob.glob("**/helpers.py", recursive=True):
            print(f"Found: {path}") 