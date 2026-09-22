import json
import sys

def walk_paths(obj, path=""):
    """Yields all leaf paths and their values."""
    if isinstance(obj, dict):
        if not obj:  # empty dict is a leaf
            yield (path, obj)
        else:
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                yield from walk_paths(v, new_path)
    elif isinstance(obj, list):
        if not obj:  # empty list is a leaf
            yield (path, obj)
        else:
            for i, v in enumerate(obj):
                new_path = f"{path}[{i}]"
                yield from walk_paths(v, new_path)
    else:
        # scalar: int, float, str, bool, None
        yield (path, obj)

def compare_jsons(orig_file, rerun_file):
    """Compares two JSON files and reports differing leaves."""
    with open(orig_file, 'r') as f:
        orig = json.load(f)
    with open(rerun_file, 'r') as f:
        rerun = json.load(f)
    
    # Build dicts of path -> value
    orig_paths = {path: val for path, val in walk_paths(orig)}
    rerun_paths = {path: val for path, val in walk_paths(rerun)}
    
    all_paths = set(orig_paths.keys()) | set(rerun_paths.keys())
    
    diffs = []
    for path in sorted(all_paths):
        orig_val = orig_paths.get(path, "MISSING")
        rerun_val = rerun_paths.get(path, "MISSING")
        if orig_val != rerun_val:
            diffs.append((path, orig_val, rerun_val))
    
    # Print first 80 diffs or all if fewer
    for i, (path, orig_val, rerun_val) in enumerate(diffs[:80]):
        print(f"{path}")
        print(f"  original: {orig_val}")
        print(f"  rerun:    {rerun_val}")
    
    print(f"differing leaves: {len(diffs)} of {len(all_paths)}")
    return len(diffs)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: jsondiff.py <orig.json> <rerun.json>")
        sys.exit(1)
    compare_jsons(sys.argv[1], sys.argv[2])
