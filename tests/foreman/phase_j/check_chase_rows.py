"""List all Chase rows from record_L.jsonl"""
import json

with open("record_L.jsonl", "r", encoding='utf-8') as f:
    lines = f.readlines()

chase_rows = []
for line_num, line in enumerate(lines, 1):
    try:
        row = json.loads(line)
        if row.get("seat") == "Chase":
            chase_rows.append((line_num, row))
            print(f"Row {line_num}: {row.get('row')}")
            print(f"  Nurse: {row.get('nurse')}")
            print(f"  Verdict: {row.get('verdict')}")
            if row.get('measured'):
                print(f"  Measured keys: {list(row.get('measured', {}).keys())[:3]}")
            print()
    except json.JSONDecodeError as e:
        print(f"Error parsing line {line_num}: {e}")

print(f"Total Chase rows: {len(chase_rows)}")
