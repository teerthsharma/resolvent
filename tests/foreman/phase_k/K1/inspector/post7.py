# Inspector pass 7: append audit lines (one JSON object per line) to the board and to audit_events_pass7.jsonl.
import json, sys
B = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
with open(B, "a", encoding="utf-8") as b, open("audit_events_pass7.jsonl", "a", encoding="utf-8") as a:
    for r in rows:
        s = json.dumps({"t": "audit", "agent": "Inspector", **r}, ensure_ascii=False)
        b.write(s + "\n"); a.write(s + "\n")
print(len(rows), "posted")
