import json,sys
F="C:/Users/seal/.claude/projects/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/subagents/agent-a84bf350cacff3358.jsonl"
res={}
rows=[json.loads(l) for l in open(F,encoding='utf-8')]
for r in rows:
    ts=r.get('timestamp','')
    if ts<'2026-09-23T02:13': continue  # UTC; 07:43 IST = 02:13 UTC
    m=r.get('message',{})
    c=m.get('content')
    if isinstance(c,list):
        for b in c:
            if b.get('type')=='tool_use':
                s=json.dumps(b.get('input'),ensure_ascii=False)
                print(ts, 'USE', b.get('id')[-6:], s[:int(sys.argv[1])].replace('\n',' | '))
            elif b.get('type')=='tool_result':
                cc=b.get('content'); s=json.dumps(cc,ensure_ascii=False) if not isinstance(cc,str) else cc
                print(ts, 'RES', b.get('tool_use_id')[-6:], s[:int(sys.argv[2])].replace('\n',' | '))
