"""Inspector N2: compare the 00:40:20 compiled rpos.py (pre-edit) against the 00:50:14 source, per function."""
import marshal, types, struct, time
pyc = r"../foreman/__pycache__/rpos.cpython-311.pyc"
b = open(pyc, "rb").read()
flags, mtime, size = struct.unpack("<III", b[4:16])
old = marshal.loads(b[16:])
src = open(r"../foreman/rpos.py", encoding="utf-8").read()
new = compile(src, r"../foreman/rpos.py", "exec")
print("pyc source mtime", time.strftime("%H:%M:%S", time.localtime(mtime)), "size", size, "current size", len(src.encode()))
def funcs(co):
    return {c.co_name: c for c in co.co_consts if isinstance(c, types.CodeType)}
fo, fn = funcs(old), funcs(new)
def sig(c):
    return (c.co_code, tuple(x if not isinstance(x, types.CodeType) else sig(x) for x in c.co_consts), c.co_names, c.co_varnames)
for k in sorted(set(fo) | set(fn)):
    same = k in fo and k in fn and sig(fo[k]) == sig(fn[k])
    print("%-20s %s" % (k, "SAME" if same else "DIFF"))
print("module consts same except funcs:", [x for x in old.co_consts if not isinstance(x, types.CodeType)] == [x for x in new.co_consts if not isinstance(x, types.CodeType)])
