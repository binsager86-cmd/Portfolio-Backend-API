"""
Scan all files in 320800594fe439050088 (PE indicator dir) to find PE value ranges
per company. Try to identify KSE companies by cross-referencing expected PE ranges.
"""
import struct, datetime, os

BASE = r'C:\Users\Sager\AppData\Local\UniTicker\TCLive\FlatFiles\aa3ba405d27847645e3d\320800594fe439050088'

def ole_to_date(v):
    try:
        d = datetime.date(1899,12,30) + datetime.timedelta(days=int(v))
        if 1990 <= d.year <= 2030: return d
    except: pass
    return None

def decode_pe_file(fpath):
    data = open(fpath,'rb').read()
    size = len(data)
    n = size // 40
    records = []
    for i in range(n):
        try:
            d_val, o, h, l, c = struct.unpack_from('<dffff', data, i*40)
            dt = ole_to_date(d_val)
            if dt:
                records.append((dt, c))  # use close value
        except: pass
    return records

files = sorted(os.listdir(BASE))
results = []
for fname in files:
    if not fname.endswith('.dat'):
        continue
    fid = fname.replace('.dat','')
    fpath = os.path.join(BASE, fname)
    mtime = os.path.getmtime(fpath)
    try:
        records = decode_pe_file(fpath)
        if not records:
            continue
        vals = [v for _,v in records]
        # Only PE-like values (3-80 for typical stocks)
        pe_vals = [v for v in vals if 3 < v < 80]
        if len(pe_vals) < 50:
            continue
        min_v, max_v = min(pe_vals), max(pe_vals)
        # Get date range
        dates = [d for d,_ in records if 3 < records[0][1] < 80]
        dates_pe = [d for d,v in records if 3 < v < 80]
        mdate = datetime.datetime.fromtimestamp(mtime).strftime('%m/%d %H:%M')
        results.append((fid, min_v, max_v, len(pe_vals), dates_pe[0], dates_pe[-1], mdate))
    except Exception as e:
        pass

results.sort(key=lambda x: x[0])
print(f'Files with PE-like values (3-80): {len(results)}')
print(f'{"FileID":>8}  {"Min PE":>7}  {"Max PE":>7}  {"Count":>6}  {"FirstDate":>12}  {"LastDate":>12}  {"Modified":>14}')
for fid, mn, mx, cnt, fd, ld, mdate in results:
    print(f'{fid:>8}  {mn:>7.2f}  {mx:>7.2f}  {cnt:>6}  {fd!s:>12}  {ld!s:>12}  {mdate:>14}')
