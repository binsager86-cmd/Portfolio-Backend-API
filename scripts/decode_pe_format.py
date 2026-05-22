import struct, datetime

fpath = r'C:\Users\Sager\AppData\Local\UniTicker\TCLive\FlatFiles\a22ce12861bfed7af141\66e5c5fe4d77037f9dd6\1003.dat'
data = open(fpath,'rb').read()
print(f'File size: {len(data)}')

def ole(v):
    try:
        d = datetime.date(1899,12,30) + datetime.timedelta(days=int(v))
        if 1990 <= d.year <= 2030: return str(d)
    except: pass
    return None

# 32-byte OHLCV: date(8)+o(4)+h(4)+l(4)+c(4)+vol(8)
print('--- 32-byte OHLCV ---')
for i in range(5):
    offs = i*32
    d_val, o, h, l, c = struct.unpack_from('<dffff', data, offs)
    vol = struct.unpack_from('<d', data, offs+24)[0]
    print(f'  [{i}] date={ole(d_val) or round(d_val,1)} O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} vol={vol:.0f}')
print(f'  32-byte: {len(data)//32} records leftover={len(data)%32}')

# 28-byte: date(8)+o(4)+h(4)+l(4)+c(4)+vol(4)
print('--- 28-byte ---')
for i in range(5):
    offs = i*28
    d_val, o, h, l, c, vol = struct.unpack_from('<dfffff', data, offs)
    print(f'  [{i}] date={ole(d_val) or round(d_val,1)} O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} vol={vol:.0f}')
print(f'  28-byte: {len(data)//28} records leftover={len(data)%28}')

# 24-byte: date(8)+o(4)+h(4)+l(4)+c(4)
print('--- 24-byte date+OHLC ---')
for i in range(5):
    offs = i*24
    d_val, o, h, l, c = struct.unpack_from('<dffff', data, offs)
    print(f'  [{i}] date={ole(d_val) or round(d_val,1)} O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}')
print(f'  24-byte: {len(data)//24} records leftover={len(data)%24}')

# The 12-byte scan gave 9490 PE-like records
# Let's check if within the 32-byte format, the CLOSE PE makes sense
print('\n--- 32-byte: Valid date scan ---')
n = len(data) // 32
valid = []
for i in range(n):
    offs = i*32
    try:
        d_val, o, h, l, c = struct.unpack_from('<dffff', data, offs)
        dt = ole(d_val)
        if dt:
            valid.append((dt, o, h, l, c))
    except: pass
print(f'  Valid records: {len(valid)}')
if valid:
    closes = [c for _,_,_,_,c in valid]
    pe_closes = [c for c in closes if 3 < c < 200]
    print(f'  Close value range: {min(closes):.2f} to {max(closes):.2f}')
    print(f'  PE-like closes (3-200): {len(pe_closes)}')
    print(f'  Sample: {valid[:3]}')
    print(f'  Last: {valid[-3:]}')
