"""Find the true record stride in the PE indicator file by scanning for OLE dates."""
import struct
import datetime

fpath = r'C:\Users\Sager\AppData\Local\UniTicker\TCLive\FlatFiles\a22ce12861bfed7af141\66e5c5fe4d77037f9dd6\1003.dat'
data = open(fpath, 'rb').read()
print(f'File size: {len(data)}')

def ole_to_date(v):
    try:
        d = datetime.date(1899, 12, 30) + datetime.timedelta(days=int(v))
        if 1990 <= d.year <= 2030:
            return d
    except:
        pass
    return None

# Scan every 8-byte aligned position for a valid OLE date
date_positions = []
for i in range(len(data) - 7):
    if i % 8 != 0:
        continue
    try:
        v = struct.unpack_from('<d', data, i)[0]
        d = ole_to_date(v)
        if d:
            date_positions.append((i, d, v))
    except:
        pass

print(f'Valid date positions (8-byte aligned): {len(date_positions)}')
if date_positions:
    # Find intervals between consecutive dates
    intervals = {}
    for idx in range(1, len(date_positions)):
        diff = date_positions[idx][0] - date_positions[idx-1][0]
        intervals[diff] = intervals.get(diff, 0) + 1
    
    # Sort by frequency
    sorted_intervals = sorted(intervals.items(), key=lambda x: -x[1])
    print('Most common byte-intervals between dates:')
    for stride, count in sorted_intervals[:10]:
        print(f'  stride={stride} bytes: {count} occurrences')
    
    # Show first 10 date positions and surrounding values
    print('\nFirst 10 date positions with neighbors:')
    for pos, d, v in date_positions[:10]:
        # Read next 4 bytes as float32 (the indicator value)
        val = 'N/A'
        if pos + 12 <= len(data):
            f32 = struct.unpack_from('<f', data, pos+8)[0]
            val = f'{f32:.4f}'
        print(f'  offset={pos:7d}  date={d}  val4={val}')
