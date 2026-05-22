"""
Watch TickerChart FlatFiles for new/changed .dat files.
Run this BEFORE opening the PE indicator in TickerChart.
When TC writes new data, this script prints the file and decodes its content.

Usage:
    python mobile-migration/backend-api/scripts/watch_flatfiles.py
"""
import os
import struct
import datetime
import time

FLATFILES = r'C:\Users\Sager\AppData\Local\UniTicker\TCLive\FlatFiles'
WATCH_IDS = {'282'}  # NBK company ID

def ole_to_date(v):
    try:
        d = datetime.date(1899, 12, 30) + datetime.timedelta(days=int(v))
        if 1990 <= d.year <= 2030:
            return d
    except:
        pass
    return None


def decode_file(fpath: str):
    data = open(fpath, 'rb').read()
    size = len(data)
    
    # Try 40-byte records (indicator OHLC format)
    if size % 40 == 0 and size >= 40:
        records = []
        for i in range(size // 40):
            offs = i * 40
            try:
                d_val, o, h, l, c = struct.unpack_from('<dffff', data, offs)
                dt = ole_to_date(d_val)
                if dt:
                    records.append((dt, o, h, l, c))
            except:
                pass
        if records:
            vals = [c for _, _, _, _, c in records]
            pe_like = [c for c in vals if 3 < c < 300]
            return f"40-byte: {len(records)} records, val range [{min(vals):.2f},{max(vals):.2f}], PE-like: {len(pe_like)}"
    
    # Try 12-byte records (OHLCV close-only format)
    n = size // 12
    records12 = []
    for i in range(n):
        try:
            d_val, v = struct.unpack_from('<df', data, i * 12)
            dt = ole_to_date(d_val)
            if dt:
                records12.append((dt, v))
        except:
            pass
    if records12:
        vals = [v for _, v in records12]
        pe_like = [v for v in vals if 3 < v < 300]
        return f"12-byte: {len(records12)} records, val range [{min(vals):.2f},{max(vals):.2f}], PE-like: {len(pe_like)}"
    
    return f"Unknown format, size={size}"


def snapshot():
    """Return dict of {filepath: mtime} for all .dat files."""
    state = {}
    for root, dirs, files in os.walk(FLATFILES):
        for fname in files:
            if fname.endswith('.dat'):
                fpath = os.path.join(root, fname)
                try:
                    state[fpath] = os.path.getmtime(fpath)
                except:
                    pass
    return state


def main():
    print(f"Watching: {FLATFILES}")
    print(f"Looking for company IDs: {WATCH_IDS}")
    print("Open TickerChart -> select NBK -> add 'Price to Earnings (LTM)' indicator")
    print("Press Ctrl+C to stop\n")
    
    prev = snapshot()
    print(f"Baseline: {len(prev)} files")
    
    while True:
        time.sleep(1)
        curr = snapshot()
        
        for fpath, mtime in curr.items():
            fname = os.path.basename(fpath)
            company_id = fname.replace('.dat', '')
            
            # New file or modified file
            if fpath not in prev or prev[fpath] != mtime:
                rel = fpath.replace(FLATFILES + '\\', '')
                action = "NEW" if fpath not in prev else "MODIFIED"
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] {action}: {rel}")
                
                if company_id in WATCH_IDS:
                    print(f"  *** COMPANY 282 (NBK) FOUND! ***")
                    try:
                        info = decode_file(fpath)
                        print(f"  {info}")
                    except Exception as e:
                        print(f"  Decode error: {e}")
                else:
                    print(f"  Company ID: {company_id}")
        
        prev = curr


if __name__ == '__main__':
    main()
