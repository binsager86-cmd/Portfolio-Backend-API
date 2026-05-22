#!/usr/bin/env python3
"""
frida_capture_cli.py
--------------------
Captures ALL HTTP requests from TickerChartLive.exe by driving frida.exe
via subprocess (avoids the Python-API enumerate_processes bug in Frida 17.x).

- Uses tasklist.exe to find the PID (no frida enumerate needed)
- Writes a FRIDA_JSON:-prefixed line per event from the JS side
- Deduplicates + categorises every URL on the fly
- On Ctrl-C: prints summary + saves frida_api_catalogue.json

Usage:  python frida_capture_cli.py
"""

import sys, os, re, json, signal, time, subprocess, threading
from collections import defaultdict

FRIDA_EXE   = r"C:\Users\Sager\OneDrive\Desktop\portfolio_app\.venv\Scripts\frida.exe"
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE    = os.path.join(OUT_DIR, "frida_api_log.json")
CAT_FILE    = os.path.join(OUT_DIR, "frida_api_catalogue.json")

# ─── Frida JS (console.log only – no send()) ──────────────────────────────────
JS = r"""
'use strict';
const whConns=new Map(), whReqs=new Map(), wiConns=new Map(), wiReqs=new Map();

function u16(p){try{return(p&&!p.isNull())?p.readUtf16String():null;}catch(_){return null;}}
function i32(p){try{return p.toInt32();}catch(_){return 0;}}
function url(s,port,path){
  const h=s||'?',pp=path||'/';
  const sp=port&&port!==80&&port!==443&&port!==0;
  return'https://'+h+(sp?':'+port:'')+pp;
}
function emit(o){console.log('FRIDA_JSON:'+JSON.stringify(o));}
function hook(m,f,cb){
  try{Interceptor.attach(Module.getExportByName(m,f),cb);emit({t:'log',msg:'[+] '+m+'!'+f});}
  catch(e){emit({t:'log',msg:'[-] '+m+'!'+f+': '+e});}
}

hook('winhttp.dll','WinHttpConnect',{
  onEnter(a){this.s=u16(a[1]);this.p=i32(a[2]);},
  onLeave(r){if(!r.isNull())whConns.set(r.toString(),{s:this.s,p:this.p});}
});
hook('winhttp.dll','WinHttpOpenRequest',{
  onEnter(a){const c=whConns.get(a[0].toString())||{};
    this.req={s:c.s,p:c.p,m:u16(a[1])||'GET',path:u16(a[2])||'/',h:''};},
  onLeave(r){if(!r.isNull())whReqs.set(r.toString(),this.req);}
});
hook('winhttp.dll','WinHttpAddRequestHeaders',{
  onEnter(a){const r=whReqs.get(a[0].toString());if(!r)return;
    const h=u16(a[1]);if(h)r.h+=(r.h?'|':'')+h.replace(/\r\n/g,'|');}
});
hook('winhttp.dll','WinHttpSendRequest',{
  onEnter(a){const r=whReqs.get(a[0].toString());if(!r)return;
    const ex=u16(a[1])||'';
    emit({t:'req',tr:'wh',m:r.m,url:url(r.s,r.p,r.path),h:r.h+(ex?'|'+ex:''),ts:Date.now()});}
});
hook('winhttp.dll','WinHttpReadData',{
  onEnter(a){this.h=a[0].toString();this.b=a[1];this.l=i32(a[2]);},
  onLeave(r){try{const req=whReqs.get(this.h);if(!req)return;
    if(this.b&&!this.b.isNull()&&this.l>0){
      const body=this.b.readUtf8String(Math.min(this.l,8192));
      if(body&&body.trim())emit({t:'resp',url:url(req.s,req.p,req.path),body:body.substring(0,4096)});
    }}catch(_){}}
});

hook('wininet.dll','InternetConnectW',{
  onEnter(a){this.s=u16(a[1]);this.p=i32(a[3]);},
  onLeave(r){if(!r.isNull())wiConns.set(r.toString(),{s:this.s,p:this.p});}
});
hook('wininet.dll','HttpOpenRequestW',{
  onEnter(a){const c=wiConns.get(a[0].toString())||{};
    this.req={s:c.s,p:c.p,m:u16(a[1])||'GET',path:u16(a[2])||'/',h:''};},
  onLeave(r){if(!r.isNull())wiReqs.set(r.toString(),this.req);}
});
hook('wininet.dll','HttpAddRequestHeadersW',{
  onEnter(a){const r=wiReqs.get(a[0].toString());if(!r)return;
    const h=u16(a[1]);if(h)r.h+=(r.h?'|':'')+h.replace(/\r\n/g,'|');}
});
hook('wininet.dll','HttpSendRequestW',{
  onEnter(a){const r=wiReqs.get(a[0].toString());if(!r)return;
    const ex=u16(a[1])||'';
    emit({t:'req',tr:'wi',m:r.m,url:url(r.s,r.p,r.path),h:r.h+(ex?'|'+ex:''),ts:Date.now()});}
});
hook('wininet.dll','InternetReadFile',{
  onEnter(a){this.h=a[0].toString();this.b=a[1];this.l=i32(a[2]);},
  onLeave(r){try{const req=wiReqs.get(this.h);if(!req)return;
    if(this.b&&!this.b.isNull()&&this.l>0){
      const body=this.b.readUtf8String(Math.min(this.l,8192));
      if(body&&body.trim())emit({t:'resp',url:url(req.s,req.p,req.path),body:body.substring(0,4096)});
    }}catch(_){}}
});

emit({t:'log',msg:'[*] Hooks ready — navigate TickerChart now.'});
"""

# ─── Categories ───────────────────────────────────────────────────────────────
CATS = [
    ("indicators / technical",    re.compile(r'indicator|technical|rsi|macd|bollinger|moving.av|ema|sma|stoch|atr|cci|obv', re.I)),
    ("financials / fundamentals", re.compile(r'financial|fundamental|balance|income|cashflow|revenue|earnings|eps|ebitda|net.income', re.I)),
    ("ratios / valuation",        re.compile(r'ratio|valuat|pe|p-e|p_e|price.earn|forward.pe|yield|dividend', re.I)),
    ("market info / lookup",      re.compile(r'market.info|company.info|lookup|symbol|companyid|ticker|search', re.I)),
    ("prices / quotes",           re.compile(r'price|quote|ohlc|candle|bar|chart|history|snapshot|close|open|high|low', re.I)),
    ("screener",                  re.compile(r'screen|filter|scan', re.I)),
    ("news / calendar",           re.compile(r'news|event|calendar|dividend.date|earning.date', re.I)),
    ("auth / session",            re.compile(r'auth|login|token|session|oauth|refresh', re.I)),
    ("other",                     re.compile(r'.*')),
]
SKIP = re.compile(r'\.(js|css|png|jpg|svg|woff|woff2|ico|gif|webp|ttf|eot|map)(\?|$)', re.I)

# ─── State ────────────────────────────────────────────────────────────────────
raw_log   = []
seen      = {}        # norm_url -> record
responses = {}        # norm_url -> body
catalogue = defaultdict(list)

def categorise(url):
    for name, rx in CATS:
        if rx.search(url):
            return name
    return "other"

def normalise(url):
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        path = re.sub(r'/\d+', '/{id}', p.path)
        return f"{p.scheme}://{p.netloc}{path}"
    except Exception:
        return url

# ─── Event handler ────────────────────────────────────────────────────────────
def handle(payload):
    t = payload.get('t')
    if t == 'log':
        print(payload.get('msg',''), flush=True)
        return

    if t == 'req':
        url   = payload.get('url','')
        meth  = payload.get('m','GET')
        hdrs  = payload.get('h','')
        if SKIP.search(url):
            return
        cat  = categorise(url)
        norm = normalise(url)
        rec  = {'url': url, 'norm': norm, 'method': meth,
                'category': cat, 'headers': hdrs[:300]}
        raw_log.append(rec)

        tag = f'  [{cat.upper()}]' if cat not in ('other','auth / session') else ''
        print(f"[{meth}] {url}{tag}", flush=True)

        if norm not in seen:
            seen[norm] = rec
            catalogue[cat].append(rec)
            if cat not in ('other', 'auth / session'):
                print(f"\n  *** NEW: [{cat}]  {meth} {url}\n", flush=True)

    elif t == 'resp':
        url  = payload.get('url','')
        body = payload.get('body','')
        norm = normalise(url)
        if norm not in responses and body.strip():
            responses[norm] = body[:2000]
            cat = categorise(url)
            if cat != 'other':
                print(f"\n  [RESP] {url}", flush=True)
                print(f"  {body[:300]}\n", flush=True)

# ─── Output ───────────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "═"*72)
    print("  TICKERCHART API CATALOGUE")
    print("═"*72)
    for name, _ in CATS:
        items = catalogue.get(name, [])
        if not items:
            continue
        print(f"\n▶ {name.upper()} ({len(items)})")
        for r in items:
            print(f"    {r['method']}  {r['url']}")
            norm = r['norm']
            if norm in responses:
                print(f"         → {responses[norm][:120]}")
    print("\n" + "═"*72)

def save():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(raw_log, f, indent=2)
    cat_out = {}
    for name, items in catalogue.items():
        cat_out[name] = []
        for r in items:
            e = {'method': r['method'], 'url': r['url']}
            if r['norm'] in responses:
                e['response_snippet'] = responses[r['norm']][:500]
            cat_out[name].append(e)
    with open(CAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cat_out, f, indent=2)
    print(f"\n[*] Log      → {LOG_FILE}")
    print(f"[*] Catalogue→ {CAT_FILE}")

# ─── Find PID via tasklist ────────────────────────────────────────────────────
def find_pid(name='TickerChartLive.exe'):
    result = subprocess.run(
        ['tasklist', '/FI', f'IMAGENAME eq {name}', '/FO', 'CSV', '/NH'],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        parts = [p.strip('"') for p in line.split(',')]
        if parts and parts[0].lower() == name.lower():
            try:
                return int(parts[1])
            except (IndexError, ValueError):
                pass
    return None

# ─── Reader thread ────────────────────────────────────────────────────────────
def reader(proc):
    prefix = 'FRIDA_JSON:'
    for line in proc.stdout:
        line = line.rstrip('\n\r')
        if line.startswith(prefix):
            try:
                handle(json.loads(line[len(prefix):]))
            except json.JSONDecodeError:
                pass
        elif line.strip():
            # Print non-JSON frida output for debugging
            print(f'[frida] {line}', flush=True)

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    # Write JS to a temp file so frida.exe can load it
    js_path = os.path.join(OUT_DIR, '_frida_tmp.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(JS)

    # Poll for TickerChart
    pid = None
    while pid is None:
        pid = find_pid()
        if pid is None:
            print("[*] Waiting for TickerChartLive.exe …", flush=True)
            time.sleep(2)

    print(f"[*] Found TickerChartLive.exe — PID {pid}", flush=True)
    print(f"[*] Attaching via PowerShell pipeline (captures WriteConsole output) …\n", flush=True)

    # PowerShell virtualises the console for child processes so its pipeline
    # captures frida's WriteConsole output that a plain Python pipe cannot.
    ps_cmd = (
        f'& "{FRIDA_EXE}" -p {pid} -l "{js_path}" 2>&1 | '
        f'ForEach-Object {{ Write-Output $_ }}'
    )
    proc = subprocess.Popen(
        ['powershell', '-NoProfile', '-Command', ps_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    t = threading.Thread(target=reader, args=(proc,), daemon=True)
    t.start()

    print("[*] Navigate TickerChart — open charts, indicators, financials, screener.")
    print("[*] Press Ctrl-C when done.\n", flush=True)

    def stop(*_):
        proc.terminate()
        print_summary()
        save()
        try:
            os.remove(js_path)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)

    try:
        proc.wait()
    except KeyboardInterrupt:
        stop()

if __name__ == '__main__':
    main()
