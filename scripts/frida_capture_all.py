#!/usr/bin/env python3
"""
Capture ALL HTTP requests from TickerChart, deduplicate, categorise, and
write a live JSON log + a final organised summary.

Drives the Frida CLI (frida.exe) via subprocess so it works regardless of
Python-API session issues in Frida 17.x.

Usage:
    python frida_capture_all.py
    → polls until TickerChartLive.exe is running, then attaches
    → navigate every section of TickerChart you care about
    → Ctrl+C to stop and print the final organised API catalogue
    → also writes: frida_api_log.json  (raw, all calls)
                   frida_api_catalogue.json  (deduplicated + categorised)
"""

import sys, signal, json, re, time, os, subprocess, threading, tempfile
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# JS payload – hooks WinHTTP + WinInet, captures requests + response bodies
# ──────────────────────────────────────────────────────────────────────────────
JS = r"""
'use strict';

const winHttpConns  = new Map();   // hConn  -> {server, port}
const winHttpReqs   = new Map();   // hReq   -> {server, port, method, path, headers}
const winInetConns  = new Map();
const winInetReqs   = new Map();

function u16(p) { try { return (p&&!p.isNull()) ? p.readUtf16String() : null; } catch(_){return null;} }
function i32(p) { try { return p.toInt32(); } catch(_){return 0;} }
function fmt(s, port, path) {
    const h = s||'?', pp = path||'/';
    const showPort = port && port !== 80 && port !== 443 && port !== 0;
    return 'https://' + h + (showPort ? ':'+port : '') + pp;
}
function hook(mod, fn, cbs) {
    try { Interceptor.attach(Module.getExportByName(mod,fn), cbs); send({t:'log',msg:'[+] '+mod+'!'+fn}); }
    catch(e){ send({t:'log',msg:'[-] '+mod+'!'+fn+': '+e}); }
}

// ── WinHTTP ──────────────────────────────────────────────────────────────────
hook('winhttp.dll','WinHttpConnect',{
    onEnter(a){ this.s=u16(a[1]); this.p=i32(a[2]); },
    onLeave(r){ if(!r.isNull()) winHttpConns.set(r.toString(),{server:this.s,port:this.p}); }
});
hook('winhttp.dll','WinHttpOpenRequest',{
    onEnter(a){
        const c=winHttpConns.get(a[0].toString())||{};
        this.req={server:c.server,port:c.port,method:u16(a[1])||'GET',path:u16(a[2])||'/',headers:''};
    },
    onLeave(r){ if(!r.isNull()) winHttpReqs.set(r.toString(),this.req); }
});
hook('winhttp.dll','WinHttpAddRequestHeaders',{
    onEnter(a){
        const req=winHttpReqs.get(a[0].toString()); if(!req) return;
        const h=u16(a[1]); if(h) req.headers+=(req.headers?'|':'')+h.replace(/\r\n/g,'|').trim();
    }
});
hook('winhttp.dll','WinHttpSendRequest',{
    onEnter(a){
        const req=winHttpReqs.get(a[0].toString()); if(!req) return;
        const extra=u16(a[1])||'';
        const hdrs=req.headers+(extra?'|'+extra.replace(/\r\n/g,'|'):'');
        send({t:'req',transport:'winhttp',method:req.method,url:fmt(req.server,req.port,req.path),headers:hdrs,ts:Date.now()});
    }
});
hook('winhttp.dll','WinHttpReadData',{
    onEnter(a){ this.h=a[0].toString(); this.buf=a[1]; this.len=i32(a[2]); },
    onLeave(r){
        try {
            const req=winHttpReqs.get(this.h); if(!req) return;
            if(this.buf&&!this.buf.isNull()&&this.len>0){
                const body=this.buf.readUtf8String(Math.min(this.len,8192));
                if(body&&body.trim()) send({t:'resp',url:fmt(req.server,req.port,req.path),body:body.substring(0,4096),ts:Date.now()});
            }
        } catch(_){}
    }
});

// ── WinInet ──────────────────────────────────────────────────────────────────
hook('wininet.dll','InternetConnectW',{
    onEnter(a){ this.s=u16(a[1]); this.p=i32(a[3]); },
    onLeave(r){ if(!r.isNull()) winInetConns.set(r.toString(),{server:this.s,port:this.p}); }
});
hook('wininet.dll','HttpOpenRequestW',{
    onEnter(a){
        const c=winInetConns.get(a[0].toString())||{};
        this.req={server:c.server,port:c.port,method:u16(a[1])||'GET',path:u16(a[2])||'/',headers:''};
    },
    onLeave(r){ if(!r.isNull()) winInetReqs.set(r.toString(),this.req); }
});
hook('wininet.dll','HttpAddRequestHeadersW',{
    onEnter(a){
        const req=winInetReqs.get(a[0].toString()); if(!req) return;
        const h=u16(a[1]); if(h) req.headers+=(req.headers?'|':'')+h.replace(/\r\n/g,'|').trim();
    }
});
hook('wininet.dll','HttpSendRequestW',{
    onEnter(a){
        const req=winInetReqs.get(a[0].toString()); if(!req) return;
        const extra=u16(a[1])||'';
        const hdrs=req.headers+(extra?'|'+extra.replace(/\r\n/g,'|'):'');
        send({t:'req',transport:'wininet',method:req.method,url:fmt(req.server,req.port,req.path),headers:hdrs,ts:Date.now()});
    }
});
hook('wininet.dll','InternetReadFile',{
    onEnter(a){ this.h=a[0].toString(); this.buf=a[1]; this.len=i32(a[2]); },
    onLeave(r){
        try {
            const req=winInetReqs.get(this.h); if(!req) return;
            if(this.buf&&!this.buf.isNull()&&this.len>0){
                const body=this.buf.readUtf8String(Math.min(this.len,8192));
                if(body&&body.trim()) send({t:'resp',url:fmt(req.server,req.port,req.path),body:body.substring(0,4096),ts:Date.now()});
            }
        } catch(_){}
    }
});

send({t:'log',msg:'[*] All hooks ready. Navigate TickerChart to capture APIs.'});
"""

# ──────────────────────────────────────────────────────────────────────────────
# Categorisation rules  (first match wins)
# ──────────────────────────────────────────────────────────────────────────────
CATEGORIES = [
    ('indicators / technical',  re.compile(r'indicator|technical|rsi|macd|bollinger|moving.av|ema|sma|stoch|atr|cci|obv|volume', re.I)),
    ('financials / fundamentals',re.compile(r'financial|fundamental|balance|income|cashflow|revenue|earnings|eps|ebitda|net.income', re.I)),
    ('ratios / valuation',      re.compile(r'ratio|valuat|pe|p-e|p_e|price.earn|forward.pe|yield|dividend', re.I)),
    ('market info / lookup',    re.compile(r'market.info|company.info|lookup|symbol|companyid|ticker|search', re.I)),
    ('prices / quotes',         re.compile(r'price|quote|ohlc|candle|bar|chart|history|snapshot|close|open|high|low', re.I)),
    ('screener',                re.compile(r'screen|filter|scan', re.I)),
    ('news / calendar',         re.compile(r'news|event|calendar|dividend.date|earning.date', re.I)),
    ('auth / session',          re.compile(r'auth|login|token|session|oauth|refresh', re.I)),
    ('static / asset',          re.compile(r'\.(js|css|png|jpg|svg|woff|ico|gif|webp)(\?|$)', re.I)),
    ('other',                   re.compile(r'.*')),
]

SKIP_EXTS = re.compile(r'\.(js|css|png|jpg|svg|woff|woff2|ico|gif|webp|ttf|eot|map)(\?|$)', re.I)

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
raw_log   = []          # every event
seen_urls = {}          # url -> first record
responses = {}          # url -> body snippet
catalogue = defaultdict(list)   # category -> [url_info, …]

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_FILE  = os.path.join(OUT_DIR, 'frida_api_log.json')
CAT_FILE  = os.path.join(OUT_DIR, 'frida_api_catalogue.json')


def categorise(url):
    for name, rx in CATEGORIES:
        if rx.search(url):
            return name
    return 'other'


def normalise_url(url):
    """Strip query-string dynamic values for deduplication, keep path structure."""
    try:
        from urllib.parse import urlparse, parse_qs
        p = urlparse(url)
        # Replace numeric path segments with {id}
        path = re.sub(r'/\d+', '/{id}', p.path)
        return f"{p.scheme}://{p.netloc}{path}"
    except Exception:
        return url


def on_message(message, data):
    if message['type'] == 'error':
        print(f"[FRIDA ERROR] {message.get('description','?')}", flush=True)
        return
    if message['type'] != 'send':
        return

    payload = message['payload']
    kind    = payload.get('t')

    if kind == 'log':
        print(payload.get('msg', ''), flush=True)
        return

    if kind == 'req':
        url    = payload.get('url', '')
        method = payload.get('method', 'GET')
        hdrs   = payload.get('headers', '')
        ts     = payload.get('ts', int(time.time()*1000))

        # Skip static assets
        if SKIP_EXTS.search(url):
            return

        norm = normalise_url(url)
        cat  = categorise(url)
        rec  = {'url': url, 'norm': norm, 'method': method,
                'category': cat, 'headers': hdrs[:300], 'ts': ts}
        raw_log.append(rec)

        # Print all non-asset requests
        tag = f'  [{cat.upper()}]' if cat not in ('static / asset', 'other') else ''
        print(f"[{method}] {url}{tag}", flush=True)

        # Track unique normalised URLs
        if norm not in seen_urls:
            seen_urls[norm] = rec
            catalogue[cat].append(rec)
            # Print discovery banner for interesting categories
            if cat not in ('static / asset', 'other', 'auth / session'):
                print(f"\n  *** NEW API DISCOVERED [{cat}] ***", flush=True)
                print(f"  {method} {url}\n", flush=True)

    elif kind == 'resp':
        url  = payload.get('url', '')
        body = payload.get('body', '')
        norm = normalise_url(url)
        if norm not in responses and body.strip():
            responses[norm] = body[:2000]
            cat = categorise(url)
            if cat not in ('static / asset',):
                print(f"\n  [RESP] {url}", flush=True)
                print(f"  Body: {body[:300]}\n", flush=True)


def print_summary():
    print("\n" + "═"*72, flush=True)
    print("  TICKERCHART API CATALOGUE", flush=True)
    print("═"*72, flush=True)
    for cat in [c for c,_ in CATEGORIES]:
        items = catalogue.get(cat, [])
        if not items:
            continue
        print(f"\n▶ {cat.upper()} ({len(items)} endpoint{'s' if len(items)!=1 else ''})", flush=True)
        for r in items:
            resp_note = ''
            if r['norm'] in responses:
                snippet = responses[r['norm']][:80].replace('\n',' ')
                resp_note = f"  → {snippet}"
            print(f"    {r['method']}  {r['url']}", flush=True)
            if resp_note:
                print(f"         {resp_note}", flush=True)
    print("\n" + "═"*72, flush=True)


def save_files():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(raw_log, f, indent=2)
    cat_out = {}
    for cat, items in catalogue.items():
        cat_out[cat] = []
        for r in items:
            entry = {'method': r['method'], 'url': r['url'], 'normalised': r['norm']}
            if r['norm'] in responses:
                entry['response_snippet'] = responses[r['norm']][:500]
            cat_out[cat].append(entry)
    with open(CAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cat_out, f, indent=2)
    print(f"\n[*] Raw log  → {LOG_FILE}", flush=True)
    print(f"[*] Catalogue→ {CAT_FILE}", flush=True)


def main():
    device   = frida.get_local_device()
    procs    = device.enumerate_processes()
    tc_procs = [p for p in procs if any(x in p.name.lower() for x in ['ticker','uniticker','tclive'])]

    if not tc_procs:
        print("[*] TickerChart not running yet — polling every 2 s until it starts …", flush=True)
        while not tc_procs:
            time.sleep(2)
            procs    = device.enumerate_processes()
            tc_procs = [p for p in procs if any(x in p.name.lower() for x in ['ticker','uniticker','tclive'])]
            if tc_procs:
                break
            print("  … still waiting", flush=True)

    proc = tc_procs[0]
    print(f"[*] Attaching to '{proc.name}' (PID {proc.pid}) …", flush=True)

    session = device.attach(proc.pid)
    script  = session.create_script(JS)
    script.on('message', on_message)
    script.load()

    print("\n[*] Hooks active — navigate EVERY section of TickerChart:", flush=True)
    print("     • Open charts (stocks, indices)", flush=True)
    print("     • Add technical indicators (RSI, MACD, Bollinger …)", flush=True)
    print("     • Open a stock's Financial tab (EPS, P/E, ratios …)", flush=True)
    print("     • Use the Screener", flush=True)
    print("     • Open any News / Calendar section", flush=True)
    print("     Press Ctrl+C when done.\n", flush=True)

    def stop(*_):
        print_summary()
        save_files()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    sys.stdin.read()


if __name__ == '__main__':
    main()
