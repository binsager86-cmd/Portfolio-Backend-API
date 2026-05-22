#!/usr/bin/env python3
"""
Frida script to capture TickerChart indicator API calls.
Hooks WinHTTP + WinInet and dumps every request URL + headers.
Responses are also captured to reveal the full indicator endpoint shape.

Usage:
    1. Run this script (with TickerChart already running OR launch it after the prompt).
    2. In TickerChart, open a chart and add/click any indicator (RSI, MACD, EPS, P/E, etc.)
    3. Watch the console for captured URLs.
    Press Ctrl+C to stop.
"""

import sys
import signal
import frida

JS = r"""
'use strict';

const winHttpConnections = new Map();  // hConnect -> {server, port}
const winHttpRequests    = new Map();  // hRequest -> {server, port, method, path}
const winInetConnections = new Map();
const winInetRequests    = new Map();

// ── helpers ────────────────────────────────────────────────────────────────
function u16(ptr) {
    try { return ptr && !ptr.isNull() ? ptr.readUtf16String() : null; } catch(_) { return null; }
}
function i32(ptr) {
    try { return ptr.toInt32(); } catch(_) { return 0; }
}
function fmt(server, port, path) {
    const p = path || '/';
    const h = server || '?';
    const usePort = port && port !== 80 && port !== 443 && port !== 0;
    return `https://${h}${usePort ? ':'+port : ''}${p}`;
}

function emit(obj) { send(obj); }

function hook(mod, fn, cbs) {
    try {
        Interceptor.attach(Module.getExportByName(mod, fn), cbs);
        send({t:'log', msg:`[+] hooked ${mod}!${fn}`});
    } catch(e) {
        send({t:'log', msg:`[-] cannot hook ${mod}!${fn}: ${e}`});
    }
}

// ── WinHTTP ────────────────────────────────────────────────────────────────
hook('winhttp.dll', 'WinHttpConnect', {
    onEnter(args) { this.server = u16(args[1]); this.port = i32(args[2]); },
    onLeave(ret) {
        if (!ret.isNull())
            winHttpConnections.set(ret.toString(), {server: this.server, port: this.port});
    }
});

hook('winhttp.dll', 'WinHttpOpenRequest', {
    onEnter(args) {
        const conn = winHttpConnections.get(args[0].toString()) || {};
        this.req = { server: conn.server, port: conn.port,
                     method: u16(args[1]) || 'GET', path: u16(args[2]) || '/' };
    },
    onLeave(ret) {
        if (!ret.isNull()) winHttpRequests.set(ret.toString(), this.req);
    }
});

hook('winhttp.dll', 'WinHttpAddRequestHeaders', {
    onEnter(args) {
        const req = winHttpRequests.get(args[0].toString());
        if (!req) return;
        const hdr = u16(args[1]);
        if (hdr) req._extraHeaders = (req._extraHeaders || '') + hdr;
    }
});

hook('winhttp.dll', 'WinHttpSendRequest', {
    onEnter(args) {
        const req = winHttpRequests.get(args[0].toString());
        if (!req) return;
        const url = fmt(req.server, req.port, req.path);
        const hdrs = u16(args[1]) || req._extraHeaders || '';
        emit({t:'req', transport:'winhttp', method: req.method, url, headers: hdrs});
    }
});

// Capture response body (WinHttpReadData)
hook('winhttp.dll', 'WinHttpReceiveResponse', {
    onEnter(args) { this.hReq = args[0].toString(); },
    onLeave(ret) {
        // just tag the handle so WinHttpReadData can associate it
        const req = winHttpRequests.get(this.hReq);
        if (req) req._awaitingRead = true;
    }
});

hook('winhttp.dll', 'WinHttpReadData', {
    onEnter(args) {
        this.hReq   = args[0].toString();
        this.buf    = args[1];
        this.bufLen = i32(args[2]);
    },
    onLeave(ret) {
        try {
            const req = winHttpRequests.get(this.hReq);
            if (!req || !req._awaitingRead) return;
            if (this.buf && !this.buf.isNull() && this.bufLen > 0) {
                const body = this.buf.readUtf8String(Math.min(this.bufLen, 4096));
                if (body && body.trim().length > 0) {
                    const url = fmt(req.server, req.port, req.path);
                    emit({t:'resp', url, body: body.substring(0, 2000)});
                }
            }
        } catch(_) {}
    }
});

// ── WinInet ────────────────────────────────────────────────────────────────
hook('wininet.dll', 'InternetConnectW', {
    onEnter(args) { this.server = u16(args[1]); this.port = i32(args[3]); },
    onLeave(ret) {
        if (!ret.isNull())
            winInetConnections.set(ret.toString(), {server: this.server, port: this.port});
    }
});

hook('wininet.dll', 'HttpOpenRequestW', {
    onEnter(args) {
        const conn = winInetConnections.get(args[0].toString()) || {};
        this.req = { server: conn.server, port: conn.port,
                     method: u16(args[1]) || 'GET', path: u16(args[2]) || '/' };
    },
    onLeave(ret) {
        if (!ret.isNull()) winInetRequests.set(ret.toString(), this.req);
    }
});

hook('wininet.dll', 'HttpAddRequestHeadersW', {
    onEnter(args) {
        const req = winInetRequests.get(args[0].toString());
        if (!req) return;
        const hdr = u16(args[1]);
        if (hdr) req._extraHeaders = (req._extraHeaders || '') + hdr;
    }
});

hook('wininet.dll', 'HttpSendRequestW', {
    onEnter(args) {
        const req = winInetRequests.get(args[0].toString());
        if (!req) return;
        const url  = fmt(req.server, req.port, req.path);
        const hdrs = u16(args[1]) || req._extraHeaders || '';
        emit({t:'req', transport:'wininet', method: req.method, url, headers: hdrs});
    }
});

hook('wininet.dll', 'InternetReadFile', {
    onEnter(args) {
        this.hFile  = args[0].toString();
        this.buf    = args[1];
        this.bufLen = i32(args[2]);
    },
    onLeave(ret) {
        try {
            const req = winInetRequests.get(this.hFile);
            if (!req) return;
            const url = fmt(req.server, req.port, req.path);
            // Only care about indicator/financial paths
            if (!url.toLowerCase().match(/indicator|financial|ratio|screener|eps|market/)) return;
            if (this.buf && !this.buf.isNull() && this.bufLen > 0) {
                const body = this.buf.readUtf8String(Math.min(this.bufLen, 4096));
                if (body && body.trim().length > 0)
                    emit({t:'resp', url, body: body.substring(0, 2000)});
            }
        } catch(_) {}
    }
});

send({t:'log', msg:'[*] All hooks installed. Now navigate to an indicator in TickerChart.'});
"""


INDICATOR_KEYWORDS = [
    'indicator', 'financial', 'ratio', 'screener', 'eps', 'earnings',
    'pe', 'p-e', 'revenue', 'net_income', 'balance', 'overview',
    'field', 'metric', 'fundamental', 'technical', 'rsi', 'macd',
    'bollinger', 'moving', 'market-financials',
]

seen_urls = set()

def on_message(message, data):
    if message['type'] == 'error':
        print(f"[FRIDA ERROR] {message.get('description', message)}", flush=True)
        return
    if message['type'] != 'send':
        return

    payload = message['payload']
    kind = payload.get('t')

    if kind == 'log':
        print(payload.get('msg', ''), flush=True)

    elif kind == 'req':
        url    = payload.get('url', '')
        method = payload.get('method', 'GET')
        hdrs   = payload.get('headers', '')
        low    = url.lower()

        # Always print non-trivial URLs (skip static assets)
        if any(ext in low for ext in ['.png', '.jpg', '.svg', '.css', '.woff', '.ico', '.js']):
            return

        tag = ''
        if any(k in low for k in INDICATOR_KEYWORDS):
            tag = '  <<< INDICATOR API'

        line = f"[REQ] {method} {url}"
        if hdrs:
            line += f"\n      Headers: {hdrs[:200]}"
        print(line + tag, flush=True)

        if tag and url not in seen_urls:
            seen_urls.add(url)
            print(f"\n{'='*70}", flush=True)
            print(f"  *** FOUND INDICATOR ENDPOINT ***", flush=True)
            print(f"  METHOD : {method}", flush=True)
            print(f"  URL    : {url}", flush=True)
            if hdrs:
                print(f"  HEADERS: {hdrs[:500]}", flush=True)
            print(f"{'='*70}\n", flush=True)

    elif kind == 'resp':
        url  = payload.get('url', '')
        body = payload.get('body', '')
        low  = url.lower()
        if any(k in low for k in INDICATOR_KEYWORDS):
            print(f"\n[RESP] {url}", flush=True)
            print(f"  Body snippet: {body[:500]}", flush=True)


def main():
    device = frida.get_local_device()
    processes = device.enumerate_processes()

    tc_procs = [p for p in processes
                if any(x in p.name.lower() for x in ['ticker', 'uniticker', 'tclive'])]

    if not tc_procs:
        print("ERROR: TickerChart process not found. Make sure TickerChartLive.exe is running.")
        print("Tip: Start TickerChart first, then re-run this script.")
        sys.exit(1)

    proc = tc_procs[0]
    print(f"[*] Attaching to '{proc.name}' (PID {proc.pid}) ...", flush=True)

    session = device.attach(proc.pid)
    script  = session.create_script(JS)
    script.on('message', on_message)
    script.load()

    print("\n[*] Frida hooks active.", flush=True)
    print("[*] >>> Open TickerChart now and navigate to any Indicator (Technical or Financial).", flush=True)
    print("[*] >>> Press Ctrl+C when done.\n", flush=True)

    signal.signal(signal.SIGINT, lambda *_: (print("\n[*] Stopping."), sys.exit(0)))
    sys.stdin.read()


if __name__ == '__main__':
    main()
