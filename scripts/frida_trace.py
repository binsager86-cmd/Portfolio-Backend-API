#!/usr/bin/env python3
"""
Attach Frida to TickerChartLive and print all HTTP requests.
Run: python frida_trace.py
"""
import sys
import signal
import frida

JS = r"""
'use strict';

const handles = {};

function hookWinInet() {
    const InternetConnectW = Module.findExportByName('wininet.dll', 'InternetConnectW');
    const HttpOpenRequestW = Module.findExportByName('wininet.dll', 'HttpOpenRequestW');
    const HttpSendRequestW = Module.findExportByName('wininet.dll', 'HttpSendRequestW');

    if (!InternetConnectW) { send({t:'log', msg:'[-] wininet.dll not found'}); return; }

    Interceptor.attach(InternetConnectW, {
        onEnter(args) { try { this.host = args[1].readUtf16String(); } catch(e) {} },
        onLeave(ret) { if (!ret.isNull() && this.host) handles[ret.toString()] = { host: this.host }; }
    });

    Interceptor.attach(HttpOpenRequestW, {
        onEnter(args) {
            try {
                this.hConn = args[0].toString();
                this.verb = args[1].readUtf16String();
                this.path = args[2].readUtf16String();
            } catch(e) {}
        },
        onLeave(ret) {
            if (!ret.isNull()) {
                const conn = handles[this.hConn] || {};
                handles[ret.toString()] = { host: conn.host || '?', verb: this.verb || 'GET', path: this.path || '/' };
            }
        }
    });

    Interceptor.attach(HttpSendRequestW, {
        onEnter(args) {
            try {
                const info = handles[args[0].toString()];
                if (!info) return;
                send({t:'req', url: 'https://' + info.host + info.path, verb: info.verb});
            } catch(e) {}
        }
    });
    send({t:'log', msg:'[+] WinInet hooks installed'});
}

function hookWinHttp() {
    const WinHttpConnect    = Module.findExportByName('winhttp.dll', 'WinHttpConnect');
    const WinHttpOpenRequest= Module.findExportByName('winhttp.dll', 'WinHttpOpenRequest');
    const WinHttpSendRequest= Module.findExportByName('winhttp.dll', 'WinHttpSendRequest');

    if (!WinHttpConnect) { send({t:'log', msg:'[-] winhttp.dll not found'}); return; }

    Interceptor.attach(WinHttpConnect, {
        onEnter(args) { try { this.host = args[1].readUtf16String(); } catch(e) {} },
        onLeave(ret) { if (!ret.isNull() && this.host) handles['wh_'+ret.toString()] = { host: this.host }; }
    });

    Interceptor.attach(WinHttpOpenRequest, {
        onEnter(args) {
            try {
                this.hConn = 'wh_' + args[0].toString();
                this.verb = args[1].readUtf16String();
                this.path = args[2].readUtf16String();
            } catch(e) {}
        },
        onLeave(ret) {
            if (!ret.isNull()) {
                const conn = handles[this.hConn] || {};
                handles['wh_'+ret.toString()] = { host: conn.host || '?', verb: this.verb || 'GET', path: this.path || '/' };
            }
        }
    });

    Interceptor.attach(WinHttpSendRequest, {
        onEnter(args) {
            try {
                const info = handles['wh_'+args[0].toString()];
                if (!info) return;
                send({t:'req', url: 'https://' + info.host + info.path, verb: info.verb});
            } catch(e) {}
        }
    });
    send({t:'log', msg:'[+] WinHTTP hooks installed'});
}

hookWinInet();
hookWinHttp();
send({t:'log', msg:'[*] Ready — open the P/E indicator chart in TickerChart'});
"""

def on_message(message, data):
    if message['type'] == 'send':
        payload = message['payload']
        if payload.get('t') == 'req':
            url = payload.get('url', '')
            verb = payload.get('verb', 'GET')
            print(f"[HTTP] {verb} {url}", flush=True)
            # Highlight PE-related URLs
            lower = url.lower()
            if any(k in lower for k in ['pe', 'earning', 'eps', 'financial', 'indicator', 'ratio']):
                print(f"  >>> POSSIBLE PE URL: {url}", flush=True)
        elif payload.get('t') == 'log':
            print(payload.get('msg', ''), flush=True)
    elif message['type'] == 'error':
        print(f"[ERROR] {message.get('description', message)}", flush=True)


def main():
    # Find TickerChart process
    device = frida.get_local_device()
    processes = device.enumerate_processes()
    tc_procs = [p for p in processes if 'ticker' in p.name.lower() or 'uniticker' in p.name.lower()]

    if not tc_procs:
        print("ERROR: TickerChartLive.exe not found. Is TickerChart running?")
        sys.exit(1)

    proc = tc_procs[0]
    print(f"[*] Attaching to {proc.name} (PID {proc.pid})...")

    session = device.attach(proc.pid)
    script = session.create_script(JS)
    script.on('message', on_message)
    script.load()

    print("[*] Hooks loaded! Now open TickerChart and add the P/E indicator to a chart.")
    print("[*] Press Ctrl+C to stop.\n")

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    sys.stdin.read()  # block forever


if __name__ == '__main__':
    main()
