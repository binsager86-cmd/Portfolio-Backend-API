'use strict';
// Frida script: trace TickerChart HTTP requests via WinInet hooks
// Run: frida -p <PID> -l tc_pe_trace.js

const handles = {};

// 1. Hook InternetConnectW — captures hostname per connection handle
const InternetConnectW = Module.findExportByName('wininet.dll', 'InternetConnectW');
if (InternetConnectW) {
    Interceptor.attach(InternetConnectW, {
        onEnter(args) {
            try { this.host = args[1].readUtf16String(); } catch(e) {}
        },
        onLeave(ret) {
            if (!ret.isNull() && this.host)
                handles[ret.toString()] = { host: this.host };
        }
    });
    console.log('[+] Hooked InternetConnectW');
} else {
    console.log('[-] InternetConnectW not found');
}

// 2. Hook HttpOpenRequestW — captures verb + path per request handle
const HttpOpenRequestW = Module.findExportByName('wininet.dll', 'HttpOpenRequestW');
if (HttpOpenRequestW) {
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
                handles[ret.toString()] = {
                    host: conn.host || '?',
                    verb: this.verb || 'GET',
                    path: this.path || '/'
                };
            }
        }
    });
    console.log('[+] Hooked HttpOpenRequestW');
} else {
    console.log('[-] HttpOpenRequestW not found');
}

// 3. Hook HttpSendRequestW — fires when request is actually sent; print full URL
const HttpSendRequestW = Module.findExportByName('wininet.dll', 'HttpSendRequestW');
if (HttpSendRequestW) {
    Interceptor.attach(HttpSendRequestW, {
        onEnter(args) {
            try {
                const info = handles[args[0].toString()];
                if (!info) return;
                const url = 'https://' + info.host + info.path;
                // Print ALL requests (remove filter to see everything)
                console.log('[REQ] ' + info.verb + ' ' + url);
            } catch(e) {}
        }
    });
    console.log('[+] Hooked HttpSendRequestW');
} else {
    console.log('[-] HttpSendRequestW not found — trying WinHTTP fallback');

    // Fallback: WinHTTP (used by .NET Core / newer apps)
    const WinHttpOpenRequest = Module.findExportByName('winhttp.dll', 'WinHttpOpenRequest');
    const WinHttpSendRequest = Module.findExportByName('winhttp.dll', 'WinHttpSendRequest');
    const WinHttpConnect   = Module.findExportByName('winhttp.dll', 'WinHttpConnect');

    if (WinHttpConnect) {
        Interceptor.attach(WinHttpConnect, {
            onEnter(args) {
                try { this.host = args[1].readUtf16String(); } catch(e) {}
            },
            onLeave(ret) {
                if (!ret.isNull() && this.host)
                    handles['wh_' + ret.toString()] = { host: this.host };
            }
        });
    }

    if (WinHttpOpenRequest) {
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
                    handles['wh_' + ret.toString()] = {
                        host: conn.host || '?',
                        verb: this.verb || 'GET',
                        path: this.path || '/'
                    };
                }
            }
        });
    }

    if (WinHttpSendRequest) {
        Interceptor.attach(WinHttpSendRequest, {
            onEnter(args) {
                try {
                    const info = handles['wh_' + args[0].toString()];
                    if (!info) return;
                    const url = 'https://' + info.host + info.path;
                    console.log('[REQ-WH] ' + info.verb + ' ' + url);
                } catch(e) {}
            }
        });
        console.log('[+] WinHTTP fallback hooks installed');
    }
}

console.log('\n[*] Hook ready. Open the P/E indicator chart in TickerChart now.\n');
