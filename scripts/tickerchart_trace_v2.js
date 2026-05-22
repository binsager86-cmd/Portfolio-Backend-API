// tickerchart_trace_v2.js
// Frida 17.x spawn-mode script — uses Process.getModuleByName(mod).findExportByName(fn)
// Run: frida.exe -f "C:\Program Files (x86)\UniTicker\TCLive\TickerChartLive.exe" -l tickerchart_trace_v2.js --no-pause

var conns = {}, reqs = {};

function hookWinHttp() {
    try {
        var wh = Process.getModuleByName('winhttp.dll');

        var fnConnect = wh.findExportByName('WinHttpConnect');
        var fnOpen    = wh.findExportByName('WinHttpOpenRequest');
        var fnSend    = wh.findExportByName('WinHttpSendRequest');

        if (fnConnect) Interceptor.attach(ptr(fnConnect), {
            onEnter: function(a) {
                this.srv  = Memory.readUtf16String(a[1]);
                this.port = a[2].toInt32();
            },
            onLeave: function(r) {
                conns[r.toString()] = { srv: this.srv, port: this.port };
            }
        });

        if (fnOpen) Interceptor.attach(ptr(fnOpen), {
            onEnter: function(a) {
                var c = conns[a[0].toString()] || {};
                this.req = {
                    srv:    c.srv,
                    port:   c.port,
                    method: Memory.readUtf16String(a[1]) || 'GET',
                    path:   Memory.readUtf16String(a[2]) || '/'
                };
            },
            onLeave: function(r) {
                reqs[r.toString()] = this.req;
            }
        });

        if (fnSend) Interceptor.attach(ptr(fnSend), {
            onEnter: function(a) {
                var r   = reqs[a[0].toString()] || {};
                var url = 'https://' + (r.srv || 'unknown') + (r.path || '/');
                console.log('URL>> ' + (r.method || 'GET') + ' ' + url);
            }
        });

        console.log('[+] WinHTTP hooks OK (connect/open/send)');
    } catch(e) {
        console.log('[-] WinHTTP hook failed: ' + e);
    }
}

function hookWinHttpReadData() {
    try {
        var wh   = Process.getModuleByName('winhttp.dll');
        var fnRd = wh.findExportByName('WinHttpReadData');
        if (!fnRd) { console.log('[-] WinHttpReadData not found'); return; }

        Interceptor.attach(ptr(fnRd), {
            onEnter: function(a) {
                this.req  = reqs[a[0].toString()];
                this.buf  = a[1];
                this.plen = a[3]; // LPDWORD lpdwNumberOfBytesRead
            },
            onLeave: function(r) {
                if (r.toInt32() === 0) return;
                try {
                    var len = this.plen.readU32();
                    if (len > 0 && this.buf && !this.buf.isNull()) {
                        var s = this.buf.readUtf8String(Math.min(len, 1024));
                        if (s) console.log('RESP>> ' + s.substring(0, 800));
                    }
                } catch(e) {}
            }
        });
        console.log('[+] WinHttpReadData hook OK');
    } catch(e) {
        console.log('[-] WinHttpReadData hook failed: ' + e);
    }
}

function hookDecryptMessage() {
    try {
        var ssp = Process.getModuleByName('sspicli.dll');
        var dm  = ssp.findExportByName('DecryptMessage');
        if (!dm) { console.log('[-] DecryptMessage not found'); return; }

        Interceptor.attach(ptr(dm), {
            onEnter: function(a) { this.msg = a[1]; },
            onLeave: function() {
                try {
                    var cnt   = this.msg.add(4).readU32();
                    var pbufs = this.msg.add(8).readPointer();
                    for (var i = 0; i < cnt; i++) {
                        var btype = pbufs.add(i * 16).readU32();
                        var blen  = pbufs.add(i * 16 + 4).readU32();
                        var pbuf  = pbufs.add(i * 16 + 8).readPointer();
                        if (btype === 1 && blen > 10 && pbuf && !pbuf.isNull()) {
                            try {
                                var s = pbuf.readUtf8String(Math.min(blen, 1024));
                                if (s && (
                                    s.indexOf('GET ')         >= 0 ||
                                    s.indexOf('POST ')        >= 0 ||
                                    s.indexOf('HTTP/1')       >= 0 ||
                                    s.indexOf('tickerchart')  >= 0 ||
                                    s.indexOf('"data"')       >= 0 ||
                                    s.indexOf('"result"')     >= 0
                                )) {
                                    console.log('DEC>> ' + s.substring(0, 800));
                                }
                            } catch(e2) {}
                        }
                    }
                } catch(e) {}
            }
        });
        console.log('[+] sspicli DecryptMessage hooked');
    } catch(e) {
        console.log('[-] DecryptMessage hook failed: ' + e);
    }
}

// Install all hooks immediately (for --attach) or on resume (for --spawn)
hookWinHttp();
hookWinHttpReadData();
hookDecryptMessage();

console.log('[*] tickerchart_trace_v2.js ready — watching for traffic');
