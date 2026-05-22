const winHttpConnections = new Map();
const winHttpRequests = new Map();
const winInetConnections = new Map();
const winInetRequests = new Map();

function safeReadUtf16(pointer) {
    if (pointer === null || pointer.isNull()) {
        return null;
    }

    try {
        return Memory.readUtf16String(pointer);
    } catch (_) {
        return null;
    }
}

function safeReadAnsi(pointer) {
    if (pointer === null || pointer.isNull()) {
        return null;
    }

    try {
        return Memory.readAnsiString(pointer);
    } catch (_) {
        return null;
    }
}

function formatUrl(server, port, path) {
    const host = server || "<unknown-host>";
    const normalizedPath = path || "/";
    const needsPort = port && port !== 80 && port !== 443;
    return `https://${host}${needsPort ? `:${port}` : ""}${normalizedPath}`;
}

function logLine(prefix, value) {
    console.log(`${prefix} ${value}`);
}

function attachExport(moduleName, exportName, callbacks) {
    try {
        const address = Module.getExportByName(moduleName, exportName);
        Interceptor.attach(address, callbacks);
        logLine("[hooked]", `${moduleName}!${exportName}`);
    } catch (error) {
        logLine("[missing]", `${moduleName}!${exportName} (${error})`);
    }
}

attachExport("winhttp.dll", "WinHttpConnect", {
    onEnter(args) {
        this.server = safeReadUtf16(args[1]);
        this.port = args[2].toInt32();
    },
    onLeave(retval) {
        const handle = retval.toString();
        winHttpConnections.set(handle, {
            server: this.server,
            port: this.port,
        });
        logLine("[winhttp.connect]", `${this.server || "<unknown-host>"}:${this.port}`);
    },
});

attachExport("winhttp.dll", "WinHttpOpenRequest", {
    onEnter(args) {
        const connectionHandle = args[0].toString();
        const connection = winHttpConnections.get(connectionHandle) || {};

        this.request = {
            transport: "winhttp",
            server: connection.server,
            port: connection.port,
            method: safeReadUtf16(args[1]) || "GET",
            path: safeReadUtf16(args[2]) || "/",
        };
    },
    onLeave(retval) {
        const handle = retval.toString();
        winHttpRequests.set(handle, this.request);
        logLine("[request]", `${this.request.method} ${formatUrl(this.request.server, this.request.port, this.request.path)}`);
    },
});

attachExport("winhttp.dll", "WinHttpSendRequest", {
    onEnter(args) {
        const requestHandle = args[0].toString();
        const request = winHttpRequests.get(requestHandle);
        if (!request) {
            return;
        }

        const extraHeaders = safeReadUtf16(args[1]);
        if (extraHeaders) {
            logLine("[headers]", `${request.method} ${request.path} :: ${extraHeaders.replace(/\r\n/g, " | ")}`);
        }
    },
});

attachExport("wininet.dll", "InternetConnectW", {
    onEnter(args) {
        this.server = safeReadUtf16(args[2]);
        this.port = args[3].toInt32();
    },
    onLeave(retval) {
        const handle = retval.toString();
        winInetConnections.set(handle, {
            server: this.server,
            port: this.port,
        });
        logLine("[wininet.connect]", `${this.server || "<unknown-host>"}:${this.port}`);
    },
});

attachExport("wininet.dll", "HttpOpenRequestW", {
    onEnter(args) {
        const connectionHandle = args[0].toString();
        const connection = winInetConnections.get(connectionHandle) || {};

        this.request = {
            transport: "wininet",
            server: connection.server,
            port: connection.port,
            method: safeReadUtf16(args[1]) || "GET",
            path: safeReadUtf16(args[2]) || "/",
        };
    },
    onLeave(retval) {
        const handle = retval.toString();
        winInetRequests.set(handle, this.request);
        logLine("[request]", `${this.request.method} ${formatUrl(this.request.server, this.request.port, this.request.path)}`);
    },
});

attachExport("wininet.dll", "HttpSendRequestW", {
    onEnter(args) {
        const requestHandle = args[0].toString();
        const request = winInetRequests.get(requestHandle);
        if (!request) {
            return;
        }

        const extraHeaders = safeReadUtf16(args[1]) || safeReadAnsi(args[1]);
        if (extraHeaders) {
            logLine("[headers]", `${request.method} ${request.path} :: ${extraHeaders.replace(/\r\n/g, " | ")}`);
        }
    },
});