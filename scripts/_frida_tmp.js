
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
