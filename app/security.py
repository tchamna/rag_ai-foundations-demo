"""ASGI boundaries: enforce limits before parsing and hold job locks through streams."""
import asyncio
import os
import tempfile
import time
from collections import defaultdict, deque
from http.cookies import SimpleCookie
from starlette.responses import JSONResponse

class GuardrailMiddleware:
    def __init__(self,app):
        self.app=app
        self.requests=defaultdict(deque)
        self.jobs=set()

    async def __call__(self,scope,receive,send):
        if scope['type']!='http': return await self.app(scope,receive,send)
        headers={k.decode().lower():v.decode() for k,v in scope.get('headers',[])}
        path=scope['path']; method=scope['method']
        async def reject(status,message):
            await JSONResponse({'detail':message},status_code=status,headers={'Cache-Control':'no-store'})(scope,receive,send)
        # Never trust X-Forwarded-For for rate limiting; configure trusted proxies at deployment.
        if path.startswith('/api/'):
            origin=headers.get('origin')
            expected=scope.get('scheme','http')+'://'+headers.get('host','')
            if (origin and origin!=expected) or headers.get('sec-fetch-site')=='cross-site':
                return await reject(403,'Cross-site API requests are not allowed.')
            client=(scope.get('client') or ('unknown',))[0]
            now=time.monotonic()
            if len(self.requests)>5000:
                for key in list(self.requests):
                    if not self.requests[key] or self.requests[key][-1]<now-60: self.requests.pop(key,None)
                if client not in self.requests and len(self.requests)>5000:
                    return await reject(429,'Server request capacity reached. Please retry later.')
            recent=self.requests[client]
            while recent and recent[0]<=now-60: recent.popleft()
            if len(recent)>=int(os.getenv('RAG_REQUESTS_PER_MINUTE','90')):
                return await reject(429,'Too many requests. Please wait a minute before trying again.')
            recent.append(now)
        job=path.startswith('/api/') and (method not in {'GET','HEAD'} or path.endswith('/questions')) and not path.endswith('/status')
        key=None
        if job:
            cookie=SimpleCookie()
            try: cookie.load(headers.get('cookie',''))
            except Exception: pass
            key=cookie['rag_workspace'].value if 'rag_workspace' in cookie else (scope.get('client') or ('unknown',))[0]
            if key in self.jobs:
                return await reject(409,'A document or AI task is already running in this workspace. Wait for it to finish.')
            if len(self.jobs)>=4:
                return await reject(429,'All processing slots are busy. Please try again shortly.')
            self.jobs.add(key)
        spool=None
        try:
            if method in {'POST','PUT','PATCH'}:
                limit=40*1024*1024 if '/upload' in path else 64*1024
                length=headers.get('content-length')
                if length is not None:
                    try:
                        if int(length)<0: raise ValueError()
                        if int(length)>limit: return await reject(413,'Request too large. Upload batches are limited to 40 MB; chat requests to 64 KB.')
                    except ValueError: return await reject(400,'Invalid request length.')
                # Spool chunked bodies too: a missing/false Content-Length cannot bypass the cap.
                spool=tempfile.SpooledTemporaryFile(max_size=1024*1024)
                total=0
                while True:
                    event=await receive()
                    if event['type']=='http.disconnect': return
                    chunk=event.get('body',b''); total+=len(chunk)
                    if total>limit: return await reject(413,'Request body exceeds the permitted size.')
                    spool.write(chunk)
                    if not event.get('more_body',False): break
                spool.seek(0)
                original_receive=receive
                body_done=False
                async def limited_receive():
                    nonlocal body_done
                    if body_done: return await original_receive()
                    chunk=spool.read(65536)
                    more=spool.tell()<total
                    if not more: body_done=True
                    return {'type':'http.request','body':chunk,'more_body':more}
                downstream_receive=limited_receive
            else: downstream_receive=receive
            async def safe_send(message):
                if message['type']=='http.response.start':
                    extra=[(b'x-content-type-options',b'nosniff'),(b'x-frame-options',b'DENY'),
                           (b'referrer-policy',b'no-referrer'),(b'permissions-policy',b'camera=(), microphone=(), geolocation=()'),
                           (b'content-security-policy',b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'")]
                    message['headers']=list(message.get('headers',[]))+extra
                await send(message)
            await self.app(scope,downstream_receive,safe_send)
        finally:
            if spool: spool.close()
            if key is not None: self.jobs.discard(key)
