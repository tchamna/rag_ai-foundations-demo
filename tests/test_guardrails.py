import json
import os
import sqlite3
import tempfile
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI,Request
from fastapi.testclient import TestClient
from app.security import GuardrailMiddleware
from app.rag.spend import protect_model_request,reserve,SpendLimitError

class HTTPGuardTests(unittest.TestCase):
    def client(self):
        app=FastAPI()
        @app.post('/api/chat')
        async def chat(request:Request): return {'size':len(await request.body())}
        @app.get('/api/status')
        async def status(): return {'ok':True}
        app.add_middleware(GuardrailMiddleware)
        return TestClient(app)

    def test_origin_body_caps_and_headers(self):
        with self.client() as client:
            self.assertEqual(client.get('/api/status',headers={'origin':'https://evil.example'}).status_code,403)
            self.assertEqual(client.get('/api/status',headers={'sec-fetch-site':'cross-site'}).status_code,403)
            self.assertEqual(client.post('/api/chat',content=b'x'*65537).status_code,413)
            self.assertEqual(client.post('/api/chat',content=iter([b'x'*40000,b'y'*40000])).status_code,413)
            response=client.post('/api/chat',content=b'hello')
            self.assertEqual(response.json()['size'],5)
            self.assertEqual(response.headers['x-content-type-options'],'nosniff')
            self.assertIn("script-src 'self'",response.headers['content-security-policy'])

    def test_rate_and_active_job_limits(self):
        with patch.dict(os.environ,{'RAG_REQUESTS_PER_MINUTE':'1'}), self.client() as client:
            self.assertEqual(client.get('/api/status').status_code,200)
            self.assertEqual(client.get('/api/status').status_code,429)

    def test_active_workspace_job_is_rejected(self):
        with self.client() as client:
            middleware=client.app.middleware_stack
            while not isinstance(middleware,GuardrailMiddleware): middleware=middleware.app
            middleware.jobs.add('testclient')
            self.assertEqual(client.post('/api/chat',content=b'test').status_code,409)
            middleware.jobs.clear()
            self.assertEqual(client.post('/api/chat',content=b'test').status_code,200)

class SpendGuardTests(unittest.TestCase):
    def request(self,**extra):
        return urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps({'model':'gpt-5-nano','messages':[{'role':'user','content':'hello'}],**extra}).encode())

    def test_caps_and_unknown_model_tool_destinations(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'RAG_SPEND_DB':str(Path(folder)/'spend.db')}):
            request=self.request(max_completion_tokens=999999)
            protect_model_request(request)
            self.assertEqual(json.loads(request.data)['max_completion_tokens'],1600)
            self.assertEqual(json.loads(request.data)['service_tier'],'default')
            for extra in [{'model':'unapproved-expensive-model'},{'tools':[{'type':'web_search'}]}]:
                with self.assertRaises(SpendLimitError): protect_model_request(self.request(**extra))
            bad=urllib.request.Request('https://unapproved.example',data=self.request().data)
            with self.assertRaises(SpendLimitError): protect_model_request(bad)

    def test_persistent_budget_is_atomic_across_threads(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'RAG_SPEND_DB':str(Path(folder)/'spend.db'),'RAG_DAILY_BUDGET_USD':'0.25'}):
            def attempt(_):
                try: reserve(.1);return True
                except SpendLimitError:return False
            with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(attempt,range(12)))
            self.assertEqual(sum(results),2)
            with self.assertRaises(SpendLimitError):reserve(.1)
            with sqlite3.connect(Path(folder)/'spend.db') as db:
                self.assertLessEqual(db.execute('SELECT reserved FROM allowance').fetchone()[0],.25)
            db.close()


    def test_exhausted_budget_prevents_any_http_call(self):
        from app.rag.synthesis import _json_request, ProviderUnavailable
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'RAG_SPEND_DB':str(Path(folder)/'spend.db'),'RAG_DAILY_BUDGET_USD':'0'}), patch('urllib.request.urlopen') as network:
            with self.assertRaises(ProviderUnavailable): _json_request(self.request(),'test')
            network.assert_not_called()


    def test_provider_errors_do_not_expose_response_secrets(self):
        import io
        from urllib.error import HTTPError
        from app.rag.synthesis import _json_request, ProviderUnavailable
        error=HTTPError('https://api.openai.com/v1/chat/completions',400,'bad request',{},io.BytesIO(b'private-token-do-not-display'))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'RAG_SPEND_DB':str(Path(folder)/'spend.db')}), patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaises(ProviderUnavailable) as raised:_json_request(self.request(),'test')
            self.assertNotIn('private-token',str(raised.exception))
