"""Run with python -m unittest discover -s tests. No model calls are made."""
import shutil
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

class WorkspaceTests(unittest.TestCase):
    def test_removed_shared_demo_routes(self):
        with TestClient(app) as client:
            for path in ['/njsharing', '/njsharing/compare', '/njsharing/freemodels',
                         '/api/status', '/api/questions', '/static/index.html']:
                self.assertEqual(client.get(path).status_code, 404, path)
            for path in ['/api/chat', '/api/compare', '/api/freemodels', '/api/upload', '/api/reindex']:
                self.assertEqual(client.post(path, json={'message':'hello'}).status_code, 404, path)

    def test_document_lifecycle_and_isolation(self):
        with patch('app.rag.llm_questions.synthesis.PROVIDER_ORDER', []), TestClient(app) as first, TestClient(app) as second:
            try:
                first.get('/generic')
                self.assertEqual(first.get('/api/generic/status').json()['files'], 0)
                self.assertEqual(second.get('/api/generic/status').json()['files'], 0)
                self.assertEqual(first.post('/api/generic/chat', json={'message':'Hello'}).status_code, 409)
                text = b'Aurora launches in November. Alex manages the project. Every team member must submit a detailed report by Friday before the official launch date.'
                result = first.post('/api/generic/upload', files={'files':('notes.txt', text)}).json()
                self.assertTrue(result['uploaded'][0]['ok'])
                self.assertEqual(result['status']['files'], 1)
                ideas = first.get('/api/generic/questions').json()['questions']
                self.assertTrue(ideas)
                self.assertTrue(all(item['source'] == 'notes.txt' for item in ideas))
                self.assertTrue(any('dates or deadlines' in item['question'] for item in ideas))
                self.assertFalse(any('employees' in item['question'] for item in ideas))
                self.assertEqual(second.get('/api/generic/status').json()['files'], 0)
                for name, content in [('empty.txt', b''), ('bad.pdf', b'invalid'), ('bad.exe', b'invalid')]:
                    result = first.post('/api/generic/upload', files={'files':(name, content)}).json()
                    self.assertFalse(result['uploaded'][0]['ok'])
                self.assertEqual(first.post('/api/generic/chat', json={'message':'   '}).status_code, 422)
                self.assertEqual(first.post('/api/generic/reindex', headers={'origin':'https://other.example'}).status_code, 403)
                with patch('app.main.synthesize_answer', return_value={'answer':'November','citations':[]}):
                    self.assertEqual(first.post('/api/generic/chat', json={'message':'When does Aurora launch?'}).status_code, 200)
                self.assertEqual(second.delete('/api/generic/documents/notes.txt').status_code, 404)
                self.assertEqual(first.delete('/api/generic/documents/notes.txt').json()['files'], 0)
                self.assertFalse(first.get('/api/generic/status').json()['ready'])
                stream = first.post('/api/generic/upload/stream', files={'files':('stream.txt', text)})
                events = [json.loads(line) for line in stream.text.splitlines()]
                self.assertEqual([e['stage'] for e in events], ['extracting', 'chunking', 'embedding', 'indexing', 'indexed'])
                self.assertTrue(events[-1]['result']['uploaded'][0]['ok'])
                first.delete('/api/generic/documents/stream.txt')

            finally:
                for client in (first, second):
                    token = client.cookies.get('rag_workspace')
                    if token:
                        root = Path('storage/workspaces') / token
                        self.assertEqual(root.resolve().parent, Path('storage/workspaces').resolve())
                        shutil.rmtree(root, ignore_errors=True)
