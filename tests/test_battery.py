import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

class BatteryTests(unittest.TestCase):
    def test_education_and_safety(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/battery').status_code, 200)
            self.assertTrue(client.get('/api/battery/status').json()['ready'])
            self.assertGreater(len(client.get('/api/battery/questions').json()['questions']), 3)
            with patch('app.battery.synthesize_answer') as model:
                result = client.post('/api/battery/chat', json={'message':'My battery is smoking and hissing'}).json()
                model.assert_not_called()
                self.assertIn('emergency services', result['answer'])
                self.assertTrue(result['citations'])
            with patch('app.battery.synthesize_answer', return_value={'answer':'Check the manufacturer instructions.', 'citations':[]}) as model:
                self.assertEqual(client.post('/api/battery/chat', json={'message':'How do I improve battery health?'}).status_code, 200)
                self.assertTrue(model.call_args.args[1])
            self.assertEqual(client.post('/api/battery/chat', json={'message':'  '}).status_code, 422)
            self.assertEqual(client.post('/api/battery/upload').status_code, 404)
