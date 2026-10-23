import json
import unittest
from unittest.mock import patch
from app.rag.models import Chunk,RetrievalHit
from app.rag.model_comparison import compare_answers

class ModelComparisonTests(unittest.TestCase):
    def hit(self,text='A simple document states that reports must be submitted by Friday before the project launch.'):
        return [RetrievalHit(Chunk('one',text,'notes.txt','notes.txt',1,'section',20),.8,1)]

    def test_pair_uses_same_context_and_output_caps(self):
        requests=[]
        def fake(request,*args,**kwargs):
            requests.append(json.loads(request.data))
            return {'choices':[{'message':{'content':'Friday [1]'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':200,'completion_tokens':20}}
        with patch.dict('os.environ',{'OPENAI_API_KEY':'test'}), patch('app.rag.model_comparison.synthesis._json_request',side_effect=fake):
            result=compare_answers('When is the deadline?',self.hit(),['gpt-5-nano','gpt-6-luna','gpt-6.1-sol'])
        self.assertEqual(len(requests),3)
        self.assertEqual(requests[0]['messages'],requests[1]['messages'])
        self.assertEqual(requests[0]['messages'],requests[2]['messages'])
        self.assertTrue(all(r['max_completion_tokens']==1200 for r in requests))
        self.assertTrue(all(item['ok'] for item in result['results']))
        self.assertLess(result['comparison_budget']['answer_cost_upper_estimate_usd'],.05)

    def test_budget_blocks_calls_and_validates_models(self):
        with patch('app.rag.model_comparison.synthesis._json_request') as api:
            with self.assertRaises(ValueError):
                compare_answers('Explain '+('x'*80000),self.hit(),['gpt-5-nano','gpt-6.1-sol'])
            with self.assertRaises(ValueError):
                compare_answers('Explain',self.hit(),['gpt-6-astra','gpt-5-nano'])
            api.assert_not_called()


    def test_large_evidence_is_fitted_for_all_three_models(self):
        requests=[]
        def fake(request,*args,**kwargs):
            requests.append(json.loads(request.data))
            return {'choices':[{'message':{'content':'Supported answer [1]'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':100,'completion_tokens':10}}
        with patch.dict('os.environ',{'OPENAI_API_KEY':'test'}), patch('app.rag.model_comparison.synthesis._json_request',side_effect=fake):
            result=compare_answers('Explain the meaning',self.hit('Unicode test \u0289\u0304 '*10000),['gpt-5-nano','gpt-6-luna','gpt-6.1-sol'])
        self.assertEqual(len(requests),3)
        self.assertTrue(result['comparison_budget']['context_reduced'])
        self.assertTrue(result['comparison_budget']['passage_clipped'])
        self.assertLess(result['comparison_budget']['answer_cost_upper_estimate_usd'],.048)
        self.assertEqual(requests[0]['messages'],requests[2]['messages'])
        self.assertLess(len(result['citations'][0]['text']),len(self.hit('Unicode test \u0289\u0304 '*10000)[0].chunk.text))
