import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app.rag.models import Chunk
from app.rag.cache import ResponseCache
from app.rag.llm_questions import suggest_questions

class LLMQuestionTests(unittest.TestCase):
    def test_generated_questions_are_validated_and_cached(self):
        chunk = Chunk('one', 'Aurora launches in November. Reports are due Friday.', 'plan.txt', 'plan.txt', 1, 'intro', 20)
        index = SimpleNamespace(chunks=[chunk])
        output = json.dumps([{'question':'When does the Aurora project launch?', 'evidence_rank':1},
                             {'question':'When must the team submit its reports?', 'evidence_rank':1},
                             {'question':'What is the unmentioned project budget?', 'evidence_rank':99}])
        with tempfile.TemporaryDirectory() as folder:
            cache = ResponseCache(Path(folder)/'questions.json')
            with patch('app.rag.llm_questions.synthesis.PROVIDER_ORDER',['openai']), patch('app.rag.llm_questions.synthesis._call_openai',return_value=output) as model:
                result = suggest_questions(index, cache)
                self.assertEqual(result['generation'],'llm')
                self.assertEqual(len(result['questions']),2)
                self.assertEqual(result['questions'][0]['source'],'plan.txt')
                suggest_questions(index, cache)
                self.assertEqual(model.call_count,1)
                suggest_questions(index, cache, batch=1)
                self.assertEqual(model.call_count,2)
                suggest_questions(index, cache, batch=9)
                self.assertEqual(model.call_count,2)  # Eight bounded cached batches.
            with patch('app.rag.llm_questions.synthesis.PROVIDER_ORDER',['openai']), patch('app.rag.llm_questions.synthesis._call_openai',return_value='invalid'):
                fallback = suggest_questions(index, ResponseCache(Path(folder)/'other.json'))
                self.assertEqual(fallback['generation'],'fallback')
                self.assertTrue(fallback['notice'])
