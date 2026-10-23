import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.rag.intent import interpret_question
from app.rag.cache import ResponseCache

class IntentTests(unittest.TestCase):
    def result(self, phrase='je suis malade'):
        return {'intent':'translation','search_query':f'How do you say "{phrase}"?','phrase':phrase,
                'response_language':'English','needs_clarification':False,'clarification':''}

    def test_informal_request_and_cache(self):
        question='hw d u say je suis malade'
        with tempfile.TemporaryDirectory() as folder, patch('app.rag.intent.llm.PROVIDER_ORDER',['openai']), patch('app.rag.intent._completion', return_value=self.result()) as model:
            cache=ResponseCache(Path(folder)/'intents.json')
            result=interpret_question(question,cache)
            self.assertEqual(result['phrase'],'je suis malade')
            self.assertEqual(result['search_query'],'How do you say "je suis malade"?')
            self.assertEqual(result['response_language'],'English')
            interpret_question(question,cache); self.assertEqual(model.call_count,1)

    def test_never_invent_or_change_target_phrase(self):
        with patch('app.rag.intent.llm.PROVIDER_ORDER',['openai']), patch('app.rag.intent._completion', return_value=self.result('I am ill')):
            result=interpret_question('hw d u say je suis malade')
            self.assertEqual(result['mode'],'original')
            self.assertEqual(result['search_query'],'hw d u say je suis malade')

    def test_ambiguous_question_requests_clarification(self):
        data={'intent':'question','search_query':'that thing','phrase':'','response_language':'English',
              'needs_clarification':True,'clarification':'Which topic in the document do you mean?'}
        with patch('app.rag.intent.llm.PROVIDER_ORDER',['openai']), patch('app.rag.intent._completion',return_value=data):
            result=interpret_question('wht abt tht')
            self.assertTrue(result['needs_clarification'])


    def test_clarification_reply_uses_previous_phrase(self):
        history=[{'role':'user','content':'comment dit on I am happy'},
                 {'role':'assistant','content':'Should I use masculine or feminine form?'}]
        data=self.result('I am happy');data['search_query']='How do you say "I am happy" for a feminine speaker?';data['response_language']='French'
        with patch('app.rag.intent.llm.PROVIDER_ORDER',['openai']), patch('app.rag.intent._completion',return_value=data) as model:
            result=interpret_question('feminin',history=history,documents=['Nufi-French dictionary.pdf'])
            self.assertEqual(result['phrase'],'I am happy')
            self.assertIn('feminine',result['search_query'])
            self.assertEqual(model.call_args.kwargs['history'],history)

    def test_known_translation_phrase_does_not_block_on_gender(self):
        data=self.result('I am happy');data['needs_clarification']=True;data['clarification']='Masculine or feminine?'
        with patch('app.rag.intent.llm.PROVIDER_ORDER',['openai']), patch('app.rag.intent._completion',return_value=data):
            result=interpret_question('comment dit on I am happy')
            self.assertFalse(result['needs_clarification'])


    def test_response_language_is_not_an_invented_target(self):
        data=self.result('I am happy');data['search_query']='How do you say I am happy in French?';data['response_language']='French'
        with patch('app.rag.intent.llm.PROVIDER_ORDER',['openai']), patch('app.rag.intent._completion',return_value=data):
            result=interpret_question('comment dit on I am happy')
            self.assertNotIn('in French',result['search_query'])


    def test_search_expansions_preserve_original_target(self):
        data=self.result('I am happy');data['search_phrases']=['je suis heureux','je suis content']
        with patch('app.rag.intent.llm.PROVIDER_ORDER',['openai']), patch('app.rag.intent._completion',return_value=data):
            result=interpret_question('comment dit on I am happy',documents=['French-Nufi dictionary.pdf'])
            self.assertEqual(result['phrase'],'I am happy')
            self.assertEqual(result['search_phrases'],['je suis heureux','je suis content'])
