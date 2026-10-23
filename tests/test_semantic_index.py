import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from app.rag.index import RagIndex, chunk_documents
from app.rag.models import Document

class SemanticIndexTests(unittest.TestCase):
    def test_sentence_components_precede_repeated_senses_and_neighbors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'docs'; data.mkdir()
            for n in range(8):
                (data/f'dog-{n}.txt').write_text('Dictionary entry chien. Le chien is a domestic animal. This passage describes its appearance and several related animal names and examples.')
            (data/'laugh.txt').write_text('Dictionary verb rire. Rire means to laugh, express amusement. This entry describes the verb and an example of someone laughing with friends.')
            index=RagIndex(data,root/'index.pkl'); index.rebuild()
            hits=index.search('Translate le chien rit',phrase='le chien rit',search_phrases=['chien','rire'],all_meanings=True)
            self.assertTrue(any('chien' in h.chunk.text for h in hits[:2]))
            self.assertTrue(any('rire' in h.chunk.text for h in hits[:2]))

    def test_semantic_search_reload_reuse_and_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); data = root/'documents'; data.mkdir()
            (data/'care.txt').write_text('Excessive heat accelerates lithium cell degradation. Keeping the device cool preserves capacity and extends its useful lifespan. Avoid leaving electronics in a hot parked vehicle.')
            (data/'cooking.txt').write_text('A chef prepares vegetable soup using carrots onions tomatoes and fresh herbs. Simmer the ingredients in a large pot and serve with toasted bread.')
            path = root/'index.pkl'
            index = RagIndex(data, path); stages=[]
            index.rebuild(lambda stage, detail: stages.append(stage))
            self.assertEqual(stages, ['chunking','embedding','indexing'])
            self.assertEqual(index.stats()['retrieval']['dimensions'],384)
            self.assertEqual(index.search('How do I prevent my battery from wearing out prematurely?',limit=1)[0].chunk.source,'care.txt')
            self.assertTrue(np.allclose(np.linalg.norm(index.vectors,axis=1),1,atol=1e-5))
            loaded=RagIndex(data,path)
            with patch('app.rag.index.encode', side_effect=AssertionError('Unexpected re-embedding')):
                loaded.load_or_build()
                loaded.rebuild()
            self.assertEqual(loaded.faiss_index.ntotal,2)
            (data/'care.txt').unlink()
            loaded.load_or_build()
            self.assertEqual(loaded.faiss_index.ntotal,1)
            self.assertTrue(all(hit.chunk.source != 'care.txt' for hit in loaded.search('battery',limit=2)))
            (data/'cooking.txt').unlink(); loaded.load_or_build()
            self.assertFalse(loaded.ready)
            self.assertEqual(loaded.stats()['retrieval']['vectors'],0)
            self.assertEqual(loaded.search('anything'),[])

    def test_legacy_migration_and_long_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'docs'; data.mkdir()
            (data/'notes.txt').write_text('This document describes a test project with useful background information and several important implementation details for the team.')
            path=root/'index.pkl'; path.write_bytes(pickle.dumps({'fingerprint':'old-keyword-index'}))
            index=RagIndex(data,path); index.load_or_build()
            self.assertEqual(index.stats()['retrieval']['engine'],'FAISS')
            self.assertEqual(pickle.loads(path.read_bytes())['schema'],3)
        chunks=chunk_documents([Document(' '.join(['word']*1500),'long.txt','long.txt')])
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(len(c.text.split())<=140 for c in chunks))


    def test_exact_translation_example_beats_related_definitions(self):
        from app.rag.synthesis import detect_language
        self.assertEqual(detect_language("how do you say je t'aime"), 'English')
        self.assertEqual(detect_language("comment dire bonjour"), 'French')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'docs'; data.mkdir()
            (data/'example.txt').write_text("Dictionary example entry. Example expression : je t'aime. This entry directly illustrates a complete sentence rather than a general related word definition.")
            (data/'related.txt').write_text('Love is a feeling of affection. This entry describes love songs and pure love but gives no complete example of a translated sentence.')
            index=RagIndex(data,root/'index.pkl'); index.rebuild()
            for query in ["how do you say je t'aime", "How do you say je t’aime in Nufi?", 'Translate "je t’aime" into Nufi']:
                self.assertEqual(index.search(query,limit=1,candidate_pool=1)[0].chunk.source,'example.txt')


    def test_all_meanings_can_exceed_top_k(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'docs'; data.mkdir()
            for i in range(10):
                (data/f'entry-{i}.txt').write_text(f'Dictionary entry number {i}. The expression target phrase occurs here with meaning number {i}, and this passage gives its specific context and source example.')
            index=RagIndex(data,root/'index.pkl');index.rebuild()
            hits=index.search('Explain target phrase',phrase='target phrase',limit=1,all_meanings=True)
            self.assertEqual(len(hits),10)
            self.assertEqual(index.meaning_coverage['matching_passages'],10)
            self.assertEqual(index.meaning_coverage['included_matches'],10)
            self.assertFalse(index.meaning_coverage['limited'])


    def test_pdf_line_wraps_do_not_split_glosses(self):
        from app.rag.synthesis import _format_context, _messages
        from app.rag.models import Chunk, RetrievalHit
        chunk=Chunk('entry', 'example : first half\n\nsecond half. 2. Another sense.', 'dictionary.txt','dictionary.txt',1,'entry',20)
        hits=[RetrievalHit(chunk,.8,1)]
        context=_format_context(hits)
        self.assertIn('example : first half second half. 2. Another sense.',context)
        prompt=_messages('Explain this word',hits)[0]['content']
        self.assertIn('ALL distinct',prompt)
        self.assertIn('a new headword',prompt)


    def test_cross_language_expansion_retrieves_french_example(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'docs';data.mkdir()
            (data/'dictionary.txt').write_text('Dictionnaire de reference. Sens: plaire, faire plaisir. Exemple du document : je suis content. Cet exemple illustre un sentiment de satisfaction et de joie.')
            (data/'other.txt').write_text('A technical guide describes electrical circuits and battery health, charging currents and the required precautions for equipment maintenance and transport.')
            index=RagIndex(data,root/'index.pkl');index.rebuild()
            hits=index.search('How do you say I am happy?',phrase='I am happy',search_phrases=['je suis heureux','je suis content'],all_meanings=True,limit=1)
            self.assertEqual(hits[0].chunk.source,'dictionary.txt')
            self.assertIn('je suis content',hits[0].chunk.text)
