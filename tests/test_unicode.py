import unittest
import unicodedata
from unittest.mock import patch, MagicMock
from pathlib import Path
from app.rag.loaders import _load_pdf
from app.rag.synthesis import _result

class UnicodeTests(unittest.TestCase):
    def test_pdf_keeps_letters_and_combining_tones(self):
        text='Mb\u0289\u0304\u0251\u0304 s\u012b mb\u00fa \u00e1 b\u0251\u0304\n\u01cd gh\u0289\u030c n\u0251\u0301 mb\u0289\u0301\u0251\u0301 y\u0101\u0101'
        page=MagicMock();page.get_text.return_value=text
        pdf=MagicMock();pdf.needs_pass=False;pdf.__iter__.return_value=iter([page])
        with patch.object(Path, 'read_bytes', return_value=b'pdf'), patch('app.rag.loaders.pymupdf.open') as opened:
            opened.return_value.__enter__.return_value=pdf
            docs=list(_load_pdf(Path('sample.pdf'),'sample.pdf'))
        self.assertEqual(docs[0].text,unicodedata.normalize('NFC',text))
        self.assertNotIn('mb\u0289\u0302p\u0251p',docs[0].text)
        self.assertEqual(docs[0].page,1)

    def test_answer_decodes_entities_without_dropping_marks(self):
        result=_result('Mb\u0289\u0304\u0251\u0304&#x20;y\u0101\u0101',{'citations':[]},'test','test')
        self.assertEqual(result['answer'],'Mb\u0289\u0304\u0251\u0304 y\u0101\u0101')
