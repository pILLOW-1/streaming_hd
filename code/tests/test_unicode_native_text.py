import hashlib,sys,types,unicodedata,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.native_inputs import exact_decoded_text
class FakeTokenizer:
    def __init__(self):self.backend_tokenizer=types.SimpleNamespace(normalizer=types.SimpleNamespace(normalize_str=lambda t:unicodedata.normalize('NFC',t)))
    def decode(self,ids,**kw):return ''.join(map(chr,ids))
    def __call__(self,text,**kw):return {'input_ids':list(map(ord,unicodedata.normalize('NFC',text)))}
class UnicodeRecoveryTests(unittest.TestCase):
    def test_recovers_exact_original_and_preserves_token_ids(self):
        tok=FakeTokenizer();prompt='Question: Cafe\u0301?\n';steps=['A\u0304 = 1.','Therefore B.'];text=prompt+'\n\n'.join(steps)+'\nAnswer: 1';ids=tok(text)['input_ids'];start=len(tok(prompt)['input_ids'])
        n=dict(input_ids=ids,original_steps=steps,question='Cafe\u0301?',answer='1',step_token_spans=[[start,start+6],[start+8,start+20]],full_text_SHA256=hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(exact_decoded_text(tok,n),text)
        n['full_text_SHA256']='bad'
        with self.assertRaises(ValueError):exact_decoded_text(tok,n)
if __name__=='__main__':unittest.main()
