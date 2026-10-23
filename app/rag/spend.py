"""Persistent, conservative reservations for paid model calls (not a billing ledger)."""
import json
from contextlib import closing
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
RATES = {'gpt-5-nano':(.05,.40), 'gpt-6-luna':(.10,.50), 'gpt-6.1-sol':(2.,10.),
         'claude-haiku-4-5-20251001':(1.,5.), 'llama-3.3-70b-versatile':(5.,20.)}

class SpendLimitError(RuntimeError):
    pass


def reserve(amount):
    limit=float(os.getenv('RAG_DAILY_BUDGET_USD','0.50'))
    path=Path(os.getenv('RAG_SPEND_DB',str(ROOT/'storage'/'spend.sqlite3')))
    path.parent.mkdir(parents=True,exist_ok=True)
    day=datetime.now(timezone.utc).date().isoformat()
    with closing(sqlite3.connect(path,timeout=5)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS allowance (day TEXT PRIMARY KEY, reserved REAL NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT reserved FROM allowance WHERE day=?',(day,)).fetchone()
        total=row[0] if row else 0.
        if total+amount>limit:
            raise SpendLimitError('The daily AI allowance has been reached. Document search is still available; try AI again after the UTC daily reset.')
        db.execute('INSERT INTO allowance(day,reserved) VALUES (?,?) ON CONFLICT(day) DO UPDATE SET reserved=excluded.reserved',(day,total+amount))
    # Reservations are deliberately retained even after failures: billing may be uncertain.


def protect_model_request(request):
    if not request.data:
        return
    data=json.loads(request.data)
    model=data.get('model','')
    if data.get('tools') or data.get('functions'):
        raise SpendLimitError('External tools are disabled for document chat.')
    host=urlparse(request.full_url).hostname
    configured_local=urlparse(os.getenv('OLLAMA_URL','http://127.0.0.1:11434')).netloc
    if urlparse(request.full_url).scheme not in {'http','https'} or (host not in {'api.openai.com','api.groq.com','api.anthropic.com'} and urlparse(request.full_url).netloc != configured_local):
        raise SpendLimitError('Model destination is not approved.')
    paid=host in {'api.openai.com','api.groq.com','api.anthropic.com'}
    if paid and model not in RATES:
        raise SpendLimitError('This model has no approved pricing allowance. Use one of the configured supported models.')
    cap=1600
    if host=='api.anthropic.com':
        data['max_tokens']=min(int(data.get('max_tokens',cap)),cap)
        output=data['max_tokens']
    elif paid:
        output=min(int(data.get('max_completion_tokens',data.get('max_tokens',cap))),cap)
        data.pop('max_tokens',None)
        data['max_completion_tokens']=output
        if host=='api.openai.com':
            data['service_tier']='default'
            if model=='gpt-5-nano': data.setdefault('reasoning_effort','minimal')
    else:
        data.setdefault('options',{})['num_predict']=min(int(data.get('options',{}).get('num_predict',cap)),cap)
        output=data['options']['num_predict']
    if output<1: raise SpendLimitError('Invalid model output limit.')
    text=json.dumps(data.get('messages',[]),ensure_ascii=False)+str(data.get('system',''))
    input_bound=len(text.encode('utf-8'))+4096
    if input_bound>100000:
        raise SpendLimitError('The AI context is too large. Narrow the question or reduce the conversation history.')
    if paid:
        rate_in,rate_out=RATES[model]
        upper=(input_bound*rate_in+output*rate_out)/1e6
        if upper>float(os.getenv('RAG_CALL_BUDGET_USD','0.05')):
            raise SpendLimitError('This AI call exceeds its cost allowance. Narrow the question or use a less expensive model.')
        reserve(upper)
    request.data=json.dumps(data).encode('utf-8')
    request.remove_header('Content-length')
