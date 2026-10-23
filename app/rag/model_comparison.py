"""Small, bounded answer-model comparisons on identical retrieved evidence."""
import json
import os
import time
import urllib.request
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from . import synthesis
from .answer import build_answer

PRICES = {'gpt-5-nano':(.05,.40), 'gpt-6-luna':(.10,.50), 'gpt-6.1-sol':(2.,10.)}
MAX_OUTPUT = 1200
BUDGET_USD = .05


def compare_answers(question, hits, models, response_language=None):
    if not 2 <= len(models) <= 3 or len(set(models)) != len(models) or any(m not in PRICES for m in models):
        raise ValueError('Choose two or three different supported models.')
    if not hits:
        raise ValueError('No supporting document passages found. Try a more specific question.')
    selected = list(hits)
    def estimate(evidence):
        prompt = synthesis._messages(question,evidence,response_language)
        input_bound = len(json.dumps(prompt,ensure_ascii=False).encode('utf-8')) + 4096
        cost = sum((input_bound*PRICES[m][0] + MAX_OUTPUT*PRICES[m][1])/1e6 for m in models)
        return prompt,cost
    messages,bound = estimate(selected)
    while bound > BUDGET_USD - .002 and len(selected)>1:
        selected.pop()  # Retrieval order puts the strongest matches first.
        messages,bound = estimate(selected)
    clipped=False
    while bound > BUDGET_USD - .002 and len(selected[0].chunk.text)>300:
        hit=selected[0]
        text=hit.chunk.text[:max(300,len(hit.chunk.text)//2)]
        # Prefer a word boundary while preserving all Unicode characters.
        if ' ' in text[200:]: text=text.rsplit(' ',1)[0]
        selected[0]=replace(hit,chunk=replace(hit.chunk,text=text))
        clipped=True
        messages,bound=estimate(selected)
    if bound > BUDGET_USD - .002:
        raise ValueError('The question itself is too long for the $0.05 comparison budget. Shorten the question and try again.')
    context_reduced=clipped or len(selected)<len(hits)
    key = os.getenv('OPENAI_API_KEY')
    if not key: raise ValueError('OpenAI is not configured.')
    def run(model):
        started=time.perf_counter()
        pricing={'input_per_million':PRICES[model][0], 'output_per_million':PRICES[model][1]}
        base={'mode':'openai','model':model,'pricing':pricing}
        payload={'model':model,'messages':messages,'max_completion_tokens':MAX_OUTPUT,'service_tier':'default',
                 'reasoning_effort':'minimal' if model=='gpt-5-nano' else 'none' if model=='gpt-6-luna' else 'low'}
        request=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps(payload).encode(),
              headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},method='POST')
        try:
            data=synthesis._json_request(request,model,timeout=90)
            choice=data.get('choices',[{}])[0]
            answer=choice.get('message',{}).get('content','') or ''
            usage=data.get('usage',{})
            tokens={'input':usage.get('prompt_tokens',0),'output':usage.get('completion_tokens',0),'estimated':False}
            if choice.get('finish_reason')=='length': answer+='\n\n[Response reached the comparison token limit.]'
            cost=(tokens['input']*PRICES[model][0]+tokens['output']*PRICES[model][1])/1e6
            return {**base,'ok':bool(answer),'answer':answer,'error':'' if answer else 'No answer within the token limit.',
                    'tokens':tokens,'estimated_cost_usd':cost,'elapsed_seconds':round(time.perf_counter()-started,2)}
        except synthesis.ProviderUnavailable as exc:
            return {**base,'ok':False,'answer':'','error':str(exc),'tokens':{'input':0,'output':0,'estimated':True},
                    'estimated_cost_usd':None,'elapsed_seconds':round(time.perf_counter()-started,2)}
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        results=list(pool.map(run,models))
    return {'results':results,'citations':build_answer(question,selected)['citations'],
            'comparison_budget':{'context_reduced':context_reduced, 'original_passages':len(hits), 'included_passages':len(selected), 'passage_clipped':clipped, 'limit_usd':BUDGET_USD,'answer_cost_upper_estimate_usd':round(bound,6),
                                 'note':'Costs use current standard token rates. Shared intent processing is additional; total request preflight reserves $0.002 for it.'}}
