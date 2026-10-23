"""Read-only battery education grounded in a small, reviewed reference library."""
from pathlib import Path
import re
from fastapi import APIRouter
from fastapi.responses import FileResponse
from .main_types import BatteryChatRequest
from .rag.index import RagIndex
from .rag.answer import build_answer
from .rag.synthesis import synthesize_answer, detect_language

ROOT = Path(__file__).resolve().parent.parent
router = APIRouter()


def battery_index():
    index = RagIndex(ROOT / 'data_battery', ROOT / 'storage' / 'battery_index.pkl')
    index.load_or_build()
    return index


@router.get('/battery')
def page():
    return FileResponse(ROOT / 'static' / 'battery.html')


@router.get('/api/battery/status')
def status():
    return battery_index().stats()


@router.get('/api/battery/questions')
def questions():
    return {'questions': [{'question': q, 'topic': 'Battery education', 'source': 'Learning guide'} for q in [
        'What is the difference between battery charge and battery health?',
        'How can I make my phone or laptop battery last longer?',
        'What are the warning signs of battery failure?',
        'What should I do if a battery is swollen?',
        'How should I store an unused device?',
        'How is lead-acid maintenance different from lithium-ion care?',
        'Why does my device shut down even when it shows a charge?',
        'When should I have a battery professionally tested or replaced?']]}


@router.post('/api/battery/chat')
def chat(request: BatteryChatRequest):
    index = battery_index()
    question = request.message
    hits = index.search(question, limit=request.top_k)
    # Always show emergency guidance before any potentially slow provider call.
    danger = re.search(r'\b(smok\w*|hiss\w*|burn\w*|on fire|flames?|sparks?|explod\w*)\b', question, re.I)
    if danger:
        hits = index.search('smoke fire hissing emergency battery safety', limit=3)
        result = build_answer(question, hits)
        result['answer'] = ('If a battery is smoking, hissing, burning or creating immediate danger, move away, evacuate, '
                            'and contact local emergency services from a safe place. Do not handle or carry it, '
                            'and do not attempt to open or repair it. Stop using or charging a damaged battery. '
                            'For general prevention, use manufacturer-approved charging equipment and keep charging away from flammable materials. [1]')
        result.update(model='Safety guidance', mode='safety')
    else:
        instruction = ('Battery education question: ' + question + '\n\n'
            'Explain in plain language using only the reference evidence. Distinguish lithium-ion, flooded lead-acid and sealed AGM. '
            'Ask for device model and battery type when unspecified and needed. Do not invent model-specific limits or diagnoses. '
            'Explain uncertainty and when professional testing is needed. Never provide internal pack repair, bypassing safety systems, '
            'or improvised high-voltage repair instructions. If the library cannot answer, say what information or official manual is needed. '
            'Cite the reference numbers. Acknowledge that maintenance cannot guarantee prevention of every failure.')
        result = synthesize_answer(instruction, hits, response_language=detect_language(question))
    return {'message':question, 'answer':result['answer'], 'citations':result.get('citations', []),
            'model':result.get('model'), 'mode':result.get('mode'),
            'warning': 'Showing reference excerpts because the AI service is unavailable.' if result.get('warning') else None,
            'status':index.stats()}
