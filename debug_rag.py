#!/usr/bin/env python
"""
Debug script to test the RAG pipeline directly and see if retrieval works.
Run this to diagnose issues without the Streamlit UI.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import VECTORSTORE_DIR, DATA_DIR, TOP_K
from src.rag_pipeline import RAGPipeline

print("Testing RAG Pipeline Directly")
print("=" * 60)

# Test 1: Check if vectorstore exists
print(f"\n1. Checking vectorstore at: {VECTORSTORE_DIR}")
if VECTORSTORE_DIR.exists():
    print(f"   ✓ Vectorstore directory exists")
    files = list(VECTORSTORE_DIR.glob("*"))
    print(f"   Files: {[f.name for f in files]}")
else:
    print(f"   ✗ Vectorstore directory NOT found!")
    sys.exit(1)

# Test 2: Initialize RAG pipeline
print(f"\n2. Initializing RAG pipeline...")
try:
    rag = RAGPipeline(VECTORSTORE_DIR, use_chatgpt=False, use_reranker=False)
    rag.ensure_loaded()
    print(f"   ✓ RAG pipeline initialized successfully")
except Exception as e:
    print(f"   ✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Try retrieval with a test query
print(f"\n3. Testing retrieval with query: 'bank limit'")
try:
    contexts = rag.retrieve("bank limit", TOP_K)
    if contexts:
        print(f"   ✓ Retrieved {len(contexts)} contexts")
        for i, ctx in enumerate(contexts[:2], 1):
            print(f"\n   Context {i}:")
            print(f"   Score: {ctx.get('score', 'N/A')}")
            print(f"   Text: {ctx.get('text', 'N/A')[:100]}...")
    else:
        print(f"   ✗ No contexts retrieved (empty list)")
except Exception as e:
    print(f"   ✗ Retrieval failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Try answer generation
print(f"\n4. Testing answer generation...")
try:
    result = rag.answer("bank limit", contexts)
    print(f"   ✓ Answer generated successfully")
    print(f"   Answer: {result.get('answer', 'N/A')[:200]}...")
except Exception as e:
    print(f"   ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n" + "=" * 60)
print("All tests passed! The RAG pipeline is working correctly.")
