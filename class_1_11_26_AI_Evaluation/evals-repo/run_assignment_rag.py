"""Run Assignment 3: City Library FAQ RAG evaluation.

Run from this directory with: uv run python run_assignment_rag.py
Requires GEMINI_API_KEY in the root .env file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import ask, embed, MODEL

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "assignment_rag.json").read_text(encoding="utf-8"))
docs, queries = DATA["documents"], DATA["queries"]
K = 3


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def make_chunks(documents: list[dict], size: int, overlap: int) -> list[dict]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    chunks = []
    step = size - overlap
    for doc in documents:
        words = doc["text"].split()
        for start in range(0, len(words), step):
            part = words[start : start + size]
            if not part:
                break
            chunks.append({"chunk_id": len(chunks), "doc_id": doc["id"], "text": " ".join(part)})
            if start + size >= len(words):
                break
    return chunks


def grade(chunk: dict, query: dict) -> int:
    if norm(query["answer_span"]) in norm(chunk["text"]):
        return 2
    return 1 if chunk["doc_id"] == query["doc_id"] else 0


def hit_at_k(grades: list[int], k: int, thr: int = 2) -> float:
    return float(any(g >= thr for g in grades[:k]))


def reciprocal_rank(grades: list[int], k: int, thr: int = 2) -> float:
    for rank, value in enumerate(grades[:k], start=1):
        if value >= thr:
            return 1.0 / rank
    return 0.0


def dcg_at_k(grades: list[int], k: int) -> float:
    return sum(value / math.log2(rank + 1) for rank, value in enumerate(grades[:k], start=1))


def ndcg_at_k(grades: list[int], k: int) -> float:
    ideal = dcg_at_k(sorted(grades[:k], reverse=True), k)
    return dcg_at_k(grades, k) / ideal if ideal else 0.0


# Verify answer spans before evaluating retrieval.
for query in queries:
    owners = [doc["id"] for doc in docs if norm(query["answer_span"]) in norm(doc["text"])]
    if query["doc_id"] not in owners:
        raise AssertionError(f"Answer span missing from source document: {query['query']}")

print("Assignment 3: City Library FAQ")
base_chunks = make_chunks(docs, 30, 0)
print("size=30, overlap=0 ->", len(base_chunks), "chunks")
test_grades = [0, 2, 0, 1]
print(
    "Metric check [0, 2, 0, 1]:",
    f"Hit@3={hit_at_k(test_grades, K):.2f}",
    f"MRR@3={reciprocal_rank(test_grades, K):.2f}",
    f"NDCG@3={ndcg_at_k(test_grades, K):.2f}",
)

# Cache vectors and embed batches of texts in one API call per batch.
_embedding_cache: dict[str, np.ndarray] = {}


def embed_texts(texts: list[str]) -> np.ndarray:
    missing = [text for text in dict.fromkeys(texts) if text not in _embedding_cache]
    for start in range(0, len(missing), 50):
        batch = missing[start : start + 50]
        vectors = np.asarray(embed(batch), dtype=float)
        for text, vector in zip(batch, vectors):
            _embedding_cache[text] = vector
    matrix = np.asarray([_embedding_cache[text] for text in texts], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def evaluate(size: int, overlap: int) -> tuple[list[dict], pd.DataFrame]:
    chunks = make_chunks(docs, size, overlap)
    chunk_vectors = embed_texts([chunk["text"] for chunk in chunks])
    rows = []
    for query in queries:
        query_vector = embed_texts([query["query"]])[0]
        order = np.argsort(-(chunk_vectors @ query_vector), kind="stable")
        grades = [grade(chunks[int(index)], query) for index in order]
        rows.append(
            {
                "query": query["query"],
                "hit": hit_at_k(grades, K),
                "rr": reciprocal_rank(grades, K),
                "ndcg": ndcg_at_k(grades, K),
            }
        )
    return chunks, pd.DataFrame(rows)


chunks_30, baseline = evaluate(30, 0)
print(
    "size=30, overlap=0 average:",
    {
        "Hit@3": round(float(baseline.hit.mean()), 2),
        "MRR@3": round(float(baseline.rr.mean()), 2),
        "NDCG@3": round(float(baseline.ndcg.mean()), 2),
    },
)

# Grid search.
grid_rows = []
for size in [30, 60]:
    for overlap in [0, 15]:
        chunks, frame = evaluate(size, overlap)
        grid_rows.append(
            {
                "chunk_size": size,
                "overlap": overlap,
                "n_chunks": len(chunks),
                "Hit@3": frame.hit.mean(),
                "MRR@3": frame.rr.mean(),
                "NDCG@3": frame.ndcg.mean(),
            }
        )
grid = pd.DataFrame(grid_rows)
print("\nGrid search")
print(grid.round(3).to_string(index=False))
best = grid.sort_values(["Hit@3", "MRR@3"], ascending=False, kind="stable").iloc[0]
BEST_SIZE, BEST_OVERLAP = int(best.chunk_size), int(best.overlap)
print(f"Best setting: size={BEST_SIZE}, overlap={BEST_OVERLAP}")

# Generate answers from the best setting's top-three retrieved chunks.
GROUNDED = (
    "Answer the question using ONLY the documents provided. If they do not contain the answer, "
    "reply exactly: I don't know based on the documents. Answer in one or two sentences."
)
best_chunks = make_chunks(docs, BEST_SIZE, BEST_OVERLAP)
best_vectors = embed_texts([chunk["text"] for chunk in best_chunks])


def retrieve(query_text: str) -> list[dict]:
    query_vector = embed_texts([query_text])[0]
    order = np.argsort(-(best_vectors @ query_vector), kind="stable")[:K]
    return [best_chunks[int(index)] for index in order]


def answer_with(query_text: str, context: str, system: str) -> str:
    return (ask(f"Documents:\n{context}\n\nQuestion: {query_text}", system=system, temperature=0) or "").strip()


JUDGE_PROMPT = """Judge whether every factual claim in the answer is supported by the documents.
Return JSON only with fields `answer` (exactly `Pass` or `Fail`) and `reasoning` (one short sentence).
Documents:
{context}
Question: {query}
Answer: {answer}"""


def judge_faithfulness(context: str, query_text: str, answer: str) -> tuple[str, str]:
    result = ask(
        JUDGE_PROMPT.format(context=context, query=query_text, answer=answer),
        as_json=True,
    )
    if isinstance(result, list):
        result = result[0]
    verdict = result.get("answer") if isinstance(result, dict) else None
    if verdict not in {"Pass", "Fail"}:
        raise ValueError(f"Judge answer must be exactly Pass or Fail; received {result!r}")
    return verdict, str(result.get("reasoning", ""))


def diagnose(hit: bool, correct: bool) -> str:
    if hit and correct:
        return "OK"
    if hit:
        return "generation problem"
    if correct:
        return "lucky"
    return "retrieval problem"


def keyword_correct(answer: str, keywords: list[str]) -> bool:
    folded = answer.casefold()
    return any(keyword.casefold() in folded for keyword in keywords)


def make_result(query: dict, chunks: list[dict], answer: str, fixture: bool = False) -> dict:
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    hit = any(norm(query["answer_span"]) in norm(chunk["text"]) for chunk in chunks)
    correct = keyword_correct(answer, query["answer_keywords"])
    faithful, reasoning = judge_faithfulness(context, query["query"], answer)
    return {
        "query": query["query"],
        "retrieval_hit": hit,
        "answer_correct": correct,
        "faithful": faithful,
        "diagnosis": diagnose(hit, correct),
        "fixture": "FIXTURE" if fixture else "",
        "answer": answer,
        "judge_reasoning": reasoning,
    }


results_rows = []
for query in queries:
    retrieved = retrieve(query["query"])
    context = "\n\n".join(chunk["text"] for chunk in retrieved)
    generated = answer_with(query["query"], context, GROUNDED)
    results_rows.append(make_result(query, retrieved, generated))

# Clearly marked hand-written unfaithful fixture for a known judge failure.
fixture_query = queries[0]
fixture_answer = "You can borrow up to 15 books at a time, and the library delivers them to your home."
results_rows.append(make_result(fixture_query, retrieve(fixture_query["query"]), fixture_answer, fixture=True))
results = pd.DataFrame(results_rows)
print("\nGeneration results (fixture is intentionally unfaithful)")
print(results[["query", "retrieval_hit", "answer_correct", "faithful", "diagnosis", "fixture", "answer"]].to_string(index=False))
print("Diagnosis counts:")
counts = results["diagnosis"].value_counts()
for label in ["OK", "generation problem", "retrieval problem", "lucky"]:
    print(f"  {label}: {int(counts.get(label, 0))}")

# Context swap: Sunday opening question with real and unrelated source documents.
swap_query = next(query for query in queries if "Sunday" in query["query"])
real_doc = next(doc["text"] for doc in docs if doc["id"] == swap_query["doc_id"])
unrelated_doc = next(doc["text"] for doc in docs if doc["id"] != swap_query["doc_id"])
plain = "You are a helpful library assistant."
answers = [
    {"prompt": "Plain", "context": "Real document", "answer": answer_with(swap_query["query"], real_doc, plain)},
    {"prompt": "Plain", "context": "Unrelated document", "answer": answer_with(swap_query["query"], unrelated_doc, plain)},
    {"prompt": "Grounded", "context": "Real document", "answer": answer_with(swap_query["query"], real_doc, GROUNDED)},
    {"prompt": "Grounded", "context": "Unrelated document", "answer": answer_with(swap_query["query"], unrelated_doc, GROUNDED)},
]
print("\nContext swap question:", swap_query["query"])
print(pd.DataFrame(answers).pivot(index="prompt", columns="context", values="answer").to_string())
