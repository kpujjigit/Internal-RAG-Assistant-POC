from __future__ import annotations

import importlib
import importlib.util
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float


class RAGEngine:
    def __init__(
        self,
        docs_dir: str = "data/docs",
        top_k: int = 3,
        min_confidence: float = 0.20,
        model_name: str = "google/flan-t5-small",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.model_name = model_name

        self._docs: List[Tuple[str, str]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._doc_matrix = None

        self._gen = pipeline(
            "text2text-generation",
            model=self.model_name,
            max_new_tokens=160,
        )

    def _get_sentry(self):
        if importlib.util.find_spec("sentry_sdk") is None:
            return None
        return importlib.import_module("sentry_sdk")

    def load(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")

        docs = []
        for path in sorted(self.docs_dir.glob("*.md")):
            docs.append((path.name, path.read_text(encoding="utf-8")))

        if not docs:
            raise ValueError(f"No docs found in {self.docs_dir}")

        self._docs = docs
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = self._vectorizer.fit_transform([text for _, text in self._docs])

    def retrieve(self, question: str) -> List[RetrievedDoc]:
        if self._vectorizer is None or self._doc_matrix is None:
            raise RuntimeError("Call load() before retrieve().")

        sentry_sdk = self._get_sentry()
        span_context = (
            sentry_sdk.start_span(op="rag.retrieve", description="TF-IDF retrieve")
            if sentry_sdk
            else nullcontext()
        )
        with span_context:
            q_vec = self._vectorizer.transform([question])
            sims = cosine_similarity(q_vec, self._doc_matrix).flatten()

            ranked = sorted(
                [(i, float(sims[i])) for i in range(len(sims))],
                key=lambda x: x[1],
                reverse=True,
            )[: self.top_k]

            results: List[RetrievedDoc] = []
            for idx, score in ranked:
                doc_id, text = self._docs[idx]
                results.append(RetrievedDoc(doc_id=doc_id, text=text, score=score))

            if sentry_sdk:
                span_context.set_tag("rag.top_k", self.top_k)
                if results:
                    span_context.set_tag("rag.best_score", results[0].score)
                    span_context.set_tag("rag.best_doc", results[0].doc_id)

            return results

    def generate(self, question: str, retrieved: List[RetrievedDoc]) -> str:
        best_score = retrieved[0].score if retrieved else 0.0

        if best_score < self.min_confidence:
            sentry_sdk = self._get_sentry()
            if sentry_sdk:
                sentry_sdk.capture_message(
                    "Low retrieval confidence; returning 'I don't know'",
                    level="info",
                )
            return (
                "I’m not confident I can answer from the available docs. "
                "Could you clarify what team/policy scope you mean, or provide more context?"
            )

        context = "\n\n".join(
            [f"[{doc.doc_id} | score={doc.score:.3f}]\n{doc.text}" for doc in retrieved]
        )

        prompt = (
            "You are a helpful internal assistant. Answer the question using ONLY the context.\n"
            "If the context is ambiguous or contradictory, say so and ask a clarifying question.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "ANSWER:"
        )

        sentry_sdk = self._get_sentry()
        span_context = (
            sentry_sdk.start_span(op="llm.generate", description=self.model_name)
            if sentry_sdk
            else nullcontext()
        )
        with span_context:
            start = time.time()
            output = self._gen(prompt)[0]["generated_text"].strip()
            if sentry_sdk:
                span_context.set_tag("llm.model", self.model_name)
                span_context.set_tag("llm.latency_ms", int((time.time() - start) * 1000))
            return output
