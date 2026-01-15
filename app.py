from __future__ import annotations

import importlib
import importlib.util
import os
import time
from contextlib import nullcontext
from types import ModuleType


def get_sentry() -> ModuleType | None:
    if importlib.util.find_spec("sentry_sdk") is None:
        return None
    return importlib.import_module("sentry_sdk")

def get_streamlit() -> ModuleType | None:
    if importlib.util.find_spec("streamlit") is None:
        return None
    return importlib.import_module("streamlit")


def get_load_dotenv():
    if importlib.util.find_spec("dotenv") is None:
        return None
    dotenv = importlib.import_module("dotenv")
    return dotenv.load_dotenv


def init_sentry() -> None:
    sentry_sdk = get_sentry()
    if sentry_sdk is None:
        return
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0")),
        environment=os.getenv("SENTRY_ENVIRONMENT", "local"),
    )


def main() -> None:
    st = get_streamlit()
    if st is None:
        raise RuntimeError("Streamlit is required. Install dependencies with: pip install -r requirements.txt")

    load_dotenv = get_load_dotenv()
    if load_dotenv is None:
        raise RuntimeError("python-dotenv is required. Install dependencies with: pip install -r requirements.txt")
    load_dotenv()
    init_sentry()

    st.set_page_config(page_title="Internal Q&A (RAG) Demo", layout="wide")
    st.title("Internal Q&A Assistant (RAG) — Demo")
    st.caption("Open-source model + TF-IDF retrieval + Sentry instrumentation")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top K docs", 1, 5, 3)
        min_conf = st.slider("Min confidence", 0.0, 0.6, 0.20, 0.01)
        simulate_slow = st.toggle("Simulate slow response (anomaly)")
        simulate_error = st.toggle("Simulate exception (anomaly)")

    from rag import RAGEngine

    rag = RAGEngine(top_k=top_k, min_confidence=min_conf)
    rag.load()

    question = st.text_input(
        "Ask a question about company policy / product info:",
        placeholder="e.g., What is our PTO policy? How is data protected?",
    )

    if st.button("Answer", type="primary", disabled=not question.strip()):
        sentry_sdk = get_sentry()
        transaction = (
            sentry_sdk.start_transaction(op="ui.query", name="rag_demo_query")
            if sentry_sdk
            else nullcontext()
        )
        with transaction:
            if sentry_sdk:
                sentry_sdk.set_tag("rag.top_k", top_k)
                sentry_sdk.set_tag("rag.min_confidence", min_conf)

            if simulate_slow:
                time.sleep(2.0)

            if simulate_error:
                error = RuntimeError("Simulated failure for Sentry demo")
                if sentry_sdk:
                    sentry_sdk.capture_exception(error)
                st.error("Simulated exception captured (check Sentry).")
                return

            retrieved = rag.retrieve(question)
            answer = rag.generate(question, retrieved)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Answer")
                st.write(answer)

            with col2:
                st.subheader("Retrieved context")
                if not retrieved:
                    st.info("No docs retrieved.")
                else:
                    for doc in retrieved:
                        st.markdown(f"**{doc.doc_id}** — score: `{doc.score:.3f}`")
                        with st.expander("View doc"):
                            st.code(doc.text, language="markdown")


if __name__ == "__main__":
    main()
