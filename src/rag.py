"""
rag.py
======
Retrieval-Augmented Generation (RAG) pipeline for the AI product assistant.

How RAG works here
------------------
1. **Corpus construction** (``build_document_corpus``):
   Each product row in the cleaned DataFrame is serialised into a single
   free-text string that combines its name, category, pricing info, star
   rating, product description, and a snippet of a customer review.

2. **Embedding + indexing** (``build_faiss_index``):
   Every document string is encoded into a 384-dimensional dense vector by
   ``sentence-transformers/all-MiniLM-L6-v2`` — a lightweight model that runs
   comfortably on CPU in ~5 seconds for 1,400 documents.  The vectors are
   stored in a FAISS ``IndexFlatL2`` (exact nearest-neighbour search via
   Euclidean distance).

3. **Retrieval** (``retrieve_relevant_docs``):
   At query time the question is encoded with the same embedding model and
   the ``top_k`` nearest documents are returned from FAISS in milliseconds.

4. **Generation** (``generate_answer``):
   The retrieved documents are concatenated into a context block and injected
   into an instruction prompt.  The prompt is fed to ``google/flan-t5-base``
   (or whichever model is configured via the ``LLM_MODEL`` environment
   variable) to produce a grounded, factual answer.

Why these choices
-----------------
- ``all-MiniLM-L6-v2`` — 22 M parameters, 384-dim embeddings, runs on CPU,
  strong semantic similarity performance for English text.
- ``google/flan-t5-base`` — open-source instruction-following seq2seq model,
  ~250 MB, no GPU required, good at extractive QA from a given context.
- FAISS ``IndexFlatL2`` — exact search (no approximation error), fast enough
  for a corpus of this size, no tuning required.

Exported public API
-------------------
build_document_corpus  -- Convert a DataFrame into a list of text documents.
build_faiss_index      -- Encode documents and build + save the FAISS index.
load_faiss_index       -- Load a previously built index from disk.
retrieve_relevant_docs -- Semantic search over the index for a text query.
generate_answer        -- Generate a grounded answer using the LLM.
"""

import os
import pickle
from typing import List

import numpy as np
import pandas as pd

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss_index.pkl")
DOCS_PATH = os.path.join(ARTIFACTS_DIR, "documents.pkl")

# Module-level cache so the LLM pipeline is only instantiated once per process,
# even across many /answer_question requests.
_llm_cache: dict = {}


# ── Corpus construction ───────────────────────────────────────────────────────

def build_document_corpus(df: pd.DataFrame) -> List[str]:
    """Convert the product DataFrame into a list of rich text documents.

    Each document is a single string that captures the most informative
    fields of one product row.  Truncation limits (120/200 chars) prevent
    excessively long category hierarchies and verbose descriptions from
    dominating the embedding vector.

    Document template::

        Product: <name>. Category: <category[:120]>.
        Price: ₹<discounted> (was ₹<actual>, <pct>% off).
        Rating: <rating>/5 from <count> reviews.
        About: <about_product[:200]>. Review: <review_content[:200]>.

    Args:
        df: A cleaned DataFrame as returned by
            ``data_preprocessing.load_and_clean_data``.  Must contain the
            columns: ``product_name``, ``category``, ``discounted_price``,
            ``actual_price``, ``discount_percentage``, ``rating``,
            ``rating_count``, ``about_product``, ``review_content``.

    Returns:
        A list of strings, one per row, in the same order as the DataFrame.
    """
    docs = []
    for _, row in df.iterrows():
        doc = (
            f"Product: {row['product_name']}. "
            f"Category: {str(row['category'])[:120]}. "
            f"Price: ₹{row['discounted_price']:.0f} (was ₹{row['actual_price']:.0f}, "
            f"{row['discount_percentage']:.0f}% off). "
            f"Rating: {row['rating']}/5 from {row['rating_count']:.0f} reviews. "
            f"About: {str(row['about_product'])[:200]}. "
            f"Review: {str(row['review_content'])[:200]}."
        )
        docs.append(doc)
    return docs


# ── Index build / load ────────────────────────────────────────────────────────

def build_faiss_index(
    documents: List[str],
    model_name: str = "all-MiniLM-L6-v2",
):
    """Encode ``documents`` with a sentence-transformer and store in FAISS.

    This is an offline step executed once by ``train.py``.  The resulting
    index and the embedding model are both pickled to ``artifacts/`` so that
    subsequent API server restarts can load them in seconds rather than
    re-encoding the whole corpus.

    Index type — ``IndexFlatL2``:
        Exact brute-force nearest-neighbour search using squared Euclidean
        (L2) distance.  For 1,400 documents this is instantaneous; an
        approximate index (e.g. ``IndexIVFFlat``) would only be needed for
        hundreds of thousands of documents.

    Args:
        documents: List of text strings produced by ``build_document_corpus``.
        model_name: Any ``sentence-transformers``-compatible model identifier.
            Defaults to ``"all-MiniLM-L6-v2"`` which offers the best
            speed/quality trade-off for CPU inference.

    Returns:
        A tuple ``(index, embed_model, documents)`` — all three are also
        written to disk:
        - ``artifacts/faiss_index.pkl`` — the FAISS index and the embedding
          model bundled together.
        - ``artifacts/documents.pkl``   — the raw text strings (needed at
          retrieval time to return readable source citations).
    """
    # Lazy imports keep startup time fast for modules that import rag.py
    # without intending to build an index.
    from sentence_transformers import SentenceTransformer
    import faiss

    embed_model = SentenceTransformer(model_name)
    print(f"Encoding {len(documents)} documents with {model_name}...")
    embeddings = embed_model.encode(documents, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    # Bundle the index and embedding model together so load_faiss_index only
    # needs to open one file.
    with open(INDEX_PATH, "wb") as f:
        pickle.dump((index, embed_model), f)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"FAISS index built: {len(documents)} documents, dim={dim}")
    return index, embed_model, documents


def load_faiss_index():
    """Load the FAISS index, embedding model, and document corpus from disk.

    Called once at API server startup (inside the ``lifespan`` context manager)
    and cached in ``app.state`` so the heavy objects are not re-loaded per
    request.

    Returns:
        A tuple ``(index, embed_model, documents)`` ready to be passed to
        ``retrieve_relevant_docs``.

    Raises:
        FileNotFoundError: If either ``artifacts/faiss_index.pkl`` or
            ``artifacts/documents.pkl`` is missing.  The caller should prompt
            the user to run ``python train.py``.
    """
    with open(INDEX_PATH, "rb") as f:
        index, embed_model = pickle.load(f)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    return index, embed_model, documents


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_relevant_docs(
    query: str,
    index,
    embed_model,
    documents: List[str],
    top_k: int = 3,
) -> List[str]:
    """Return the ``top_k`` most semantically relevant documents for a query.

    Process:
    1. Encode the query into the same 384-dim embedding space as the corpus.
    2. Run an exact nearest-neighbour search (L2 distance) over all indexed
       document vectors.
    3. Return the corresponding raw text strings.

    The L2 distance in the embedding space correlates with semantic similarity:
    documents about charging cables will cluster close together, and a query
    about "fast charging" will land near those documents even if it shares no
    keywords with them.

    Args:
        query: The user's free-text question.
        index: A FAISS index populated with document embeddings.
        embed_model: The ``SentenceTransformer`` instance used to build the
            index — must be the same model to ensure compatible vector spaces.
        documents: The original text strings in the same order they were
            added to the index.
        top_k: Number of documents to retrieve.  Higher values give the LLM
            more context but also more noise.  Defaults to 3.

    Returns:
        A list of up to ``top_k`` document strings, ordered by relevance
        (most relevant first).
    """
    # Encode the query; shape (1, 384). Cast to float32 to match the index.
    query_embedding = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_embedding, top_k)

    # Guard against index returning -1 for unfilled slots (not expected with
    # IndexFlatL2 but defensive).
    return [documents[i] for i in indices[0] if i < len(documents)]


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer(query: str, context_docs: List[str]) -> str:
    """Generate a grounded answer from retrieved product documents.

    The function builds an instruction prompt that places the retrieved
    documents before the question, then passes it to the configured LLM.
    Constraining the model to answer *only* from the provided context is a
    standard RAG technique for reducing hallucination.

    Prompt structure::

        You are a helpful e-commerce assistant.
        Answer the question using only the product information provided below.

        Product Information:
        <doc1>

        <doc2>

        <doc3>

        Question: <query>

        Answer:

    Args:
        query: The user's original question (unchanged from the request).
        context_docs: List of retrieved document strings from
            ``retrieve_relevant_docs``.

    Returns:
        A string containing the model's generated answer.  May be a short
        phrase or a few sentences depending on the question complexity and
        the model's confidence in the retrieved context.
    """
    llm = _load_llm()
    context = "\n\n".join(context_docs)

    prompt = (
        f"You are a helpful e-commerce assistant. "
        f"Answer the question using only the product information provided below.\n\n"
        f"Product Information:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    output = llm(prompt, max_new_tokens=200, do_sample=False)

    # Normalise the output regardless of whether the pipeline returns a list
    # of dicts (standard HuggingFace format) or a plain string.
    if isinstance(output, list):
        output = output[0]
    if isinstance(output, dict):
        text = output.get("generated_text", "")
    else:
        text = str(output)

    # flan-t5 returns only the generated continuation, but some other models
    # echo the full prompt.  Strip anything up to and including "Answer:" so
    # the API response contains only the model's answer.
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    return text


# ── LLM loader ────────────────────────────────────────────────────────────────

def _load_llm():
    """Lazily load and cache the HuggingFace text-generation pipeline.

    The model is downloaded on first call (cached by HuggingFace to
    ``~/.cache/huggingface``) and kept in the module-level ``_llm_cache``
    dict for the lifetime of the process.  Subsequent calls return the cached
    pipeline instantly.

    The model is configurable via the ``LLM_MODEL`` environment variable,
    which makes it easy to swap in a larger model (e.g. ``google/flan-t5-large``)
    for better answer quality or a smaller one for faster responses.

    Returns:
        A ``transformers.Pipeline`` object configured for
        ``"text2text-generation"`` (seq2seq models like flan-t5) or
        ``"text-generation"`` for causal LMs — the caller should choose a
        model that matches ``"text2text-generation"``.
    """
    from transformers import pipeline

    model_name = os.environ.get("LLM_MODEL", "google/flan-t5-base")
    if model_name not in _llm_cache:
        print(f"Loading LLM: {model_name} (first call — may take a moment)")
        _llm_cache[model_name] = pipeline("text2text-generation", model=model_name)
    return _llm_cache[model_name]
