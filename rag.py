# rag.py
"""
Step 3 — OpenSearch RAG pipeline.

Handles:
  - Text chunking
  - Embedding with HuggingFace
  - Bulk indexing into OpenSearch (KNN vectors)
  - Hybrid retrieval: KNN semantic + BM25 lexical
  - MMR re-ranking for diversity
  - CrossEncoder reranker for final scoring

Functions:
    build_index(text, doc_id, doc_name)              → index_name
    retrieve(query, doc_id, doc_name, index_name)    → List[chunk_dicts]
    load_text_from_s3(s3_key)                        → str
"""
import time
import numpy as np
import boto3
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError
from opensearchpy.helpers import bulk
from requests_aws4auth import AWS4Auth

import settings


# ── AWS / embedding clients (module-level, lazy where heavy) ──────────────────
_session  = boto3.Session()
_s3       = _session.client("s3")
_emb: Optional[HuggingFaceEmbeddings] = None
_reranker = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    global _emb
    if _emb is None:
        print(f"[RAG] Loading embedding model: {settings.EMBEDDING_MODEL}")
        _emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return _emb


def _get_reranker():
    global _reranker
    if _reranker is None and settings.USE_RERANKER:
        from sentence_transformers import CrossEncoder
        print(f"[RAG] Loading reranker: {settings.RERANKER_MODEL}")
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


# ── OpenSearch client ─────────────────────────────────────────────────────────

def _build_os_client() -> Optional[OpenSearch]:
    host = settings.OPENSEARCH_HOST
    if not host:
        print("[RAG] OPENSEARCH_HOST not set — OpenSearch disabled.")
        return None
    parsed   = urlparse(host if "://" in host else f"https://{host}")
    hostname = parsed.hostname
    if not hostname:
        print(f"[RAG] Invalid OPENSEARCH_HOST: {host}")
        return None
    creds   = _session.get_credentials().get_frozen_credentials()
    awsauth = AWS4Auth(
        creds.access_key, creds.secret_key,
        settings.AWS_REGION, "es",
        session_token=creds.token,
    )
    client = OpenSearch(
        hosts=[{"host": hostname, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=settings.OPENSEARCH_TIMEOUT_SEC,
        max_retries=settings.OPENSEARCH_MAX_RETRIES,
        retry_on_timeout=settings.OPENSEARCH_RETRY_ON_TIMEOUT,
    )
    print(f"[RAG] OpenSearch client ready: {hostname}")
    return client


try:
    _os = _build_os_client()
except Exception as e:
    print(f"[RAG] OpenSearch init failed: {e}")
    _os = None


# ── Index management ──────────────────────────────────────────────────────────

def _ensure_index(index_name: str, dim: int) -> None:
    if _os is None:
        return
    try:
        if _os.indices.exists(index=index_name):
            return
    except NotFoundError:
        pass

    body = {
        "settings": {
            "index.knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
        "mappings": {
            "properties": {
                "content":   {"type": "text"},
                "doc_id":    {"type": "keyword"},
                "doc_name":  {"type": "keyword"},
                "chunk_id":  {"type": "integer"},
                "position":  {"type": "integer"},
                "page":      {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": int(dim),
                    "method": {"name": "hnsw", "engine": "nmslib", "space_type": "cosinesimil"},
                },
            }
        },
    }
    _os.indices.create(index=index_name, body=body, ignore=400)
    print(f"[RAG] Index created: {index_name} (dim={dim})")


# ── Text chunking ─────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    docs = splitter.create_documents([text])
    for i, d in enumerate(docs):
        d.metadata = d.metadata or {}
        d.metadata["chunk_id"] = i
        d.metadata["position"] = i
    return docs


# ── Bulk indexing ─────────────────────────────────────────────────────────────

def _bulk_index(index_name: str, docs: List[Any], doc_id: str, doc_name: str) -> None:
    if _os is None:
        print("[RAG] OpenSearch unavailable — skipping index.")
        return

    emb_model = _get_embedding_model()
    texts     = [d.page_content for d in docs if d.page_content.strip()]
    vectors   = emb_model.embed_documents(texts)
    dim       = len(vectors[0])
    _ensure_index(index_name, dim)

    actions = []
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        m = (docs[i].metadata or {})
        actions.append({
            "_op_type": "index",
            "_index":   index_name,
            "_id":      f"{doc_id}-{i}",
            "_source": {
                "content":   text,
                "doc_id":    doc_id,
                "doc_name":  doc_name,
                "chunk_id":  int(m.get("chunk_id", i)),
                "position":  int(m.get("position", i)),
                "page":      int(m.get("page", -1)),
                "embedding": [float(x) for x in vec],
            },
        })

    success, failed = bulk(_os, actions, raise_on_error=False, raise_on_exception=False, stats_only=True)
    try:
        _os.indices.refresh(index=index_name)
    except Exception as e:
        print(f"[RAG] Refresh warning: {e}")

    print(f"[RAG] Indexed {success} chunks into '{index_name}' (failed={failed})")
    if failed:
        raise RuntimeError(f"OpenSearch bulk index had {failed} failure(s)")


# ── Similarity helpers ────────────────────────────────────────────────────────

def _cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def _mmr_select(q_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_mult: float) -> List[int]:
    if cand_vecs.shape[0] == 0:
        return []
    rel = _cosine_sim(cand_vecs, q_vec.reshape(1, -1)).squeeze(-1)
    selected, remaining = [], list(range(cand_vecs.shape[0]))
    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)
    while remaining and len(selected) < k:
        sel_vecs = cand_vecs[selected]
        div = _cosine_sim(cand_vecs[remaining], sel_vecs).max(axis=1)
        scores = lambda_mult * rel[remaining] - (1 - lambda_mult) * div
        pick = remaining[int(np.argmax(scores))]
        selected.append(pick)
        remaining.remove(pick)
    return selected


# ── OpenSearch retrieval ──────────────────────────────────────────────────────

def _os_knn(q_vec: List[float], k: int, doc_id: str, index_name: str) -> List[Dict]:
    filters = [{"term": {"doc_id": doc_id}}]
    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": filters,
                "must": [{"knn": {"embedding": {"vector": q_vec, "k": k}}}],
            }
        },
        "_source": ["content", "doc_id", "doc_name", "page", "chunk_id", "position"],
    }
    return _os.search(index=index_name, body=body)["hits"]["hits"]


def _os_bm25(query: str, k: int, doc_id: str, index_name: str) -> List[Dict]:
    filters = [{"term": {"doc_id": doc_id}}]
    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": filters,
                "must": [{"match": {"content": {"query": query, "operator": "or"}}}],
            }
        },
        "_source": ["content", "doc_id", "doc_name", "page", "chunk_id", "position"],
    }
    return _os.search(index=index_name, body=body)["hits"]["hits"]


# ── Reranker ──────────────────────────────────────────────────────────────────

def _rerank(query: str, texts: List[str], top_n: int) -> Tuple[List[str], List[Optional[float]]]:
    if not texts:
        return [], []
    top_n = min(top_n, len(texts))
    reranker = _get_reranker()
    if reranker is None:
        return texts[:top_n], [None] * top_n
    scores = reranker.predict([(query, t) for t in texts])
    scored = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
    top = scored[:top_n]
    return [t for t, _ in top], [float(s) for _, s in top]


# ── Public: build index ───────────────────────────────────────────────────────

def build_index(text: str, doc_id: str, doc_name: str) -> str:
    """
    Chunk text, embed, and bulk-index into OpenSearch.
    Returns the index name used.
    """
    docs = _chunk_text(text)
    print(f"[RAG] Chunked into {len(docs)} chunk(s)")

    base_index = (settings.OPENSEARCH_INDEX or "rag_chunks").lower()
    if settings.OS_PER_DOC_INDEX:
        index_name = f"{base_index}-{doc_id[:16]}".lower()
    else:
        index_name = base_index

    _bulk_index(index_name, docs, doc_id=doc_id.lower(), doc_name=doc_name.lower().replace(" ", "_"))
    return index_name


# ── Public: retrieve ──────────────────────────────────────────────────────────

def retrieve(query: str, doc_id: str, index_name: str) -> List[Dict[str, Any]]:
    """
    Hybrid KNN + BM25 retrieval with MMR diversity re-ranking and CrossEncoder scoring.
    Returns up to TOP_N chunk dicts ready to be used as LLM context.
    """
    if _os is None:
        raise RuntimeError("[RAG] OpenSearch client is not available.")

    emb_model = _get_embedding_model()
    fetch_k   = max(2 * settings.TOP_K, settings.TOP_K + 4)
    q_vec     = emb_model.embed_query(query)

    # ── Dual retrieval ──
    def _search():
        sem_hits = _os_knn(q_vec, k=fetch_k, doc_id=doc_id.lower(), index_name=index_name)
        lex_hits = _os_bm25(query,  k=fetch_k, doc_id=doc_id.lower(), index_name=index_name)
        return sem_hits, lex_hits

    sem_hits, lex_hits = _search()

    # Retry once after a short wait if both are empty (index may not be refreshed yet)
    if not sem_hits and not lex_hits:
        try:
            _os.indices.refresh(index=index_name)
        except Exception:
            pass
        time.sleep(0.25)
        sem_hits, lex_hits = _search()

    print(f"[RAG] Retrieved knn={len(sem_hits)} bm25={len(lex_hits)} from '{index_name}'")

    # ── De-duplicate pool ──
    pool: List[Dict] = []
    seen: set = set()
    for hit in sem_hits + lex_hits:
        if hit["_id"] in seen:
            continue
        seen.add(hit["_id"])
        src = hit["_source"] or {}
        pool.append({
            "_id":      hit["_id"],
            "text":     src.get("content", ""),
            "chunk_id": src.get("chunk_id"),
            "position": src.get("position"),
            "page":     src.get("page"),
            "doc_id":   src.get("doc_id"),
            "doc_name": src.get("doc_name"),
        })

    if not pool:
        return []

    # ── Semantic scores + MMR selection ──
    q_vec_np  = np.array(q_vec, dtype=np.float32)
    pool_vecs = np.array(emb_model.embed_documents([p["text"] for p in pool]), dtype=np.float32)
    sem_scores = _cosine_sim(pool_vecs, q_vec_np.reshape(1, -1)).squeeze(-1)
    for p, s in zip(pool, sem_scores):
        p["semantic_score"] = float(s)

    sel_idx  = _mmr_select(q_vec_np, pool_vecs, k=settings.TOP_K, lambda_mult=settings.MMR_LAMBDA)
    selected = [pool[i] for i in sel_idx]

    # ── CrossEncoder reranking ──
    reranked_texts, rerank_scores = _rerank(query, [x["text"] for x in selected], top_n=settings.TOP_K)
    score_map = dict(zip(reranked_texts, rerank_scores))

    out: List[Dict] = []
    for text in reranked_texts[: settings.TOP_N]:
        item = next((x for x in selected if x["text"] == text), None)
        chunk = item.copy() if item else {"text": text, "chunk_id": None, "page": None, "position": None, "doc_id": doc_id, "doc_name": None, "semantic_score": None}
        chunk["reranker_score"] = score_map.get(text)
        out.append(chunk)

    print(f"[RAG] Returning {len(out)} chunk(s) after MMR + reranking")
    return out


# ── Utility ───────────────────────────────────────────────────────────────────

def load_text_from_s3(s3_key: str) -> str:
    """Load UTF-8 text from S3."""
    obj = _s3.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
    return obj["Body"].read().decode("utf-8")
