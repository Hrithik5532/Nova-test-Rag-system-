import os
from typing import Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "nova_product_knowledge"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 5
RERANK_TOP_K = 3


def load_documents(data_dir: str = DATA_DIR) -> list[Document]:
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n==============================================================================\n",
                     "\n---", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def create_vector_store(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vector_store


def retrieve_documents(
    vector_store: Chroma,
    query: str,
    top_k: int = TOP_K,
) -> list[tuple[Document, float]]:
    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    return results


def rerank_documents(
    query: str,
    doc_score_pairs: list[tuple[Document, float]],
    top_k: int = RERANK_TOP_K,
) -> list[tuple[Document, float]]:
    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc.page_content) for doc, _ in doc_score_pairs]
        scores = reranker.predict(pairs)

        reranked = list(zip([doc for doc, _ in doc_score_pairs], scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    except ImportError:
        return doc_score_pairs[:top_k]


_vector_store_cache: Optional[Chroma] = None


def build_rag_pipeline(force_rebuild: bool = False) -> Chroma:
    global _vector_store_cache

    if _vector_store_cache is not None and not force_rebuild:
        return _vector_store_cache

    embeddings = get_embeddings()

    if os.path.exists(CHROMA_PERSIST_DIR) and not force_rebuild:
        _vector_store_cache = load_vector_store(embeddings)
        return _vector_store_cache

    documents = load_documents()
    chunks = split_documents(documents)
    _vector_store_cache = create_vector_store(chunks, embeddings)
    return _vector_store_cache


def query_knowledge_base(question: str) -> str:
    """
    Query the NOVA product knowledge base.
    Args:
        question: The customer's product-related question.
    Returns:
        A string containing the relevant product knowledge base excerpts.
    """
    vector_store = build_rag_pipeline()

    raw_results = retrieve_documents(vector_store, question, top_k=TOP_K)

    if not raw_results:
        return "No relevant product information found in our knowledge base."

    reranked = rerank_documents(question, raw_results, top_k=RERANK_TOP_K)

    output_parts = [f"Found {len(reranked)} relevant knowledge base entries:\n"]

    for i, (doc, score) in enumerate(reranked, 1):
        source = doc.metadata.get("source", "unknown")
        output_parts.append(f"--- Source #{i} (relevance: {score:.4f}) from {os.path.basename(source)} ---")
        output_parts.append(doc.page_content.strip())
        output_parts.append("")

    return "\n".join(output_parts)


if __name__ == "__main__":
    build_rag_pipeline(force_rebuild=True)

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        result = query_knowledge_base(question)
        print("\n" + result)
