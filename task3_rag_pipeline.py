import json
from rag_module import build_rag_pipeline, query_knowledge_base


def main():
    vector_store = build_rag_pipeline(force_rebuild=True)

    sample_queries = [
        "What ingredients are in the HydraGlow Vitamin C Serum?",
    ]

    results = []
    for i, query in enumerate(sample_queries, 1):
        result = query_knowledge_base(query)
        print(result)
        results.append({"query": query, "result": result})

    eval_results = {
        "pipeline_config": {
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_store": "ChromaDB",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "chunk_size": 800,
            "chunk_overlap": 200,
            "top_k_retrieval": 5,
            "top_k_rerank": 3,
        },
        "queries_tested": len(sample_queries),
        "results": [
            {"query": r["query"], "answer_length": len(r["result"])}
            for r in results
        ],
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()
