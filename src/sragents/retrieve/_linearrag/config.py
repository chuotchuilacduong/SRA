from dataclasses import dataclass, field


@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: object = None  # SentenceTransformer instance (passed by adapter)
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_sm"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False
    # Adds passage_i ↔ passage_{i+1} edges (weight=1.0) by input order. Useful when
    # corpus order is semantically meaningful (Wikipedia paragraphs, clustered skills).
    # Default OFF for SR-Agents because 97.6% of corpus (web skills) has random order.
    enable_passage_adjacency: bool = False
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])
