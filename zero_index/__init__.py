"""LLM-free index construction and source-backed tree navigation."""

from .index import DocumentIndex, build_index, reselect_representatives
from .embeddings import CentroidRepresentatives, EmbeddingScorer
from .jev import JevScorer
from .model import Config, LexicalJaccard, Similarity, posterior_same
from .retrieval import navigate, retrieve
from .ranking import LexicalReranker, find
from .structure import HeadingHint
from .separation import normalized_separation
from .configuration import IndexingConfig, RetrievalConfig, SearchConfig
from .hybrid_retrieval import EmbeddingPassageRetriever, fuse_retrieval_paths, tree_passages
from .tree_search import (EmbeddingTreeRouter, TreeSearchConfig,
                         pack_tree_context, propose_needs, search_tree)

__all__ = [
    "Config", "DocumentIndex", "EmbeddingScorer", "JevScorer", "LexicalJaccard", "Similarity",
    "HeadingHint", "LexicalReranker", "build_index", "find", "navigate", "posterior_same", "retrieve",
    "CentroidRepresentatives", "EmbeddingTreeRouter", "TreeSearchConfig",
    "EmbeddingPassageRetriever", "fuse_retrieval_paths", "tree_passages",
    "normalized_separation",
    "IndexingConfig", "RetrievalConfig", "SearchConfig",
    "pack_tree_context", "propose_needs", "reselect_representatives", "search_tree",
]
