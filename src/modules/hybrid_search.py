"""
Module for hybrid search combining FAISS vector search with BM25 keyword search.
"""
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import os
import numpy as np
from langchain.docstore.document import Document

class HybridSearcher:
    """
    Implements hybrid search combining FAISS vector search and BM25 keyword search.
    """
    
    def __init__(self, documents=None):
        """
        Initialize the hybrid searcher with optional documents.
        
        Args:
            documents: List of document dictionaries with "content" and "metadata"
        """
        self.documents = []
        self.doc_contents = []
        self.doc_ids = {}
        self.bm25 = None
        
        if documents:
            self.index_documents(documents)
    
    def index_documents(self, documents):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of document dictionaries with "content" and "metadata"
        """
        # Reset existing data
        self.documents = documents
        self.doc_contents = []
        self.doc_ids = {}
        
        # Create document mapping
        for i, doc in enumerate(documents):
            content = doc.page_content if isinstance(doc, Document) else doc["content"]
            self.doc_contents.append(content.lower())
            self.doc_ids[i] = doc
        
        # Create BM25 index
        tokenized_corpus = [text.split() for text in self.doc_contents]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of documents sorted by relevance
        """
        if not self.bm25:
            return []
            
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by score
        sorted_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Return documents
        results = []
        for i in sorted_indices:
            if bm25_scores[i] > 0:  # Only include relevant documents
                doc = self.doc_ids[i]
                results.append((doc, bm25_scores[i]))
                
        return results

def combine_search_results(vector_docs: List[Document], 
                          keyword_docs: List[Tuple[Document, float]], 
                          vector_weight: float = 0.7, 
                          keyword_weight: float = 0.3) -> List[Document]:
    """
    Combine results from vector search and keyword search using reciprocal rank fusion.
    
    Args:
        vector_docs: List of documents from vector search
        keyword_docs: List of (document, score) tuples from keyword search
        vector_weight: Weight to apply to vector search results (0.0-1.0)
        keyword_weight: Weight to apply to keyword search results (0.0-1.0)
        
    Returns:
        Combined and reranked list of documents
    """
    # Normalize weights
    total_weight = vector_weight + keyword_weight
    vector_weight = vector_weight / total_weight
    keyword_weight = keyword_weight / total_weight
    
    # Create a mapping of documents to their scores
    doc_scores = {}
    
    # Add vector search results with position-based scoring
    for i, doc in enumerate(vector_docs):
        # Score based on position (higher rank = higher score)
        vector_score = 1.0 / (i + 1)
        doc_id = id(doc)  # Use object ID as key
        doc_scores[doc_id] = {
            "doc": doc,
            "score": vector_score * vector_weight,
            "in_vector": True,
            "in_keyword": False
        }
    
    # Add keyword search results
    for doc, keyword_score in keyword_docs:
        doc_id = id(doc)  # Use object ID as key
        if doc_id in doc_scores:
            # Document already in results, update score
            doc_scores[doc_id]["score"] += keyword_score * keyword_weight
            doc_scores[doc_id]["in_keyword"] = True
        else:
            # New document
            doc_scores[doc_id] = {
                "doc": doc,
                "score": keyword_score * keyword_weight,
                "in_vector": False,
                "in_keyword": True
            }
    
    # Sort by combined score
    sorted_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Return documents
    return [item["doc"] for item in sorted_results]
