from typing import List, Dict, Optional
from termcolor import colored

import os
import faiss
import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from models import Tool
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError  # Ad


class MovieVectorDatabaseImpl:
    """Implementation for the movie vector database using FAISS and Sentence Transformers."""

    def __init__(self, persist_directory: str = "./db_index"):
        self.persist_directory = persist_directory
        self.index_path = os.path.join(
            self.persist_directory, "wiki_movies_index_final.faiss"
        )
        self.texts_path = os.path.join(
            self.persist_directory, "wiki_movies_texts_final.pkl"
        )
        self.metadata_path = os.path.join(
            self.persist_directory, "wiki_movies_metadata_final.pkl"
        )
        self.index = None
        self.texts = []
        self.metadata = []
        self.embedder = None
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize TF-IDF vectorizer for hybrid search
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", max_features=10000, ngram_range=(1, 2)
        )

    def initialize(self):
        """Initialize the database, loading or creating the index."""
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self.reranker_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                .to(self.device)
                .eval()
            )
            if (
                os.path.exists(self.index_path)
                and os.path.exists(self.texts_path)
                and os.path.exists(self.metadata_path)
            ):
                self.load()
            else:
                print(
                    colored(
                        "[MovieVectorDatabaseImpl] Index not found. Creating from HuggingFace dataset...",
                        "cyan",
                    )
                )
                self.create_index_from_dataset()
        except Exception as e:
            print(
                colored(f"[MovieVectorDatabaseImpl] Initialization error: {e}", "red")
            )
            print(
                colored(
                    "[MovieVectorDatabaseImpl] Initializing with empty index due to error.",
                    "red",
                )
            )
            self._create_empty_index()

    def create_index_from_dataset(self):
        """Create the FAISS index from the HuggingFace dataset."""
        try:
            from datasets import load_dataset

            # --- USE THE REQUESTED DATASET ---
            # dataset = load_dataset("vishnupriyavr/wiki-movie-plots-with-summaries", split="train[:5000]")
            dataset = load_dataset(
                "vishnupriyavr/wiki-movie-plots-with-summaries", split="train[:]"
            )
            texts = []
            metadata = []
            count = 0
            for item in dataset:
                # TODO: Remove this limit if you want to index the entire dataset
                # if count >= 5000: break
                title = item.get("Title", "Unknown Title")
                genre = item.get("Genre", "Unknown Genre")
                plot = item.get("Plot", "No plot summary available.")
                text = f"Title: {title}. Genre: {genre}. Plot: {plot}"
                texts.append(text)
                metadata.append(
                    {
                        "title": title,
                        "genre": genre,
                        "plot": plot,
                        "id": hash(
                            title + str(count)
                        ),  # Simple ID based on title and count
                    }
                )
                count += 1
            if not texts:
                print(
                    colored(
                        "[MovieVectorDatabaseImpl] No texts to index after processing.",
                        "yellow",
                    )
                )
                self._create_empty_index()
                return
            print(
                colored(
                    f"[MovieVectorDatabaseImpl] Loaded {len(texts)} movies from dataset.",
                    "cyan",
                )
            )
            # Fit TF-IDF vectorizer
            documents_for_tfidf = [f"{m['title']} {m['plot']}" for m in metadata]
            self.tfidf_vectorizer.fit(documents_for_tfidf)
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            self.texts = texts
            self.metadata = metadata
            self.save()
        except Exception as e:
            print(
                colored(
                    f"[MovieVectorDatabaseImpl] Error creating index from dataset: {e}",
                    "red",
                )
            )
            print(
                colored(
                    "[MovieVectorDatabaseImpl] Initializing with empty index due to error.",
                    "red",
                )
            )
            self._create_empty_index()

    def _create_empty_index(self):
        """Helper to create an empty FAISS index."""
        dimension = 384
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []
        self.metadata = []
        # Fit TF-IDF vectorizer on empty data
        self.tfidf_vectorizer.fit([""])
        os.makedirs(self.persist_directory, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.texts, f)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def save(self):
        """Save the index and metadata."""
        os.makedirs(self.persist_directory, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.texts, f)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(
            colored(
                f"[MovieVectorDatabaseImpl] Index saved to {self.persist_directory}",
                "green",
            )
        )

    def load(self):
        """Load the index and metadata."""
        self.index = faiss.read_index(self.index_path)
        with open(self.texts_path, "rb") as f:
            self.texts = pickle.load(f)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        # Fit TF-IDF vectorizer after loading
        documents_for_tfidf = [f"{m['title']} {m['plot']}" for m in self.metadata]
        if documents_for_tfidf:
            self.tfidf_vectorizer.fit(documents_for_tfidf)
        else:
            self.tfidf_vectorizer.fit([""])  # Fit on empty if no documents
        print(
            colored(
                f"[MovieVectorDatabaseImpl] Index loaded from {self.persist_directory}",
                "green",
            )
        )

    def semantic_search(self, query: str, k: int = 20) -> List[Dict]:
        """Perform semantic search."""
        if self.index is None or not self.texts:
            print(
                colored(
                    "[MovieVectorDatabaseImpl] Database not initialized for search.",
                    "red",
                )
            )
            return []
        try:
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            distances, indices = self.index.search(query_embedding, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result["similarity_score"] = float(distances[0][i])
                    results.append(result)
            return results
        except Exception as e:
            print(
                colored(f"[MovieVectorDatabaseImpl] Semantic search error: {e}", "red")
            )
            return []

    def get_movie_by_title(self, title: str) -> Optional[Dict]:
        """Get a movie by its exact title."""
        if not self.metadata:  # Fixed typo: was 'self.meta'
            return None
        for movie in self.metadata:
            if movie.get("title", "").lower() == title.lower():
                return movie
        return None

    def rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank search results using a cross-encoder."""
        if not candidates:
            return []
        try:
            pairs = [[query, cand.get("plot", "")] for cand in candidates]
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                logits = self.reranker_model(**inputs).logits.squeeze()
                scores = logits.tolist() if isinstance(logits, torch.Tensor) else logits
            reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [
                {**item, "rerank_score": round(score, 4)} for item, score in reranked
            ]
        except Exception as e:
            print(colored(f"Error during reranking: {e}", "red"))
            return sorted(
                candidates, key=lambda x: x.get("similarity_score", 0), reverse=True
            )

    def find_similar_to_movie(self, movie_title: str, k: int = 10) -> List[Dict]:
        """Find movies similar to a given movie title."""
        target_movie = self.get_movie_by_title(movie_title)
        if not target_movie:
            print(
                colored(
                    f"[MovieVectorDatabaseImpl] Movie '{movie_title}' not found for similarity search.",
                    "yellow",
                )
            )
            return []
        target_index = None
        for i, meta in enumerate(self.metadata):
            if meta.get("id") == target_movie.get("id"):
                target_index = i
                break
        if target_index is None:
            print(
                colored(
                    f"[MovieVectorDatabaseImpl] Index for movie '{movie_title}' not found.",
                    "yellow",
                )
            )
            return []
        try:
            target_plot = target_movie.get("plot", "")
            target_embedding = self.embedder.encode([target_plot])
            faiss.normalize_L2(target_embedding)
            distances, indices = self.index.search(target_embedding, k + 1)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == target_index:
                    continue
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result["similarity_score"] = float(distances[0][i])
                    results.append(result)
                if len(results) == k:
                    break
            return results
        except Exception as e:
            print(
                colored(
                    f"[MovieVectorDatabaseImpl] Similarity search error for '{movie_title}': {e}",
                    "red",
                )
            )
            return []

    def hybrid_search(self, query: str, k: int = 20, alpha: float = 0.5) -> List[Dict]:
        """
        Perform a hybrid search combining semantic similarity and keyword matching (TF-IDF).
        Args:
            query: The user's search query.
            k: The number of top results to return.
            alpha: Weight for semantic score (1-alpha for keyword score). Default 0.7 (favor semantic).
        Returns:
            A list of dictionaries representing movies, with added 'hybrid_score', 'semantic_score', 'keyword_score'.
        """
        print(
            colored(
                f"[VectorDB] Performing hybrid search for: '{query}' (k={k}, alpha={alpha})",
                "magenta",
            )
        )
        # --- Semantic Search ---
        semantic_results = self.semantic_search(
            query, k * 2
        )  # Fetch more for re-ranking
        if not semantic_results:
            print(colored("[VectorDB] No semantic results found.", "yellow"))
            return []
        # --- Keyword Search (TF-IDF) ---
        # Prepare documents (movie titles + plots) for TF-IDF
        documents = [
            f"{item.get('title', '')} {item.get('plot', '')}"
            for item in semantic_results
        ]
        if not documents:
            print(
                colored(
                    "[VectorDB] No documents to perform keyword search on.", "yellow"
                )
            )
            # Return semantic results if keyword search cannot be performed
            for item in semantic_results:
                item["hybrid_score"] = item.get("similarity_score", 0)
                item["semantic_score"] = item.get("similarity_score", 0)
                item["keyword_score"] = 0.0
            return sorted(
                semantic_results, key=lambda x: x.get("hybrid_score", 0), reverse=True
            )[:k]
        try:
            # --- FIX: Check if vectorizer is fitted ---
            try:
                # This call will raise NotFittedError if not fitted
                self.tfidf_vectorizer.transform(["test"])  # Dummy transform to check
                # If no error, proceed with actual transform
                query_vec = self.tfidf_vectorizer.transform([query])
                doc_vecs = self.tfidf_vectorizer.transform(documents)
                keyword_similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            except NotFittedError:
                print(
                    colored(
                        "[VectorDB] TF-IDF vectorizer is not fitted (e.g., empty database). Falling back to semantic.",
                        "red",
                    )
                )
                # Fallback to semantic results if TF-IDF fails
                for item in semantic_results:
                    item["hybrid_score"] = item.get("similarity_score", 0)
                    item["semantic_score"] = item.get("similarity_score", 0)
                    item["keyword_score"] = 0.0
                return sorted(
                    semantic_results,
                    key=lambda x: x.get("hybrid_score", 0),
                    reverse=True,
                )[:k]
            # --- End of FIX ---
        except (
            Exception
        ) as e:  # Catch other potential errors during transform/similarity
            print(
                colored(
                    f"[VectorDB] Error during TF-IDF keyword search: {e}. Falling back to semantic.",
                    "red",
                )
            )
            # Fallback to semantic results if TF-IDF fails
            for item in semantic_results:
                item["hybrid_score"] = item.get("similarity_score", 0)
                item["semantic_score"] = item.get("similarity_score", 0)
                item["keyword_score"] = 0.0
            return sorted(
                semantic_results, key=lambda x: x.get("hybrid_score", 0), reverse=True
            )[:k]
        # Map keyword scores back to items
        keyword_scores_dict = {
            item["id"]: score
            for item, score in zip(semantic_results, keyword_similarities)
        }
        # --- Normalize Scores ---
        max_semantic_score = max(
            (item.get("similarity_score", 0) for item in semantic_results), default=1
        )
        if max_semantic_score == 0:
            max_semantic_score = 1  # Avoid division by zero
        max_keyword_score = max(keyword_scores_dict.values(), default=1)
        if max_keyword_score == 0:
            max_keyword_score = 1  # Avoid division by zero
        # --- Combine Scores ---
        hybrid_scores = []
        for item in semantic_results:
            movie_id = item["id"]
            norm_semantic = item.get("similarity_score", 0) / max_semantic_score
            norm_keyword = keyword_scores_dict.get(movie_id, 0) / max_keyword_score
            # Weighted combination
            hybrid_score = (alpha * norm_semantic) + ((1 - alpha) * norm_keyword)
            item_copy = item.copy()  # Don't modify original metadata
            item_copy["hybrid_score"] = hybrid_score
            item_copy["semantic_score"] = norm_semantic
            item_copy["keyword_score"] = norm_keyword
            hybrid_scores.append(item_copy)
        # --- Rerank based on Hybrid Score ---
        # Sort by hybrid score descending
        reranked_by_hybrid = sorted(
            hybrid_scores, key=lambda x: x.get("hybrid_score", 0), reverse=True
        )
        # --- Return Top K ---
        print(
            colored(
                f"[VectorDB] Hybrid search completed. Returning top {min(k, len(reranked_by_hybrid))} results.",
                "magenta",
            )
        )
        return reranked_by_hybrid[:k]


class MovieVectorDatabaseTool(Tool):
    """Tool wrapper for MovieVectorDatabaseImpl."""

    def __init__(self):
        super().__init__("MovieVectorDatabaseTool")
        self.db: Optional[MovieVectorDatabaseImpl] = None

    def initialize(self) -> bool:
        try:
            self.db = MovieVectorDatabaseImpl()
            self.db.initialize()
            return self.is_available()
        except Exception as e:
            print(colored(f"[{self.name}] Initialization error: {e}", "red"))
            return False

    def is_available(self) -> bool:
        return self.db is not None

    def semantic_search(self, query: str, k: int = 20) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.db.semantic_search(query, k)

    def get_movie_by_title(self, title: str) -> Optional[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.db.get_movie_by_title(title)

    def rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.db.rerank_results(query, candidates)

    def find_similar_to_movie(self, movie_title: str, k: int = 10) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.db.find_similar_to_movie(movie_title, k)

    def hybrid_search(self, query: str, k: int = 20, alpha: float = 0.5) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.db.hybrid_search(query, k, alpha)
