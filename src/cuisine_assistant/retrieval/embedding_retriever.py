from __future__ import annotations
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from cuisine_assistant.retrieval.models import RecipeSearchEvidence, RecipeSearchHit, RecipeSearchResult
from cuisine_assistant.retrieval.ports import RecipeDocumentRepository, RecipeRetriever

class TextEmbedder(ABC):
    """Creates compatible vectors for queries and documents."""
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: raise NotImplementedError
    @abstractmethod
    def embed_query(self, text: str) -> tuple[float, ...]: raise NotImplementedError

class SentenceTransformerTextEmbedder(TextEmbedder):
    """Creates local normalized embeddings with Sentence Transformers."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", *, device: str | None = None) -> None:
        if not model_name.strip(): raise ValueError("model_name must not be empty")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("sentence-transformers is required for embedding retrieval") from error
        self._model = SentenceTransformer(model_name, device=device)
    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts: return ()
        encoder = getattr(self._model, "encode_document", self._model.encode)
        vectors = encoder(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return tuple(tuple(float(x) for x in vector) for vector in vectors)
    def embed_query(self, text: str) -> tuple[float, ...]:
        if not text.strip(): raise ValueError("query text must not be empty")
        encoder = getattr(self._model, "encode_query", self._model.encode)
        vectors = encoder([text], normalize_embeddings=True, show_progress_bar=False)
        return tuple(float(x) for x in vectors[0])

class EmbeddingRecipeRetriever(RecipeRetriever):
    """Ranks recipe documents by cosine similarity and structured boosts."""
    def __init__(self, repository: RecipeDocumentRepository, embedder: TextEmbedder, *, minimum_similarity: float = 0.0, cuisine_weight: float = 0.15, ingredient_weight: float = 0.25) -> None:
        for name, value in (("minimum_similarity", minimum_similarity),("cuisine_weight", cuisine_weight),("ingredient_weight", ingredient_weight)):
            if not 0 <= value <= 1: raise ValueError(f"{name} must be between 0.0 and 1.0")
        if cuisine_weight + ingredient_weight > 1: raise ValueError("cuisine_weight and ingredient_weight must not exceed 1.0")
        self._documents = repository.list_all(); self._embedder = embedder
        self._minimum_similarity = minimum_similarity; self._cuisine_weight = cuisine_weight; self._ingredient_weight = ingredient_weight
        self._vectors = embedder.embed_documents(tuple(d.searchable_text for d in self._documents))
        if len(self._vectors) != len(self._documents): raise ValueError("embedder must return one vector for every recipe document")
        self._dimension = self._validate_vectors(self._vectors)
    def search(self, query):
        if not self._documents: return RecipeSearchResult(query=query)
        q = self._embedder.embed_query(query.text)
        if len(q) != self._dimension: raise ValueError("query embedding dimension must match document embeddings")
        scored=[]
        for doc, vector in zip(self._documents, self._vectors):
            if query.cuisine_id and doc.cuisine_id != query.cuisine_id: continue
            semantic=max(0.0,min(1.0,(self._cosine(q,vector)+1)/2))
            if semantic < self._minimum_similarity: continue
            matched=self._matched_ingredients(query.required_ingredient_ids, doc.ingredient_text)
            ingredient=len(matched)/len(query.required_ingredient_ids) if query.required_ingredient_ids else 0.0
            cuisine=1.0 if query.cuisine_id and doc.cuisine_id == query.cuisine_id else 0.0
            cw=self._cuisine_weight if query.cuisine_id else 0.0; iw=self._ingredient_weight if query.required_ingredient_ids else 0.0
            score=min(1.0,(1-cw-iw)*semantic+cw*cuisine+iw*ingredient)
            evidence=RecipeSearchEvidence((),matched,0.0,semantic,cuisine)
            scored.append((doc,score,evidence))
        scored.sort(key=lambda x:(-x[1],x[0].recipe_id))
        hits=tuple(RecipeSearchHit(d,s,i,e) for i,(d,s,e) in enumerate(scored[:query.maximum_results],1))
        return RecipeSearchResult(query=query,hits=hits)
    @staticmethod
    def _validate_vectors(vectors):
        if not vectors: return 0
        dimension=len(vectors[0])
        if dimension < 1: raise ValueError("embedding vectors must not be empty")
        if any(len(v)!=dimension or not all(math.isfinite(x) for x in v) for v in vectors): raise ValueError("embedding vectors must have equal finite dimensions")
        return dimension
    @staticmethod
    def _cosine(a,b):
        na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
        return 0.0 if not na or not nb else sum(x*y for x,y in zip(a,b))/(na*nb)
    @staticmethod
    def _matched_ingredients(required, ingredient_text):
        tokens=set(ingredient_text.lower().replace(";"," ").split())
        return tuple(i for i in required if set(i.lower().replace("_"," ").split()).issubset(tokens))
