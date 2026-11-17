import pytest
from cuisine_assistant.retrieval import EmbeddingRecipeRetriever, RecipeDocument, RecipeDocumentRepository, RecipeSearchQuery, TextEmbedder
class Repo(RecipeDocumentRepository):
    """Provides deterministic documents."""
    def __init__(self): self.docs=(RecipeDocument("r1","d1","Paprikash","Hungarian paprika chicken","chicken; paprika","Cook","hungarian","manual"),RecipeDocument("r2","d2","Curry","Thai coconut chicken","chicken; coconut milk","Cook","thai","manual"),RecipeDocument("r3","d3","Caprese","Italian cheese salad","tomato; mozzarella","Serve","italian","manual"))
    def get_by_id(self,r): return next((d for d in self.docs if d.recipe_id==r),None)
    def list_all(self): return self.docs
class Embedder(TextEmbedder):
    """Returns controlled vectors."""
    def embed_documents(self,texts): return ((1.,0.,0.),(.8,.2,0.),(0.,0.,1.))
    def embed_query(self,text): return (0.,0.,1.) if "italian" in text.lower() else (1.,0.,0.)
def test_semantic_ranking():
    result=EmbeddingRecipeRetriever(Repo(),Embedder()).search(RecipeSearchQuery("creamy poultry"))
    assert result.best_hit.document.recipe_id=="r1"
def test_without_shared_words():
    result=EmbeddingRecipeRetriever(Repo(),Embedder()).search(RecipeSearchQuery("Italian fresh starter"))
    assert result.best_hit.document.recipe_id=="r3"
def test_filters_and_boosts():
    result=EmbeddingRecipeRetriever(Repo(),Embedder()).search(RecipeSearchQuery("curry",cuisine_id="thai",required_ingredient_ids=("chicken","coconut_milk")))
    assert tuple(h.document.recipe_id for h in result.hits)==("r2",)
    assert result.hits[0].evidence.matched_ingredient_ids==("chicken","coconut_milk")
def test_minimum_similarity_and_top_k():
    result=EmbeddingRecipeRetriever(Repo(),Embedder(),minimum_similarity=.9).search(RecipeSearchQuery("poultry",maximum_results=1))
    assert len(result.hits)==1 and result.hits[0].document.recipe_id=="r1"
def test_dimension_validation():
    class Bad(Embedder):
        """Returns incompatible query dimensions."""
        def embed_query(self,text): return (1.,0.)
    with pytest.raises(ValueError,match="dimension"): EmbeddingRecipeRetriever(Repo(),Bad()).search(RecipeSearchQuery("x"))
def test_invalid_configuration():
    with pytest.raises(ValueError,match="minimum_similarity"): EmbeddingRecipeRetriever(Repo(),Embedder(),minimum_similarity=1.1)
    with pytest.raises(ValueError,match="must not exceed"): EmbeddingRecipeRetriever(Repo(),Embedder(),cuisine_weight=.7,ingredient_weight=.5)
