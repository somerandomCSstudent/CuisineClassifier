from abc import ABC, abstractmethod
class RecipeDocumentRepository(ABC):
    """Provides recipe documents."""
    @abstractmethod
    def get_by_id(self, recipe_id: str): raise NotImplementedError
    @abstractmethod
    def list_all(self): raise NotImplementedError
class RecipeRetriever(ABC):
    """Retrieves ranked recipe documents."""
    @abstractmethod
    def search(self, query): raise NotImplementedError
