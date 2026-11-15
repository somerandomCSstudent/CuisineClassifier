from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote

from rdflib import Graph, RDF, URIRef

from cuisine_assistant.domain.models import OntologyEntityReference
from cuisine_assistant.domain.ports import KnowledgeGraphRepository


class RdfLibKnowledgeGraphRepository(KnowledgeGraphRepository):
    """Reads cuisine knowledge from an RDF/XML OWL ontology using RDFLib."""

    def __init__(
        self,
        ontology_path: str | Path,
        base_iri: str,
        *,
        rdf_format: str = "xml",
        has_ingredient_property: str = "hasIngredient",
        belongs_to_cuisine_property: str = "belongsToCuisine",
    ) -> None:
        self._ontology_path = Path(ontology_path)
        self._base_iri = self._normalize_base_iri(base_iri)
        self._has_ingredient = self._entity_iri(has_ingredient_property)
        self._belongs_to_cuisine = self._entity_iri(
            belongs_to_cuisine_property
        )
        self._graph = self._load_graph(rdf_format)

    def get_entity_reference(
        self,
        ontology_id: str,
    ) -> OntologyEntityReference | None:
        entity_iri = self._entity_iri(ontology_id)
        if not self._entity_exists(entity_iri):
            return None
        return OntologyEntityReference(
            ontology_id=ontology_id,
            iri=str(entity_iri),
            rdf_types=self.get_entity_types(ontology_id),
            exists_in_knowledge_graph=True,
        )

    def get_entity_types(self, ontology_id: str) -> tuple[str, ...]:
        entity_iri = self._entity_iri(ontology_id)
        type_ids = {
            self._local_id(type_iri)
            for type_iri in self._graph.objects(entity_iri, RDF.type)
            if isinstance(type_iri, URIRef)
        }
        return tuple(sorted(type_ids))

    def find_dish_ids_by_ingredient_ids(
        self,
        ingredient_ontology_ids: Iterable[str],
    ) -> tuple[str, ...]:
        ingredient_iris = {
            self._entity_iri(ontology_id)
            for ontology_id in ingredient_ontology_ids
            if ontology_id and ontology_id.strip()
        }
        if not ingredient_iris:
            return ()

        dish_ids = {
            self._local_id(dish_iri)
            for ingredient_iri in ingredient_iris
            for dish_iri in self._graph.subjects(
                self._has_ingredient,
                ingredient_iri,
            )
            if isinstance(dish_iri, URIRef)
        }
        return tuple(sorted(dish_ids))

    def get_dish_ingredient_ids(
        self,
        dish_ontology_id: str,
    ) -> tuple[str, ...]:
        dish_iri = self._entity_iri(dish_ontology_id)
        ingredient_ids = {
            self._local_id(ingredient_iri)
            for ingredient_iri in self._graph.objects(
                dish_iri,
                self._has_ingredient,
            )
            if isinstance(ingredient_iri, URIRef)
        }
        return tuple(sorted(ingredient_ids))

    def get_dish_cuisine_id(self, dish_ontology_id: str) -> str | None:
        dish_iri = self._entity_iri(dish_ontology_id)
        cuisine_iris = sorted(
            (
                cuisine_iri
                for cuisine_iri in self._graph.objects(
                    dish_iri,
                    self._belongs_to_cuisine,
                )
                if isinstance(cuisine_iri, URIRef)
            ),
            key=str,
        )
        if not cuisine_iris:
            return None
        if len(cuisine_iris) > 1:
            raise ValueError(
                f"Dish has multiple cuisine relations: {dish_ontology_id}"
            )
        return self._local_id(cuisine_iris[0])

    def _load_graph(self, rdf_format: str) -> Graph:
        if not self._ontology_path.is_file():
            raise FileNotFoundError(
                f"Ontology file does not exist: {self._ontology_path}"
            )
        graph = Graph()
        try:
            graph.parse(self._ontology_path, format=rdf_format)
        except Exception as error:
            raise ValueError(
                f"Failed to parse ontology file: {self._ontology_path}"
            ) from error
        return graph

    def _entity_exists(self, entity_iri: URIRef) -> bool:
        return any(self._graph.triples((entity_iri, None, None))) or any(
            self._graph.triples((None, None, entity_iri))
        )

    def _entity_iri(self, ontology_id: str) -> URIRef:
        if not ontology_id or not ontology_id.strip():
            raise ValueError("ontology_id must not be empty")
        if ontology_id.startswith(("http://", "https://", "urn:")):
            return URIRef(ontology_id)
        return URIRef(f"{self._base_iri}{ontology_id.strip()}")

    @staticmethod
    def _normalize_base_iri(base_iri: str) -> str:
        if not base_iri or not base_iri.strip():
            raise ValueError("base_iri must not be empty")
        value = base_iri.strip()
        if not value.endswith(("#", "/")):
            value += "#"
        return value

    @staticmethod
    def _local_id(iri: URIRef) -> str:
        value = str(iri)
        if "#" in value:
            return unquote(value.rsplit("#", 1)[1])
        return unquote(value.rstrip("/").rsplit("/", 1)[-1])
