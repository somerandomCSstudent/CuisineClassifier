from __future__ import annotations

import csv
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

from cuisine_assistant.rag.models import RagGraphFact
from cuisine_assistant.rag.ports import RagKnowledgeGraphFactProvider


class RdfLibRagFactProvider(RagKnowledgeGraphFactProvider):
    """Extracts dish, cuisine, ingredient, type, and label facts from RDFLib."""

    def __init__(
        self,
        ontology_path: str | Path,
        dishes_csv: str | Path,
        ontology_base_iri: str,
    ) -> None:
        self._base_iri = ontology_base_iri
        self._graph = Graph()
        self._graph.parse(Path(ontology_path), format="xml")
        self._dish_ontology_ids = self._load_dish_ontology_ids(
            Path(dishes_csv)
        )

    def facts_for_dishes(
        self,
        dish_ids: tuple[str, ...],
        maximum_facts: int,
    ) -> tuple[RagGraphFact, ...]:
        """Returns deterministic outgoing facts for selected dish entities."""
        if maximum_facts < 1:
            raise ValueError("maximum_facts must be at least 1")
        facts: list[RagGraphFact] = []
        seen: set[tuple[str, str, str]] = set()
        for dish_id in dish_ids:
            ontology_id = self._dish_ontology_ids.get(dish_id)
            if not ontology_id:
                continue
            subject = URIRef(self._base_iri + ontology_id)
            predicates = (
                URIRef(self._base_iri + "belongsToCuisine"),
                URIRef(self._base_iri + "hasIngredient"),
                RDF.type,
                RDFS.label,
            )
            for predicate in predicates:
                objects = sorted(
                    self._graph.objects(subject, predicate),
                    key=str,
                )
                for obj in objects:
                    predicate_name = self._local_name(str(predicate))
                    object_value = self._display_value(obj)
                    signature = (
                        ontology_id,
                        predicate_name,
                        object_value,
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    facts.append(
                        RagGraphFact(
                            subject_id=ontology_id,
                            predicate=predicate_name,
                            object_value=object_value,
                            source_iri=str(subject),
                        )
                    )
                    if len(facts) >= maximum_facts:
                        return tuple(facts)
        return tuple(facts)

    @staticmethod
    def _load_dish_ontology_ids(path: Path) -> dict[str, str]:
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = csv.DictReader(source)
            return {
                row["dish_id"].strip(): row["ontology_id"].strip()
                for row in rows
                if row.get("dish_id") and row.get("ontology_id")
            }

    def _display_value(self, value) -> str:
        text = str(value)
        if isinstance(value, URIRef):
            label = next(self._graph.objects(value, RDFS.label), None)
            return str(label) if label is not None else self._local_name(text)
        return text

    @staticmethod
    def _local_name(iri: str) -> str:
        if "#" in iri:
            return iri.rsplit("#", 1)[1]
        return iri.rstrip("/").rsplit("/", 1)[-1]
