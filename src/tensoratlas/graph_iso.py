
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ContractionGraph:
    nodes: Tuple[str, ...]
    edges: Tuple[Tuple[int,int], ...]
    free_indices: Tuple[str, ...]

def canonical_label(graph: ContractionGraph):
    return (
        tuple(sorted(graph.nodes)),
        tuple(sorted(tuple(sorted(e)) for e in graph.edges)),
        tuple(sorted(graph.free_indices)),
    )

def are_isomorphic(g1: ContractionGraph, g2: ContractionGraph):
    return canonical_label(g1) == canonical_label(g2)
