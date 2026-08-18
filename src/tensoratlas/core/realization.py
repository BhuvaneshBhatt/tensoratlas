"""Bridge abstract semantic tensor expressions to component tensors."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Mapping

from .components import Basis, ComponentTensor, Scalar, generalized_delta_component_tensor, levi_civita_symbol_component_tensor
from .indices import AbstractIndex
from .manifolds import TensorKernelError
from .tensor_expr import TensorExpr, TensorFactor, TensorTerm
from .tensor_heads import TensorHead


@dataclass(slots=True)
class ComponentRealizationRegistry:
    """Registry mapping abstract tensor heads to component realizations."""

    basis: Basis
    tensors: dict[tuple[TensorHead, tuple[str | None, ...]], ComponentTensor] = field(default_factory=dict)

    def register(self, tensor: ComponentTensor) -> None:
        if tensor.basis != self.basis:
            raise TensorKernelError("Registered component tensor must use the registry basis.")
        self.tensors[(tensor.head, tuple(tensor.variance))] = tensor

    def lookup(self, head: TensorHead, variance: tuple[str | None, ...]) -> ComponentTensor:
        key = (head, tuple(variance))
        if key in self.tensors:
            return self.tensors[key]
        if head.role == "epsilon":
            tensor = levi_civita_symbol_component_tensor(head.name, self.basis, variance=tuple(v or "down" for v in variance))
            self.register(tensor)
            return tensor
        if head.role == "generalized_delta":
            tensor = generalized_delta_component_tensor(head.name, self.basis, head.rank // 2)
            self.register(tensor)
            return tensor
        raise TensorKernelError(f"No component realization registered for tensor head {head.name!r} with variance {variance!r}.")


def realize_tensor_expression(expr: TensorExpr, registry: ComponentRealizationRegistry, *, head_name: str = "Realized") -> ComponentTensor | Scalar:
    """Realize an abstract tensor expression as components in ``registry.basis``.

    Free abstract indices become output component slots, ordered by the semantic
    expression free-index signature.  Dummy indices are summed over the basis
    dimension.  Scalar expressions return a scalar-like value.
    """

    expr = expr.canonicalized()
    free_signature = expr.free_index_signature
    free_indices = tuple(AbstractIndex(name, itype, variance) for name, variance, itype in free_signature)
    dim = registry.basis.dimension
    if not free_indices:
        total = 0
        for term in expr.terms:
            total += _realize_term_scalar(term, registry, {})
        return total
    variance = tuple(idx.variance for idx in free_indices)
    index_type = registry.basis.index_type
    head = TensorHead(head_name, (index_type,) * len(free_indices), variance=variance)
    comps: dict[tuple[int, ...], Scalar] = {}
    for out_key in product(range(dim), repeat=len(free_indices)):
        assignment = {(idx.name, idx.index_type): value for idx, value in zip(free_indices, out_key)}
        value = 0
        for term in expr.terms:
            value += _realize_term_scalar(term, registry, assignment)
        if value != 0:
            comps[out_key] = value
    return ComponentTensor(head, registry.basis, comps, variance=variance)


def _realize_term_scalar(term: TensorTerm, registry: ComponentRealizationRegistry, free_assignment: Mapping[tuple[str, object], int]) -> Scalar:
    dim = registry.basis.dimension
    dummy_names = _dummy_families(term)
    total = 0
    for dummy_values in product(range(dim), repeat=len(dummy_names)):
        assignment = dict(free_assignment)
        assignment.update({key: value for key, value in zip(dummy_names, dummy_values)})
        value = Fraction(term.coefficient)
        for factor in term.factors:
            component_tensor = registry.lookup(factor.head, tuple(idx.variance for idx in factor.indices))
            key = tuple(assignment[(idx.name, idx.index_type)] for idx in factor.indices)
            value *= component_tensor.component(*key)
        total += value
    return total


def _dummy_families(term: TensorTerm) -> tuple[tuple[str, object], ...]:
    counts: dict[tuple[str, object], list[str]] = {}
    for factor in term.factors:
        for idx in factor.indices:
            counts.setdefault((idx.name, idx.index_type), []).append(idx.variance)
    out = []
    for key, variances in counts.items():
        if len(variances) == 2 and set(variances) == {"up", "down"}:
            out.append(key)
        elif len(variances) > 1:
            raise TensorKernelError(f"Invalid repeated component-realization index {key[0]!r}.")
    return tuple(out)
