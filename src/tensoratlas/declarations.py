from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .semantic_ir import TensorExpr, ir_node, canonical_ir_key


class DeclarationError(ValueError):
    """Raised when declarations are inconsistent or incomplete."""


class IndexVariance(str, Enum):
    CONTRAVARIANT = "u"
    COVARIANT = "l"

    @classmethod
    def parse(cls, value: str | "IndexVariance") -> "IndexVariance":
        if isinstance(value, cls):
            return value
        normalized = str(value).lower()
        aliases = {
            "up": cls.CONTRAVARIANT,
            "upper": cls.CONTRAVARIANT,
            "contravariant": cls.CONTRAVARIANT,
            "u": cls.CONTRAVARIANT,
            "+": cls.CONTRAVARIANT,
            "down": cls.COVARIANT,
            "lower": cls.COVARIANT,
            "covariant": cls.COVARIANT,
            "l": cls.COVARIANT,
            "-": cls.COVARIANT,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise DeclarationError(f"Unsupported variance {value!r}.") from exc


class CurvatureSignConvention(str, Enum):
    XACT = "xact"
    CADABRA = "cadabra"
    MTW = "mtw"
    WALD = "wald"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ManifoldDeclaration:
    name: str
    dimension: int
    signature: tuple[int, ...] | None = None
    orientation: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise DeclarationError("Manifold name must be nonempty.")
        if self.dimension <= 0:
            raise DeclarationError("Manifold dimension must be positive.")
        if self.signature is not None and len(self.signature) not in {2, 3, self.dimension}:
            raise DeclarationError("Signature must be a pair, triple, or explicit per-dimension tuple.")

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:manifold",
            payload=self.name,
            dimension=self.dimension,
            signature=self.signature,
            orientation=self.orientation,
            assumptions=self.assumptions,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class ChartDeclaration:
    name: str
    manifold: str
    coordinates: tuple[Any, ...]
    domain: str | None = None
    transition_maps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.manifold:
            raise DeclarationError("Chart and manifold names must be nonempty.")
        if not self.coordinates:
            raise DeclarationError("Chart must declare at least one coordinate.")

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:chart",
            payload=self.name,
            manifold=self.manifold,
            coordinates=self.coordinates,
            domain=self.domain,
            transition_maps=self.transition_maps,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class BundleDeclaration:
    name: str
    manifold: str
    rank: int
    kind: str = "vector"
    dual_of: str | None = None
    metric: str | None = None
    structure_group: str | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise DeclarationError("Bundle rank must be positive.")

    def dual_name(self) -> str:
        return self.dual_of or f"{self.name}*"

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:bundle",
            payload=self.name,
            manifold=self.manifold,
            rank=self.rank,
            bundle_kind=self.kind,
            dual_of=self.dual_of,
            metric=self.metric,
            structure_group=self.structure_group,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class IndexFamilyDeclaration:
    name: str
    bundle: str
    symbols: tuple[str, ...]
    variance: IndexVariance | str | None = None
    dummy_prefix: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(str(s) for s in self.symbols))
        if self.variance is not None:
            object.__setattr__(self, "variance", IndexVariance.parse(self.variance))
        if not self.symbols:
            raise DeclarationError("Index family must have at least one symbol.")

    def dummy_symbol(self, number: int) -> str:
        prefix = self.dummy_prefix or self.name
        return f"{prefix}{number}"

    def to_ir(self) -> TensorExpr:
        variance = self.variance.value if isinstance(self.variance, IndexVariance) else self.variance
        return ir_node(
            "declaration:index_family",
            payload=self.name,
            bundle=self.bundle,
            symbols=self.symbols,
            variance=variance,
            dummy_prefix=self.dummy_prefix,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class DummyIndexPool:
    name: str
    family: str
    prefix: str
    counter: int = 0
    reserved: tuple[str, ...] = ()

    def allocate(self, count: int = 1) -> tuple["DummyIndexPool", tuple[str, ...]]:
        if count < 0:
            raise DeclarationError("Cannot allocate a negative number of dummy indices.")
        allocated = tuple(f"{self.prefix}{self.counter + offset}" for offset in range(count))
        return replace(self, counter=self.counter + count, reserved=self.reserved + allocated), allocated

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:dummy_index_pool",
            payload=self.name,
            family=self.family,
            prefix=self.prefix,
            counter=self.counter,
            reserved=self.reserved,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class TensorSymmetryDeclaration:
    kind: str
    slots: tuple[int, ...]

    def to_ir(self) -> TensorExpr:
        return ir_node("declaration:tensor_symmetry", payload=self.kind, slots=self.slots, provenance={"origin": "declaration_registry"})


@dataclass(frozen=True)
class TensorDeclaration:
    name: str
    bundle_slots: tuple[str, ...]
    variance: tuple[IndexVariance | str, ...]
    symmetries: tuple[TensorSymmetryDeclaration, ...] = ()
    density_weight: int | float = 0
    dependencies: tuple[str, ...] = ()
    role: str = "tensor"

    def __post_init__(self) -> None:
        object.__setattr__(self, "variance", tuple(IndexVariance.parse(v) for v in self.variance))
        if len(self.bundle_slots) != len(self.variance):
            raise DeclarationError("Tensor bundle_slots and variance must have the same length.")
        rank = len(self.bundle_slots)
        for sym in self.symmetries:
            if any(slot < 0 or slot >= rank for slot in sym.slots):
                raise DeclarationError(f"Symmetry {sym} refers to a nonexistent slot.")

    @property
    def rank(self) -> int:
        return len(self.bundle_slots)

    def variance_string(self) -> str:
        return "".join(v.value if isinstance(v, IndexVariance) else str(v) for v in self.variance)

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:tensor",
            *(sym.to_ir() for sym in self.symmetries),
            payload=self.name,
            bundle_slots=self.bundle_slots,
            variance=self.variance_string(),
            density_weight=self.density_weight,
            dependencies=self.dependencies,
            role=self.role,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class MetricDeclaration:
    name: str
    manifold: str
    bundle: str
    signature: tuple[int, ...] | None = None
    inverse_name: str | None = None
    determinant_name: str | None = None
    density_weight: int | float = 0

    def to_tensor_declaration(self) -> TensorDeclaration:
        return TensorDeclaration(
            self.name,
            bundle_slots=(self.bundle, self.bundle),
            variance=(IndexVariance.COVARIANT, IndexVariance.COVARIANT),
            symmetries=(TensorSymmetryDeclaration("symmetric", (0, 1)),),
            density_weight=self.density_weight,
            role="metric",
        )

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:metric",
            self.to_tensor_declaration().to_ir(),
            payload=self.name,
            manifold=self.manifold,
            bundle=self.bundle,
            signature=self.signature,
            inverse_name=self.inverse_name,
            determinant_name=self.determinant_name,
            density_weight=self.density_weight,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class TorsionDeclaration:
    name: str
    connection: str
    tensor_name: str | None = None
    vanishes: bool = False

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:torsion",
            payload=self.name,
            connection=self.connection,
            tensor_name=self.tensor_name,
            vanishes=self.vanishes,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class NonmetricityDeclaration:
    name: str
    connection: str
    metric: str
    tensor_name: str | None = None
    vanishes: bool = False

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:nonmetricity",
            payload=self.name,
            connection=self.connection,
            metric=self.metric,
            tensor_name=self.tensor_name,
            vanishes=self.vanishes,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class CurvatureConventionPolicy:
    name: str = "default"
    sign: CurvatureSignConvention | str = CurvatureSignConvention.XACT
    riemann_slot_order: tuple[str, ...] = ("up", "down", "down", "down")
    ricci_contraction: tuple[int, int] = (0, 2)
    commutator_sign: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.sign, str):
            try:
                object.__setattr__(self, "sign", CurvatureSignConvention(self.sign.lower()))
            except ValueError:
                object.__setattr__(self, "sign", CurvatureSignConvention.CUSTOM)
        if self.commutator_sign not in {-1, 1}:
            raise DeclarationError("commutator_sign must be +1 or -1.")

    def to_ir(self) -> TensorExpr:
        sign = self.sign.value if isinstance(self.sign, CurvatureSignConvention) else str(self.sign)
        return ir_node(
            "declaration:curvature_convention",
            payload=self.name,
            sign=sign,
            riemann_slot_order=self.riemann_slot_order,
            ricci_contraction=self.ricci_contraction,
            commutator_sign=self.commutator_sign,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class ConnectionDeclaration:
    name: str
    bundle: str
    manifold: str
    metric: str | None = None
    torsion: TorsionDeclaration | None = None
    nonmetricity: NonmetricityDeclaration | None = None
    curvature_policy: str = "default"
    coefficients: str | None = None

    def is_torsion_free(self) -> bool:
        return bool(self.torsion and self.torsion.vanishes)

    def is_metric_compatible(self) -> bool:
        return bool(self.nonmetricity and self.nonmetricity.vanishes)

    def to_ir(self) -> TensorExpr:
        children = tuple(x.to_ir() for x in (self.torsion, self.nonmetricity) if x is not None)
        return ir_node(
            "declaration:connection",
            *children,
            payload=self.name,
            bundle=self.bundle,
            manifold=self.manifold,
            metric=self.metric,
            curvature_policy=self.curvature_policy,
            coefficients=self.coefficients,
            torsion_free=self.is_torsion_free(),
            metric_compatible=self.is_metric_compatible(),
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class CovariantDerivativeCommutationRule:
    name: str
    connection: str
    applies_to: str | None = None
    curvature: str = "Riemann"
    torsion_term: bool = True
    nonmetricity_term: bool = False
    sign: int = 1

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise DeclarationError("Commutation-rule sign must be +1 or -1.")

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:covariant_commutation_rule",
            payload=self.name,
            connection=self.connection,
            applies_to=self.applies_to,
            curvature=self.curvature,
            torsion_term=self.torsion_term,
            nonmetricity_term=self.nonmetricity_term,
            sign=self.sign,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class TensorDependency:
    tensor: str
    depends_on: tuple[str, ...]
    kind: str = "functional"

    def to_ir(self) -> TensorExpr:
        return ir_node(
            "declaration:tensor_dependency",
            payload=self.tensor,
            depends_on=self.depends_on,
            dependency_kind=self.kind,
            provenance={"origin": "declaration_registry"},
        )


@dataclass(frozen=True)
class DeclarationSnapshot:
    manifolds: Mapping[str, ManifoldDeclaration]
    charts: Mapping[str, ChartDeclaration]
    bundles: Mapping[str, BundleDeclaration]
    index_families: Mapping[str, IndexFamilyDeclaration]
    dummy_pools: Mapping[str, DummyIndexPool]
    tensors: Mapping[str, TensorDeclaration]
    metrics: Mapping[str, MetricDeclaration]
    connections: Mapping[str, ConnectionDeclaration]
    commutation_rules: Mapping[str, CovariantDerivativeCommutationRule]
    curvature_policies: Mapping[str, CurvatureConventionPolicy]
    dependencies: tuple[TensorDependency, ...]


@dataclass(frozen=True)
class DeclarationRegistry:
    manifolds: Mapping[str, ManifoldDeclaration] = field(default_factory=dict)
    charts: Mapping[str, ChartDeclaration] = field(default_factory=dict)
    bundles: Mapping[str, BundleDeclaration] = field(default_factory=dict)
    index_families: Mapping[str, IndexFamilyDeclaration] = field(default_factory=dict)
    dummy_pools: Mapping[str, DummyIndexPool] = field(default_factory=dict)
    tensors: Mapping[str, TensorDeclaration] = field(default_factory=dict)
    metrics: Mapping[str, MetricDeclaration] = field(default_factory=dict)
    connections: Mapping[str, ConnectionDeclaration] = field(default_factory=dict)
    commutation_rules: Mapping[str, CovariantDerivativeCommutationRule] = field(default_factory=dict)
    curvature_policies: Mapping[str, CurvatureConventionPolicy] = field(default_factory=lambda: {"default": CurvatureConventionPolicy()})
    dependencies: tuple[TensorDependency, ...] = ()

    def _replace_map(self, field_name: str, name: str, value: Any) -> "DeclarationRegistry":
        existing = dict(getattr(self, field_name))
        if name in existing:
            raise DeclarationError(f"Duplicate declaration {name!r} in {field_name}.")
        existing[name] = value
        return replace(self, **{field_name: existing})

    def declare_manifold(self, name: str, dimension: int, *, signature: Sequence[int] | None = None, orientation: str | None = None, assumptions: Sequence[str] = ()) -> "DeclarationRegistry":
        return self._replace_map("manifolds", name, ManifoldDeclaration(name, dimension, tuple(signature) if signature is not None else None, orientation, tuple(assumptions)))

    def declare_chart(self, name: str, manifold: str, coordinates: Sequence[Any], *, domain: str | None = None, transition_maps: Sequence[str] = ()) -> "DeclarationRegistry":
        self.require_manifold(manifold)
        if len(tuple(coordinates)) != self.manifolds[manifold].dimension:
            raise DeclarationError("Chart coordinate count must match manifold dimension.")
        return self._replace_map("charts", name, ChartDeclaration(name, manifold, tuple(coordinates), domain, tuple(transition_maps)))

    def declare_bundle(self, name: str, manifold: str, rank: int | None = None, *, kind: str = "vector", dual_of: str | None = None, metric: str | None = None, structure_group: str | None = None) -> "DeclarationRegistry":
        self.require_manifold(manifold)
        actual_rank = self.manifolds[manifold].dimension if rank is None else rank
        return self._replace_map("bundles", name, BundleDeclaration(name, manifold, actual_rank, kind, dual_of, metric, structure_group))

    def declare_index_family(self, name: str, bundle: str, symbols: Sequence[str], *, variance: str | IndexVariance | None = None, dummy_prefix: str | None = None) -> "DeclarationRegistry":
        self.require_bundle(bundle)
        return self._replace_map("index_families", name, IndexFamilyDeclaration(name, bundle, tuple(symbols), variance, dummy_prefix))

    def declare_dummy_pool(self, name: str, family: str, *, prefix: str | None = None) -> "DeclarationRegistry":
        self.require_index_family(family)
        fam = self.index_families[family]
        return self._replace_map("dummy_pools", name, DummyIndexPool(name, family, prefix or fam.dummy_prefix or fam.name))

    def allocate_dummy_indices(self, pool: str, count: int = 1) -> tuple["DeclarationRegistry", tuple[str, ...]]:
        try:
            dummy_pool = self.dummy_pools[pool]
        except KeyError as exc:
            raise DeclarationError(f"Unknown dummy-index pool {pool!r}.") from exc
        new_pool, allocated = dummy_pool.allocate(count)
        pools = dict(self.dummy_pools)
        pools[pool] = new_pool
        return replace(self, dummy_pools=pools), allocated

    def declare_tensor(self, name: str, bundle_slots: Sequence[str], variance: Sequence[str | IndexVariance], *, symmetries: Sequence[TensorSymmetryDeclaration | tuple[str, Sequence[int]]] = (), density_weight: int | float = 0, dependencies: Sequence[str] = (), role: str = "tensor") -> "DeclarationRegistry":
        for bundle in bundle_slots:
            self.require_bundle(bundle)
        parsed_sym = tuple(s if isinstance(s, TensorSymmetryDeclaration) else TensorSymmetryDeclaration(str(s[0]), tuple(s[1])) for s in symmetries)
        dep_objs = tuple(TensorDependency(name, (dep,)) for dep in dependencies)
        registry = self._replace_map("tensors", name, TensorDeclaration(name, tuple(bundle_slots), tuple(variance), parsed_sym, density_weight, tuple(dependencies), role))
        return replace(registry, dependencies=registry.dependencies + dep_objs)

    def declare_metric(self, name: str, manifold: str, bundle: str, *, signature: Sequence[int] | None = None, inverse_name: str | None = None, determinant_name: str | None = None, density_weight: int | float = 0) -> "DeclarationRegistry":
        self.require_manifold(manifold)
        self.require_bundle(bundle)
        metric = MetricDeclaration(name, manifold, bundle, tuple(signature) if signature is not None else self.manifolds[manifold].signature, inverse_name, determinant_name, density_weight)
        tensors = dict(self.tensors)
        tensors[name] = metric.to_tensor_declaration()
        bundles = dict(self.bundles)
        bundles[bundle] = replace(bundles[bundle], metric=name)
        return replace(self, metrics={**dict(self.metrics), name: metric}, tensors=tensors, bundles=bundles)

    def declare_curvature_policy(self, policy: CurvatureConventionPolicy) -> "DeclarationRegistry":
        return self._replace_map("curvature_policies", policy.name, policy)

    def declare_connection(self, name: str, bundle: str, *, manifold: str | None = None, metric: str | None = None, torsion_free: bool = False, metric_compatible: bool = False, curvature_policy: str = "default", coefficients: str | None = None) -> "DeclarationRegistry":
        self.require_bundle(bundle)
        if curvature_policy not in self.curvature_policies:
            raise DeclarationError(f"Unknown curvature convention policy {curvature_policy!r}.")
        actual_manifold = manifold or self.bundles[bundle].manifold
        self.require_manifold(actual_manifold)
        if metric is not None:
            self.require_metric(metric)
        torsion = TorsionDeclaration(f"T[{name}]", name, vanishes=torsion_free)
        nonmetricity = None
        if metric is not None:
            nonmetricity = NonmetricityDeclaration(f"Q[{name},{metric}]", name, metric, vanishes=metric_compatible)
        conn = ConnectionDeclaration(name, bundle, actual_manifold, metric, torsion, nonmetricity, curvature_policy, coefficients)
        return self._replace_map("connections", name, conn)

    def declare_commutation_rule(self, name: str, connection: str, *, applies_to: str | None = None, curvature: str = "Riemann", torsion_term: bool | None = None, nonmetricity_term: bool | None = None, sign: int | None = None) -> "DeclarationRegistry":
        self.require_connection(connection)
        conn = self.connections[connection]
        policy = self.curvature_policies[conn.curvature_policy]
        rule = CovariantDerivativeCommutationRule(
            name,
            connection,
            applies_to,
            curvature,
            torsion_term=(not conn.is_torsion_free()) if torsion_term is None else torsion_term,
            nonmetricity_term=(not conn.is_metric_compatible()) if nonmetricity_term is None else nonmetricity_term,
            sign=policy.commutator_sign if sign is None else sign,
        )
        return self._replace_map("commutation_rules", name, rule)

    def declare_dependency(self, tensor: str, depends_on: Sequence[str], *, kind: str = "functional") -> "DeclarationRegistry":
        if tensor not in self.tensors and tensor not in self.metrics:
            raise DeclarationError(f"Unknown tensor {tensor!r} in dependency declaration.")
        return replace(self, dependencies=self.dependencies + (TensorDependency(tensor, tuple(depends_on), kind),))

    def require_manifold(self, name: str) -> ManifoldDeclaration:
        try:
            return self.manifolds[name]
        except KeyError as exc:
            raise DeclarationError(f"Unknown manifold {name!r}.") from exc

    def require_bundle(self, name: str) -> BundleDeclaration:
        try:
            return self.bundles[name]
        except KeyError as exc:
            raise DeclarationError(f"Unknown bundle {name!r}.") from exc

    def require_index_family(self, name: str) -> IndexFamilyDeclaration:
        try:
            return self.index_families[name]
        except KeyError as exc:
            raise DeclarationError(f"Unknown index family {name!r}.") from exc

    def require_metric(self, name: str) -> MetricDeclaration:
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise DeclarationError(f"Unknown metric {name!r}.") from exc

    def require_connection(self, name: str) -> ConnectionDeclaration:
        try:
            return self.connections[name]
        except KeyError as exc:
            raise DeclarationError(f"Unknown connection {name!r}.") from exc

    def tensor_expr(self, tensor: str, indices: Sequence[Any] = ()) -> TensorExpr:
        decl = self.tensors.get(tensor)
        if decl is None and tensor in self.metrics:
            decl = self.metrics[tensor].to_tensor_declaration()
        if decl is None:
            raise DeclarationError(f"Unknown tensor {tensor!r}.")
        return ir_node(
            "indexed_tensor",
            payload=tensor,
            tensor_name=tensor,
            indices=tuple(indices),
            variance_spec=decl.variance_string(),
            declaration_key=canonical_ir_key(decl.to_ir()),
            provenance={"origin": "declaration_registry"},
        )

    def connection_ir(self, connection: str) -> TensorExpr:
        return self.require_connection(connection).to_ir()

    def commutator_ir(self, rule: str, left_index: Any, right_index: Any, operand: TensorExpr) -> TensorExpr:
        rule_decl = self.commutation_rules.get(rule)
        if rule_decl is None:
            raise DeclarationError(f"Unknown commutation rule {rule!r}.")
        return ir_node(
            "covariant_derivative_commutator",
            operand,
            payload=rule,
            left_index=left_index,
            right_index=right_index,
            connection=rule_decl.connection,
            curvature=rule_decl.curvature,
            torsion_term=rule_decl.torsion_term,
            nonmetricity_term=rule_decl.nonmetricity_term,
            sign=rule_decl.sign,
            provenance={"origin": "declaration_registry"},
        )

    def to_ir(self) -> TensorExpr:
        children: list[TensorExpr] = []
        for collection in (
            self.manifolds,
            self.charts,
            self.bundles,
            self.index_families,
            self.dummy_pools,
            self.tensors,
            self.metrics,
            self.connections,
            self.commutation_rules,
            self.curvature_policies,
        ):
            children.extend(value.to_ir() for _, value in sorted(collection.items()))
        children.extend(dep.to_ir() for dep in self.dependencies)
        return ir_node("declaration:registry", *children, payload="registry", provenance={"origin": "declaration_registry"})

    def snapshot(self) -> DeclarationSnapshot:
        return DeclarationSnapshot(
            manifolds=dict(self.manifolds),
            charts=dict(self.charts),
            bundles=dict(self.bundles),
            index_families=dict(self.index_families),
            dummy_pools=dict(self.dummy_pools),
            tensors=dict(self.tensors),
            metrics=dict(self.metrics),
            connections=dict(self.connections),
            commutation_rules=dict(self.commutation_rules),
            curvature_policies=dict(self.curvature_policies),
            dependencies=self.dependencies,
        )


def declaration_registry() -> DeclarationRegistry:
    return DeclarationRegistry()


def standard_riemannian_registry(name: str = "M", dimension: int = 4, *, signature: Sequence[int] | None = None) -> DeclarationRegistry:
    """Create a minimal Riemannian/Lorentzian declaration environment."""
    reg = declaration_registry().declare_manifold(name, dimension, signature=signature)
    reg = reg.declare_bundle("TM", name).declare_bundle("T*M", name, kind="covector", dual_of="TM")
    reg = reg.declare_index_family("latin", "TM", tuple("abcdefghijklmnopqrstuvwxyz"), variance=None, dummy_prefix="d")
    reg = reg.declare_dummy_pool("latin_dummies", "latin", prefix="d")
    reg = reg.declare_metric("g", name, "TM", signature=signature, inverse_name="g_inv", determinant_name="det_g")
    reg = reg.declare_connection("CD", "TM", metric="g", torsion_free=True, metric_compatible=True)
    return reg.declare_commutation_rule("riemann_commutator", "CD", applies_to="tensor")
