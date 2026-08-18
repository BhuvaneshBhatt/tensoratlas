# Exterior/spin breadth and polish

This layer adds a compact but practical breadth layer on top of the earlier algebra and geometry core:

- typed spin and spinor descriptors,
- a lightweight Clifford algebra helper for generator relations and simple reductions,
- differential-form convenience wrappers built on the existing `DifferentialForm` API,
- JSON import/export for geometry descriptors and chart-backed objects,
- targeted regression tests for those workflows.

The Clifford layer is intentionally modest: it supports diagonal signatures and generator-level reductions, not a full gamma-matrix or spinor-calculus engine.

The import/export layer is designed for portability and inspection rather than maximal fidelity for every runtime object. In particular, `FrameDef` deserialization is left explicit because reconstructing a basis generally requires a user-supplied basis constructor.
