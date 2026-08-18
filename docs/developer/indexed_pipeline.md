# Indexed canonicalization pipeline

The indexed-expression normalizer follows a documented stage order:

1. expand the expression tree
2. alpha-rename dummy indices
3. lower metric and Kronecker-delta contractions
4. canonicalize special tensors
5. sort commutative factors
6. rebuild the final expression

See `tensoratlas.indexed_pipeline.INDEXED_NORMALIZATION_STAGES`.
