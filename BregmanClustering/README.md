# BregmanClustering (core package)

Core implementation of the Bregman divergence-based graph clustering models.

## Files

- `models.py` — the clustering estimators (see the root
  [README](../README.md) for which class to use). All models follow the
  scikit-learn estimator API: `fit(...)` then `predict(...)`.
- `divergences.py` — Bregman divergence functions (`KL_div`,
  `euclidean_distance`, `logistic_loss`, ...) and the `phi`/`psi`
  function tables used to derive them from exponential-family distributions.
- `phi.py` — small helper convex functions (`phi_kl`, `phi_euclidean`) used
  when building custom divergences.

## Architecture notes

`models.py` shares logic between estimator classes via two mixins:

- `_AttributedClusteringMixin` — shared by every model that clusters nodes
  using an attribute matrix (`spectralEmbedding`, `computeAttributeMeans`,
  `likelihoodAttributes`, `predict`).
- `_SparseGraphClusteringMixin` — shared by models that represent the graph
  as a sparse edge list and delegate the initial membership guess to
  `BregmanInitializer` (constructor, `initialize`, `assignInitialLabels`,
  `assignments`/`assignments_joblib`, `singleNodeAssignment`,
  `precompute_edge_divergences`, `index_to_mask`).

Subclasses only implement what's genuinely specific to them (e.g. how
cluster means are computed — loop-based vs. vectorized/tensordot-based).
