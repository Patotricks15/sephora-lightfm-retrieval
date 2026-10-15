# sephora-lightfm-retrieval

Hybrid, feature-aware matrix factorization with a WARP ranking loss (the
LightFM algorithm, implemented from scratch in NumPy — no PyTorch/TensorFlow,
and no dependency on the `lightfm` PyPI package) for cold-start product
retrieval on the Sephora Products and Skincare Reviews dataset.

> Status: scaffolding in progress. See `paper/main.pdf` for the full
> technical report once built, and the sections below for the quickstart,
> architecture and evaluation results.

## Why this project exists

The two sibling portfolio repositories (`anime-bpr-recommender`,
`movielens-svd-ranking`) factorize **identities**: a user is a row index, an
item is a column index, and a brand-new user or item has no row/column to
score at all. This project factorizes **features** instead: a user's vector
is the sum of embeddings of their known attributes (skin type, skin tone,
age range), and an item's vector is the sum of embeddings of its attributes
(category, brand, price bucket, ingredient tags). A user or item the model has
never seen individually can still be scored at request time as long as it
shares known features with the training population — the classical,
pre-deep-learning solution to cold start.

Full problem statement, architecture, quickstart and results are documented
below as the repository is built out commit by commit.
