# AI Use and Ownership

AI coding assistance was used to accelerate repository scaffolding, implementation, test generation, documentation editing, and review.

Human-owned decisions in this project are:

- choosing a capacity-first KV placement question instead of a generic semiconductor chatbot;
- defining the first-order scope and refusing product-performance claims;
- selecting equations, units, rejection conditions, and sensitivity variables;
- requiring deterministic tests and regenerated evidence;
- documenting what evidence would falsify the conclusion.

The PyTorch measurement path follows the same boundary: raw copy, GEMM, and SDPA timing samples plus environment metadata are committed, while fitted parameters and held-out error metrics are recomputed from those samples.

The acceptance boundary is `make verify`: lint, type checking, tests with coverage, and reproducible evidence. Generated code is treated as a draft until it passes that boundary and its assumptions can be explained.
