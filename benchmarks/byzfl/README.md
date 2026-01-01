# ByzFL Benchmarks

This directory will host benchmarks and adapters targeting the
[ByzFL](https://github.com/LPD-EPFL/byzfl) framework. The plan is to:

1. Map our existing workloads (e.g., MDA, CwTM) onto the corresponding ByzFL
   aggregators/pre-aggregators via a lightweight adapter so they can be timed
   with the exact same synthetic gradient inputs we use for ByzPy. The first
   example (`mda_compare.py`) contrasts ByzPy vs ByzFL for Minimum Diameter
   Averaging.
2. Record installation steps (`pip install byzfl` or an editable checkout) and
   any additional dependency pinning required for CPU/GPU environments.
3. Provide CLI entry-points (e.g., `python benchmarks/byzfl/mda_trmean.py`)
   that mirror the interfaces under `benchmarks/` but compare:
   - PyTorch/ByzPy implementation
   - ByzFL implementation
   - ActorPool-accelerated variations when relevant.
4. Optionally integrate ByzFL's own FL benchmark harness (JSON-driven) so we can
   run small federated simulations for apples-to-apples comparisons.

Until the scripts land, this README acts as the placeholder documentation for
where ByzFL-specific work will live.

## Available workloads

- `python benchmarks/byzfl/mda_compare.py`: Generates synthetic gradients and
  times the ByzFL Minimum Diameter Averaging aggregator.
- `python benchmarks/byzfl/cwtm_compare.py`: Generates synthetic gradients and
  times the ByzFL Coordinate-wise Trimmed Mean (TrMean) aggregator.
- `python benchmarks/byzfl/multikrum_compare.py`: Generates synthetic gradients
  and times the ByzFL Multi-Krum aggregator.
- `python benchmarks/byzfl/centered_clipping_compare.py`: Generates synthetic
  gradients and times the ByzFL Centered Clipping aggregator.
- `python benchmarks/byzfl/clipping_compare.py`: Generates synthetic gradients
  and times the ByzFL static clipping pre-aggregator.
- `python benchmarks/byzfl/nnm_compare.py`: Generates synthetic gradients and
  times the ByzFL Nearest Neighbor Mixing pre-aggregator.
- `python benchmarks/byzfl/bucketing_compare.py`: Generates synthetic gradients
  and times the ByzFL Bucketing pre-aggregator.
- `python benchmarks/byzfl/caf_compare.py`: Generates synthetic gradients and
  times the ByzFL Covariance-bound Agnostic Filter aggregator.
- `python benchmarks/byzfl/meamed_compare.py`: Generates synthetic gradients and
  times the ByzFL Meamed aggregator.
- `python benchmarks/byzfl/monna_compare.py`: Generates synthetic gradients and
  times the ByzFL MoNNA aggregator.
- `python benchmarks/byzfl/ipm_attack_compare.py`: Generates synthetic
  honest gradients and times the ByzFL Inner Product Manipulation attack
  (ByzFL's analogue for the Empire attack).
- `python benchmarks/byzfl/little_attack_compare.py`: Generates synthetic
  honest gradients and times the ByzFL A Little Is Enough (ALIE) attack.
- Comparative Gradient Elimination (CGE): **not available in ByzFL** (no
  reference implementation to benchmark).
