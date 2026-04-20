"""Unit tests for FlatTensorBatch intermediate representation (feature A1).

Covers:
- Round-trip ``list -> FlatTensorBatch -> materialize_list`` parity.
- Sequence protocol: ``__len__``, ``__getitem__``, ``__iter__``.
- ``release()`` lifecycle (idempotent, post-release access raises).
- Numerical parity for ``Clipping -> Bucketing -> NNM -> Median`` pipeline
  against the ``BYZPY_DISABLE_FLAT_BATCH`` baseline.
- Scheduler-driven eviction: intermediate SHM is unlinked after last consumer.
"""

from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import pytest
import torch

from byzpy.aggregators.coordinate_wise._tiling import as_flat_batch
from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.engine.graph.batch import FlatTensorBatch, release_if_batch
from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.scheduler import NodeScheduler
from byzpy.engine.storage.shared_store import cleanup_tensor, register_tensor
from byzpy.pre_aggregators import Bucketing, NearestNeighborMixing
from byzpy.pre_aggregators.clipping import Clipping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gradients(n: int, dim: int, seed: int = 0) -> List[torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen, dtype=torch.float32) for _ in range(n)]


def _flatten_like_reference(xs: List[torch.Tensor]) -> np.ndarray:
    """Replicate the current ``flatten_gradients`` semantics for parity."""
    return np.stack([x.detach().cpu().numpy().reshape(-1) for x in xs], axis=0)


# ---------------------------------------------------------------------------
# Basic FlatTensorBatch behavior
# ---------------------------------------------------------------------------


class TestFlatTensorBatchBasics:
    def test_round_trip_materialize_matches_input(self) -> None:
        xs = _make_gradients(5, 10, seed=42)
        batch = as_flat_batch(xs)
        try:
            materialized = batch.materialize_list()
            assert len(materialized) == len(xs)
            for original, restored in zip(xs, materialized):
                assert torch.allclose(original, restored, atol=0)
        finally:
            batch.release()

    def test_len_matches_row_count(self) -> None:
        xs = _make_gradients(7, 4)
        batch = as_flat_batch(xs)
        try:
            assert len(batch) == 7
        finally:
            batch.release()

    def test_indexing_returns_like_typed_tensor(self) -> None:
        xs = _make_gradients(3, 6)
        batch = as_flat_batch(xs)
        try:
            first = batch[0]
            assert isinstance(first, torch.Tensor)
            assert first.dtype == xs[0].dtype
            assert torch.allclose(first, xs[0])
        finally:
            batch.release()

    def test_iteration_yields_all_rows(self) -> None:
        xs = _make_gradients(4, 5, seed=1)
        batch = as_flat_batch(xs)
        try:
            out = list(batch)
            assert len(out) == 4
            stacked = torch.stack(out)
            assert torch.allclose(stacked, torch.stack(xs))
        finally:
            batch.release()

    def test_negative_index_supported(self) -> None:
        xs = _make_gradients(3, 4, seed=2)
        batch = as_flat_batch(xs)
        try:
            assert torch.allclose(batch[-1], xs[-1])
        finally:
            batch.release()

    def test_out_of_range_index_raises(self) -> None:
        xs = _make_gradients(2, 3, seed=3)
        batch = as_flat_batch(xs)
        try:
            with pytest.raises(IndexError):
                _ = batch[5]
        finally:
            batch.release()

    def test_as_flat_batch_passes_through_existing_batch(self) -> None:
        xs = _make_gradients(4, 8, seed=7)
        original = as_flat_batch(xs)
        try:
            second = as_flat_batch(original)
            assert second is original
        finally:
            original.release()

    def test_release_is_idempotent(self) -> None:
        xs = _make_gradients(2, 3)
        batch = as_flat_batch(xs)
        batch.release()
        batch.release()

    def test_access_after_release_raises(self) -> None:
        xs = _make_gradients(2, 3)
        batch = as_flat_batch(xs)
        batch.release()
        with pytest.raises(RuntimeError):
            _ = batch[0]
        with pytest.raises(RuntimeError):
            list(batch)
        with pytest.raises(RuntimeError):
            batch.materialize_list()

    def test_disown_suppresses_release(self) -> None:
        from byzpy.engine.storage.shared_store import open_tensor

        xs = _make_gradients(2, 3)
        batch = as_flat_batch(xs)
        batch.disown()
        batch.release()
        with open_tensor(batch.handle) as arr:
            assert arr.shape == (2, 3)
        cleanup_tensor(batch.handle)

    def test_release_if_batch_noop_for_non_batch(self) -> None:
        release_if_batch([1, 2, 3])
        release_if_batch(torch.zeros(3))
        release_if_batch(None)

    def test_as_numpy_returns_copy(self) -> None:
        xs = _make_gradients(3, 4, seed=9)
        batch = as_flat_batch(xs)
        try:
            arr = batch.as_numpy()
            assert arr.shape == (3, 4)
            arr[0, 0] = 1e9
            assert batch[0][0].item() != 1e9
        finally:
            batch.release()


# ---------------------------------------------------------------------------
# Pipeline parity: feature flag on vs off
# ---------------------------------------------------------------------------


def _run_pipeline(gradients: List[torch.Tensor]) -> torch.Tensor:
    clip = Clipping(threshold=2.0, chunk_size=256)
    bucket = Bucketing(bucket_size=2, feature_chunk_size=256, perm=list(range(len(gradients))))
    nnm = NearestNeighborMixing(f=1, feature_chunk_size=256)
    median = CoordinateWiseMedian(chunk_size=256)

    graph = ComputationGraph(
        nodes=[
            GraphNode(
                name="clip",
                op=clip,
                inputs={"vectors": graph_input("vectors")},
            ),
            GraphNode(
                name="bucket",
                op=bucket,
                inputs={"vectors": "clip"},
            ),
            GraphNode(
                name="nnm",
                op=nnm,
                inputs={"vectors": "bucket"},
            ),
            GraphNode(
                name="median",
                op=median,
                inputs={"gradients": "nnm"},
            ),
        ],
        outputs=["median"],
    )
    scheduler = NodeScheduler(graph)

    import asyncio

    result = asyncio.run(scheduler.run({"vectors": gradients}))
    return result["median"]


class TestPipelineParity:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_ftb_vs_baseline_median(self, seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
        xs = _make_gradients(8, 32, seed=seed)

        monkeypatch.setenv("BYZPY_DISABLE_FLAT_BATCH", "1")
        baseline = _run_pipeline([g.clone() for g in xs])

        monkeypatch.delenv("BYZPY_DISABLE_FLAT_BATCH", raising=False)
        enabled = _run_pipeline([g.clone() for g in xs])

        assert torch.allclose(baseline, enabled, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------


def _record_emitted_handles(operator: Any, sink: List[str]) -> None:
    """Wrap ``operator.reduce_subtasks`` so each emitted FlatTensorBatch's
    handle name is appended to ``sink`` before the value is returned.
    """
    original = operator.reduce_subtasks

    def wrapper(partials, inputs, *, context):
        result = original(partials, inputs, context=context)
        if isinstance(result, FlatTensorBatch):
            sink.append(result.handle.name)
        return result

    operator.reduce_subtasks = wrapper  # type: ignore[method-assign]


class TestSchedulerLifecycle:
    def test_intermediate_ftb_is_released_after_last_consumer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The specific FlatTensorBatch emitted by the first operator must be
        released by the scheduler after the downstream operator consumes it.
        """
        released: List[str] = []
        original_cleanup = cleanup_tensor

        def tracking_cleanup(handle) -> None:
            released.append(handle.name)
            original_cleanup(handle)

        monkeypatch.setattr(
            "byzpy.engine.graph.batch.cleanup_tensor", tracking_cleanup, raising=True
        )
        monkeypatch.delenv("BYZPY_DISABLE_FLAT_BATCH", raising=False)

        xs = _make_gradients(6, 16, seed=11)

        clip = Clipping(threshold=2.0, chunk_size=64)
        median = CoordinateWiseMedian(chunk_size=64)
        emitted: List[str] = []
        _record_emitted_handles(clip, emitted)

        graph = ComputationGraph(
            nodes=[
                GraphNode(
                    name="clip",
                    op=clip,
                    inputs={"vectors": graph_input("vectors")},
                ),
                GraphNode(
                    name="median",
                    op=median,
                    inputs={"gradients": "clip"},
                ),
            ],
            outputs=["median"],
        )

        import asyncio

        from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig

        async def _run() -> None:
            pool = ActorPool([ActorPoolConfig(backend="thread", count=2)])
            await pool.start()
            try:
                scheduler = NodeScheduler(graph, pool=pool)
                await scheduler.run({"vectors": xs})
            finally:
                await pool.shutdown()

        asyncio.run(_run())

        assert emitted, "Clipping did not emit a FlatTensorBatch intermediate"
        clip_handle_name = emitted[0]
        assert clip_handle_name in released, (
            f"Intermediate handle {clip_handle_name!r} was not released by the "
            f"scheduler; recorded releases: {released}"
        )

    def test_parallel_scheduler_releases_intermediate_ftb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from byzpy.engine.graph.parallel_scheduler import ParallelScheduler
        from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig

        released: List[str] = []
        original_cleanup = cleanup_tensor

        def tracking_cleanup(handle) -> None:
            released.append(handle.name)
            original_cleanup(handle)

        monkeypatch.setattr(
            "byzpy.engine.graph.batch.cleanup_tensor", tracking_cleanup, raising=True
        )
        monkeypatch.delenv("BYZPY_DISABLE_FLAT_BATCH", raising=False)

        xs = _make_gradients(6, 16, seed=33)

        clip = Clipping(threshold=2.0, chunk_size=64)
        median = CoordinateWiseMedian(chunk_size=64)
        emitted: List[str] = []
        _record_emitted_handles(clip, emitted)

        graph = ComputationGraph(
            nodes=[
                GraphNode(
                    name="clip",
                    op=clip,
                    inputs={"vectors": graph_input("vectors")},
                ),
                GraphNode(
                    name="median",
                    op=median,
                    inputs={"gradients": "clip"},
                ),
            ],
            outputs=["median"],
        )

        import asyncio

        async def _run() -> None:
            pool = ActorPool([ActorPoolConfig(backend="thread", count=2)])
            await pool.start()
            try:
                scheduler = ParallelScheduler(graph, pool=pool)
                await scheduler.run({"vectors": xs})
            finally:
                await pool.shutdown()

        asyncio.run(_run())

        assert emitted, "Clipping did not emit a FlatTensorBatch intermediate"
        clip_handle_name = emitted[0]
        assert clip_handle_name in released, (
            f"Intermediate handle {clip_handle_name!r} was not released by the "
            f"parallel scheduler; recorded releases: {released}"
        )

    def test_graph_output_batch_is_not_released(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a FlatTensorBatch is a graph output, the scheduler must not
        call cleanup_tensor on its handle during ``run()``.
        """
        released_during_run: List[str] = []
        original_cleanup = cleanup_tensor

        def tracking_cleanup(handle) -> None:
            released_during_run.append(handle.name)
            original_cleanup(handle)

        monkeypatch.setattr(
            "byzpy.engine.graph.batch.cleanup_tensor", tracking_cleanup, raising=True
        )
        monkeypatch.delenv("BYZPY_DISABLE_FLAT_BATCH", raising=False)

        xs = _make_gradients(4, 8, seed=22)

        clip = Clipping(threshold=1.0, chunk_size=32)
        emitted: List[str] = []
        _record_emitted_handles(clip, emitted)

        graph = ComputationGraph(
            nodes=[
                GraphNode(
                    name="clip",
                    op=clip,
                    inputs={"vectors": graph_input("vectors")},
                ),
            ],
            outputs=["clip"],
        )

        import asyncio

        from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig

        async def _run() -> Any:
            pool = ActorPool([ActorPoolConfig(backend="thread", count=2)])
            await pool.start()
            try:
                scheduler = NodeScheduler(graph, pool=pool)
                return await scheduler.run({"vectors": xs})
            finally:
                await pool.shutdown()

        result = asyncio.run(_run())
        out = result["clip"]

        assert isinstance(out, FlatTensorBatch), (
            "Expected the graph output to be a FlatTensorBatch; " f"got {type(out).__name__}"
        )
        assert emitted, "Clipping did not emit a FlatTensorBatch intermediate"
        output_handle_name = out.handle.name
        assert output_handle_name == emitted[0], (
            "Emitted intermediate should match the graph output handle; "
            f"emitted={emitted[0]!r} output={output_handle_name!r}"
        )

        releases_before_manual = list(released_during_run)
        try:
            assert output_handle_name not in releases_before_manual, (
                f"Graph output handle {output_handle_name!r} was released "
                f"during run(); recorded releases: {releases_before_manual}"
            )
            rows = out.materialize_list()
            assert len(rows) == len(xs)
        finally:
            release_if_batch(out)

        assert (
            output_handle_name in released_during_run
        ), "After manual release, the output handle should have been unlinked."


# ---------------------------------------------------------------------------
# last_use_map
# ---------------------------------------------------------------------------


class TestLastUseMap:
    def test_chain_last_use_is_final_consumer(self) -> None:
        from byzpy.engine.graph.operator import Operator

        class _Noop(Operator):
            def compute(self, inputs, *, context):  # type: ignore[override]
                return next(iter(inputs.values()))

        graph = ComputationGraph(
            nodes=[
                GraphNode(name="a", op=_Noop(), inputs={"x": graph_input("src")}),
                GraphNode(name="b", op=_Noop(), inputs={"x": "a"}),
                GraphNode(name="c", op=_Noop(), inputs={"x": "b"}),
            ],
            outputs=["c"],
        )
        last_use = graph.last_use_map()
        assert last_use.get("b") == ["a"]
        assert last_use.get("c") == ["b"]
        assert "c" not in {k for vals in last_use.values() for k in vals}

    def test_fan_out_last_consumer_is_latest_in_topo(self) -> None:
        from byzpy.engine.graph.operator import Operator

        class _Noop(Operator):
            def compute(self, inputs, *, context):  # type: ignore[override]
                return next(iter(inputs.values()))

        graph = ComputationGraph(
            nodes=[
                GraphNode(name="src", op=_Noop(), inputs={"x": graph_input("inp")}),
                GraphNode(name="a", op=_Noop(), inputs={"x": "src"}),
                GraphNode(name="b", op=_Noop(), inputs={"x": "src"}),
            ],
            outputs=["a", "b"],
        )
        last_use = graph.last_use_map()
        consumers_of_src = [node for node, keys in last_use.items() if "src" in keys]
        assert len(consumers_of_src) == 1
