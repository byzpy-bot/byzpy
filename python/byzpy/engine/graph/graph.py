from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Union,
)

if TYPE_CHECKING:
    from .scheduler import MessageSource

from .operator import Operator


@dataclass(frozen=True)
class GraphInput:
    """
    Opaque reference representing data supplied by the application layer
    """

    name: str

    @classmethod
    def from_message(
        cls,
        message_type: str,
        field: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Create a graph input that reads from a message.

        Returns a MessageSource object that can be used in graph node inputs.
        """
        from .scheduler import MessageSource

        return MessageSource(message_type=message_type, field=field, timeout=timeout)


def graph_input(name: str) -> GraphInput:
    return GraphInput(name)


@dataclass(frozen=True)
class GraphNode:
    name: str
    op: Operator
    inputs: Mapping[str, Union[str, GraphInput, "MessageSource"]] = field(default_factory=dict)


class ComputationGraph:
    """
    Lightweight DAG that records which operator runs after which dependencies
    """

    def __init__(
        self,
        nodes: Sequence[GraphNode],
        *,
        outputs: Sequence[str] | None = None,
    ) -> None:
        if not nodes:
            raise ValueError("ComputationGraph requires at least one node.")

        self._nodes: Dict[str, GraphNode] = {}
        for node in nodes:
            if node.name in self._nodes:
                raise ValueError(f"Duplicate graph node: {node.name}")
            self._nodes[node.name] = node

        self._order = self._topological_order(nodes)
        self.outputs = list(outputs) if outputs is not None else [self._order[-1]]
        for out in self.outputs:
            if out not in self._nodes:
                raise ValueError(f"Unknown output node: {out}")
        self.required_inputs = frozenset(self._collect_inputs())

    def nodes_in_order(self) -> Iterable[GraphNode]:
        for name in self._order:
            yield self._nodes[name]

    def last_use_map(self) -> Dict[str, List[str]]:
        """
        Compute, for each node in execution order, the list of cache entries
        whose last consumer is this node.

        The returned mapping associates ``consumer_name`` with the list of
        dependency names (either other node names or graph input names) whose
        last consumer (in topological order) is ``consumer_name``. After the
        consumer executes, those entries can be safely evicted from the cache.

        Returns
        -------
        Dict[str, List[str]]
            Mapping from node name to list of cache keys that are no longer
            needed after the node finishes. Graph outputs are never scheduled
            for eviction. User-provided graph inputs are included, so callers
            should filter them out if they must remain alive.
        """
        order_index: Dict[str, int] = {name: i for i, name in enumerate(self._order)}
        last_consumer: Dict[str, str] = {}
        for node in self._nodes.values():
            consumer_idx = order_index[node.name]
            for dep in node.inputs.values():
                if isinstance(dep, GraphInput):
                    key = dep.name
                elif isinstance(dep, str):
                    key = dep
                else:
                    continue
                existing = last_consumer.get(key)
                if existing is None or order_index.get(existing, -1) < consumer_idx:
                    last_consumer[key] = node.name

        outputs = set(self.outputs)
        evict: Dict[str, List[str]] = {}
        for key, consumer in last_consumer.items():
            if key in outputs:
                continue
            evict.setdefault(consumer, []).append(key)
        return evict

    def _collect_inputs(self) -> Set[str]:
        req: Set[str] = set()
        node_names = set(self._nodes)
        for node in self._nodes.values():
            for dep in node.inputs.values():
                if isinstance(dep, GraphInput):
                    req.add(dep.name)
                elif hasattr(dep, "message_type"):
                    # MessageSource - not a required input (comes from messages)
                    pass
                elif dep not in node_names:
                    raise ValueError(f"Node {node.name} depends on unknown node {dep!r}")
        return req

    def _topological_order(self, nodes: Sequence[GraphNode]) -> List[str]:
        deps: Dict[str, Set[str]] = {node.name: set() for node in nodes}
        for node in nodes:
            for dep in node.inputs.values():
                if isinstance(dep, GraphInput):
                    continue
                if hasattr(dep, "message_type"):
                    # MessageSource - not a node dependency
                    continue
                deps[node.name].add(dep)

        ready = [name for name, parents in deps.items() if not parents]
        order: List[str] = []
        while ready:
            name = ready.pop()
            order.append(name)
            for child in nodes:
                if name in deps[child.name]:
                    deps[child.name].remove(name)
                    if not deps[child.name]:
                        ready.append(child.name)

        if len(order) != len(nodes):
            raise ValueError("ComputationGraph contains a cycle; cannot determine order.")
        return order


__all__ = ["ComputationGraph", "GraphInput", "GraphNode", "graph_input"]
