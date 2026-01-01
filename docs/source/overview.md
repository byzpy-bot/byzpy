# Architecture Overview

ByzPy is organized into three cooperating tiers that work together to provide
Byzantine-robust distributed learning:

1. **Application Layer** – user-facing APIs such as aggregators, attacks,
   pre-aggregators, parameter-server helpers, peer-to-peer training, and the
   honest/byzantine node classes.
2. **Scheduling Layer** – the computation-graph primitives (`GraphInput`,
   `GraphNode`, `Operator`, `ComputationGraph`) plus the `NodeScheduler` and
   `ActorPool` that orchestrate execution.
3. **Actor Layer** – lightweight worker backends (threads, processes, GPUs,
   remote TCP/UCX actors) that actually run tasks.

All higher-level concepts build on top of these tiers, so understanding their
responsibilities makes it easier to extend the framework.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  Aggregators │ Attacks │ Pre-Aggregators │ PS/P2P Helpers  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Scheduling Layer                           │
│  ComputationGraph │ NodeScheduler │ ActorPool │ Operators   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Actor Layer                             │
│  Thread │ Process │ GPU │ Remote (TCP/UCX) │ Channels      │
└─────────────────────────────────────────────────────────────┘
```

## 1. Application Layer

The application layer exposes the APIs most users interact with. This is where
you define what computations happen, but not how or where they execute.

### Aggregators

Aggregators (`byzpy.aggregators.*`) combine multiple gradient vectors into a
single aggregated gradient. They are the core of Byzantine-robust learning:

- **Coordinate-wise aggregators**: Median, Trimmed Mean, Mean of Medians
- **Geometric aggregators**: Krum, Multi-Krum, Geometric Median, MDA, MoNNA, SMEA
- **Norm-wise aggregators**: Center Clipping, CGE, CAF

Each aggregator can be dropped into a node pipeline and supports optional
parallel execution via subtasks.

### Attacks

Attacks (`byzpy.attacks.*`) simulate malicious behavior by generating
adversarial gradients:

- **Empire**: Scaled mean of honest gradients
- **Sign Flip**: Negated gradients
- **Label Flip**: Gradients computed with flipped labels
- **Little**: Small random perturbations
- **Gaussian**: Gaussian noise injection
- **Inf**: Extreme gradient values
- **Mimic**: Mimics honest gradients

Attacks are used in simulations to test aggregator robustness.

### Pre-Aggregators

Pre-aggregators (`byzpy.pre_aggregators.*`) transform vectors before
aggregation:

- **Bucketing**: Groups vectors into buckets and averages within each
- **Clipping**: Clips vectors to a maximum norm
- **ARC**: Adaptive Robust Clustering
- **NNM**: Nearest-Neighbor Mixing

### Training Orchestrators

- **Parameter Server** (`byzpy.engine.parameter_server.ps.ParameterServer`):
  Centralized training where a server aggregates gradients from all nodes
- **Peer-to-Peer** (`byzpy.engine.peer_to_peer.train.PeerToPeer`):
  Decentralized training where nodes communicate directly with neighbors

### Node Applications

- **HonestNodeApplication**: Manages honest node pipelines (gradient computation,
  aggregation)
- **ByzantineNodeApplication**: Manages Byzantine node pipelines (attack
  generation)

These components rely on the scheduling layer to wire their computation graphs,
but they define the semantics of "what" gets computed.

## 2. Scheduling Layer (Engine Overview)

The scheduling layer combines graphs and actors to orchestrate execution. It
decides "when" things run and how to parallelize them.

### Graph Module

The graph module provides declarative computation graph primitives:

- **`GraphInput` / `graph_input()`** – Typed handles that identify data the
  application must supply. These are placeholders in the graph that get
  filled with actual values at runtime.
- **`GraphNode`** – A single operator plus its named inputs. Nodes can depend
  on graph inputs or other nodes' outputs.
- **`Operator`** – Reusable compute primitive. All aggregators, attacks, and
  pre-aggregators are operators. Operators can support optional subtasks for
  distributed execution.
- **`ComputationGraph`** – Validates nodes, enforces topological order, tracks
  required inputs/outputs. Ensures the graph is a valid DAG.

### Scheduling & Execution

- **`ActorPool`** – Manages a pool of worker actors from one or more backend
  configurations. Workers can be threads, processes, GPUs, or remote actors.
  The pool routes tasks to appropriate workers based on capabilities.
- **`NodeScheduler`** – Walks the DAG in topological order, hydrates inputs,
  and runs each operator. When operators support subtasks, the scheduler
  fans out work across the actor pool and fans in results.
- **`MessageAwareNodeScheduler`** – Extended scheduler that supports
  message-driven execution, allowing nodes to wait for messages from other
  nodes before proceeding.

### Execution Flow

1. Build a `ComputationGraph` with nodes and their dependencies
2. Create a `NodeScheduler` with an optional `ActorPool`
3. Call `await scheduler.run({...})` with input values
4. The scheduler:
   - Validates all required inputs are present
   - Executes nodes in topological order
   - Routes operators to the actor pool for parallel execution
   - Returns the requested outputs

Example:

```python
from byzpy.engine.graph.ops import make_single_operator_graph
from byzpy.engine.graph.scheduler import NodeScheduler
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian

# Create graph
aggregator = CoordinateWiseMedian()
graph = make_single_operator_graph(
    node_name="agg",
    operator=aggregator,
    input_keys=("gradients",)
)

# Create pool and scheduler
pool = ActorPool([ActorPoolConfig(backend="thread", count=4)])
await pool.start()
scheduler = NodeScheduler(graph, pool=pool)

# Execute
results = await scheduler.run({"gradients": gradients})
aggregated = results["agg"]
```

## 3. Actor Layer

The actor subsystem (`byzpy.engine.actor`) manages workers on different
backends. It is responsible for "where" work runs.

### Actor Backends

| Backend | Description | Typical Use | Communication |
|---------|-------------|-------------|---------------|
| `ThreadActorBackend` | Dedicated thread per worker in the current process. | Low-latency CPU tasks, I/O-bound operations. | Shared memory |
| `ProcessActorBackend` | Separate Python process per worker. | CPU-heavy work needing true parallelism, avoiding GIL. | Multiprocessing pipes |
| `GPUActorBackend` | CUDA-aware workers for local GPU execution. | GPU-accelerated computations. | CUDA memory |
| `RemoteActorBackend` | TCP-based remote actors. | Scaling beyond one machine, distributed training. | TCP sockets |
| `UCXRemoteActorBackend` | UCX-based remote actors. | High-speed GPU clusters, RDMA communication. | UCX/InfiniBand |

### Supporting Infrastructure

- **`ActorRef`** – Async proxy that turns attribute access into remote calls.
  Provides a transparent interface for calling methods on remote actors.
- **Channels** – Mailbox-style queues for streaming data between actors.
  Supports point-to-point and broadcast communication patterns.
- **Capability routing** – The `ActorPool` routes tasks to workers that satisfy
  declared capabilities (e.g., `"gpu"`, `"cpu"`). Tasks can specify affinity
  hints to prefer certain workers.

### Actor Pool Architecture

The `ActorPool` manages multiple workers and routes subtasks to them:

- Workers are created lazily when the pool starts
- Tasks are queued and dispatched to available workers
- Workers can be from different backends (heterogeneous pools)
- Task affinity hints route tasks to preferred workers
- Channels enable communication between workers

The actor layer is responsible for "where" work runs, while the scheduling layer
decides "when" it runs and the application layer defines "what" runs.

## Message-Driven Execution

ByzPy supports message-driven execution for decentralized training. Nodes can:

- Wait for messages from other nodes before proceeding
- Broadcast messages to all neighbors
- Send messages to specific nodes
- Use topology-aware routing

This enables fully asynchronous peer-to-peer training where nodes progress
independently based on received messages.

## Decentralized Scheduling

Each node can run its own scheduler independently:

- Nodes execute computation graphs based on local state
- Messages trigger graph execution when dependencies are met
- No central coordinator required
- True parallelism across nodes

This architecture supports both centralized (parameter server) and
decentralized (peer-to-peer) training patterns.
