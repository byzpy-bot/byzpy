# Decentralized Scheduler

ByzPy supports fully decentralized execution where each node runs in its own
process and exchanges messages without a central orchestrator. This enables
true peer-to-peer training with independent node progress and message-driven
communication. The existing actor-based examples remain available for
different use cases.

## Key pieces
- `NodeRunner`: runs a user-provided step function and message handler in a
  separate process. Supports manual steps, auto-stepping, and message delivery.
- `NodeCluster`: minimal manager for multiple `NodeRunner` instances (start/stop,
  send messages, read state).
- `DecentralizedPeerToPeer`: spawns each P2P participant as its own `NodeRunner`
  and exchanges half-steps/attacks via the cluster. Honest and byzantine nodes
  implement the same p2p_* methods as before.
- `ParameterServerRunner`: similar process-backed runner for a simple PS loop;
  a `DecentralizedParameterServer` wrapper is provided for API parity.

## Examples
- `python examples/p2p/decentralized_demo.py`
- `python examples/ps/decentralized_demo.py`

These demos avoid actor backends entirely and use the process-based runners to
show independent node progress and message passing.

## Notes
- The decentralized execution path is production-ready and fully integrated.
- The actor-backed P2P/PS scripts remain available for different execution models.
- Runners can host a `NodeScheduler` + `ActorPool` inside a node process (see
  `test_node_runner.py`) for intra-node parallelism.
- The decentralized wrappers (`PeerToPeer`, `DecentralizedParameterServer`) are
  thin facades that delegate to the process-based runners.
