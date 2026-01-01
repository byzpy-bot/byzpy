# Examples Guide

This folder contains end-to-end scripts that demonstrate the distributed training
primitives provided by ByzPy. All examples are self-contained and can be
invoked directly with `python` once dependencies are installed and the working
copy is in the project root.

Below is the status of each example as verified in this environment, along with
notes on how to run them yourself.

> **Note**
> Running the examples downloads MNIST the first time, so the initial launch is
a bit slower. Subsequent runs reuse the cached data under `./data`.

## Peer-to-Peer (P2P) examples

Each P2P script spins up a set of *honest* and optional *Byzantine* nodes, drives
the P2P training loop, and prints evaluation metrics every 50 rounds.

| Example | Command | Status |
|---------|---------|--------|
| Thread backend | `P2P_ROUNDS=1 P2P_HONEST=1 P2P_BYZ=0 P2P_BATCH=16 python examples/p2p/thread/mnist.py` | ✅ Verified (runs end to end) |
| Process backend | `P2P_ROUNDS=1 P2P_HONEST=1 P2P_BYZ=0 P2P_BATCH=16 python examples/p2p/process/mnist.py` | ✅ Verified |
| Heterogeneous backends | `P2P_ROUNDS=1 P2P_HONEST=1 P2P_BYZ=0 P2P_BATCH=16 python examples/p2p/heterogeneous/mnist.py` | ✅ Verified |
| Decentralized demo | `python examples/p2p/decentralized_demo.py` | ✅ Local demo (no actors) |
| Remote backends | see below | ⚠️ Requires remote actor server |

### P2P Remote example

Remote runs require one or more long-lived actor servers listening on the ports
listed in `P2P_REMOTES` (defaults to `tcp://127.0.0.1:29000`). In this repo we
provide a helper script that sets up the import path for the remote worker:

```bash
# Terminal 1 – start a remote actor server (run once per remote target)
python examples/p2p/remote/server.py --host 0.0.0.0 --port 29000
```

At the moment the helper script has the port hard-coded to `29001`. Start the
server once, then in a second terminal run the training script pointing at it:

```bash
# Terminal 2 – run the client
P2P_REMOTES=tcp://127.0.0.1:29001 \
P2P_ROUNDS=1 P2P_HONEST=1 P2P_BYZ=0 P2P_BATCH=16 \
python examples/p2p/remote/mnist.py
```

If you prefer a different port, edit `examples/p2p/remote/server.py` and the
`P2P_REMOTES` environment variable accordingly.

## Parameter-Server (PS) examples

These scripts instantiate a single parameter server and a configurable set of
worker nodes. Just like the P2P examples, all nodes are constructed on top of
actor pools.

| Example | Command | Status |
|---------|---------|--------|
| Thread backend | `PS_ROUNDS=1 PS_HONEST=1 PS_BYZ=0 PS_BATCH=16 python examples/ps/thread/mnist.py` | ✅ Verified |
| Process backend | `PS_ROUNDS=1 PS_HONEST=1 PS_BYZ=0 PS_BATCH=16 python examples/ps/process/mnist.py` | ✅ Verified |
| Heterogeneous backends | `PS_ROUNDS=1 PS_HONEST=1 PS_BYZ=0 PS_BATCH=16 python examples/ps/heterogenous/mnist.py` | ✅ Verified |
| Decentralized demo | `python examples/ps/decentralized_demo.py` | ✅ Local demo (no actors) |
| Remote backends | see below | ⚠️ Requires remote actor server |

### PS Remote example

Just like the P2P remote example, the PS remote variant needs one or more remote
actor servers. Start a server on every host/port listed in `PS_REMOTES`
(defaults to `tcp://127.0.0.1:29000`):

```bash
# Terminal 1 – start remote actor server(s)
python examples/ps/remote/server.py --host 0.0.0.0 --port 29000
```

Then in a second terminal run the training script:

```bash
# Terminal 2 – run the PS client
PS_REMOTES=tcp://127.0.0.1:29000 \
PS_ROUNDS=1 PS_HONEST=1 PS_BYZ=0 PS_BATCH=16 \
python examples/ps/remote/mnist.py
```

If the port is already in use you can adjust both the server script and the
`PS_REMOTES` environment variable to match a free port.

## Environment variables

All examples honour the following optional knobs:

- `*_ROUNDS`: number of training rounds (default 200)
- `*_HONEST`: number of honest workers (default 4)
- `*_BYZ`: number of Byzantine workers (default 1 for P2P, 0 for PS)
- `*_BATCH`: local batch size (default 64)
- `{P2P,PS}_DATA`: dataset root (defaults to `./data`)

Remote examples also honour `{P2P,PS}_REMOTES`, a comma-separated list of
`tcp://host:port` addresses.

## GPU/UCX examples

The heterogeneous P2P example attempts to create GPU and UCX actors if available.
If you do not have UCX or CUDA installed, trim the `P2P_BACKENDS` environment
variable to only include the backends you can support.

## Distributed Training Example

A comprehensive example demonstrating training across multiple machines:

| Example | Description | Status |
|---------|-------------|--------|
| Distributed PS | ParameterServer training with nodes on separate machines | ✅ New |

### Distributed Example

The distributed example shows how to run ByzPy training with nodes distributed across multiple machines:

1. **Start remote actor servers** on each machine:
   ```bash
   # On each machine
   python examples/distributed/server.py --host 0.0.0.0 --port 29000
   ```

2. **Run the training script** from a client machine:
   ```bash
   python examples/distributed/mnist.py \
       --remote-hosts tcp://machine1:29000,tcp://machine2:29000,tcp://machine3:29000 \
       --num-honest 3 --num-byz 1 --rounds 50
   ```

See `examples/distributed/README.md` for detailed setup instructions and examples.

## Summary

- ✅ Thread-based P2P / PS examples
- ✅ Process-based P2P / PS examples
- ✅ Heterogeneous P2P / PS examples (CPU-safe defaults)
- ⚠️ Remote examples require running `examples/*/remote/server.py` separately
- ✅ Distributed example with comprehensive documentation

When in doubt, start with the thread backend: it requires no extra setup and
runs entirely in the current Python process.
