# Distributed ByzPy Training Examples

This directory contains examples demonstrating how to run ByzPy training with nodes distributed across multiple machines.

## Architecture

- **Remote Actor Servers**: Each machine runs a remote actor server that hosts ByzPy nodes
- **Client Script**: The training script runs on a client machine and connects to remote servers
- **Communication**: Nodes communicate via TCP using `RemoteActorBackend`
- **Aggregation**: Gradient aggregation happens on the client machine

## Quick Start

### Local Testing (Single Machine)

**Yes, running on localhost validates the distributed architecture!** You can test the distributed setup on a single machine by running multiple servers on different ports. This proves:
- ✅ TCP communication works (same protocol as remote machines)
- ✅ RemoteActorBackend abstraction works correctly
- ✅ Node spawning and lifecycle management work
- ✅ The architecture is ready to extend to multiple physical machines

**Quick test script:**
```bash
# Run the automated test script
./examples/distributed/test_local.sh
```

**Manual local testing:**
```bash
# Terminal 1: Start server on port 29000
python examples/distributed/server.py --host 127.0.0.1 --port 29000

# Terminal 2: Start server on port 29001
python examples/distributed/server.py --host 127.0.0.1 --port 29001

# Terminal 3: Start server on port 29002
python examples/distributed/server.py --host 127.0.0.1 --port 29002

# Terminal 4: Run training client
python examples/distributed/mnist.py \
    --remote-hosts tcp://127.0.0.1:29000,tcp://127.0.0.1:29001,tcp://127.0.0.1:29002 \
    --num-honest 3 --num-byz 1 --rounds 50
```

**Note**: Local testing validates the architecture. Real multi-machine deployment may have additional considerations (network latency, firewall rules, different machine configurations), but the core protocol and architecture are the same.

### Multi-Machine Setup

On each machine that will host nodes, start a remote actor server:

```bash
# On machine 1 (e.g., 192.168.1.10)
python examples/distributed/server.py --host 0.0.0.0 --port 29000

# On machine 2 (e.g., 192.168.1.11)
python examples/distributed/server.py --host 0.0.0.0 --port 29000

# On machine 3 (e.g., 192.168.1.12)
python examples/distributed/server.py --host 0.0.0.0 --port 29000
```

**Note**: Use `0.0.0.0` to bind to all interfaces, allowing connections from other machines. Use a specific IP if you want to restrict access.

Then run the client script:

```bash
python examples/distributed/mnist.py \
    --remote-hosts tcp://192.168.1.10:29000,tcp://192.168.1.11:29000,tcp://192.168.1.12:29000 \
    --num-honest 3 \
    --num-byz 1 \
    --rounds 50 \
    --batch-size 64 \
    --lr 0.05 \
    --f 1 \
    --seed 42
```

## Configuration

### Server Options

- `--host`: Host to bind to (default: `0.0.0.0` for all interfaces)
- `--port`: Port to listen on (default: `29000`)

### Client Options

- `--remote-hosts`: **Required**. Comma-separated list of remote server addresses
  - Format: `tcp://HOST:PORT`
  - Example: `tcp://192.168.1.10:29000,tcp://192.168.1.11:29000`
- `--num-honest`: Number of honest nodes (default: `3`)
- `--num-byz`: Number of Byzantine nodes (default: `1`)
- `--rounds`: Number of training rounds (default: `50`)
- `--batch-size`: Batch size per node (default: `64`)
- `--lr`: Learning rate (default: `0.05`)
- `--f`: MultiKrum fault tolerance parameter (default: `1`)
- `--q`: MultiKrum parameter q (defaults to `n - f - 1`)
- `--chunk-size`: MultiKrum chunk size (default: `32`)
- `--seed`: Random seed (default: `42`)
- `--data-root`: MNIST data directory (default: `./data`)
- `--eval-interval`: Evaluate model every N rounds (default: `10`, set to `0` to disable)

## Node Distribution

Nodes are distributed across remote hosts using round-robin assignment:
- Node 0 → Remote host 0
- Node 1 → Remote host 1
- Node 2 → Remote host 2
- Node 3 → Remote host 0 (wraps around)
- ...

If you have fewer remote hosts than nodes, nodes will be distributed across the available hosts.

## Requirements

### All Machines

- Python 3.11+ (recommended: same version on all machines)
- ByzPy installed: `pip install -e .` (or install from PyPI)
- PyTorch and torchvision installed
- Network connectivity between machines
- Firewall rules allowing TCP connections on the specified ports

### Remote Machines (Running Servers)

- ByzPy installed
- PyTorch and torchvision installed
- Access to MNIST dataset (will download automatically if needed)

### Client Machine (Running Training Script)

- ByzPy installed
- PyTorch and torchvision installed
- Access to MNIST dataset (will download automatically if needed)
- Network access to all remote machines

## Example: 3-Machine Setup

### Machine 1 (192.168.1.10)
```bash
# Terminal 1: Start server
python examples/distributed/server.py --host 0.0.0.0 --port 29000
```

### Machine 2 (192.168.1.11)
```bash
# Terminal 1: Start server
python examples/distributed/server.py --host 0.0.0.0 --port 29000
```

### Machine 3 (192.168.1.12)
```bash
# Terminal 1: Start server
python examples/distributed/server.py --host 0.0.0.0 --port 29000

# Terminal 2: Run training (or run from a 4th machine)
python examples/distributed/mnist.py \
    --remote-hosts tcp://192.168.1.10:29000,tcp://192.168.1.11:29000,tcp://192.168.1.12:29000 \
    --num-honest 3 \
    --num-byz 1 \
    --rounds 50
```

## Troubleshooting

### Connection Refused

- Ensure remote servers are running before starting the client
- Check firewall rules allow TCP connections on the specified ports
- Verify the host addresses are correct and reachable

### Import Errors on Remote Machines

- Ensure ByzPy is installed on all remote machines
- Check that Python paths are set correctly
- Verify all dependencies (PyTorch, torchvision) are installed

### Slow Performance

- Network latency affects performance - use machines on the same network when possible
- Consider using UCX backend (`ucx://host:port`) for GPU clusters with InfiniBand
- Reduce batch size or number of rounds for testing

## Advanced: Custom Ports

If you need to use different ports on different machines:

```bash
# Machine 1: port 29000
python examples/distributed/server.py --host 0.0.0.0 --port 29000

# Machine 2: port 29001
python examples/distributed/server.py --host 0.0.0.0 --port 29001

# Machine 3: port 29002
python examples/distributed/server.py --host 0.0.0.0 --port 29002

# Client
python examples/distributed/mnist.py \
    --remote-hosts tcp://192.168.1.10:29000,tcp://192.168.1.11:29001,tcp://192.168.1.12:29002 \
    --num-honest 3 --num-byz 1 --rounds 50
```

## Security Considerations

- The remote actor servers accept connections from any host by default (`0.0.0.0`)
- For production deployments, consider:
  - Using firewall rules to restrict access
  - Binding to specific IP addresses instead of `0.0.0.0`
  - Using VPN or private networks
  - Implementing authentication (not currently supported)
