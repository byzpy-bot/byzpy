# Getting Started

This guide walks through installing ByzPy, verifying your setup, running
examples, and understanding the main concepts.

## Installation

### System Requirements

- Python 3.9 or higher
- pip (Python package manager)
- For GPU support: CUDA-capable GPU and CUDA toolkit (optional)
- For UCX support: UCX library and UCX Python bindings (optional)

### PyPI Installation

Install the base package:

```bash
pip install byzpy
```

Install with optional GPU support (CUDA/UCX):

```bash
pip install "byzpy[gpu]"
```

Install with development dependencies (pytest, coverage, etc.):

```bash
pip install "byzpy[dev]"
```

Install all extras:

```bash
pip install "byzpy[gpu,dev]"
```

### Installation from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/Byzpy/byzpy.git
cd byzpy
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e python[dev]
pip install -e python[gpu]      # optional CUDA extras
```

### Verify Installation

Check that ByzPy is installed correctly:

```bash
byzpy version
```

Run the diagnostic tool to check your environment:

```bash
byzpy doctor
```

For JSON output:

```bash
byzpy doctor --format json
```

This will show:
- Python version and platform
- PyTorch availability and CUDA support
- CuPy availability (for GPU operations)
- UCX availability (for high-speed GPU communication)

## Quick Start Examples

### Parameter Server Training

Run a simple parameter server example with threaded workers:

```bash
python examples/ps/thread/mnist.py
```

This trains a CNN on MNIST using a parameter server architecture with
honest and Byzantine nodes.

### Peer-to-Peer Training

For peer-to-peer training with heterogeneous backends, start a UCX server
in one terminal:

```bash
python examples/p2p/heterogeneous/server.py
```

Then run the training client in another terminal:

```bash
python examples/p2p/heterogeneous/mnist.py
```

### Decentralized Process-Based Training

Run fully decentralized training where each node runs in a separate process:

```bash
python examples/p2p/decentralized_process_mnist.py
```

## Basic Usage

### Using Aggregators

```python
from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
import torch

# Create aggregator
aggregator = CoordinateWiseMedian(chunk_size=4096)

# Aggregate gradients
gradients = [torch.randn(1000) for _ in range(10)]
result = aggregator.aggregate(gradients)
print(f"Aggregated gradient shape: {result.shape}")
```

### Using Attacks

```python
from byzpy.attacks.empire import EmpireAttack
import torch

# Create attack
attack = EmpireAttack(scale=-1.0)

# Generate malicious gradient
honest_grads = [torch.randn(100) for _ in range(5)]
malicious = attack.apply(honest_grads=honest_grads)
```

### Using Pre-Aggregators

```python
from byzpy.pre_aggregators.bucketing import Bucketing
import torch

# Create pre-aggregator
preagg = Bucketing(bucket_size=4)

# Pre-aggregate vectors
vectors = [torch.randn(100) for _ in range(10)]
buckets = preagg.pre_aggregate(vectors)
print(f"Number of buckets: {len(buckets)}")
```

## Command-Line Interface

ByzPy provides a CLI tool for common tasks:

```bash
# Check version
byzpy version

# Run diagnostics
byzpy doctor

# List available aggregators
byzpy list aggregators

# List available attacks
byzpy list attacks

# List available pre-aggregators
byzpy list pre-aggregators

# JSON output
byzpy list aggregators --format json
```

## Running ByzPy Locally

### Parameter Server Examples

#### Thread Backend (Single Machine)

```bash
python examples/ps/thread/mnist.py
```

This runs a parameter server with threaded workers on a single machine.

#### Process Backend

```bash
python examples/ps/process/mnist.py
```

Uses separate processes for each worker, enabling true parallelism.

#### Remote Backend (Multiple Machines)

First, start remote actor servers on each machine:

```bash
# On machine 1
python examples/ps/remote/server.py --host 0.0.0.0 --port 29000

# On machine 2
python examples/ps/remote/server.py --host 0.0.0.0 --port 29001
```

Then run the training client:

```bash
python examples/distributed/mnist.py \
    --remote-hosts tcp://machine1:29000,tcp://machine2:29001 \
    --num-honest 2 --num-byz 1 --rounds 50
```

### Peer-to-Peer Examples

#### Thread Backend

```bash
python examples/p2p/thread/mnist.py
```

#### Process Backend

```bash
python examples/p2p/process/mnist.py
```

#### Heterogeneous Backends

Start a UCX server:

```bash
python examples/p2p/heterogeneous/server.py
```

Then run the client:

```bash
python examples/p2p/heterogeneous/mnist.py
```

#### Decentralized Process-Based

Each node runs in a separate OS process:

```bash
python examples/p2p/decentralized_process_mnist.py
```

### Configuration Options

Many examples support environment variables for configuration:

```bash
# Set number of training rounds
P2P_ROUNDS=100 python examples/p2p/thread/mnist.py

# Set number of honest and Byzantine nodes
P2P_HONEST=5 P2P_BYZ=2 python examples/p2p/thread/mnist.py

# Set batch size
P2P_BATCH=128 python examples/p2p/thread/mnist.py
```

### Running Benchmarks

ByzPy includes benchmarks for aggregator performance:

```bash
cd benchmarks
python pytorch/parameter_server_actor_pool.py \
    --num-honest 10 --num-byz 2 --rounds 100
```

Generate benchmark plots:

```bash
python plot_benchmarks.py
```

## Viewing Documentation Locally

After building the documentation (see [Building Documentation](../../docs/BUILDING.md)), you can view it locally:

### Quick Start

1. **Build the documentation:**
   ```bash
   sphinx-build -b html docs/source docs/_build/html
   ```

2. **View in browser:**
   ```bash
   # Option 1: Open HTML file
   open docs/_build/html/index.html        # macOS
   xdg-open docs/_build/html/index.html    # Linux

   # Option 2: HTTP server (recommended)
   cd docs/_build/html
   python -m http.server 8000
   # Then visit http://localhost:8000
   ```

3. **For live reload during development:**
   ```bash
   pip install sphinx-autobuild
   sphinx-autobuild docs/source docs/_build/html
   ```

See [Building and Viewing Documentation](../../docs/BUILDING.md) for detailed instructions.

## Contributing to Documentation

We welcome documentation contributions! See the [Documentation Contribution Guide](../../docs/CONTRIBUTING.md) for:
- Documentation structure and style
- How to add new documentation
- Code example guidelines
- Review checklist

Quick start for documentation contributions:
1. Make your changes to docstrings or Markdown files
2. Build and preview: `sphinx-build -b html docs/source docs/_build/html`
3. Test your changes locally
4. Submit a pull request

## Next Steps

- Read the [Architecture Overview](overview.md) to understand ByzPy's design
- Explore the [API Reference](api_reference.md) for detailed API documentation
- Check out the [Examples](../examples/README.md) for more complex use cases
- See the [Installation Guide](installation.md) for detailed setup instructions
- See the [Developer Guide](developer_guide.md) for contributing
- Read [Building Documentation](../../docs/BUILDING.md) for building docs locally
- Read [Documentation Contribution Guide](../../docs/CONTRIBUTING.md) to contribute

## Extending ByzPy

### Creating Custom Aggregators

Subclass `byzpy.aggregators.base.Aggregator` and implement `aggregate`:

```python
from byzpy.aggregators.base import Aggregator

class MyAggregator(Aggregator):
    name = "my-aggregator"

    def aggregate(self, gradients):
        # Your aggregation logic here
        return aggregated_gradient
```

### Creating Custom Attacks

Subclass `byzpy.attacks.base.Attack`, set the `uses_*` flags, and implement `apply`:

```python
from byzpy.attacks.base import Attack

class MyAttack(Attack):
    uses_honest_grads = True

    def apply(self, *, honest_grads=None, **kwargs):
        # Your attack logic here
        return malicious_gradient
```

### Creating Custom Pre-Aggregators

Subclass `byzpy.pre_aggregators.base.PreAggregator` and implement `pre_aggregate`:

```python
from byzpy.pre_aggregators.base import PreAggregator

class MyPreAggregator(PreAggregator):
    name = "my-pre-aggregator"

    def pre_aggregate(self, xs):
        # Your transformation logic here
        return transformed_vectors
```

## Testing

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=byzpy --cov-report=term-missing
```

Run specific test files:

```bash
pytest python/byzpy/aggregators/tests/
```
