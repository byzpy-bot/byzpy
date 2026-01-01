# Developer Guide

This page captures the standard development workflow.

## Environment Setup

```bash
git clone https://github.com/Byzpy/byzpy.git
cd byzpy
python -m venv .venv
source .venv/bin/activate
pip install -e python[dev]
pip install -e python[gpu]            # optional CUDA extras
pip install -r docs/requirements.txt  # to build docs
```

## Linting & Tests

```bash
cd python
pytest
pytest --cov=byzpy --cov-report=term-missing   # coverage goal ≥ 80%
```

## Documentation

### Building Documentation

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Build HTML documentation:

```bash
sphinx-build -b html docs/source docs/_build/html
```

Or from the repository root:

```bash
sphinx-build -b html docs/source docs/_build/html
```

### Viewing Documentation Locally

**Option 1: Open HTML file directly**
```bash
open docs/_build/html/index.html        # macOS
xdg-open docs/_build/html/index.html    # Linux
start docs/_build/html/index.html       # Windows
```

**Option 2: Use HTTP server (recommended)**
```bash
cd docs/_build/html
python -m http.server 8000
# Then visit http://localhost:8000 in your browser
```

**Option 3: Live reload during development**
```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/_build/html
```

This automatically rebuilds when files change and serves at `http://127.0.0.1:8000`.

### Documentation Contribution

See [docs/CONTRIBUTING.md](../../docs/CONTRIBUTING.md) for detailed guidelines on:
- Documentation structure
- Writing style guidelines
- How to add new documentation
- Documentation review checklist

### Documentation Quality Checks

Before submitting documentation changes:

```bash
# Build with warnings as errors
sphinx-build -W -b html docs/source docs/_build/html

# Check for broken links
sphinx-build -b linkcheck docs/source docs/_build/linkcheck

# Verify all examples work
pytest docs/source/  # if examples are in test files
```

## Submitting Changes

1. Fork + branch.
2. Update docs/tests when touching public APIs.
3. Run `pytest` and `sphinx-build`.
4. Open a PR with a clear summary, test output, and coverage notes.
