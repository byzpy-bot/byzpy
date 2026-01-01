#!/usr/bin/env python
"""
Remote Actor Server for Distributed ByzPy Training

This script starts a remote actor server that can host ByzPy nodes on a separate machine.
Run this script on each machine that will participate in distributed training.

Usage:
    # On machine 1 (e.g., 192.168.1.10)
    python examples/distributed/server.py --host 0.0.0.0 --port 29000

    # On machine 2 (e.g., 192.168.1.11)
    python examples/distributed/server.py --host 0.0.0.0 --port 29000

    # On machine 3 (e.g., 192.168.1.12)
    python examples/distributed/server.py --host 0.0.0.0 --port 29000

Then run the client script with:
    python examples/distributed/mnist.py \
        --remote-hosts tcp://192.168.1.10:29000,tcp://192.168.1.11:29000,tcp://192.168.1.12:29000 \
        --num-honest 3 --num-byz 1 --rounds 50
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from byzpy.engine.actor.backends.remote import start_actor_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a remote actor server for distributed ByzPy training."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (0.0.0.0 for all interfaces, or specific IP)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=29000,
        help="Port to listen on",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ByzPy Remote Actor Server")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Address: {args.host}:{args.port}")
    print("=" * 70)
    print("Server is ready to accept connections.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 70)

    try:
        asyncio.run(start_actor_server(host=args.host, port=args.port))
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
