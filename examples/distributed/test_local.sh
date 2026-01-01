#!/bin/bash
# Quick local test script for distributed ByzPy training
# This demonstrates the distributed architecture on a single machine

set -e

echo "=========================================="
echo "ByzPy Distributed Training - Local Test"
echo "=========================================="
echo ""
echo "This script starts 3 remote actor servers on different ports"
echo "and runs a distributed training example."
echo ""
echo "Press Ctrl+C to stop all servers."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    pkill -f "examples/distributed/server.py" || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start servers in background
echo -e "${GREEN}Starting remote actor servers...${NC}"
python examples/distributed/server.py --host 127.0.0.1 --port 29000 > /tmp/byzpy_server_29000.log 2>&1 &
SERVER1_PID=$!
sleep 1

python examples/distributed/server.py --host 127.0.0.1 --port 29001 > /tmp/byzpy_server_29001.log 2>&1 &
SERVER2_PID=$!
sleep 1

python examples/distributed/server.py --host 127.0.0.1 --port 29002 > /tmp/byzpy_server_29002.log 2>&1 &
SERVER3_PID=$!
sleep 2

echo -e "${GREEN}Servers started (PIDs: $SERVER1_PID, $SERVER2_PID, $SERVER3_PID)${NC}"
echo ""

# Run training
echo -e "${GREEN}Starting distributed training...${NC}"
python examples/distributed/mnist.py \
    --remote-hosts tcp://127.0.0.1:29000,tcp://127.0.0.1:29001,tcp://127.0.0.1:29002 \
    --num-honest 3 \
    --num-byz 1 \
    --rounds 10 \
    --batch-size 64 \
    --lr 0.05 \
    --f 1 \
    --eval-interval 5

# Cleanup
cleanup
