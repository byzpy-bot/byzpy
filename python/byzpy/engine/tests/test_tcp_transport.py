from __future__ import annotations

import time

from byzpy.engine.transport.tcp_simple import TcpMailbox, send_message


def test_tcp_mailbox_send_recv():
    mbox = TcpMailbox()
    try:
        addr = ("127.0.0.1", mbox.port)
        send_message(addr, {"hello": "world"})
        msg = mbox.recv(timeout=1.0)
        assert msg == {"hello": "world"}
    finally:
        mbox.close()


def test_tcp_mailbox_multiple_messages():
    mbox = TcpMailbox()
    try:
        addr = ("127.0.0.1", mbox.port)
        for i in range(3):
            send_message(addr, i)
        seen = []
        start = time.time()
        while len(seen) < 3 and (time.time() - start) < 2.0:
            seen.append(mbox.recv(timeout=1.0))
        assert sorted(seen) == [0, 1, 2]
    finally:
        mbox.close()
