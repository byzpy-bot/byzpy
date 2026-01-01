from __future__ import annotations

import time

from byzpy.engine.transport.tcp import TcpTransport


def test_tcp_transport_loopback():
    recv = []
    transport = TcpTransport()
    transport.register("a", lambda msg: recv.append(("a", msg)))
    transport.register("b", lambda msg: recv.append(("b", msg)))
    try:
        transport.send("b", {"hi": "there"})
        start = time.time()
        while len(recv) < 1 and (time.time() - start) < 1.5:
            time.sleep(0.05)
        assert ("b", {"hi": "there"}) in recv
    finally:
        transport.close()
