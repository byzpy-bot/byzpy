from __future__ import annotations

import time

from byzpy.engine.node_runner import NodeRunner


def test_pump_once_advances_state():
    def step(state: dict) -> dict:
        state["count"] = state.get("count", 0) + 1
        return state

    def on_msg(state: dict, msg):
        msgs = state.get("msgs", [])
        msgs.append(msg)
        state["msgs"] = msgs
        return state

    r = NodeRunner(step, on_msg, init_state={})
    r.start()
    try:
        r.send_message("hello")
        r.pump_once()
        st = r.state()
        assert st["count"] == 1
        assert st["msgs"] == ["hello"]
    finally:
        r.stop()


def test_start_async_runs_continuously():
    def step(state: dict) -> dict:
        state["count"] = state.get("count", 0) + 1
        return state

    def on_msg(state: dict, msg):
        msgs = state.get("msgs", [])
        msgs.append(msg)
        state["msgs"] = msgs
        return state

    r = NodeRunner(step, on_msg, init_state={})
    r.start()
    try:
        r.start_async(interval_sec=0.02)
        r.send_message("hi")
        time.sleep(0.1)
        st = r.state()
        assert st["count"] >= 2
        assert "hi" in st.get("msgs", [])
    finally:
        r.stop()
