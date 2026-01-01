from __future__ import annotations
from typing import Any, Dict, Optional, Protocol

from .channels import ChannelRef, Endpoint


class ActorBackend(Protocol):
    async def start(self) -> None: ...
    async def construct(self, cls_or_factory: Any, *, args: tuple, kwargs: dict) -> None: ...
    async def call(self, method: str, *args, **kwargs) -> Any: ...
    async def close(self) -> None: ...
    async def get_endpoint(self) -> Endpoint: ...
    async def chan_open(self, name: str) -> Endpoint: ...
    async def chan_put(self, *, from_ep: Endpoint, to_ep: Endpoint, name: str, payload: Any) -> None: ...
    async def chan_get(self, *, ep: Endpoint, name: str, timeout: Optional[float]) -> Any: ...


class ActorRef:
    """
    Thin async proxy: attribute access becomes an async RPC to backend.call(name, ...).
    Also exposes channel helpers.
    """
    def __init__(self, backend: ActorBackend):
        self._backend = backend

    def __getattr__(self, name: str):
        async def _remote(*args, **kwargs):
            return await self._backend.call(name, *args, **kwargs)
        return _remote

    async def __aenter__(self):
        await self._backend.start()
        return self

    async def __aexit__(self, *exc):
        await self._backend.close()
        return False

    async def open_channel(self, name: str) -> ChannelRef:
        ep = await self._backend.chan_open(name)
        return ChannelRef(self._backend, ep, name)

    async def endpoint(self) -> Endpoint:
        return await self._backend.get_endpoint()
