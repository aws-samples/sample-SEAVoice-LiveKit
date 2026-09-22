"""Persistent-client, keep-alive aioboto3 Session for the ``aws.LLM`` Bedrock plugin.

The stock ``livekit-plugins-aws`` LLM opens and closes a fresh bedrock-runtime
client on every turn, and aiobotocore's default 12 s connection keep-alive is
shorter than a conversational turn gap — so each turn pays a fresh TLS handshake,
adding first-token latency. This ``aioboto3.Session`` subclass fixes both, using
only the plugin's public ``session=`` parameter (no monkeypatching):

  - its client context manager's ``__aexit__`` is a no-op, so the client and its
    connection pool survive across the plugin's per-turn ``async with``;
  - it rebuilds the per-call config as an ``AioConfig`` with a longer
    ``keepalive_timeout`` so the pooled socket survives the inter-turn gap.

An ``asyncio.Lock`` guards the one-time lazy open. The owner must call
``await session.aclose_all()`` on shutdown to close the pooled client.
"""

from __future__ import annotations

import asyncio
import logging

import aioboto3  # type: ignore
from aiobotocore.config import AioConfig  # type: ignore

logger = logging.getLogger("bedrock_session")

# Idle keep-alive for the pooled bedrock-runtime TCP connection. Chosen to sit
# ABOVE normal conversational turn cadence (~18-21 s gaps) so the socket is reused
# turn-to-turn, but comfortably BELOW a typical server/NAT idle-close window so a
# rare long pause expires the socket on OUR side first -> a clean client reconnect
# rather than the plugin being handed a dead socket. aiobotocore's default is 12 s
# (too short for our gaps).
_KEEPALIVE_TIMEOUT_S = 75.0


class _PersistentClientCM:
    """An async CM that opens a client once and keeps it (no-op ``__aexit__``).

    Wraps the *real* ``aioboto3.Session.client(...)`` context manager. The first
    ``__aenter__`` actually enters it (opening the client + its connection pool);
    later ``__aenter__``s return the same cached client; ``__aexit__`` deliberately
    does nothing so the pool is not torn down between the plugin's per-turn
    ``async with`` blocks. ``aclose()`` performs the deferred real exit.
    """

    def __init__(self, real_cm_factory) -> None:
        self._factory = real_cm_factory
        self._client = None
        self._real_cm = None
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        # Guard the one-time open: preemptive generation can enter concurrently
        # before the first client finished opening — without the lock that would
        # build two clients and leak one.
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._real_cm = self._factory()
                    self._client = await self._real_cm.__aenter__()
        return self._client

    async def __aexit__(self, *exc) -> bool:
        # No-op on purpose: keep the client + connection pool alive across turns.
        # Never suppress exceptions from the plugin's body.
        return False

    async def aclose(self) -> None:
        """Perform the deferred real ``__aexit__`` and drop the cached client."""
        if self._real_cm is not None:
            real_cm, self._real_cm, self._client = self._real_cm, None, None
            try:
                await real_cm.__aexit__(None, None, None)
            except Exception as e:  # shutdown best-effort
                logger.debug("persistent client close error: %s", e)


class PersistentSession(aioboto3.Session):
    """An ``aioboto3.Session`` whose ``.client()`` hands out persistent, keep-alive clients.

    Drop-in for the default session the ``aws.LLM`` plugin creates — pass it via
    ``aws.LLM(session=PersistentSession(region_name=...))``. Keyed per service name
    so a client for one service never shadows another (only ``"bedrock-runtime"``
    is used here, but the shape is general).
    """

    def __init__(
        self, *args, keepalive_timeout: float = _KEEPALIVE_TIMEOUT_S, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._persist: dict[str, _PersistentClientCM] = {}
        self._keepalive_timeout = keepalive_timeout

    def _with_keepalive(self, incoming):
        """Rebuild the plugin's per-call ``botocore.Config`` as an ``AioConfig`` that
        keeps the pooled TCP connection alive across the inter-turn gap.

        ``keepalive_timeout`` lives ONLY on ``AioConfig.connector_args`` — a plain
        ``botocore.Config`` (what the plugin passes) has no such field — so we must
        rebuild it here. Preserves the incoming config's ``user_agent_extra`` and any
        pre-existing ``connector_args`` (so we override only the keepalive knob).
        """
        ua = (
            getattr(incoming, "user_agent_extra", None)
            if incoming is not None
            else None
        )
        connector_args = dict(getattr(incoming, "connector_args", None) or {})
        connector_args["keepalive_timeout"] = self._keepalive_timeout
        return AioConfig(connector_args=connector_args, user_agent_extra=ua)

    def client(self, service_name: str, **kwargs) -> _PersistentClientCM:  # type: ignore[override]
        # Inject keepalive into the plugin's per-call config before it reaches the
        # real client. (The plugin passes an identical config every call, and we
        # cache one CM per service, so the first call's injected config persists —
        # which is correct, it's the same each time.)
        kwargs["config"] = self._with_keepalive(kwargs.get("config"))
        cm = self._persist.get(service_name)
        if cm is None:
            # Capture the bound super().client via the explicit 2-arg super()
            # form (a lambda has no zero-arg-super magic).
            factory = lambda: super(PersistentSession, self).client(  # noqa: E731
                service_name, **kwargs
            )
            cm = _PersistentClientCM(factory)
            self._persist[service_name] = cm
        return cm

    async def aclose_all(self) -> None:
        """Close every persistent client (idempotent). Call on session shutdown."""
        for cm in list(self._persist.values()):
            await cm.aclose()
        self._persist.clear()
