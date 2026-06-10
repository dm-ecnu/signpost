"""
Minimal compatibility layer for trio features not available in asyncio.
Only includes features that don't have direct asyncio equivalents.
"""

import asyncio


class CapacityLimiter:
    """Asynchronous capacity limiter supporting weighted (multi-token) reservations.

    Usage examples
    --------------
    # Borrow a single token (legacy style)
    async with limiter:
        ...

    # Borrow multiple tokens
    async with limiter.acquire(3):
        ...

    # Manual acquire / release
    lease = limiter.acquire(2)
    async with lease:
        ...
    await lease.release()
    """

    class _Lease:
        def __init__(self, limiter: "CapacityLimiter", tokens: int):
            self._limiter = limiter
            self._tokens = tokens
            self._released = False

        async def __aenter__(self):
            # Reserve the requested tokens when entering the context.
            await self._limiter._reserve(self._tokens)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Always release on exit, even if an exception occurred.
            await self.release()

        async def release(self):
            """Release the leased tokens early (idempotent)."""
            if not self._released:
                await self._limiter._release(self._tokens)
                self._released = True

    def __init__(self, total_tokens: int):
        if total_tokens <= 0:
            raise ValueError("total_tokens must be a positive integer")
        self._total = total_tokens
        self._available = total_tokens
        self._cond = asyncio.Condition()

    # ---------- Public API -------------------------------------------------

    def acquire(self, tokens: int = 1) -> "_Lease":
        """Return an async context manager that will reserve *tokens*.

        Example:
            async with limiter.acquire(3):
                ...
        """
        if tokens <= 0 or tokens > self._total:
            raise ValueError("invalid tokens")
        return self._Lease(self, tokens)

    async def release(self, tokens: int = 1) -> None:
        """Release *tokens* previously acquired via the legacy style."""
        if tokens <= 0 or tokens > self._total:
            raise ValueError("invalid tokens")
        await self._release(tokens)

    # Legacy `async with limiter:` (single token) ---------------------------
    async def __aenter__(self):
        await self._reserve(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._release(1)

    # ---------- Internal helpers ------------------------------------------
    async def _reserve(self, tokens: int):
        async with self._cond:
            await self._cond.wait_for(lambda: self._available >= tokens)
            self._available -= tokens

    async def _release(self, tokens: int):
        async with self._cond:
            self._available += tokens
            # Guard against over-release; clamp to total capacity.
            if self._available > self._total:
                self._available = self._total
            self._cond.notify_all()

    # ---------- Introspection ---------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self._total

    @property
    def borrowed_tokens(self) -> int:
        return self._total - self._available

    @property
    def available_tokens(self) -> int:
        return self._available
