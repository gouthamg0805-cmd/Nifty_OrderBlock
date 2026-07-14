"""
core/message_bus.py
Async message bus for inter-agent communication.
Uses asyncio.Queue — lightweight, no external dependencies.
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, Callable, List
from loguru import logger


class MessageBus:
    """
    Simple pub-sub bus. Agents subscribe to topics and receive messages.
    Topics: market_state | trade_signal | trade_order | executed_trade |
            trailing_update | closed_trade | system_event
    """

    def __init__(self):
        self._queues: Dict[str, List[asyncio.Queue]] = {}
        self._running = False
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None):
        """
        Record the main event loop so code running on OTHER threads (e.g.
        blocking broker calls executed via loop.run_in_executor, which run on
        a worker thread with no event loop of its own) can still publish
        safely. Call this once, from inside the running loop, at startup.
        """
        self.loop = loop or asyncio.get_running_loop()

    def subscribe(self, topic: str) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self._queues.setdefault(topic, []).append(q)
        logger.debug(f"Subscribed to '{topic}'")
        return q

    async def publish(self, topic: str, message: Any):
        if topic not in self._queues:
            return
        for q in self._queues[topic]:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for topic '{topic}', dropping message")

    async def publish_event(self, event_type: str, data: Any = None):
        await self.publish("system_event", {"event": event_type, "data": data})

    def publish_event_threadsafe(self, event_type: str, data: Any = None):
        """
        Safe to call from ANY thread, including non-event-loop worker threads
        (e.g. inside a blocking call running via run_in_executor). Schedules
        the publish onto the bound main loop instead of trying to fetch/create
        an event loop on the calling thread, which raises
        'There is no current event loop in thread ...' on worker threads.
        """
        if self.loop is None or not self.loop.is_running():
            logger.debug(
                f"[MessageBus] No bound running loop — dropping event '{event_type}'"
            )
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.publish_event(event_type, data), self.loop
            )
        except Exception as e:
            logger.debug(f"[MessageBus] publish_event_threadsafe failed: {e}")


# Global singleton
bus = MessageBus()
