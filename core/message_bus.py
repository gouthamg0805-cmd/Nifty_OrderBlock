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


# Global singleton
bus = MessageBus()
