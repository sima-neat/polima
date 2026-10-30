"""Blend a newly arriving plan across its overlap with the queued plan."""
from __future__ import annotations

import threading

NAME = "crossfade"


class Crossfade:
    def __init__(self):
        self.local = threading.local()

    def arm(self, count: int) -> None:
        self.local.index, self.local.count = 0, count

    def __call__(self, old, new):
        index = getattr(self.local, "index", None)
        if index is None:
            raise RuntimeError("crossfade used outside its queue merge")
        self.local.index = index + 1
        weight = min(index + 1, self.local.count) / (self.local.count + 1)
        return old + (new - old) * weight


def install(configs, client_cls) -> Crossfade:
    aggregate = Crossfade()
    configs.AGGREGATE_FUNCTIONS[NAME] = aggregate
    original = client_cls._aggregate_action_queues

    def merge(self, incoming_actions, aggregate_fn=None):
        with self.action_queue_lock:
            queued = {action.get_timestep() for action in self.action_queue.queue}
        with self.latest_action_lock:
            latest = self.latest_action
        aggregate.arm(sum(
            latest < action.get_timestep() and action.get_timestep() in queued
            for action in incoming_actions
        ))
        return original(self, incoming_actions, aggregate_fn)

    client_cls._aggregate_action_queues = merge
    return aggregate
