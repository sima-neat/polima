"""Position-aware hand-off between consecutive action plans (POLIMA_AGGREGATE=crossfade).

LeRobot merges an arriving plan into the queue with one fixed old/new weight for every
overlapping timestep, so each arrival steps the target by most of the two plans'
disagreement at once. Crossfade gives the new plan weight i/(n+1) at the i-th of the n
overlapping timesteps; timesteps only the new plan covers are taken as-is, exactly as
before. Only the merge weights change: queue timing, dropping of executed timesteps and
the motor path are LeRobot's.
"""
from __future__ import annotations

import threading

NAME = "crossfade"


class Crossfade:
    """LeRobot aggregate function whose weight depends on the overlap position."""

    def __init__(self):
        self.local = threading.local()

    def arm(self, count: int) -> None:
        self.local.index, self.local.count = 0, count

    def __call__(self, old, new):
        index = getattr(self.local, "index", None)
        if index is None:
            raise RuntimeError("crossfade used without the merge hook that counts the overlap")
        self.local.index = index + 1
        # LeRobot re-reads latest_action per item, so fewer than `count` calls can occur;
        # min() only guards the opposite case.
        weight = min(index + 1, self.local.count) / (self.local.count + 1)
        return old + (new - old) * weight


def install(configs, client_cls) -> Crossfade:
    """Register the aggregate name and count each merge's overlap before LeRobot's merge.

    Must run before the bundle's robot-client script is executed: that script captures
    RobotClient._aggregate_action_queues at import and wraps it.
    """
    aggregate = Crossfade()
    configs.AGGREGATE_FUNCTIONS[NAME] = aggregate
    original = client_cls._aggregate_action_queues

    def merge(self, incoming_actions, aggregate_fn=None):
        with self.action_queue_lock:
            queued = {action.get_timestep() for action in self.action_queue.queue}
        with self.latest_action_lock:
            latest = self.latest_action
        aggregate.arm(sum(1 for action in incoming_actions
                          if latest < action.get_timestep() and action.get_timestep() in queued))
        return original(self, incoming_actions, aggregate_fn)

    client_cls._aggregate_action_queues = merge
    return aggregate
