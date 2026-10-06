"""Passive first-cycle SO101 provenance; no extra motor reads or control changes."""
from __future__ import annotations

import inspect
import threading
import time


def joint_values(values):
    return {key if key.endswith('.pos') else key + '.pos': float(value)
            for key, value in values.items()}


def request_client(som_client):
    adapter = getattr(som_client, 'adapter', None)
    return adapter.client if adapter is not None else som_client


class ActionTrace:
    def __init__(self, metrics):
        self.metrics = metrics
        self.local = threading.local()
        self.tensors = {}  # Keep identity stable until this short diagnostic closes.
        self.started = False
        self.ended = False
        self.sequence = 0
        self.pending_after = None
        self.restores = []

    def now(self):
        return time.perf_counter() - self.metrics.started

    def patch(self, obj, name, replacement):
        # Restore instance attributes without shadowing descriptors afterward.
        own = name in vars(obj)
        previous = getattr(obj, name)
        setattr(obj, name, replacement)
        def restore():
            if own:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self.restores.append(restore)
        return previous

    def remember(self, tensor, info):
        self.tensors[id(tensor)] = (tensor, info)

    def lookup(self, tensor):
        item = self.tensors.get(id(tensor))
        return dict(item[1]) if item is not None and item[0] is tensor else {}

    def end_cycle(self, reason):
        if self.started and not self.ended:
            self.ended = True
            self.pending_after = None
            self.metrics.record('first_cycle_end', reason=reason)

    def install_robot(self, robot):
        original_read = robot.bus.sync_read
        original_send = robot.send_action
        try:
            source = inspect.getsource(original_send)
        except (OSError, TypeError):
            source = None
        self.metrics.record('action_trace_driver', send_action_source=source,
                            robot_class=type(robot).__module__ + '.' + type(robot).__name__,
                            measurement_mode='existing reads only; after is next available read')

        def read(register, *args, **kwargs):
            began = self.now()
            result = original_read(register, *args, **kwargs)
            ended = self.now()
            if register == 'Present_Position' and not self.ended:
                values = joint_values(result)
                pending = self.pending_after
                if pending is not None:
                    self.metrics.record('action_measured_after', action_id=pending,
                                        read_started_s=began, read_finished_s=ended,
                                        measured=values)
                    self.pending_after = None
                command = getattr(self.local, 'command', None)
                if command is not None:
                    command.update(measured_before=values, before_read_started_s=began,
                                   before_read_finished_s=ended)
            return result

        def send(action, *args, **kwargs):
            provenance = getattr(self.local, 'selected', None)
            self.local.selected = None  # A later reset send is not a policy action.
            if provenance is None or self.ended:
                return original_send(action, *args, **kwargs)
            if not self.started:
                self.started = True
                self.metrics.record('first_cycle_start')
                print('ACTION TRACE: first policy command (t=0 for plots)', flush=True)
            self.sequence += 1
            command = dict(provenance, action_id=self.sequence,
                           command_started_s=self.now(), requested_targets=joint_values(action))
            self.local.command = command
            try:
                result = original_send(action, *args, **kwargs)
            except BaseException:
                self.metrics.record('action_command_failed', **command)
                raise
            finally:
                self.local.command = None
            final = joint_values(result)
            command.update(command_finished_s=self.now(), final_targets=final,
                           clamp_delta={k: final[k] - command['requested_targets'][k] for k in final})
            command['clamp_amount'] = {k: abs(v) for k, v in command['clamp_delta'].items()}
            self.metrics.record('action_command', **command)
            self.pending_after = self.sequence
            return result

        self.patch(robot.bus, 'sync_read', read)
        self.patch(robot, 'send_action', send)

    def install(self, cls):
        original_init = cls.__init__
        def init(client, *args, **kwargs):
            original_init(client, *args, **kwargs)
            self.install_robot(client.robot)
        self.patch(cls, '__init__', init)

        original_merge = cls._aggregate_action_queues
        def merge(client, incoming, aggregate_fn=None):
            if self.ended:
                return original_merge(client, incoming, aggregate_fn)
            names = tuple(client.robot.action_features)
            request_id = request_client(client.som_client).request_id
            def values(tensor):
                return dict(zip(names, map(float, tensor.tolist())))
            for index, action in enumerate(incoming):
                tensor = action.get_action()
                self.remember(tensor, {"chunk_id": request_id, "action_index": index,
                                       "timestep": action.get_timestep(),
                                       "raw_targets": values(tensor)})
            self.metrics.record('prediction_chunk', chunk_id=request_id,
                                actions=[dict(self.lookup(a.get_action())) for a in incoming])
            def blend(old, new):
                result = aggregate_fn(old, new) if aggregate_fn else new
                info = self.lookup(new)
                self.metrics.record('action_merge', **info, old_targets=values(old),
                                    blended_targets=values(result), old_chunk_id=self.lookup(old).get('chunk_id'))
                self.remember(result, info)
                return result
            return original_merge(client, incoming, blend)
        self.patch(cls, '_aggregate_action_queues', merge)

        original_convert = cls._action_tensor_to_action_dict
        def convert(client, tensor):
            info = self.lookup(tensor)
            blended = dict(zip(client.robot.action_features, map(float, tensor.tolist())))
            result = original_convert(client, tensor)
            if not self.ended:
                self.local.selected = dict(info, blended_targets=blended)
            return result
        self.patch(cls, '_action_tensor_to_action_dict', convert)

        # The established script calls this for startup, auto-repeat and shutdown.
        # Startup precedes first_cycle_start and is intentionally ignored.
        namespace = getattr(cls.start, '__globals__', {})
        reset = namespace.get('_return_to_reset_position')
        if reset is not None:
            def reset_traced(*args, **kwargs):
                self.end_cycle('return_to_reset_position')
                return reset(*args, **kwargs)
            namespace['_return_to_reset_position'] = reset_traced
            self.restores.append(lambda: namespace.__setitem__('_return_to_reset_position', reset))
        # Record availability of the exact deployed boundary rather than guessing.
        self.metrics.record('action_trace_boundary', reset_hook_installed=reset is not None)
        if reset is None:
            self.close()
            raise RuntimeError('Action trace cannot identify the existing return-to-rest boundary')
        return self.close

    def close(self):
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()
        self.tensors.clear()
