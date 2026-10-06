import json
from types import SimpleNamespace

import numpy as np
import pytest

from polima.robot.action_trace import ActionTrace
from polima.robot.metrics import TrialMetrics


@pytest.mark.parametrize('rest_snap_delta', [0, -1])
@pytest.mark.parametrize('transport', ['groot', 'act'])
def test_provenance_clamp_reads_and_first_cycle(tmp_path, monkeypatch, rest_snap_delta, transport):
    calls = []
    class Bus:
        position = 0
        def sync_read(self, register, **kwargs):
            calls.append(('read', register))
            return {'joint': self.position}
        def sync_write(self, register, values):
            calls.append(('write', register, dict(values)))
    class Robot:
        def __init__(self): self.bus, self.action_features = Bus(), {'joint.pos': float}
        def send_action(self, action):
            before = self.bus.sync_read('Present_Position')['joint']
            value = max(before - 12, min(before + 12, action['joint.pos']))
            self.bus.sync_write('Goal_Position', {'joint': value})
            return {'joint.pos': value}
    class Client:
        def __init__(self):
            self.robot = Robot()
            wire = SimpleNamespace(request_id=1)
            self.som_client = SimpleNamespace(adapter=SimpleNamespace(client=wire)) if transport == "groot" else wire
            self.queued = None
        def start(self): pass
        def _aggregate_action_queues(self, incoming, aggregate_fn=None):
            tensor = incoming[0].get_action()
            self.queued = tensor if self.queued is None else aggregate_fn(self.queued, tensor)
        def _action_tensor_to_action_dict(self, tensor):
            return {'joint.pos': float(tensor[0]) + rest_snap_delta}
    def action(value):
        tensor = np.array([value], dtype=float)
        return SimpleNamespace(get_action=lambda: tensor, get_timestep=lambda: 3)
    def reset(*args, **kwargs):
        calls.append(('reset',))
        return True
    monkeypatch.setitem(Client.start.__globals__, '_return_to_reset_position', reset)
    metrics = TrialMetrics(tmp_path/'trace', {})
    trace = ActionTrace(metrics)
    restore = trace.install(Client)
    original = Robot.send_action
    client = Client()
    client._aggregate_action_queues([action(10)], lambda old,new: .3*old+.7*new)
    from polima.robot.action_trace import request_client
    request_client(client.som_client).request_id = 2
    client._aggregate_action_queues([action(20)], lambda old,new: .3*old+.7*new)
    result = client.robot.send_action(client._action_tensor_to_action_dict(client.queued))
    assert result == {'joint.pos': 12}
    assert calls == [('read','Present_Position'),('write','Goal_Position',{'joint':12})]
    client.robot.bus.position = 5
    client.robot.bus.sync_read('Present_Position')
    Client.start.__globals__['_return_to_reset_position']()
    client.robot.send_action({'joint.pos': 0})  # rest command is excluded
    restore()
    assert client.robot.send_action.__func__ is original
    assert Client.start.__globals__['_return_to_reset_position'] is reset
    metrics.close()
    events=[json.loads(l) for l in (metrics.root/'events.jsonl').read_text().splitlines()]
    commands=[e for e in events if e['event']=='action_command']
    assert len(commands)==1
    command=commands[0]
    assert (command['chunk_id'],command['action_index'],command['timestep'])==(2,0,3)
    assert command['raw_targets']=={'joint.pos':20}
    assert command['blended_targets']=={'joint.pos':17}
    assert command['requested_targets']=={'joint.pos':17+rest_snap_delta}
    assert command['final_targets']=={'joint.pos':12}
    assert command['clamp_delta']=={'joint.pos':-5-rest_snap_delta}
    assert command['measured_before']=={'joint.pos':0}
    after=next(e for e in events if e['event']=='action_measured_after')
    assert after['measured']=={'joint.pos':5}
    assert after['read_started_s']>=command['command_finished_s']
    assert any(e['event']=='first_cycle_end' for e in events)
