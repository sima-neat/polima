"""Ease the packaged script's return to rest in and out (POLIMA_RESET_EASE=smoothstep).

The SO101 robot-client script (shared byte-for-byte with ACT) returns to rest in two
constant-speed segments: lift 1.5 s, then return 5 s, each command `start + (end - start)
* step / steps` at 50 Hz, so the arm starts, stops and turns the corner abruptly. This
module leaves the template untouched. When the GR00T launcher runs it, the script's own
_return_to_reset_position is recompiled from its deployed source with exactly those two
`ratio = step / ...` assignments passed through an ease curve; durations, rates, the lift
logic, the settle loop, the final write and every robot.send_action (with its clamp) are
the script's own. Each segment then starts and ends at zero speed (the corner becomes a
brief stop instead of an instant turn). If the deployed function does not contain exactly
those two assignments, nothing is changed and the launch stops with an error.
"""
from __future__ import annotations
import __future__

import ast
import inspect
import os
import textwrap

ENV = "POLIMA_RESET_EASE"
NAME = "_return_to_reset_position"
EASES = {
    # Zero speed at both ends; peak speed 1.5x the linear speed.
    "smoothstep": lambda x: x * x * (3.0 - 2.0 * x),
    # Zero speed and acceleration at both ends; peak speed 1.875x.
    "smootherstep": lambda x: x * x * x * (x * (6.0 * x - 15.0) + 10.0),
}
_EASE_GLOBAL = "_polima_reset_ease"


def ease_from_env(environ=os.environ) -> str:
    value = environ.get(ENV, "linear").strip() or "linear"
    if value != "linear" and value not in EASES:
        raise ValueError(f"{ENV} must be linear (unchanged) or one of {sorted(EASES)}")
    return value


class _EaseRatio(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Assign(self, node):
        value = node.value
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "ratio" and isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Div) and isinstance(value.left, ast.Name)
                and value.left.id == "step" and isinstance(value.right, ast.Name)
                and value.right.id in ("lift_steps", "steps")):
            self.count += 1
            node.value = ast.Call(func=ast.Name(id=_EASE_GLOBAL, ctx=ast.Load()), args=[value],
                                  keywords=[])
        return node


def eased(function, ease: str):
    """The script's own function, recompiled with eased segment ratios."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError(f"{NAME}: unexpected source; not eased")
    transformer = _EaseRatio()
    tree = ast.fix_missing_locations(transformer.visit(tree))
    if transformer.count != 2:
        raise RuntimeError(f"{NAME}: expected the lift and return ratio assignments, found "
                           f"{transformer.count}; the deployed script differs; not eased")
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    namespace = function.__globals__
    namespace[_EASE_GLOBAL] = EASES[ease]
    code = compile(tree, function.__code__.co_filename, "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    local = {}
    exec(code, namespace, local)  # noqa: S102 - the deployed script's own source
    result = local[NAME]
    result.__polima_reset_ease__ = ease
    return result


def install(robot_client_module, ease: str):
    """Swap the script's function when LeRobot's async_client is entered.

    The packaged script defines _return_to_reset_position and patches RobotClient at
    import, then calls async_client from its __main__ block; the client is constructed and
    started (startup return to rest) inside async_client. Its namespace is reached through
    RobotClient.start, which the script replaces with its own function. Install this after
    polima.robot.metrics.run_instrumented has set its entry point, so the action trace's
    return-to-rest hook wraps the eased function. Returns a restore callable.
    """
    original = robot_client_module.async_client

    def entry(*args, **kwargs):
        namespace = getattr(robot_client_module.RobotClient.start, "__globals__", {})
        if NAME not in namespace:
            raise RuntimeError(f"{ENV}: the robot-client script's {NAME} was not found")
        namespace[NAME] = eased(namespace[NAME], ease)
        return original(*args, **kwargs)

    robot_client_module.async_client = entry

    def restore():
        robot_client_module.async_client = original

    return restore
