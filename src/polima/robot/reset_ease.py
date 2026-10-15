"""Ease the control client's two return-to-rest segments."""
from __future__ import annotations
import __future__

import ast
import inspect
import os
import textwrap

ENV = "POLIMA_RESET_EASE"
NAME = "_return_to_reset_position"
EASES = {

    "smoothstep": lambda x: x * x * (3.0 - 2.0 * x),

    "smootherstep": lambda x: x * x * x * (x * (6.0 * x - 15.0) + 10.0),
}
_EASE_GLOBAL = "_polima_reset_ease"


def ease_from_env(environ=os.environ) -> str:
    value = environ.get(ENV, "linear").strip() or "linear"
    if value != "linear" and value not in EASES:
        raise ValueError(f"{ENV} must be linear or one of {sorted(EASES)}")
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
    exec(code, namespace, local)  # noqa: S102 - compile the packaged control function
    result = local[NAME]
    result.__polima_reset_ease__ = ease
    return result


def install(robot_client_module, ease: str):
    """Swap the script's function when LeRobot's async_client is entered.

    The packaged script defines _return_to_reset_position and patches RobotClient at
    import, then calls async_client. Returns a callable that restores the entry point.
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
