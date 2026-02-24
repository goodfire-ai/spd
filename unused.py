"""
Finds potentially redundant type options across a codebase by analyzing call sites.

Reports:
1. Params typed as Optional/X|None where None is never actually passed
2. Params with defaults where the arg is always explicitly provided (default never used)

Limitations:
- Name-based function matching (false positives with same-name functions)
- No *args/**kwargs support
- No dynamic calls (getattr, functools.partial, etc.)
- Single-pass, no cross-module type inference
"""

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParamInfo:
    has_none_type: bool = False
    has_default: bool = False


@dataclass
class FuncInfo:
    params: dict[str, ParamInfo] = field(default_factory=dict)
    # param_name -> list of bools: was None passed at this call site?
    none_passed: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))
    # param_name -> list of bools: was arg explicitly provided?
    explicitly_provided: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))
    call_count: int = 0


def annotation_includes_none(node: ast.expr | None) -> bool:
    if node is None:
        return False
    # X | None or None | X
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return annotation_includes_none(node.left) or annotation_includes_none(node.right)
    # None constant
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    # Optional[X]
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "Optional":
            return True
        if isinstance(node.value, ast.Attribute) and node.value.attr == "Optional":
            return True
    return False


def is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def collect_functions(tree: ast.AST) -> dict[str, FuncInfo]:
    funcs = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        info = FuncInfo()
        args = node.args
        # Build param list (positional only + regular args), skip *args/**kwargs
        all_params = args.posonlyargs + args.args
        defaults_offset = len(all_params) - len(args.defaults)

        for i, arg in enumerate(all_params):
            has_default = i >= defaults_offset
            has_none = annotation_includes_none(arg.annotation)
            info.params[arg.arg] = ParamInfo(has_none_type=has_none, has_default=has_default)

        funcs[node.name] = info
    return funcs


def process_calls(tree: ast.AST, funcs: dict[str, FuncInfo]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Get function name
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue

        if name not in funcs:
            continue

        info = funcs[name]
        info.call_count += 1
        param_names = list(info.params.keys())

        # Track positional args
        provided = set()
        for i, arg_node in enumerate(node.args):
            if i < len(param_names):
                pname = param_names[i]
                provided.add(pname)
                info.none_passed[pname].append(is_none(arg_node))

        # Track keyword args
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs unpacking, skip
                continue
            provided.add(kw.arg)
            info.none_passed[kw.arg].append(is_none(kw.value))

        # Record which params were explicitly provided
        for pname in info.params:
            info.explicitly_provided[pname].append(pname in provided)


def analyze(root: Path) -> None:
    all_funcs: dict[str, FuncInfo] = {}

    files = list(root.rglob("*.py"))
    trees = []
    for f in files:
        try:
            source = f.read_text()
            tree = ast.parse(source, filename=str(f))
            trees.append(tree)
            for name, info in collect_functions(tree).items():
                all_funcs[name] = info  # last definition wins on collision
        except SyntaxError:
            continue

    for tree in trees:
        process_calls(tree, all_funcs)

    print("=== Redundant | None ===")
    for name, info in sorted(all_funcs.items()):
        if info.call_count == 0:
            continue
        for pname, param in info.params.items():
            if not param.has_none_type:
                continue
            calls_with_data = info.none_passed.get(pname, [])
            if not calls_with_data:
                continue
            if not any(calls_with_data):
                print(
                    f"  {name}({pname}): None never passed ({len(calls_with_data)} call sites checked)"
                )

    print("\n=== Default never used (always explicitly provided) ===")
    for name, info in sorted(all_funcs.items()):
        if info.call_count == 0:
            continue
        for pname, param in info.params.items():
            if not param.has_default:
                continue
            provided_list = info.explicitly_provided.get(pname, [])
            if not provided_list:
                continue
            if all(provided_list):
                print(
                    f"  {name}({pname}): default never used ({len(provided_list)} call sites checked)"
                )


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    analyze(root)
