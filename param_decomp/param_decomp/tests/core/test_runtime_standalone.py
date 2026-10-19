"""The layering rule, as a test: subpackages of `param_decomp` import only DOWNWARD.

`param_decomp` is JUST a library — mostly pure functions, mostly logic; everything
infra-ish (schedulers, submission, cluster paths, code-shipping) belongs to whatever
launcher invokes it. A launcher may import the library; the library may never import a
launcher. The head check below names the in-house deployment packages
(`param_decomp_goodfire`, `goodfire_ops` and explain's packages) forbidden import roots
everywhere in the library — so the direction stays pinned whether or not those packages
are installed alongside. The full principle is in the root README's "Code style and
architecture". Within it, the library is enumerated layers. Each subpackage declares the
`param_decomp.*` prefixes it may import (`_LAYER_ALLOWED`); anything outside that set —
including `torch`, banned everywhere (the runtime is JAX; the torch oracle lives at git
tag `torch-oracle`) — fails this test. A subpackage that is not enumerated at all fails
collection: a new layer is added here deliberately, never absorbed silently.

The load-bearing directions:
  * `core` (the engine) sees a target only through the `DecomposedModel` protocol and
    the `ArchFamily` grammar contract — it must never import `targets`, nor anything
    composition-shaped built on top.
  * `targets` implements the engine's protocol per architecture — engine + vendored
    numerics only.
  * `target_ports` (verbatim numeric mirrors of the target architectures) and `routed`
    (the expert-parallel routed-compute machinery) are leaves.

The runtime is every `.py` that ships in a wheel — i.e. not the per-layer `tests/` and
`tools/` dirs. Test suites are exempt on purpose: engine tests may use a concrete target
as a fixture.
"""

import ast
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_NON_RUNTIME_DIRS = {"tests", "tools"}

_FORBIDDEN_ROOTS = (
    "torch",
    "param_decomp_goodfire",
    "goodfire_ops",
    "explain",
    "explain_goodfire",
    "explain_viewer",
)
"""Import roots the library never imports: torch, and the in-house deployment packages."""

_ANY = ("param_decomp",)
"""The composition layers (the merged lab): free to import anything in the library.
Tightening these to real per-layer sets is deliberate follow-up work."""

_LAYER_ALLOWED: dict[str, tuple[str, ...]] = {
    "routed": ("param_decomp.routed",),
    "target_ports": (
        "param_decomp.attention",
        "param_decomp.target_ports",
        "param_decomp.sequence",
        "param_decomp.lm.batch",
    ),
    "core": (
        "param_decomp.attention",
        "param_decomp.attention_kl",
        "param_decomp.core",
        "param_decomp.metric_schema",
        "param_decomp.routed",
        "param_decomp.target_ports",
        "param_decomp.sequence",
    ),
    "lm": (
        "param_decomp.lm.batch",
        "param_decomp.lm.batch_schedule",
        "param_decomp.lm.inputs",
        "param_decomp.sequence",
        "param_decomp.core.components",
        "param_decomp.core.placement",
    ),
    # `pretrain/train.py` is a composition root with its own `__main__`, so it reads the
    # pure contract every composition root reads: the dataset store's layout + ref schema.
    # Named module by module — the rest of `infra` (wandb and run files) stays firmly above
    # this layer.
    "pretrain": (
        "param_decomp.attention",
        "param_decomp.pretrain",
        "param_decomp.metric_schema",
        "param_decomp.sequence",
        "param_decomp.core",
        "param_decomp_formats.dataset_store",
        "param_decomp.lm",
        "param_decomp.target_ports",
    ),
    "targets": (
        "param_decomp.attention",
        "param_decomp.attention_kl",
        "param_decomp.lm.batch",
        "param_decomp.sequence",
        "param_decomp.core",
        "param_decomp.routed",
        "param_decomp.target_ports",
        "param_decomp.targets",
    ),
    "clustering": (),  # removal marker only; see its docstring
    "experiments": _ANY,
    "infra": _ANY,
    "migrations": _ANY,
    "topology": _ANY,
}


def _subpackages() -> list[str]:
    subs = sorted(
        p.name
        for p in _PACKAGE_ROOT.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and p.name not in _NON_RUNTIME_DIRS
    )
    unlisted = [s for s in subs if s not in _LAYER_ALLOWED]
    assert not unlisted, (
        f"subpackages {unlisted} are not enumerated in _LAYER_ALLOWED — declare each new "
        "layer's allowed imports deliberately"
    )
    return subs


def _bad_imports(path: Path, allowed: tuple[str, ...]) -> list[str]:
    def is_bad(module: str) -> bool:
        head = module.split(".", 1)[0]
        if head in _FORBIDDEN_ROOTS:
            return True
        if head != "param_decomp":
            return False
        return not any(module == p or module.startswith(p + ".") for p in allowed)

    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                found += [alias.name for alias in names if is_bad(alias.name)]
            case ast.ImportFrom(module=module) if module is not None:
                if is_bad(module):
                    found.append(module)
            case _:
                pass
    return found


def _runtime_python_files() -> list[tuple[Path, tuple[str, ...]]]:
    cases: list[tuple[Path, tuple[str, ...]]] = [
        (_PACKAGE_ROOT / "sequence.py", ()),
        (_PACKAGE_ROOT / "attention.py", ("param_decomp.sequence",)),
        (_PACKAGE_ROOT / "attention_kl.py", ("param_decomp.attention",)),
    ]
    for sub in _subpackages():
        for path in sorted((_PACKAGE_ROOT / sub).rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT / sub)
            if rel.parts[0] in _NON_RUNTIME_DIRS:
                continue
            layer = next(
                parent.as_posix()
                for parent in path.relative_to(_PACKAGE_ROOT).parents
                if parent.as_posix() in _LAYER_ALLOWED
            )
            cases.append((path, _LAYER_ALLOWED[layer]))
    return cases


@pytest.mark.parametrize(
    ("path", "allowed"),
    _runtime_python_files(),
    ids=lambda v: str(v.relative_to(_PACKAGE_ROOT)) if isinstance(v, Path) else None,
)
def test_runtime_imports_only_downward(path: Path, allowed: tuple[str, ...]):
    bad = _bad_imports(path, allowed)
    assert not bad, (
        f"{path.relative_to(_PACKAGE_ROOT)} imports {bad}, outside its layer's allowed set "
        f"{allowed} — runtime imports only point downward"
    )
