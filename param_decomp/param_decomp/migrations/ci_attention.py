"""Move CI-transformer attention parameters into their attention module.

Run once per Orbax item (both decomposition and training for resumable runs):

    python -m param_decomp.migrations.ci_attention SOURCE_ITEM DESTINATION_ITEM

The source is read-only and the destination must not exist. The same path transform
moves optimizer moments, including None leaves in optimizer partitions. Runtime restore
only accepts the new representation. No array values, shapes or dtypes change.
"""

import argparse
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp


def migrate_tree(value: object) -> object:
    """Move every legacy CI block's attention leaves, preserving all other data."""
    match value:
        case dict():
            migrated = {key: migrate_tree(child) for key, child in value.items()}
            projections = {"wq", "wk", "wv", "wo"}
            is_block = {"w1", "w2"} <= migrated.keys() or {
                "selected_gate",
                "shared_gate",
            } <= migrated.keys()
            if projections <= migrated.keys() and is_block:
                assert "attention" not in migrated, "mixed old/new attention checkpoint paths"
                attention = {key: migrated.pop(key) for key in ("wq", "wk", "wv", "wo")}
                gains = migrated.pop("qk_norm_scales", None)
                match gains:
                    case None:
                        normalization = {}
                    case [q_gain, k_gain] | (q_gain, k_gain):
                        normalization = {"q_gain": q_gain, "k_gain": k_gain}
                    case _:
                        raise ValueError("legacy qk_norm_scales must contain query and key gains")
                attention["query_key_norm"] = normalization
                migrated["attention"] = attention
            return migrated
        case list():
            return [migrate_tree(child) for child in value]
        case tuple():
            return tuple(migrate_tree(child) for child in value)
        case _:
            return value


def migrate_item(source: Path, destination: Path) -> None:
    """Copy an Orbax item to a new path with the attention tree migrated."""
    assert source.resolve() != destination.resolve(), "migration must leave the source intact"
    assert not destination.exists(), f"destination already exists: {destination}"
    # Host arrays make migration independent of the saved GPU mesh. Allocate enough
    # host RAM for the item; no target model or training runtime is constructed.
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(source.resolve())
        restore_args = jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata.item_metadata.tree
        )
        tree = reader.restore(
            source.resolve(), args=ocp.args.PyTreeRestore(restore_args=restore_args)
        )
    with ocp.StandardCheckpointer() as writer:
        writer.save(destination.resolve(), migrate_tree(tree))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    migrate_item(args.source, args.destination)


if __name__ == "__main__":
    main()
