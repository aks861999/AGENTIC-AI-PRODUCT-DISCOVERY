"""
Index alias management for zero-downtime reindexing.

Usage (the alias is INDEX_NAME, e.g. "products"):
    python -m ingestion --index products_v1
    python -m ingestion.ensure_alias --create            # products -> products_v1
    python -m ingestion --index products_v2              # rebuild while v1 serves traffic
    python -m ingestion.ensure_alias --switch products_v2
    python -m ingestion.ensure_alias --delete-old        # drop products_v* not behind the alias

Best practice per OpenSearch architecture: always search via alias,
never directly against index names. This allows full reindexing without
API downtime.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from ingestion.pipeline import get_client, wait_for_opensearch
from shared.config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_alias(client, alias_name: str, index_name: str) -> None:
    """Create alias pointing to index if alias doesn't exist."""
    # Check if alias exists
    if client.indices.exists_alias(name=alias_name):
        alias_info = client.indices.get_alias(name=alias_name)
        logger.info("Alias '%s' exists, points to: %s", alias_name, list(alias_info.keys()))
        return

    # An alias can't share its name with a concrete index (the default ingestion target)
    if client.indices.exists(index=alias_name):
        raise RuntimeError(
            f"'{alias_name}' is a concrete index, so it can't also be an alias. "
            f"Ingest into a versioned index (python -m ingestion --index {index_name}), "
            f"delete index '{alias_name}', then run --create again."
        )

    # Check if index exists
    if not client.indices.exists(index=index_name):
        logger.error("Index '%s' does not exist. Run ingestion first.", index_name)
        raise RuntimeError(f"Index {index_name} not found")
    
    # Create alias
    client.indices.put_alias(index=index_name, name=alias_name)
    logger.info("Created alias '%s' -> '%s'", alias_name, index_name)


def switch_alias(client, alias_name: str, new_index: str) -> None:
    """Atomically switch alias from old index to new index."""
    # Get current alias targets
    old_indices = []
    try:
        alias_info = client.indices.get_alias(name=alias_name)
        old_indices = list(alias_info.keys())
    except Exception:
        pass
    
    actions = []
    for old in old_indices:
        actions.append({"remove": {"index": old, "alias": alias_name}})
    actions.append({"add": {"index": new_index, "alias": alias_name}})
    
    client.indices.update_aliases(body={"actions": actions})
    logger.info("Switched alias '%s' from %s -> '%s'", alias_name, old_indices, new_index)


def delete_old_indices(client, alias_name: str) -> None:
    """Delete <alias>_v* indices that are NOT the current alias target."""
    alias_info = client.indices.get_alias(name=alias_name)
    current = list(alias_info.keys())

    all_indices = client.indices.get(index=f"{alias_name}_v*")
    to_delete = [name for name in all_indices.keys() if name not in current]
    
    for name in to_delete:
        client.indices.delete(index=name)
        logger.info("Deleted old index: %s", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage OpenSearch index alias")
    parser.add_argument("--create", nargs="?", const="", metavar="INDEX",
                        help="Create the alias if missing, pointing to INDEX (default: <INDEX_NAME>_v1)")
    parser.add_argument("--switch", metavar="INDEX", help="Switch alias to new index")
    parser.add_argument("--delete-old", action="store_true", help="Delete <INDEX_NAME>_v* indices not behind the alias")
    args = parser.parse_args()

    settings = get_settings()
    client = get_client(settings)
    wait_for_opensearch(client)

    alias = settings.index_name  # e.g., "products"

    if args.create is not None:
        # Convention: first versioned index is alias + "_v1"
        ensure_alias(client, alias, args.create or f"{alias}_v1")
    elif args.switch:
        switch_alias(client, alias, args.switch)
    elif args.delete_old:
        delete_old_indices(client, alias)
    else:
        parser.print_help()