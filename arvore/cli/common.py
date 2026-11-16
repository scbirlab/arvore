"""Checkpoint loading and row selection shared by CLI commands."""

from argparse import Namespace

from ..box import Box

SKLEARN_TREE: str = "sklearn.tree._tree.Tree"


def load_box(args: Namespace) -> Box:
    return Box.load(
        args.input_file,
        cache_dir=args.cache,
        trusted_types=(SKLEARN_TREE,) if args.trust_sklearn_trees else (),
    )


def selected_rows(args: Namespace):
    """Rows ``--start`` to ``--stop`` of ``--data``, or None for the training data."""
    from providence import DataResolver

    if args.data is not None:
        return DataResolver(
            args.data, 
            cache_dir=args.cache,
        ).select_rows(
            args.start, 
            args.stop,
        ).data
    if args.start is not None or args.stop is not None:
        raise ValueError("--start and --stop require --data")
    return None
