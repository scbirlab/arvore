"""Mapping between dataset columns and model inputs."""

from typing import Any
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os

from ..checkpoint_utils import load_json, save_json

def resolve_input(
    source: str | tuple[str, ...],
    batch: Mapping[str, ...]
):
    import torch
    if isinstance(source, str):
        return batch[source]
    return torch.cat(
        [batch[column] for column in source],
        dim=-1,
    )


def _columns(source):
    if isinstance(source, str):
        return (source,)
    return tuple(source)


@dataclass(frozen=True)
class ColumnMap:
    """Map batch fields onto model inputs.

    Parameters
    ==========
    inputs
        Mapping of model input names to dataset column names.
    target
        Dataset column containing the training target.
    """

    inputs: Mapping[str, str | tuple[str, ...]] | None = None
    target: str = "labels"

    def map_inputs(
        self,
        batch: Mapping[str, ...]
    ) -> dict[str, ...]:
        if self.inputs is None:
            return {
                column: resolve_input(batch=batch, source=column)
                for column in batch
            }
        missing = [
            column
            for source in self.inputs.values()
            for column in _columns(source)
            if column not in batch
        ]

        if missing:
            raise KeyError(
                "Missing input columns from batch: "
                + ", ".join(missing)
            )
        return {
            argument: resolve_input(batch=batch, source=column)
            for argument, column in self.inputs.items()
        }

    def map_target(
        self,
        batch: Mapping[str, ...]
    ):
        try:
            return batch[self.target]
        except KeyError as e:
            raise KeyError(
                f"Missing target column {self.target!r} from batch: {batch}."
            ) from e

    def map_batch(
        self,
        batch: Mapping[str, ...]
    ) -> tuple[dict[str, ...], ...]:
        return (
            self.map_inputs(batch),
            self.map_target(batch),
        )

    def __call__(self, *args, **kwargs):
        return self.map_batch(*args, **kwargs)

    @classmethod
    def from_config(
        cls, 
        config: Mapping[str, Any] | str
    ) -> 'ColumnMap':
        """Construct a ColumnMap from JSON-native configuration."""
        if isinstance(config, str):
            if os.path.exists(config):
                config = load_json(config)
            else:
                raise FileNotFoundError(
                    "`config` was string, but filename "
                    f"called `{config}` not found."
                )
        return cls(**config)

    def to_config(
        self, 
        filename: str | None = None
    ) -> dict[str, str]:
        config = asdict(self)
        if filename is not None:
            save_json(config, filename)
        return config
