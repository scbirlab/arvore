""""Utilities for loading and saving checkpoints."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tempfile import TemporaryDirectory
import os
import json

from carabiner import print_err

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
else:
    Dataset, IterableDataset = Any, Any

from .utils.package_data import CACHE_DIR


def _load_json(checkpoint: str, filename: str | None = None) -> dict[str, ...]:
    if filename is not None:
        path = os.path.join(checkpoint, filename)
    else:
        path = checkpoint
    with open(path, "r") as f:
        obj = json.load(f)
    return obj

load_json = _load_json
   

def save_json(obj, filename: str) -> None:
    _dir = os.path.dirname(filename)
    if _dir != "." and len(_dir) > 0:
        os.makedirs(_dir, exist_ok=True)
    with open(filename, "w") as f:
        try:
            json.dump(obj, f, sort_keys=True, indent=4)
        except TypeError as e:
            print_err(f"{obj=}")
            raise e
    return None


def _load_hf_dataset(checkpoint, filename) -> Union[Dataset, IterableDataset]:
    from datasets import load_from_disk
    return load_from_disk(os.path.join(checkpoint, filename))


def _load_torch_weights(checkpoint, filename):
    import torch
    return torch.load(
        os.path.join(checkpoint, filename),
        weights_only=True,
    )


FILE_LOADING_CALLBACKS = {
    "json": _load_json,
    "hf-dataset": _load_hf_dataset, 
    "pt": _load_torch_weights,
}


def load_checkpoint_file(
    checkpoint: str,
    filename: str,
    callback: Union[str, Callable] = "json",
    none_on_error: bool = False,
    allow_none: bool = False,
    cache_dir: Optional[str] = None,
    allow_empty: bool = False,
    *args, **kwargs
) -> Union[Any, None]:
    cache_dir = cache_dir or CACHE_DIR
    obj = None
    if isinstance(callback, str):
        try:
            callback = FILE_LOADING_CALLBACKS[callback.casefold()]
        except KeyError:
            raise ValueError(
                """
                File loading callback must be callable or name.
                """
            )
    if os.path.exists(checkpoint):
        try:
            obj = callback(checkpoint, filename)
        except Exception as e:
            print_err(e)
            if none_on_error:
                return None
            else:
                raise e
    elif checkpoint.startswith("hf://"):
        from huggingface_hub import snapshot_download
        checkpoint = checkpoint.split("hf://")[-1]
        if filename.endswith(".hf"):
            filename_pattern = [filename + '/*.arrow', filename + '/*.json']
        else:
            filename_pattern = filename
        with TemporaryDirectory() as tmpdirname:
            try:
                print_err(f"[INFO] Looking up: {checkpoint} :: {filename}")
                snapshot_download(
                    repo_id=checkpoint,
                    allow_patterns=filename_pattern,
                    local_dir=tmpdirname,
                    cache_dir=cache_dir,
                    *args, **kwargs
                )
            except Exception as e:
                print_err("[ERROR]", e)
                if none_on_error:
                    return None
                else:
                    raise e
            else:
                try:
                    obj = callback(tmpdirname, filename)
                except FileNotFoundError as e:
                    print_err("[ERROR]", e)
                    if none_on_error:
                        return None
                    else:
                        raise e

    if allow_empty or obj is not None:
        return obj
    else:
        raise AttributeError(
            f"Could not load anything from {checkpoint=}, {filename=} with {callback=}."
        )
