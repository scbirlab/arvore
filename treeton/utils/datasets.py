"""Utilities for dealing with Hugging Face datasets."""

from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple, Union

import os
from tempfile import TemporaryDirectory

from carabiner import print_err

if TYPE_CHECKING:
    from datasets import (
        Dataset, 
        DatasetDict, 
        IterableDataset
    )
else:
    Dataset, DatasetDict, IterableDataset = Any, Any, Any

import numpy as np
from tqdm.auto import tqdm

from ..checkpoint_utils import save_json

def to_dataset(
    ds: IterableDataset,
    batch_size: int = 1024,
    nrows: Optional[int] = None,
    cache: str = "./cache"
) -> Dataset:
    from datasets import Dataset, concatenate_datasets
    if isinstance(ds, Dataset):
        return ds
    new_ds = None
    total_iter = np.ceil(nrows / batch_size).astype(int) if nrows is not None else None
    for record in tqdm(ds.iter(batch_size=batch_size), total=total_iter, desc="Building dataset"):
        if new_ds is not None:
            with TemporaryDirectory() as tmpdirname:
                filename = os.path.join(tmpdirname, "add-record.json")
                save_json(record, filename)
                new_ds = concatenate_datasets([new_ds, Dataset.from_json(filename, cache_dir=cache)])
        else:
            with TemporaryDirectory() as tmpdirname:
                filename = os.path.join(tmpdirname, "init-record.json")
                save_json(record, filename)
                new_ds = Dataset.from_json(filename, cache_dir=cache)
    return new_ds


def top_n(
    ds: Union[Dataset, IterableDataset],
    column: str,
    n: int = 100,
    ds_rows: Optional[int] = None
) -> list:
    buffer = set()
    for record in tqdm(ds, total=ds_rows):
        if len(buffer) < n:
            buffer.add(record[column])
        else:
            test_value = record[column]
            for item in buffer:
                if test_value < item:
                    buffer.remove(item)
                    buffer.add(test_value)
                    break
    return list(buffer)


def _stream_and_subsample(
    ds: Dataset,
    min_subsample: int,
    total_rows: int,
    subsample: Optional[int] = None,
    seed: int = 0,
    shuffle_buffer: int = 10_000,
    n_shards: int = 1024
) -> IterableDataset:

    ds_stream = (
        ds
        .to_iterable_dataset(num_shards=n_shards)
        .shuffle(seed=seed, buffer_size=shuffle_buffer)
    )
    
    if subsample is not None:
        if subsample < min_subsample:
            subsample = min_subsample + 1
        if subsample <= total_rows:
            ds_stream = ds_stream.take(subsample)
    else:
        subsample = total_rows

    return subsample, ds_stream


def _get_cutoff(
    ds: Union[Dataset, IterableDataset],
    total_rows: int,
    column: str,
    n_hits: int,
    subsample: Optional[int] = None
) -> Union[float, int]:
    subsample = subsample or total_rows
    subsample = min(subsample, total_rows)
    top_n_values = top_n(
        ds.take(subsample),
        column,
        n=int(n_hits),
        ds_rows=subsample,
    )
    return max(top_n_values)
    
    
def split_3way(
    ds: Union[Dataset, IterableDataset],
    n_train: Optional[int] = None, 
    n_val: int = 0, 
    n_test: Optional[int] = None, 
    filter_column: Optional[str] = None,
    ascending: bool = False,
    sample_for_cutoff: Optional[int] = None,
    hit_frac: float = .01,
    initial_subsample: Optional[int] = None, 
    seed: int = 0,
    shuffle_buffer: int = 10_000,
    n_shards: int = 1024
):  
    from datasets import DatasetDict, concatenate_datasets

    total_rows = ds.num_rows
    n_train = n_train or total_rows
    n_train = n_train if n_train <= total_rows else total_rows
    n_train_and_val = n_train + n_val
    assert (n_train_and_val <= total_rows)
    n_test = n_test or (total_rows - n_train - n_val)

    ds_stream = (
        ds
        .to_iterable_dataset(num_shards=n_shards)
        .shuffle(seed=seed, buffer_size=shuffle_buffer)
    )
    
    if initial_subsample is not None:
        if initial_subsample < n_train_and_val:
            initial_subsample = n_train_and_val + 1
        if initial_subsample <= total_rows:
            ds_stream = ds_stream.take(initial_subsample)
    else:
        initial_subsample = total_rows
        
    if filter_column is not None:
        ds_train_and_val = ds_stream.take(n_train_and_val)
        ds_potential_test = ds_stream.skip(n_train_and_val)#.take(ds_shuffled.num_rows - n_train_and_val)
        ds_potential_test_nrows = initial_subsample - n_train_and_val
        n_hits = int(np.ceil(hit_frac * n_test))
        print_err(f"[INFO] Getting hit cutoff from {sample_for_cutoff} / {ds_potential_test_nrows} rows")

        hit_cutoff = _get_cutoff(
            ds=ds_potential_test,
            total_rows=ds_potential_test_nrows,
            column=filter_column,
            n_hits=n_hits,
            subsample=sample_for_cutoff
        )
        print_err(f"[INFO] Hit cutoff is {hit_cutoff:.2f}")
        print_err(f"[INFO] Taking {n_hits} rows with {filter_column} < {hit_cutoff:.2f}")
        ds_hits = (
            ds_potential_test
            # .with_format("numpy")
            .filter(
                lambda x: x[filter_column] <= hit_cutoff,
                batched=False,
                # desc=f"Filtering {filter_column} <= {hit_cutoff:.2f}"
            )
            .shuffle(seed=seed, buffer_size=shuffle_buffer)
            .take(n_hits)
        )
        ds_test = (
            ds_potential_test
            # .with_format("numpy")
            .filter(
                lambda x: x[filter_column] > hit_cutoff,
                batched=False,
                # desc=f"Filtering {filter_column} > {hit_cutoff:.2f}"
            )
            .shuffle(seed=seed, buffer_size=shuffle_buffer)
            .take(n_test - n_hits)
        )
        ds_test = (
            concatenate_datasets([ds_hits, ds_test])
            .shuffle(seed=seed, buffer_size=shuffle_buffer)
        )
        ds = {
            "train": to_dataset(ds_train_and_val.take(n_train), ds_rows=n_train),
            "val": to_dataset(ds_train_and_val.skip(n_train), ds_rows=n_val),
            "test": to_dataset(ds_test, ds_rows=n_test),
        }
    else:
        ds = {
            "train": to_dataset(ds_stream.take(n_train), ds_rows=n_train),
            "val": to_dataset(ds_stream.skip(n_train).take(n_val), ds_rows=n_val),
            "test": to_dataset(ds_stream.skip(n_train + n_val).take(n_test), ds_rows=n_test),
        }
    return DatasetDict(ds)


def split_al_pools(
    ds: Dataset, 
    n_initial: int, 
    candidate_filter: Optional[Mapping] = None,
    n_candidates: Optional[int] = None, 
    n_val: Optional[int] = None, 
    batch_size: Optional[int] = None, 
    n_test: Optional[int] = None, 
    n_batches: int = 20,
    filter_column: Optional[str] = None,
    hit_frac: float = .05,
    sample_for_cutoff: Optional[int] = None, 
    initial_subsample: Optional[int] = None, 
    ascending: bool = False,
    seed: int = 0,
    shuffle_buffer: int = 10_000,
    n_shards: int = 1024
) -> Tuple[Tuple[int], DatasetDict]:

    from datasets import DatasetDict, concatenate_datasets
    
    total_rows = ds.num_rows

    n_initial = min(n_initial, total_rows)
    batch_size = batch_size or n_initial
    n_candidates = n_candidates or (n_batches * batch_size)
    assert n_candidates > (n_batches * batch_size), "You need at least as many candidates as will be sampled!"
    n_val = n_val or n_initial
    n_test = n_test or n_candidates

    total_rows_final = n_initial + n_candidates + n_val + n_test

    initial_subsample, ds_stream = _stream_and_subsample(
        ds,
        min_subsample=total_rows_final,
        total_rows=total_rows,
        subsample=initial_subsample,
        shuffle_buffer=shuffle_buffer,
        n_shards=n_shards,
        seed=seed,
    )
    n_noncandidates = total_rows_final - n_candidates

    if candidate_filter is not None:
        ds_potential_candidates = ds_stream
        ds_noncandidates = ds_stream
        for key, val in candidate_filter.items():
            ds_potential_candidates = ds_potential_candidates.filter(
                lambda x: x[key] < val,
            )
            ds_noncandidates = ds_noncandidates.filter(
                lambda x: x[key] >= val,
            )
        ds_noncandidates = ds_noncandidates.take(n_noncandidates)
    else:
        ds_noncandidates = ds_stream.take(n_noncandidates)
        ds_potential_candidates = ds_stream.skip(n_noncandidates)
        
    if filter_column is not None:        
        ds_potential_candidates_nrows = initial_subsample - n_noncandidates
        print_err(f"[INFO] Getting hit cutoff from {sample_for_cutoff} / {ds_potential_candidates_nrows} rows")
        n_hits = int(np.ceil(hit_frac * n_test))

        hit_cutoff = _get_cutoff(
            ds=ds_potential_candidates,
            total_rows=ds_potential_candidates_nrows,
            column=filter_column,
            n_hits=n_hits,
            subsample=sample_for_cutoff
        )
        print_err(f"[INFO] Hit cutoff is {hit_cutoff:.2f}")
        print_err(f"[INFO] Taking {n_hits} rows with {filter_column} < {hit_cutoff:.2f}")
        ds_hits = (
            ds_potential_candidates
            # .with_format("numpy")
            .filter(
                lambda x: x[filter_column] <= hit_cutoff,
                batched=False,
                # batch_size=10_000,
                # desc=f"Filtering {filter_column} <= {hit_cutoff:.2f}"
            )
            .shuffle(seed=seed, buffer_size=shuffle_buffer)
            .take(n_hits)
        )
        ds_candidates = (
            ds_potential_candidates
            # .with_format("numpy")
            .filter(
                lambda x: x[filter_column] > hit_cutoff,
                batched=False,
                # desc=f"Filtering {filter_column} > {hit_cutoff:.2f}"
            )
            .shuffle(seed=seed, buffer_size=shuffle_buffer)
            .take(int(n_candidates - n_hits))
        )
        ds_candidates = (
            concatenate_datasets([ds_hits, ds_candidates])
            .shuffle(seed=seed, buffer_size=shuffle_buffer)
            .take(n_candidates)
        )#.flatten_indices()
        ds = {
            "initial": to_dataset(ds_noncandidates.take(n_initial), ds_rows=n_initial),
            "candidates": to_dataset(ds_candidates, ds_rows=n_candidates),
            "val": to_dataset(ds_noncandidates.skip(n_initial).take(n_val), ds_rows=n_val),
            "test": to_dataset(ds_noncandidates.skip(n_initial + n_val).take(n_test), ds_rows=n_test),
        }
    else:
        ds = {
            "initial": to_dataset(ds_noncandidates.take(n_initial), ds_rows=n_initial),
            "candidates":  to_dataset(ds_potential_candidates.take(n_candidates), ds_rows=n_candidates),
            "val":  to_dataset(ds_noncandidates.skip(n_initial).take(n_val), ds_rows=n_val),
            "test":  to_dataset(ds_noncandidates.skip(n_initial + n_val).take(n_test), ds_rows=n_test),
        }
    return tuple(range(n_candidates)), DatasetDict(ds)
