"""CLI dataset I/O."""

from pathlib import Path


def save_dataset(
    dataset,
    output: str,
) -> None:
    """Save a Hugging Face Dataset based on filename extension."""

    output = str(output)
    path = Path(output)

    if path.parent != Path("."):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    dataset = dataset.with_format(None)

    uncompressed_output = output.removesuffix(".gz")
    to_compress = output.endswith(".gz")
    
    if uncompressed_output.endswith((".csv", ".tsv", ".txt")):
        sep = "," if uncompressed_output.endswith(".csv") else "\t"
        dataset.to_csv(
            output,
            sep=sep,
            compression=(
                "gzip"
                if to_compress
                else None
            ),
        )

    elif output.endswith(".json"):
        dataset.to_json(output)

    elif output.endswith(".parquet"):
        dataset.to_parquet(output)

    elif output.endswith(".hf"):
        dataset.save_to_disk(output)

    else:
        dataset.save_to_disk(output)
