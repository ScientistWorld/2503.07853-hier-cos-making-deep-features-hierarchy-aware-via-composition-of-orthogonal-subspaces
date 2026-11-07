"""Prepare the scientist-visible CIFAR-100 files (standard library only).

`scripts/download.sh` fetches the official CIFAR-100 python archive and then
calls this script. It copies the labelled training pickle and the label-name
metadata, and converts the official test pickle into an *unlabelled* raw image
file. The test labels are the hidden benchmark's answers: they are never
written to `data/downloads/`.

Outputs in ``--dest``:
  train              official training pickle (50,000 labelled images)
  meta               official label-name metadata
  test_images.u8     10,000 x 3,072 uint8 rows in official test order, same
                     channel-major layout as the pickle's ``data`` array
  test_images.json   shape/dtype header for ``test_images.u8``

This file runs without numpy so it works on any download node.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

NUM_TEST = 10_000
ROW_BYTES = 3 * 32 * 32


class _Stub:
    """Stand-in for numpy objects; records reconstruction state only."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.state = None

    def __setstate__(self, state) -> None:  # numpy ndarray / dtype BUILD
        self.state = state


class _NoNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy"):
            if name in {"_reconstruct", "ndarray", "dtype", "scalar"}:
                return _Stub
        return super().find_class(module, name)


def _raw_array_bytes(obj) -> bytes:
    """Extract raw bytes from a stubbed numpy ndarray reconstruction."""
    if not isinstance(obj, _Stub) or obj.state is None:
        raise ValueError("unexpected CIFAR pickle layout: 'data' is not an ndarray")
    state = obj.state
    # ndarray.__setstate__ = (version, shape, dtype, is_fortran, rawdata)
    raw = state[-1]
    if isinstance(raw, str):
        raw = raw.encode("latin1")
    return bytes(raw)


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = _NoNumpyUnpickler(handle, encoding="latin1").load()
    return {(k.decode() if isinstance(k, bytes) else k): v for k, v in payload.items()}


def prepare(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train", "meta"):
        target = dest / name
        if target.is_symlink() or target.exists():
            target.unlink()
        shutil.copyfile(source / name, target)

    test = _load(source / "test")
    raw = _raw_array_bytes(test["data"])
    if len(raw) != NUM_TEST * ROW_BYTES:
        raise ValueError(f"unexpected CIFAR-100 test image size: {len(raw)} bytes")
    (dest / "test_images.u8").write_bytes(raw)
    (dest / "test_images.json").write_text(json.dumps({
        "num_examples": NUM_TEST,
        "shape": [NUM_TEST, 3, 32, 32],
        "dtype": "uint8",
        "order": "official CIFAR-100 test order",
        "labels": "withheld (hidden benchmark answers)",
    }, indent=2))
    # Never leave the labelled test pickle in the scientist-visible folder.
    stale = dest / "test"
    if stale.is_symlink() or stale.exists():
        stale.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="extracted cifar-100-python directory")
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    prepare(Path(args.source), Path(args.dest))


if __name__ == "__main__":
    main()
