#!/bin/bash
# Download all data needed for this environment.
#
# Run on a node with internet access BEFORE building the container or
# submitting compute jobs. Idempotent: skips work whose outputs already exist.
#
# Estimated total download size: 0.17 GB (official CIFAR-100 python archive)
# Estimated disk usage after preparation: 0.19 GB
#
# Writes into data/downloads/cifar-100-python/ only:
#   train, meta        official labelled training pickle and label names
#   test_images.u8     the 10,000 official test images WITHOUT labels
#   test_images.json   header for test_images.u8
# The labelled test pickle is deleted after conversion: test labels are the
# hidden benchmark's answers and are never placed in data/.

set -e

cd /home/user

DEST=data/downloads/cifar-100-python
mkdir -p data/downloads

if [ ! -e "$DEST/train" ] || [ ! -e "$DEST/meta" ] || [ ! -e "$DEST/test_images.u8" ] || [ -e "$DEST/test" ]; then
  tmpdir="$(mktemp -d)"
  wget -q https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz -O "$tmpdir/cifar-100-python.tar.gz" \
    || wget -q https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -O "$tmpdir/cifar-100-python.tar.gz"
  tar -xzf "$tmpdir/cifar-100-python.tar.gz" -C "$tmpdir"
  rm -rf "$DEST"
  python3 -I data/prepare_cifar100.py --source "$tmpdir/cifar-100-python" --dest "$DEST"
  rm -rf "$tmpdir"
fi
