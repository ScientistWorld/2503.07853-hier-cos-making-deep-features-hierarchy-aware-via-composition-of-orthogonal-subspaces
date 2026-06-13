#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0.17 GB
# Estimated disk usage after extraction: 0.17 GB
#
# This portable script writes into data/downloads/ only. Local development may
# use an equivalent symlink there, but fresh clones can repopulate it here.

set -e

cd /home/user

mkdir -p data/downloads

if [ ! -e data/downloads/cifar-100-python/train ]; then
  tmpdir="$(mktemp -d)"
  wget -nc https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz -O "$tmpdir/cifar-100-python.tar.gz"
  tar -xzf "$tmpdir/cifar-100-python.tar.gz" -C "$tmpdir"
  rm -rf data/downloads/cifar-100-python
  mv "$tmpdir/cifar-100-python" data/downloads/cifar-100-python
  rm -rf "$tmpdir"
fi
