#!/bin/bash

echo "Starting F3EO-Bench experiments..."

echo "Running CIFAR-10 experiment..."
python -m scripts.train --config config/cifar10.toml

echo "Running WikiText-2 experiment..."
python -m scripts.train --config config/wikitext2.toml

echo "All experiments completed!"