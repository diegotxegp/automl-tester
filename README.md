# Ludwig AutoML Benchmark

A Python script for benchmarking Ludwig's AutoML capabilities on OpenML tasks.

## Overview

This script automatically downloads datasets from OpenML, trains machine learning models using Ludwig's AutoML functionality, and evaluates their performance across multiple random seeds.

## Features

- Downloads OpenML tasks with predefined train/test splits
- Runs Ludwig AutoML training with configurable time limits
- Evaluates models on separate test sets
- Performs multiple runs with different random seeds
- Computes statistics (mean, std) across runs
- Saves results to CSV files

## Configuration

- **TEST_MODE**: Quick testing mode with reduced tasks and seeds
- **TASK_IDS**: OpenML task identifiers to benchmark
- **SEEDS**: Random seeds for reproducible experiments
- **Time limits**: 5 minutes (test mode) or 2 hours (full mode)

## Output

The script generates:
- Individual run results with metrics for each seed
- Statistical summaries across all runs
- CSV files saved to `results/` directory

## Usage

```bash
python benchmark_script.py
```

Results are displayed in the console and saved as CSV files for further analysis.
