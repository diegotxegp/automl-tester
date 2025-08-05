# AutoML Benchmark: Ludwig, AutoGluon, and H2O

A Python script for benchmarking the AutoML capabilities of Ludwig, AutoGluon, and H2O on OpenML tasks.

## Overview

This script automatically downloads datasets from OpenML, trains machine learning models using Ludwig, AutoGluon, and H2O AutoML frameworks, and evaluates their performance across multiple random seeds.

## Features

- Downloads OpenML tasks with predefined train/test splits  
- Trains models using Ludwig, AutoGluon, and H2O with configurable time limits  
- Evaluates models on separate test sets  
- Runs multiple experiments with different random seeds  
- Computes statistics (mean, standard deviation) across runs  
- Saves results to CSV files  

## Configuration

- **TEST_MODE**: Quick test mode with fewer tasks and seeds  
- **TASK_IDS**: OpenML task identifiers to benchmark  
- **SEEDS**: Random seeds for reproducibility  
- **Time limits**: 5 minutes (test mode) or 2 hours (full mode)  

## Output

The script generates:  
- Individual run results with metrics for each model and seed  
- Statistical summaries for each framework  
- CSV files saved in the `results/` directory

## Usage

```bash
python main.py
