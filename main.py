import os
import pandas as pd
import time
import openml
from ludwig.automl import auto_train

# List of task IDs
# Set to True for quick testing
TEST_MODE = True
TASK_IDS = [31] if TEST_MODE else [15, 23, 37, 57, 8]
SEEDS = [42] if TEST_MODE else [42, 123, 2023, 1, 99]

def download_openml_task_with_splits(task_id):
    try:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=task.target_name
        )
        
        # Get predefined train/test splits from OpenML
        train_indices, test_indices = task.get_train_test_split_indices()
        
        # Create train and test datasets
        X_train = X.iloc[train_indices].copy()
        X_test = X.iloc[test_indices].copy()
        y_train = y.iloc[train_indices].copy()
        y_test = y.iloc[test_indices].copy()
        
        # Create DataFrames with target column
        train_df = X_train.copy()
        train_df[task.target_name] = y_train
        
        test_df = X_test.copy()
        test_df[task.target_name] = y_test
        
        # Get evaluation measure, dataset name and task type
        evaluation_measure = task.evaluation_measure
        dataset_name = dataset.name
        task_type = task.task_type
        
        return train_df, test_df, task.target_name, evaluation_measure, dataset_name, task_type
        
    except Exception as e:
        print(f"Error downloading task {task_id}: {e}")
        return None, None, None, None, None, None

def run_ludwig_experiment(train_dataset, test_dataset, target_column, seed):
    try:
        time_limit = 300 if TEST_MODE else 7200

        # Train only on training data
        results = auto_train(
            dataset=train_dataset,
            target=target_column,
            time_limit_s=time_limit,
            tune_for_memory=False,
            random_seed=seed
        )
        
        # Evaluate on separate test set
        eval_results = results.best_model.evaluate(test_dataset)
        
        return results, eval_results
        
    except Exception as e:
        print(f"Error in training with seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_metrics(eval_results, target_column):
    metrics_dict = {}
    try:
        if isinstance(eval_results, tuple) and len(eval_results) > 0:
            eval_stats = eval_results[0]  # evaluation statistics
            
            # Look for the target feature's metrics
            if target_column in eval_stats:
                feature_metrics = eval_stats[target_column]
            elif 'combined' in eval_stats:
                feature_metrics = eval_stats['combined']
            else:
                # If none, take the first feature's metrics
                feature_metrics = next(iter(eval_stats.values()))
            
            # Extract all available metrics
            for metric_name, metric_value in feature_metrics.items():
                metrics_dict[metric_name] = metric_value
                
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics_dict

def compute_statistics(runs_list):
    # Extract metrics from each run
    metrics_list = [run['metrics'] for run in runs_list]
    df = pd.DataFrame(metrics_list)
    stats = {}
    for col in df.columns:
        stats[f"{col}_mean"] = df[col].mean()
        stats[f"{col}_std"] = df[col].std()
    return stats

def display_and_save_results(all_results):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    for task_id, data in all_results.items():
        print(f"\nTask {task_id} ({data['dataset_name']}) - {data['task_type']} Results:")
        print(f"Evaluation measure: {data['evaluation_measure']}")
        
        runs_df = pd.DataFrame([{
            'seed': run['seed'],
            'task_id': task_id,
            'dataset_name': data['dataset_name'],
            'task_type': data['task_type'],
            'evaluation_measure': data['evaluation_measure'],
            **run['metrics']
        } for run in data["runs"]])
        
        print(runs_df)
        print("\nStatistics:")
        stats_series = pd.Series(data["stats"])
        print(stats_series)

        # Save to CSV with task_id, dataset_name, task_type and evaluation_measure
        runs_df.to_csv(os.path.join(save_dir, f"task_{task_id}_runs.csv"), index=False)
        
        # Add task info to stats and save
        stats_with_info = data["stats"].copy()
        stats_with_info['task_id'] = task_id
        stats_with_info['dataset_name'] = data['dataset_name']
        stats_with_info['task_type'] = data['task_type']
        stats_with_info['evaluation_measure'] = data['evaluation_measure']
        stats_series_with_info = pd.Series(stats_with_info)
        stats_series_with_info.to_csv(os.path.join(save_dir, f"task_{task_id}_stats.csv"), 
                           header=['value'], 
                           index_label='metric')

def main():
    all_results = {}

    for task_id in TASK_IDS:
        print(f"Processing task {task_id}...")
        train_df, test_df, target_column, evaluation_measure, dataset_name, task_type = download_openml_task_with_splits(task_id)
        if train_df is None:
            continue

        print(f"Dataset: {dataset_name}")
        print(f"Task type: {task_type}")
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        print(f"Evaluation measure: {evaluation_measure}")

        task_results = []
        for seed in SEEDS:
            print(f"Running seed {seed}...")
            start_time = time.time()
            results, eval_results = run_ludwig_experiment(train_df, test_df, target_column, seed)
            end_time = time.time()
            time_taken = end_time - start_time

            if results and eval_results:
                metrics_dict = extract_metrics(eval_results, target_column)
                metrics_dict['time_taken'] = time_taken

                run_data = {
                    "seed": seed,
                    "metrics": metrics_dict
                }
                task_results.append(run_data)
                print(f"Run completed in {time_taken:.2f} seconds. Test metrics: {metrics_dict}")

        if task_results:
            stats = compute_statistics(task_results)
            all_results[task_id] = {
                "runs": task_results,
                "stats": stats,
                "evaluation_measure": evaluation_measure,
                "dataset_name": dataset_name,
                "task_type": task_type
            }

    display_and_save_results(all_results)
    print("All tasks processed.")

if __name__ == "__main__":
    main()