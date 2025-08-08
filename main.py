import os
import sys
import pandas as pd
import time
import openml
from ludwig.automl import auto_train
from autogluon.tabular import TabularPredictor
import h2o
from h2o.automl import H2OAutoML
import torch
import gc
import ray

# Set to True to run in test mode (only one task and one seed)
TEST_MODE = False

# List of task IDs
TASK_IDS = [52948] if TEST_MODE else [
    10101, 15, 67141, 31, 37, 9957,  # Binary classification
    3560, 23, 3011, 18, 53,          # Multi-class classification
    2295, 2301, 52948, 4839          # Regression
]

"""
10101, 15, 67141, 31, 37, 9957,  # Binary classification
    3560, 23, 3011, 18, 53,          # Multi-class classification
    2295, 2301, 52948, 4839          # Regression
"""

SEEDS = [123] if TEST_MODE else [123, 2027, 99]

# Time limit for each method in seconds
TIME_LIMIT = 300 if TEST_MODE else 600

def download_openml_task_with_splits(task_id):
    try:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=task.target_name
        )
        
        train_indices, test_indices = task.get_train_test_split_indices()
        
        X_train = X.iloc[train_indices].copy()
        X_test = X.iloc[test_indices].copy()
        y_train = y.iloc[train_indices].copy()
        y_test = y.iloc[test_indices].copy()
        
        train_df = X_train.copy()
        train_df[task.target_name] = y_train
        
        test_df = X_test.copy()
        test_df[task.target_name] = y_test
        
        evaluation_measure = task.evaluation_measure
        dataset_name = dataset.name
        task_type = task.task_type
        
        return train_df, test_df, task.target_name, evaluation_measure, dataset_name, task_type
        
    except Exception as e:
        print(f"Error downloading task {task_id}: {e}")
        return None, None, None, None, None, None
    
def clear_resources():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if ray.is_initialized():
            ray.shutdown()

def run_ludwig_experiment(train_dataset, test_dataset, target_column, seed, task_type):
    try:
        clear_resources()
        results = auto_train(
            dataset=train_dataset,
            target=target_column,
            time_limit_s=TIME_LIMIT,
            tune_for_memory=False,
            random_seed=seed,
            user_config={
                'hyperopt': {
                    'executor': {
                        'max_concurrent_trials': 4
                    }
                }
            }
        )
        eval_results = results.best_model.evaluate(test_dataset)[0]
        metrics = extract_ludwig_metrics(eval_results, target_column, task_type)
        clear_resources()
        return results, metrics
    except Exception as e:
        print(f"Error in Ludwig training with seed {seed}: {e}")
        return None, None

def run_autogluon_experiment(train_dataset, test_dataset, target_column, seed, task_type):
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"autogluon_models/task_seed_{seed}")
        os.makedirs(model_dir, exist_ok=True)
        
        eval_metric = 'roc_auc' if 'classification' in task_type.lower() and train_dataset[target_column].nunique() == 2 else 'accuracy' if 'classification' in task_type.lower() else 'rmse'
        
        predictor = TabularPredictor(
            label=target_column,
            path=model_dir,
            eval_metric=eval_metric,
            verbosity=2
        ).fit(
            train_data=train_dataset,
            time_limit=TIME_LIMIT,
            presets='medium_quality'
        )
        
        eval_results = predictor.evaluate(test_dataset, silent=False)
        metrics = extract_autogluon_metrics(eval_results, task_type)
        return predictor, metrics
    except Exception as e:
        print(f"Error in AutoGluon training with seed {seed}: {e}")
        return None, None

def run_h2o_experiment(train_dataset, test_dataset, target_column, seed, task_type):
    try:
        h2o_train = h2o.H2OFrame(train_dataset)
        h2o_test = h2o.H2OFrame(test_dataset)
        
        if 'classification' in task_type.lower():
            h2o_train[target_column] = h2o_train[target_column].asfactor()
            h2o_test[target_column] = h2o_test[target_column].asfactor()
        
        x = h2o_train.columns
        x.remove(target_column)
        
        aml = H2OAutoML(
            max_runtime_secs=TIME_LIMIT,
            seed=seed,
            project_name=f"automl_task_{seed}"
        )
        
        aml.train(x=x, y=target_column, training_frame=h2o_train)
        
        perf = aml.leader.model_performance(h2o_test)
        metrics = extract_h2o_metrics(perf, task_type)
        return aml, metrics
    except Exception as e:
        print(f"Error in H2O training with seed {seed}: {e}")
        return None, None

def safe_metric_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in ['nan', 'inf', '-inf', 'null', 'none']:
            return None
        try:
            return float(value)
        except:
            return None
    if isinstance(value, (int, float)):
        if str(value).lower() in ['nan', 'inf', '-inf']:
            return None
    return value

def extract_ludwig_metrics(eval_results, target_column, task_type):
    metrics_dict = {}
    try:
        print(f"Ludwig eval_results: {eval_results}")  # Debug output
        feature_metrics = eval_results.get(target_column, eval_results.get('combined', next(iter(eval_results.values()))))
        if 'classification' in task_type.lower():
            for metric in ['loss', 'roc_auc', 'accuracy', 'accuracy_micro', 'hits_at_k', 'precision', 'recall', 'specificity']:
                if metric in feature_metrics:
                    metrics_dict[metric] = safe_metric_value(feature_metrics[metric])
        else:
            for metric in ["loss", "root_mean_squared_error", "root_mean_squared_percentage_error", "r2", "mean_absolute_error", "mean_squared_error", "mean_absolute_percentage_error"]:
                if metric in feature_metrics:
                    metrics_dict[metric] = safe_metric_value(feature_metrics[metric])
    except Exception as e:
        print(f"Error extracting Ludwig metrics: {e}")
    return metrics_dict

def extract_autogluon_metrics(eval_results, task_type):
    metrics_dict = {}
    try:
        print(f"AutoGluon eval_results: {eval_results}")  # Debug output
        if isinstance(eval_results, dict):
            if 'classification' in task_type.lower():
                for metric in ['accuracy', 'balanced_accuracy', 'mcc']:
                    if metric in eval_results:
                        metrics_dict[metric] = safe_metric_value(eval_results[metric])
            else:
                for metric in ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'r2', 'pearsonr', 'median_absolute_error']:
                    if metric in eval_results:
                        metrics_dict[metric] = safe_metric_value(eval_results[metric])
        else:
            metrics_dict['score'] = safe_metric_value(eval_results)
    except Exception as e:
        print(f"Error extracting AutoGluon metrics: {e}")
    return metrics_dict

def safe_h2o_metric(metric_func):
    try:
        value = metric_func()
        print(f"H2O metric output: {value}")  # Debug output
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], list) and len(value[0]) > 1:
                value = value[0][1]
        return safe_metric_value(value)
    except Exception as e:
        print(f"Error in safe_h2o_metric: {e}")
        return None

def extract_h2o_metrics(perf, task_type):
    metrics_dict = {}
    try:
        if 'classification' in task_type.lower():
            metrics_dict['accuracy'] = safe_h2o_metric(lambda: perf.accuracy())
            metrics_dict['f1'] = safe_h2o_metric(lambda: perf.F1())
            metrics_dict['precision'] = safe_h2o_metric(lambda: perf.precision())
            metrics_dict['recall'] = safe_h2o_metric(lambda: perf.recall())
            metrics_dict['roc_auc'] = safe_h2o_metric(lambda: perf.auc())
            metrics_dict['logloss'] = safe_h2o_metric(lambda: perf.logloss())
            metrics_dict['aucpr'] = safe_h2o_metric(lambda: perf.aucpr())
            metrics_dict['mean_per_class_error'] = safe_h2o_metric(lambda: perf.mean_per_class_error())
        else:
            metrics_dict['r2'] = safe_h2o_metric(lambda: perf.r2())
            metrics_dict['rmse'] = safe_h2o_metric(lambda: perf.rmse())
            metrics_dict['mae'] = safe_h2o_metric(lambda: perf.mae())
            metrics_dict['mse'] = safe_h2o_metric(lambda: perf.mse())
            metrics_dict['rmsle'] = safe_h2o_metric(lambda: perf.rmsle())
            metrics_dict['mean_residual_deviance'] = safe_h2o_metric(lambda: perf.mean_residual_deviance())
    except Exception as e:
        print(f"Error extracting H2O metrics: {e}")
    return metrics_dict

def compute_statistics(runs_list):
    metrics_list = [run['metrics'] for run in runs_list]
    df = pd.DataFrame(metrics_list)
    stats = {}
    for col in df.columns:
        stats[f"{col}_mean"] = df[col].mean()
        stats[f"{col}_std"] = df[col].std()
    return stats

def compare_results(all_results):
    comparison_data = []
    for task_id, methods_data in all_results.items():
        for method, data in methods_data.items():
            if method in ['dataset_name', 'task_type', 'evaluation_measure']:
                continue
            if 'stats' not in data or not data['stats']:
                continue
            row = {
                'task_id': task_id,
                'method': method,
                'dataset_name': methods_data['dataset_name'],
                'task_type': methods_data['task_type'],
                'evaluation_measure': methods_data['evaluation_measure']
            }
            for key, value in data['stats'].items():
                if key.endswith('_mean'):
                    row[key] = value
            comparison_data.append(row)
    return pd.DataFrame(comparison_data)

def display_and_save_results(all_results):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    for task_id, data in all_results.items():
        print(f"\nTask {task_id} ({data['dataset_name']}) - {data['task_type']} Results:")
        print(f"Evaluation measure: {data['evaluation_measure']}")
        for method in ['ludwig', 'autogluon', 'h2o']:
            if method in data and data[method]['runs']:
                print(f"\n{method.upper()} Results:")
                runs_df = pd.DataFrame([{
                    'seed': run['seed'],
                    **run['metrics']
                } for run in data[method]["runs"]])
                print(runs_df)
                runs_df.to_csv(os.path.join(save_dir, f"task_{task_id}_{method}_runs.csv"), index=False)
                if 'stats' in data[method] and data[method]['stats']:
                    print(f"{method.upper()} Statistics:")
                    print(pd.Series(data[method]['stats']))
    
    comparison_df = compare_results(all_results)
    comparison_df.to_csv(os.path.join(save_dir, "methods_comparison.csv"), index=False)
    print("\nComparison Summary:")
    print(comparison_df)

def main():
    h2o.init()
    all_results = {}
    for task_id in TASK_IDS:
        print(f"Processing task {task_id}...")
        train_df, test_df, target_column, evaluation_measure, dataset_name, task_type = download_openml_task_with_splits(task_id)
        if train_df is None:
            continue
        print(f"Dataset: {dataset_name}")
        print(f"Task type: {task_type}")
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        task_results = {
            'dataset_name': dataset_name,
            'task_type': task_type,
            'evaluation_measure': evaluation_measure,
            'ludwig': {'runs': []},
            'autogluon': {'runs': []},
            'h2o': {'runs': []}
        }
        for seed in SEEDS:
            print(f"Running seed {seed}...")
            print("Running Ludwig...")
            start_time = time.time()
            ludwig_results, ludwig_metrics = run_ludwig_experiment(train_df, test_df, target_column, seed, task_type)
            ludwig_time = time.time() - start_time
            if ludwig_results and ludwig_metrics:
                ludwig_metrics['time_taken'] = ludwig_time
                task_results['ludwig']['runs'].append({
                    'seed': seed,
                    'metrics': ludwig_metrics
                })
                print(f"Ludwig completed in {ludwig_time:.2f}s")
            
            print("Running AutoGluon...")
            start_time = time.time()
            ag_results, ag_metrics = run_autogluon_experiment(train_df, test_df, target_column, seed, task_type)
            ag_time = time.time() - start_time
            if ag_results and ag_metrics:
                ag_metrics['time_taken'] = ag_time
                task_results['autogluon']['runs'].append({
                    'seed': seed,
                    'metrics': ag_metrics
                })
                print(f"AutoGluon completed in {ag_time:.2f}s")
            
            print("Running H2O...")
            start_time = time.time()
            h2o_results, h2o_metrics = run_h2o_experiment(train_df, test_df, target_column, seed, task_type)
            h2o_time = time.time() - start_time
            if h2o_results and h2o_metrics:
                h2o_metrics['time_taken'] = h2o_time
                task_results['h2o']['runs'].append({
                    'seed': seed,
                    'metrics': h2o_metrics
                })
                print(f"H2O completed in {h2o_time:.2f}s")
        for method in ['ludwig', 'autogluon', 'h2o']:
            if task_results[method]['runs']:
                task_results[method]['stats'] = compute_statistics(task_results[method]['runs'])
        all_results[task_id] = task_results
    display_and_save_results(all_results)
    print("All tasks processed.")
    h2o.shutdown()

if __name__ == "__main__":
    main()