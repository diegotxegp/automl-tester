import os
import pandas as pd
import time
import openml
from ludwig.automl import auto_train
from autogluon.tabular import TabularPredictor
import h2o
from h2o.automl import H2OAutoML

# List of task IDs
TEST_MODE = True
TASK_IDS = [31] if TEST_MODE else [2295, 167141, 2301, 23, 31, 37, 3560, 52948, 18, 4839, 9957, 53] # Cholesterol, Churn, Cloud, Cmc, Credit-g, Diabetes, analcatdata_dmft, liver-disorders, mfeat-morphological,plasma_retinol, qsar-biodeg, vehicle
SEEDS = [42] if TEST_MODE else [42, 123, 2023, 1, 99]

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

def run_ludwig_experiment(train_dataset, test_dataset, target_column, seed):
    try:
        time_limit = 300 if TEST_MODE else 7200
        results = auto_train(
            dataset=train_dataset,
            target=target_column,
            time_limit_s=time_limit,
            tune_for_memory=False,
            random_seed=seed
        )
        eval_results = results.best_model.evaluate(test_dataset)
        return results, eval_results
    except Exception as e:
        print(f"Error in Ludwig training with seed {seed}: {e}")
        return None, None

def run_autogluon_experiment(train_dataset, test_dataset, target_column, seed):
    try:
        time_limit = 300 if TEST_MODE else 7200
        
        # Create temporary directory for AutoGluon
        model_dir = f"./autogluon_models/task_seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine eval metric based on target type
        if train_dataset[target_column].dtype == 'object' or train_dataset[target_column].nunique() <= 10:
            # Classification
            if train_dataset[target_column].nunique() == 2:
                eval_metric = 'roc_auc'  # Binary classification
            else:
                eval_metric = 'accuracy'  # Multi-class classification
        else:
            # Regression
            eval_metric = 'rmse'
        
        predictor = TabularPredictor(
            label=target_column,
            path=model_dir,
            eval_metric=eval_metric,
            verbosity=2
        ).fit(
            train_data=train_dataset,
            time_limit=time_limit,
            presets='medium_quality'
        )
        
        # Evaluate on test set
        eval_results = predictor.evaluate(test_dataset, silent=False)
        
        return predictor, eval_results
    except Exception as e:
        print(f"Error in AutoGluon training with seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_h2o_experiment(train_dataset, test_dataset, target_column, seed):
    try:
        time_limit = 300 if TEST_MODE else 7200
        
        # Convert to H2O frames
        h2o_train = h2o.H2OFrame(train_dataset)
        h2o_test = h2o.H2OFrame(test_dataset)
        
        # Set target as factor for classification
        if train_dataset[target_column].dtype == 'object' or train_dataset[target_column].nunique() < 10:
            h2o_train[target_column] = h2o_train[target_column].asfactor()
            h2o_test[target_column] = h2o_test[target_column].asfactor()
        
        x = h2o_train.columns
        x.remove(target_column)
        
        aml = H2OAutoML(
            max_runtime_secs=time_limit,
            seed=seed,
            project_name=f"automl_task_{seed}"
        )
        
        aml.train(x=x, y=target_column, training_frame=h2o_train)
        
        # Evaluate on test set
        perf = aml.leader.model_performance(h2o_test)
        
        return aml, perf
    except Exception as e:
        print(f"Error in H2O training with seed {seed}: {e}")
        return None, None

def extract_ludwig_metrics(eval_results, target_column):
    metrics_dict = {}
    try:
        if isinstance(eval_results, tuple) and len(eval_results) > 0:
            eval_stats = eval_results[0]
            
            if target_column in eval_stats:
                feature_metrics = eval_stats[target_column]
            elif 'combined' in eval_stats:
                feature_metrics = eval_stats['combined']
            else:
                feature_metrics = next(iter(eval_stats.values()))
            
            for metric_name, metric_value in feature_metrics.items():
                metrics_dict[metric_name] = metric_value
                
    except Exception as e:
        print(f"Error extracting Ludwig metrics: {e}")
    
    return metrics_dict

def extract_autogluon_metrics(eval_results):
    metrics_dict = {}
    try:
        print(f"AutoGluon eval_results type: {type(eval_results)}")
        print(f"AutoGluon eval_results content: {eval_results}")
        
        if isinstance(eval_results, dict):
            for metric_name, metric_value in eval_results.items():
                metrics_dict[metric_name] = metric_value
        elif isinstance(eval_results, (int, float)):
            metrics_dict['score'] = eval_results
        else:
            # Try to convert to dict if it has attributes
            if hasattr(eval_results, '__dict__'):
                metrics_dict = eval_results.__dict__
            else:
                metrics_dict['score'] = float(eval_results) if eval_results is not None else 0.0
                
    except Exception as e:
        print(f"Error extracting AutoGluon metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics_dict

def extract_h2o_metrics(perf, task_type):
    metrics_dict = {}
    try:
        if 'classification' in task_type.lower():
            metrics_dict['auc'] = perf.auc() if perf.auc() else None
            metrics_dict['accuracy'] = perf.accuracy()[0][1] if perf.accuracy() else None
            metrics_dict['logloss'] = perf.logloss() if perf.logloss() else None
        else:
            metrics_dict['rmse'] = perf.rmse() if perf.rmse() else None
            metrics_dict['mae'] = perf.mae() if perf.mae() else None
            metrics_dict['r2'] = perf.r2() if perf.r2() else None
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
            
            # Skip if no stats (no successful runs)
            if 'stats' not in data or not data['stats']:
                continue
                
            row = {
                'task_id': task_id,
                'method': method,
                'dataset_name': methods_data['dataset_name'],
                'task_type': methods_data['task_type'],
                'evaluation_measure': methods_data['evaluation_measure']
            }
            
            # Add mean metrics
            for key, value in data['stats'].items():
                if key.endswith('_mean'):
                    row[key] = value
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def display_and_save_results(all_results):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    for task_id, data in all_results.items():
        print(f"\nTask {task_id} ({data['dataset_name']}) - {data['task_type']} Results:")
        print(f"Evaluation measure: {data['evaluation_measure']}")
        
        # Display results for each method
        for method in ['ludwig', 'autogluon', 'h2o']:
            if method in data and data[method]['runs']:
                print(f"\n{method.upper()} Results:")
                runs_df = pd.DataFrame([{
                    'seed': run['seed'],
                    **run['metrics']
                } for run in data[method]["runs"]])
                print(runs_df)
                
                # Save individual results
                runs_df.to_csv(os.path.join(save_dir, f"task_{task_id}_{method}_runs.csv"), index=False)
                
                # Print stats if available
                if 'stats' in data[method] and data[method]['stats']:
                    print(f"{method.upper()} Statistics:")
                    stats_series = pd.Series(data[method]['stats'])
                    print(stats_series)
    
    # Create and save comparison
    comparison_df = compare_results(all_results)
    comparison_df.to_csv(os.path.join(save_dir, "methods_comparison.csv"), index=False)
    print("\nComparison Summary:")
    print(comparison_df)

def main():
    # Initialize H2O
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
            
            # Ludwig
            print("Running Ludwig...")
            start_time = time.time()
            ludwig_results, ludwig_eval = run_ludwig_experiment(train_df, test_df, target_column, seed)
            ludwig_time = time.time() - start_time
            
            if ludwig_results and ludwig_eval:
                metrics = extract_ludwig_metrics(ludwig_eval, target_column)
                metrics['time_taken'] = ludwig_time
                task_results['ludwig']['runs'].append({
                    'seed': seed,
                    'metrics': metrics
                })
                print(f"Ludwig completed in {ludwig_time:.2f}s")
            
            # AutoGluon
            print("Running AutoGluon...")
            start_time = time.time()
            ag_results, ag_eval = run_autogluon_experiment(train_df, test_df, target_column, seed)
            ag_time = time.time() - start_time
            
            if ag_results and ag_eval is not None:
                metrics = extract_autogluon_metrics(ag_eval)
                metrics['time_taken'] = ag_time
                task_results['autogluon']['runs'].append({
                    'seed': seed,
                    'metrics': metrics
                })
                print(f"AutoGluon completed in {ag_time:.2f}s")
            else:
                print(f"AutoGluon failed for seed {seed}")
            
            # H2O
            print("Running H2O...")
            start_time = time.time()
            h2o_results, h2o_eval = run_h2o_experiment(train_df, test_df, target_column, seed)
            h2o_time = time.time() - start_time
            
            if h2o_results and h2o_eval:
                metrics = extract_h2o_metrics(h2o_eval, task_type)
                metrics['time_taken'] = h2o_time
                task_results['h2o']['runs'].append({
                    'seed': seed,
                    'metrics': metrics
                })
                print(f"H2O completed in {h2o_time:.2f}s")

        # Compute statistics for each method
        for method in ['ludwig', 'autogluon', 'h2o']:
            if task_results[method]['runs']:
                task_results[method]['stats'] = compute_statistics(task_results[method]['runs'])

        all_results[task_id] = task_results

    display_and_save_results(all_results)
    print("All tasks processed.")
    
    # Shutdown H2O
    h2o.shutdown()

if __name__ == "__main__":
    main()