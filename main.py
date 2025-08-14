import os
import pandas as pd
import time
import openml
from ludwig.automl import auto_train
from autogluon.tabular import TabularPredictor
import h2o
from h2o.automl import H2OAutoML

# Set to True to run in test mode (only one task and one seed)
TEST_MODE = False

# FRAMEWORKS TO RUN - Set to True/False to enable/disable each framework
# To avoid running all frameworks, set the ones you want to test to True
RUN_LUDWIG = True
RUN_AUTOGLUON = False
RUN_H2O = False

# List of task IDs
TASK_IDS = [52948] if TEST_MODE else [
    10101, 15, 31, 37, 9957,         # Binary classification
    3560, 23, 3011, 18, 53,          # Multi-class classification
    2295, 2301, 52948, 4839          # Regression
]

SEEDS = [123] if TEST_MODE else [123, 2027, 99]

# Time limit for each method in seconds
TIME_LIMIT = 300 if TEST_MODE else 1000

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

def save_incremental_results(task_id, method, task_results, save_dir):
    """Save results incrementally for each framework and task"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        if task_results[method]['runs']:
            # Create DataFrame with all information
            runs_data = []
            for run in task_results[method]['runs']:
                row = {
                    'task_id': task_id,
                    'method': method,
                    'seed': run['seed'],
                    'dataset_name': task_results['dataset_name'],
                    'task_type': task_results['task_type'],
                    'evaluation_measure': task_results['evaluation_measure']
                }
                # Add metrics
                row.update(run['metrics'])
                runs_data.append(row)
            
            runs_df = pd.DataFrame(runs_data)
            filename = f"task_{task_id}_{method}_all_seeds.csv"
            filepath = os.path.join(save_dir, filename)
            runs_df.to_csv(filepath, index=False)
            print(f"Saved incremental results: {filename}")
    except Exception as e:
        print(f"Error saving incremental results for {method}: {e}")

def run_ludwig_experiment(train_dataset, test_dataset, target_column, seed, task_type):
    try:
        results = auto_train(
            dataset=train_dataset,
            target=target_column,
            time_limit_s=TIME_LIMIT,
            tune_for_memory=False,
            random_seed=seed
        )
        eval_results = results.best_model.evaluate(test_dataset)[0]
        metrics = extract_ludwig_metrics(eval_results, target_column, task_type)
        
        del results
        return None, metrics
    except Exception as e:
        print(f"Error in Ludwig training with seed {seed}: {e}")
        return None, None

def run_autogluon_experiment(train_dataset, test_dataset, target_column, seed, task_type):
    import shutil
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
        
        del predictor
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        return None, metrics
    except Exception as e:
        print(f"Error in AutoGluon training with seed {seed}: {e}")
        return None, None

def run_h2o_experiment(train_dataset, test_dataset, target_column, seed, task_type):
    try:
        h2o_train = h2o.H2OFrame(train_dataset)
        h2o_test = h2o.H2OFrame(test_dataset)
        
        num_classes = None
        if 'classification' in task_type.lower():
            h2o_train[target_column] = h2o_train[target_column].asfactor()
            h2o_test[target_column] = h2o_test[target_column].asfactor()
            # Get number of classes for multi-class detection
            num_classes = h2o_train[target_column].nlevels()[0]
        
        x = h2o_train.columns
        x.remove(target_column)
        
        aml = H2OAutoML(
            max_runtime_secs=TIME_LIMIT,
            seed=seed,
            project_name=f"automl_task_{seed}"
        )
        
        aml.train(x=x, y=target_column, training_frame=h2o_train)
        
        perf = aml.leader.model_performance(h2o_test)
        metrics = extract_h2o_metrics(perf, task_type, num_classes)
        
        h2o.remove([h2o_train, h2o_test])
        try:
            h2o.remove(aml.leaderboard.as_data_frame()['model_id'].tolist())
        except:
            pass
        return None, metrics
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
        print(f"Ludwig eval_results: {eval_results}")
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
        print(f"AutoGluon eval_results: {eval_results}")
        if isinstance(eval_results, dict):
            if 'classification' in task_type.lower():
                for metric in ['roc_auc', 'accuracy', 'balanced_accuracy', 'mcc', 'f1', 'precision', 'recall']:
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

def safe_h2o_metric(metric_func, is_multiclass=False, metric_name=None):
    try:
        value = metric_func()
        print(f"H2O metric output for {metric_name}: {value} (type: {type(value)})")
        
        if is_multiclass:
            # For multi-class classification, H2O returns different formats
            if metric_name == 'accuracy':
                # For multi-class accuracy, H2O should return the overall accuracy directly
                # If it returns a matrix, extract the overall accuracy
                if isinstance(value, list):
                    if len(value) == 1 and isinstance(value[0], (int, float)):
                        return safe_metric_value(value[0])
                    # If it's a confusion matrix format, calculate accuracy
                    elif len(value) > 0 and isinstance(value[0], list):
                        # Try to calculate accuracy from confusion matrix
                        total_correct = 0
                        total_samples = 0
                        for i, row in enumerate(value[:-1]):  # Exclude totals row if present
                            for j, cell in enumerate(row[:-1]):  # Exclude totals column if present
                                if isinstance(cell, (int, float)):
                                    total_samples += cell
                                    if i == j:  # Diagonal elements are correct predictions
                                        total_correct += cell
                        if total_samples > 0:
                            return safe_metric_value(total_correct / total_samples)
                return safe_metric_value(value)
                
            elif metric_name in ['f1', 'precision', 'recall']:
                # For F1, precision, recall in multi-class, calculate macro average
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], list):
                        # Each row represents a class, extract the metric value for each class
                        class_values = []
                        for row in value:
                            if len(row) >= 2 and isinstance(row[1], (int, float)):
                                class_values.append(row[1])
                        if class_values:
                            return safe_metric_value(sum(class_values) / len(class_values))
                    else:
                        # If it's already a list of values, calculate mean
                        numeric_values = [v for v in value if isinstance(v, (int, float))]
                        if numeric_values:
                            return safe_metric_value(sum(numeric_values) / len(numeric_values))
            
            # For other metrics, try to extract a single value
            if isinstance(value, list):
                if len(value) == 1:
                    return safe_metric_value(value[0])
                elif len(value) > 1 and isinstance(value[0], list):
                    # Look for a summary value (often in the last position)
                    if len(value[-1]) > 0:
                        return safe_metric_value(value[-1][-1])
        else:
            # For binary classification, handle as before
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list) and len(value[0]) > 1:
                    value = value[0][1]
                elif len(value) == 1:
                    value = value[0]
        
        return safe_metric_value(value)
    except Exception as e:
        print(f"Error in safe_h2o_metric for {metric_name}: {e}")
        return None

def extract_h2o_metrics(perf, task_type, num_classes=None):
    metrics_dict = {}
    try:
        if 'classification' in task_type.lower():
            # Detect if it's multi-class (more than 2 classes)
            is_multiclass = num_classes is not None and num_classes > 2
            
            if is_multiclass:
                # For multi-class classification, extract metrics from confusion matrix
                print(f"Processing multi-class classification with {num_classes} classes")
                
                try:
                    # Get confusion matrix for detailed metrics calculation
                    cm = perf.confusion_matrix()
                    if cm is not None:
                        cm_df = cm.as_data_frame()
                        print(f"Confusion Matrix shape: {cm_df.shape}")
                        print(f"Confusion Matrix columns: {cm_df.columns.tolist()}")
                        
                        if not cm_df.empty and len(cm_df) > 1:
                            # Calculate overall accuracy
                            total_correct = 0
                            total_samples = 0
                            class_metrics = []
                            
                            # Process each class (exclude last row/column which are totals)
                            num_classes_cm = min(len(cm_df) - 1, len(cm_df.columns) - 1)
                            
                            for i in range(num_classes_cm):
                                class_stats = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0, 'true_negative': 0}
                                
                                for j in range(num_classes_cm):
                                    cell_value = cm_df.iloc[i, j]
                                    if isinstance(cell_value, (int, float)):
                                        total_samples += cell_value
                                        
                                        if i == j:  # True positives for class i
                                            class_stats['true_positive'] = cell_value
                                            total_correct += cell_value
                                        else:
                                            if i < num_classes_cm:  # False negatives for class i (predicted as other classes)
                                                class_stats['false_negative'] += cell_value
                                            if j < num_classes_cm:  # False positives for class i (other classes predicted as i)
                                                class_stats['false_positive'] += cm_df.iloc[j, i] if isinstance(cm_df.iloc[j, i], (int, float)) else 0
                                
                                # Calculate per-class metrics
                                tp = class_stats['true_positive']
                                fp = class_stats['false_positive'] 
                                fn = class_stats['false_negative']
                                
                                # Precision = TP / (TP + FP)
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                
                                # Recall = TP / (TP + FN)
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                
                                # F1 = 2 * (precision * recall) / (precision + recall)
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                                
                                class_metrics.append({
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1
                                })
                            
                            # Calculate macro averages
                            if class_metrics:
                                metrics_dict['accuracy'] = total_correct / total_samples if total_samples > 0 else 0
                                metrics_dict['precision'] = sum(cm['precision'] for cm in class_metrics) / len(class_metrics)
                                metrics_dict['recall'] = sum(cm['recall'] for cm in class_metrics) / len(class_metrics)
                                metrics_dict['f1'] = sum(cm['f1'] for cm in class_metrics) / len(class_metrics)
                                
                                print(f"Calculated metrics - Accuracy: {metrics_dict['accuracy']:.4f}, "
                                      f"Precision: {metrics_dict['precision']:.4f}, "
                                      f"Recall: {metrics_dict['recall']:.4f}, "
                                      f"F1: {metrics_dict['f1']:.4f}")
                                
                except Exception as e:
                    print(f"Could not extract detailed metrics from confusion matrix: {e}")
                    # Fallback: set accuracy only and None for others
                    try:
                        # Try alternative method for accuracy
                        if hasattr(perf, 'mean_per_class_accuracy'):
                            metrics_dict['accuracy'] = safe_h2o_metric(lambda: perf.mean_per_class_accuracy(), is_multiclass, 'mean_per_class_accuracy')
                    except:
                        pass
                    
                    # Set unavailable metrics to None
                    if 'accuracy' not in metrics_dict:
                        metrics_dict['accuracy'] = None
                    metrics_dict['f1'] = None
                    metrics_dict['precision'] = None  
                    metrics_dict['recall'] = None
                
                # These metrics should be available for multi-class
                metrics_dict['roc_auc'] = safe_h2o_metric(lambda: perf.auc(), is_multiclass, 'roc_auc')
                
            else:
                # For binary classification, use standard methods
                metrics_dict['accuracy'] = safe_h2o_metric(lambda: perf.accuracy(), is_multiclass, 'accuracy')
                metrics_dict['f1'] = safe_h2o_metric(lambda: perf.F1(), is_multiclass, 'f1')
                metrics_dict['precision'] = safe_h2o_metric(lambda: perf.precision(), is_multiclass, 'precision')
                metrics_dict['recall'] = safe_h2o_metric(lambda: perf.recall(), is_multiclass, 'recall')
                metrics_dict['roc_auc'] = safe_h2o_metric(lambda: perf.auc(), is_multiclass, 'roc_auc')
            
            # These metrics should be available for both binary and multi-class
            metrics_dict['logloss'] = safe_h2o_metric(lambda: perf.logloss(), is_multiclass, 'logloss')
            metrics_dict['aucpr'] = safe_h2o_metric(lambda: perf.aucpr(), is_multiclass, 'aucpr')
            metrics_dict['mean_per_class_error'] = safe_h2o_metric(lambda: perf.mean_per_class_error(), is_multiclass, 'mean_per_class_error')
            
        else:
            # For regression
            metrics_dict['r2'] = safe_h2o_metric(lambda: perf.r2(), False, 'r2')
            metrics_dict['rmse'] = safe_h2o_metric(lambda: perf.rmse(), False, 'rmse')
            metrics_dict['mae'] = safe_h2o_metric(lambda: perf.mae(), False, 'mae')
            metrics_dict['mse'] = safe_h2o_metric(lambda: perf.mse(), False, 'mse')
            metrics_dict['rmsle'] = safe_h2o_metric(lambda: perf.rmsle(), False, 'rmsle')
            metrics_dict['mean_residual_deviance'] = safe_h2o_metric(lambda: perf.mean_residual_deviance(), False, 'mean_residual_deviance')
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
            # Include both mean and std metrics
            for key, value in data['stats'].items():
                row[key] = value  # Changed to include all stats (mean and std)
            comparison_data.append(row)
    return pd.DataFrame(comparison_data)

def display_and_save_final_results(all_results):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    # Get enabled frameworks
    enabled_frameworks = []
    if RUN_LUDWIG:
        enabled_frameworks.append('ludwig')
    if RUN_AUTOGLUON:
        enabled_frameworks.append('autogluon')
    if RUN_H2O:
        enabled_frameworks.append('h2o')
    
    for task_id, data in all_results.items():
        print(f"\nTask {task_id} ({data['dataset_name']}) - {data['task_type']} Results:")
        print(f"Evaluation measure: {data['evaluation_measure']}")
        for method in enabled_frameworks:
            if method in data and data[method]['runs'] and 'stats' in data[method]:
                print(f"\n{method.upper()} Statistics:")
                print(pd.Series(data[method]['stats']))
    
    comparison_df = compare_results(all_results)
    comparison_df.to_csv(os.path.join(save_dir, "methods_comparison.csv"), index=False)
    print("\nComparison Summary saved to methods_comparison.csv:")
    print(comparison_df)

def main():
    # Check which frameworks are enabled
    enabled_frameworks = []
    if RUN_LUDWIG:
        enabled_frameworks.append('ludwig')
    if RUN_AUTOGLUON:
        enabled_frameworks.append('autogluon')
    if RUN_H2O:
        enabled_frameworks.append('h2o')
    
    if not enabled_frameworks:
        print("No frameworks selected to run. Please enable at least one framework.")
        return
    
    print(f"Running frameworks: {', '.join(enabled_frameworks)}")
    
    if RUN_H2O:
        h2o.init()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results")
    
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
        }
        
        # Initialize only enabled frameworks
        for framework in enabled_frameworks:
            task_results[framework] = {'runs': []}
        
        for seed in SEEDS:
            print(f"Running seed {seed}...")
            
            # Ludwig
            if RUN_LUDWIG:
                print("Running Ludwig...")
                start_time = time.time()
                ludwig_results, ludwig_metrics = run_ludwig_experiment(train_df, test_df, target_column, seed, task_type)
                ludwig_time = time.time() - start_time
                if ludwig_metrics:
                    ludwig_metrics['time_taken'] = ludwig_time
                    task_results['ludwig']['runs'].append({
                        'seed': seed,
                        'metrics': ludwig_metrics
                    })
                    print(f"Ludwig completed in {ludwig_time:.2f}s")
                    save_incremental_results(task_id, 'ludwig', task_results, save_dir)
            
            # AutoGluon
            if RUN_AUTOGLUON:
                print("Running AutoGluon...")
                start_time = time.time()
                ag_results, ag_metrics = run_autogluon_experiment(train_df, test_df, target_column, seed, task_type)
                ag_time = time.time() - start_time
                if ag_metrics:
                    ag_metrics['time_taken'] = ag_time
                    task_results['autogluon']['runs'].append({
                        'seed': seed,
                        'metrics': ag_metrics
                    })
                    print(f"AutoGluon completed in {ag_time:.2f}s")
                    save_incremental_results(task_id, 'autogluon', task_results, save_dir)
            
            # H2O
            if RUN_H2O:
                print("Running H2O...")
                start_time = time.time()
                h2o_results, h2o_metrics = run_h2o_experiment(train_df, test_df, target_column, seed, task_type)
                h2o_time = time.time() - start_time
                if h2o_metrics:
                    h2o_metrics['time_taken'] = h2o_time
                    task_results['h2o']['runs'].append({
                        'seed': seed,
                        'metrics': h2o_metrics
                    })
                    print(f"H2O completed in {h2o_time:.2f}s")
                    save_incremental_results(task_id, 'h2o', task_results, save_dir)
        
        # Calculate statistics for current task (only for enabled frameworks)
        for method in enabled_frameworks:
            if task_results[method]['runs']:
                task_results[method]['stats'] = compute_statistics(task_results[method]['runs'])
        
        all_results[task_id] = task_results
    
    display_and_save_final_results(all_results)
    print("All tasks processed.")
    
    if RUN_H2O:
        h2o.shutdown()

if __name__ == "__main__":
    main()