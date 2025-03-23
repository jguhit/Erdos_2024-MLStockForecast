import numpy as np

def calculate_growth_percentage(initial_value, final_value):
    """Calculate percentage growth from initial to final value"""
    return ((final_value - initial_value) / initial_value) * 100

def calculate_model_performance(model_trades, model_opens, accuracy_scores, rmse_scores, model_name):
    """
    Calculate performance metrics for a model across all folds
    
    Parameters:
    -----------
    model_trades : dict
        Dictionary with fold indices as keys and trade results as values
    model_opens : dict
        Dictionary with fold indices as keys and opening prices as values
    accuracy_scores : list
        List of accuracy scores for each fold
    rmse_scores : list
        List of RMSE scores for each fold
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    dict: Dictionary containing all performance metrics
    """
    num_folds = len(model_trades)
    fold_results = {}
    total_model_value = 1
    
    # Calculate per-fold metrics
    for i in range(num_folds):
        final_value = model_trades[i]
        growth_pct = calculate_growth_percentage(1, final_value)
        total_model_value *= final_value
        
        fold_results[i] = {
            "final_value": final_value,
            "growth_percentage": growth_pct,
            "accuracy": accuracy_scores[i] if i < len(accuracy_scores) else None,
            "rmse": rmse_scores[i] if i < len(rmse_scores) else None
        }
    
    # Calculate overall metrics
    total_growth = calculate_growth_percentage(1, total_model_value)
    mean_accuracy = np.mean(accuracy_scores) if accuracy_scores else None
    mean_rmse = np.mean(rmse_scores) if rmse_scores else None
    
    return {
        "model_name": model_name,
        "fold_results": fold_results,
        "total_value": total_model_value,
        "total_growth": total_growth,
        "mean_accuracy": mean_accuracy,
        "mean_rmse": mean_rmse
    }

def print_performance_report(performance_results, title=None):
    """
    Print a formatted performance report
    
    Parameters:
    -----------
    performance_results : dict
        Dictionary of performance metrics from calculate_model_performance
    title : str, optional
        Title for the report section
    """
    if title:
        print(f"\n{title}")
    
    model_name = performance_results["model_name"]
    fold_results = performance_results["fold_results"]
    
    # Print per-fold metrics
    for i in fold_results:
        res = fold_results[i]
        print(f"Fold {i+1}:")
        print(f"  {model_name}: Final Value = {res['final_value']:.4f}, Growth = {res['growth_percentage']:.2f}%")
        
        if res['accuracy'] is not None:
            print(f"  Accuracy: {res['accuracy']:.4f}")
        if res['rmse'] is not None:
            print(f"  RMSE: {res['rmse']:.4f}")
    
    # Print overall metrics
    print(f"\nTotal Performance for {model_name}:")
    print(f"  Final Value = {performance_results['total_value']:.4f}, Growth = {performance_results['total_growth']:.2f}%")
    
    if performance_results['mean_accuracy'] is not None:
        print(f"  Mean Accuracy: {performance_results['mean_accuracy']:.4f}")
    if performance_results['mean_rmse'] is not None:
        print(f"  Mean RMSE: {performance_results['mean_rmse']:.4f}")

def compare_models(model_results_list):
    """
    Compare multiple models side by side
    
    Parameters:
    -----------
    model_results_list : list
        List of performance result dictionaries from calculate_model_performance
    """
    print("\nModel Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} | {'Final Value':<15} | {'Growth %':<10} | {'Accuracy':<10} | {'RMSE':<10}")
    print("-" * 80)
    
    for results in model_results_list:
        name = results["model_name"]
        print(f"{name:<20} | {results['total_value']:<15.4f} | {results['total_growth']:<10.2f} | "
              f"{results['mean_accuracy']:<10.4f} | {results['mean_rmse']:<10.4f}")
        
def calculate_test_performance(test_score, accuracy, rmse, model_name):
    """
    Calculate performance metrics for test set
    
    Parameters:
    -----------
    test_score : float
        Final portfolio value from simulation
    accuracy : float
        Accuracy score on test set
    rmse : float
        RMSE score on test set
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    dict: Dictionary containing test performance metrics
    """
    growth_pct = calculate_growth_percentage(1, test_score)
    
    return {
        "model_name": model_name,
        "final_value": test_score,
        "growth_percentage": growth_pct,
        "accuracy": accuracy,
        "rmse": rmse
    }

def print_test_performance(performance_results, title=None):
    """
    Print a formatted test performance report
    
    Parameters:
    -----------
    performance_results : dict
        Dictionary of performance metrics from calculate_test_performance
    title : str, optional
        Title for the report section
    """
    if title:
        print(f"\n{title}")
    
    model_name = performance_results["model_name"]
    
    print(f"{model_name} Test Performance:")
    print(f"  Final Value = {performance_results['final_value']:.4f}")
    print(f"  Growth = {performance_results['growth_percentage']:.2f}%")
    print(f"  Accuracy: {performance_results['accuracy']:.4f}")
    print(f"  RMSE: {performance_results['rmse']:.4f}")

def compare_test_models(test_results_list):
    """
    Compare multiple test models side by side
    
    Parameters:
    -----------
    test_results_list : list
        List of test performance dictionaries from calculate_test_performance
    """
    print("\nTest Model Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} | {'Final Value':<15} | {'Growth %':<10} | {'Accuracy':<10} | {'RMSE':<10}")
    print("-" * 80)
    
    for results in test_results_list:
        name = results["model_name"]
        print(f"{name:<20} | {results['final_value']:<15.4f} | {results['growth_percentage']:<10.2f} | "
              f"{results['accuracy']:<10.4f} | {results['rmse']:<10.4f}")