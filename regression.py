import pandas as pd
import numpy as np
from utils import (
    load_data, explore_data, preprocess_data, 
    evaluate_model, plot_predictions, get_baseline_models,
    save_results_to_file, compare_models
)

def main():
    """Main function to run regression analysis"""
    print("Loading Boston Housing Dataset...")
    df = load_data()
    
    print("\nExploring the dataset...")
    correlation_matrix = explore_data(df)
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    print("\nTraining baseline models...")
    models = get_baseline_models()
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
        
        # Plot predictions
        plot_predictions(y_test, result['predictions'], name)
    
    # Compare models
    results_df = compare_models(results)
    
    # Save results
    save_results_to_file(results, 'baseline_results.txt')
    
    print("\nBaseline model comparison completed!")
    print("Check the generated plots and baseline_results.txt for detailed results.")

if __name__ == "__main__":
    main()
