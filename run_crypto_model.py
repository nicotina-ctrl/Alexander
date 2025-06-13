#!/usr/bin/env python3
"""
Run the Crypto LOB Prediction Model
"""

# Import the model class from the main file
# Make sure crypto_lob_model.py is in the same directory
from crypto_lob_model import CryptoLOBPredictor

def run_prediction():
    """Execute the full prediction pipeline"""
    
    # Initialize predictor with database path
    predictor = CryptoLOBPredictor(
        db_path='/content/drive/MyDrive/crypto_pipeline_whale/crypto_data_RAW_FULL.db'
    )
    
    # Store all results
    all_results = {}
    
    # Process both symbols
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print('='*60)
        
        try:
            # Prepare data with feature engineering
            print("Preparing data and engineering features...")
            X, y = predictor.prepare_data(symbol)
            print(f"Total samples: {len(X)}")
            print(f"Features created: {len(X.columns)}")
            
            # Train models for all horizons
            print("\nTraining models...")
            results, data_splits = predictor.train_models(X, y, symbol)
            
            # Store results
            all_results[symbol] = results
            
            # Display results
            print(f"\n{'='*60}")
            print(f"RESULTS FOR {symbol}")
            print('='*60)
            
            for horizon, metrics in results.items():
                print(f"\n{horizon.upper()} Prediction Horizon:")
                print("-" * 40)
                print(f"MAE:                {metrics['MAE']:.6f}")
                print(f"RMSE:               {metrics['RMSE']:.6f}")
                print(f"R²:                 {metrics['R²']:.4f}")
                print(f"Direction Accuracy: {metrics['Direction Accuracy']:.4f} ({metrics['Direction Accuracy']*100:.2f}%)")
                print(f"Sharpe Ratio:       {metrics['Sharpe Ratio']:.3f}")
                print(f"Max Drawdown:       {metrics['Max Drawdown']:.2%}")
                print(f"Win Rate:           {metrics['Win Rate']:.4f} ({metrics['Win Rate']*100:.2f}%)")
                print(f"Profit Factor:      {metrics['Profit Factor']:.2f}")
                print(f"Information Ratio:  {metrics['Information Ratio']:.3f}")
        
        except Exception as e:
            print(f"\nError processing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all models
    print("\n" + "="*60)
    print("Saving models...")
    predictor.save_models('crypto_lob_models.pkl')
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for symbol in symbols:
        if symbol in all_results:
            print(f"\n{symbol}:")
            avg_accuracy = np.mean([
                metrics['Direction Accuracy'] 
                for metrics in all_results[symbol].values()
            ])
            print(f"  Average Direction Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
            
            avg_sharpe = np.mean([
                metrics['Sharpe Ratio'] 
                for metrics in all_results[symbol].values()
            ])
            print(f"  Average Sharpe Ratio: {avg_sharpe:.3f}")
    
    return predictor, all_results


def predict_new_data(predictor, symbol='BTCUSDT', horizon='30min'):
    """
    Example of using the trained model for predictions
    """
    # Prepare new data
    X_new, _ = predictor.prepare_data(symbol)
    
    # Get the last 100 samples
    X_latest = X_new.tail(100)
    
    # Scale features
    X_scaled = predictor.scalers[symbol].transform(X_latest)
    
    # Get model
    model = predictor.models[f'{symbol}_{horizon}']
    
    # Make predictions
    predictions = model.predict_proba(X_scaled)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'timestamp': X_latest.index,
        'prediction_probability': predictions,
        'prediction': (predictions > 0.5).astype(int),
        'predicted_direction': ['UP' if p > 0.5 else 'DOWN' for p in predictions]
    })
    
    return results


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Run the full pipeline
    predictor, results = run_prediction()
    
    # Example: Make predictions with the trained model
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    try:
        # Get predictions for BTCUSDT 30min horizon
        predictions = predict_new_data(predictor, 'BTCUSDT', '30min')
        print("\nLatest predictions for BTCUSDT (30min horizon):")
        print(predictions.tail(10))
    except Exception as e:
        print(f"Error making predictions: {e}")