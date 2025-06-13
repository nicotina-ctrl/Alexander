"""
Run the enhanced crypto prediction pipeline with whale and social data
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the pipeline directory to path
sys.path.append('/content/drive/MyDrive/crypto_pipeline_whale')

from pipeline import EnhancedCryptoPipeline
from config import *

def main():
    """Main function to run the enhanced pipeline"""
    
    print("\n" + "="*80)
    print("ENHANCED CRYPTO PREDICTION PIPELINE V2.0")
    print("With Whale Activity Prioritization and Deadband Targets")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print("="*80 + "\n")
    
    # Configuration
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    horizons = [6, 12, 24]  # 30min, 60min, 120min
    
    # Set to None to process all files (including new 2 days of data)
    max_files = None
    
    # Initialize pipeline
    pipeline = EnhancedCryptoPipeline(
        data_path=GDRIVE_DIR,
        use_sqlite=False  # Use fresh data files
    )
    
    # Clear cache to ensure fresh processing with new data
    print("Clearing cache to process new data...")
    cache_files = [
        "aggregated_data_latest_whale_social.parquet",
        "features_latest_whale_social.parquet",
        "whale_features_latest.parquet",
        "social_features_latest.parquet"
    ]
    
    for cache_file in cache_files:
        cache_path = os.path.join(CACHE_DIR, cache_file)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"  Removed {cache_file}")
    
    # Run pipeline with optimization enabled
    print("\nRunning enhanced pipeline with hyperparameter optimization...")
    results = pipeline.run_enhanced_pipeline(
        symbols=symbols,
        horizons=horizons,
        max_files=max_files,
        optimize_params=True,  # Enable hyperparameter optimization
        batch_size=100,  # Larger batch size for efficiency
        include_whale_data=True,
        include_social_data=True
    )
    
    # Save results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    summary = {}
    for symbol in results:
        summary[symbol] = {}
        for metric_key in results[symbol]:
            if isinstance(results[symbol][metric_key], dict):
                # Extract key metrics
                metrics = results[symbol][metric_key]
                summary[symbol][metric_key] = {
                    'direction_accuracy': metrics.get('direction_accuracy', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0)
                }
    
    # Print summary
    for symbol in summary:
        print(f"\n{symbol}:")
        for model_horizon in summary[symbol]:
            metrics = summary[symbol][model_horizon]
            print(f"  {model_horizon}:")
            print(f"    Direction Accuracy: {metrics['direction_accuracy']:.2%}")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Save detailed results
    results_path = os.path.join(MODEL_SAVE_DIR, "results_enhanced.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Save models
    print("\nSaving enhanced models...")
    pipeline.save_enhanced_models()
    
    # Feature importance analysis
    print("\n" + "="*80)
    print("TOP WHALE FEATURES BY IMPORTANCE")
    print("="*80)
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        for horizon in horizons:
            safe_sym = symbol.replace('/', '_').replace(':', '_')
            importance_key = f"{safe_sym}_importance_{horizon}"
            
            if importance_key in pipeline.feature_importances:
                importance_df = pipeline.feature_importances[importance_key]
                
                # Filter whale features
                whale_features = importance_df[
                    importance_df['feature'].str.contains('whale|inst_|retail_|smart_dumb', na=False)
                ]
                
                if not whale_features.empty:
                    print(f"\n  Horizon {horizon*5} min:")
                    top_whale = whale_features.nlargest(10, 'importance')
                    for idx, row in top_whale.iterrows():
                        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Performance comparison
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    best_models = {}
    for symbol in results:
        best_models[symbol] = {}
        for horizon in horizons:
            best_acc = 0
            best_model = None
            
            # Check each model type
            for model_type in ['xgb', 'lgb', 'ridge', 'mlp', 'ensemble']:
                key = f"{model_type}_{horizon}"
                if key in results[symbol]:
                    acc = results[symbol][key].get('direction_accuracy', 0)
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model_type
            
            best_models[symbol][horizon] = {
                'model': best_model,
                'accuracy': best_acc
            }
    
    # Print best models
    for symbol in best_models:
        print(f"\n{symbol}:")
        for horizon in best_models[symbol]:
            info = best_models[symbol][horizon]
            print(f"  {horizon*5} min: {info['model'].upper()} (Acc: {info['accuracy']:.2%})")
    
    # Trading recommendations
    print("\n" + "="*80)
    print("TRADING RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. Focus on high-confidence signals:")
    print(f"   - Only trade when |predicted_return| > {DIRECTION_CONFIDENCE_THRESHOLD*100:.2f}%")
    print(f"   - This filters out noise from the {DIRECTION_DEADBAND*100:.2f}% deadband")
    
    print("\n2. Best performing horizons:")
    for symbol in best_models:
        best_horizon = max(best_models[symbol].items(), 
                          key=lambda x: x[1]['accuracy'])[0]
        print(f"   - {symbol}: {best_horizon*5} minutes")
    
    print("\n3. Whale signal priority:")
    print("   - inst_participation_zscore_12: Strong institutional activity indicator")
    print("   - whale_sentiment_composite: Combined whale behavior signal")
    print("   - smart_dumb_divergence: Smart money positioning")
    
    print("\n4. Risk management:")
    print("   - Use volatility-based position sizing")
    print("   - Set stops at 2x ATR from entry")
    print("   - Scale out at profit targets: 1x, 2x, 3x expected move")
    
    print("\n" + "="*80)
    print(f"Pipeline completed at: {datetime.now()}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()