"""
Main script to run the enhanced crypto prediction pipeline with whale and social data
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils import install_required_packages
from pipeline import EnhancedCryptoPipeline
from data_loader import DataLoader


def run_smoke_test():
    """Run a quick smoke test with limited data"""
    print("\n" + "="*80)
    print("RUNNING SMOKE TEST")
    print("="*80)
    
    pipeline = EnhancedCryptoPipeline(LOCAL_DIR)
    
    # Run with limited data
    results = pipeline.run_enhanced_pipeline(
        symbols=['BTC/USDT'],
        horizons=[1],  # Just 5-min horizon
        max_files=MAX_FILES_SMOKE_TEST,
        optimize_params=False,
        include_whale_data=True,
        include_social_data=True
    )
    
    if results:
        print("\n✅ Smoke test passed!")
        return True
    else:
        print("\n❌ Smoke test failed!")
        return False


def run_full_pipeline():
    """Run the full pipeline with all features"""
    print("\n" + "="*80)
    print("RUNNING FULL PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    # Use SQLite if database exists, otherwise load from parquet files
    use_sqlite = os.path.exists(SQLITE_DB_PATH)
    if use_sqlite:
        print(f"Found SQLite database at {SQLITE_DB_PATH}")
        print("Will load integrated data from database")
    else:
        print("No SQLite database found")
        print("Will load and integrate data from parquet files")
    
    pipeline = EnhancedCryptoPipeline(GDRIVE_DIR, use_sqlite=use_sqlite)
    
    # Run full pipeline
    results = pipeline.run_enhanced_pipeline(
        symbols=DEFAULT_SYMBOLS,
        horizons=DEFAULT_HORIZONS,
        max_files=None,  # Process all files
        optimize_params=False,
        batch_size=DEFAULT_BATCH_SIZE,
        include_whale_data=True,
        include_social_data=True
    )
    
    # Save models
    pipeline.save_enhanced_models()
    
    # Print results summary
    if results:
        print("\n" + "="*80)
        print("PIPELINE RESULTS SUMMARY")
        print("="*80)
        
        pipeline.evaluator.print_results_summary(results)
        
        # Save results
        results_data = {
            'results': results,
            'feature_importances': pipeline.feature_importances,
            'selected_features': pipeline.selected_features,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_path = os.path.join(MODEL_SAVE_DIR, "enhanced_results_with_whale_social.pkl")
        joblib.dump(results_data, results_path)
        print(f"\nResults saved to: {results_path}")
        
        # Analyze whale and social feature importance
        analyze_whale_social_features(pipeline.feature_importances)
    
    return results


def analyze_whale_social_features(feature_importances):
    """Analyze the importance of whale and social features"""
    print("\n" + "="*80)
    print("WHALE AND SOCIAL FEATURE ANALYSIS")
    print("="*80)
    
    for key, importance_df in feature_importances.items():
        if importance_df is None or importance_df.empty:
            continue
            
        # Extract symbol and horizon from key
        parts = key.split('_')
        symbol = parts[0] + '/' + parts[1]
        horizon = int(parts[-1]) * 5
        
        print(f"\n{symbol} @ {horizon}min:")
        
        # Get whale features
        whale_features = importance_df[
            importance_df['feature'].str.contains('whale|inst|retail|smart_dumb', regex=True)
        ].sort_values('importance', ascending=False)
        
        # Get social features
        social_features = importance_df[
            importance_df['feature'].str.contains('mention|sentiment|social|viral', regex=True)
        ].sort_values('importance', ascending=False)
        
        if not whale_features.empty:
            print("\n  Top Whale Features:")
            for idx, row in whale_features.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            print(f"  Total whale features: {len(whale_features)}")
            print(f"  Whale features importance sum: {whale_features['importance'].sum():.4f}")
        
        if not social_features.empty:
            print("\n  Top Social Features:")
            for idx, row in social_features.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            print(f"  Total social features: {len(social_features)}")
            print(f"  Social features importance sum: {social_features['importance'].sum():.4f}")
        
        # Calculate percentage of total importance
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            whale_pct = (whale_features['importance'].sum() / total_importance) * 100
            social_pct = (social_features['importance'].sum() / total_importance) * 100
            print(f"\n  Whale features: {whale_pct:.1f}% of total importance")
            print(f"  Social features: {social_pct:.1f}% of total importance")


def demonstrate_prediction():
    """Demonstrate making predictions with the trained model"""
    print("\n" + "="*80)
    print("PREDICTION DEMONSTRATION")
    print("="*80)
    
    # Load the trained pipeline
    pipeline = EnhancedCryptoPipeline(GDRIVE_DIR)
    pipeline.load_enhanced_models()
    
    print(f"Loaded {len(pipeline.models)} models")
    
    # Load recent data for prediction
    print("\nLoading recent data for predictions...")
    data_loader = DataLoader(GDRIVE_DIR)
    
    # Load last 100 5-minute bars
    recent_data = data_loader.load_and_aggregate_batched(
        symbols=['BTC/USDT'],
        max_files=5,  # Just load recent files
        use_cache=False,
        include_whale_data=True,
        include_social_data=True
    )
    
    if len(recent_data) == 0:
        print("No recent data available for prediction")
        return
    
    # Make predictions
    symbol = 'BTC/USDT'
    horizon = 12  # 60 minutes
    
    print(f"\nMaking predictions for {symbol} @ {horizon*5}min horizon...")
    
    try:
        predictions_df = pipeline.predict_new_data(
            recent_data, 
            symbol, 
            horizon,
            include_whale_data=True,
            include_social_data=True
        )
        
        # Display last 10 predictions
        print("\nLast 10 predictions:")
        print(predictions_df.tail(10)[['time', 'ensemble_pred', 'direction', 'confidence']])
        
        # Analyze prediction distribution
        print("\nPrediction Statistics:")
        print(f"Mean prediction: {predictions_df['ensemble_pred'].mean():.6f}")
        print(f"Std prediction: {predictions_df['ensemble_pred'].std():.6f}")
        print(f"UP predictions: {(predictions_df['direction'] == 'UP').sum()} ({(predictions_df['direction'] == 'UP').mean()*100:.1f}%)")
        print(f"DOWN predictions: {(predictions_df['direction'] == 'DOWN').sum()} ({(predictions_df['direction'] == 'DOWN').mean()*100:.1f}%)")
        print(f"Avg confidence: {predictions_df['confidence'].mean():.6f}")
        
    except Exception as e:
        print(f"Error making predictions: {e}")


def check_data_availability():
    """Check what data is available"""
    print("\n" + "="*80)
    print("DATA AVAILABILITY CHECK")
    print("="*80)
    
    # Check for SQLite database
    if os.path.exists(SQLITE_DB_PATH):
        print(f"✓ SQLite database found: {SQLITE_DB_PATH}")
        
        # Get basic info from database
        import sqlite3
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", 
                conn
            )
            print(f"  Tables: {tables['name'].tolist()}")
            
            # Check date range
            try:
                date_range = pd.read_sql_query(
                    "SELECT MIN(time_bucket) as min_date, MAX(time_bucket) as max_date FROM integrated_raw",
                    conn
                )
                print(f"  Date range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
            except:
                pass
    else:
        print("✗ No SQLite database found")
    
    # Check for whale data
    whale_file = os.path.join(WHALE_DATA_DIR, "whale.csv")
    if os.path.exists(whale_file):
        print(f"✓ Whale data found: {whale_file}")
        whale_df = pd.read_csv(whale_file, nrows=5)
        print(f"  Columns: {whale_df.columns.tolist()}")
    else:
        print("✗ Whale data not found")
    
    # Check for social data
    mentions4h_file = os.path.join(WHALE_DATA_DIR, "mentions4h.csv")
    mentions14d_file = os.path.join(WHALE_DATA_DIR, "mentions14d.csv")
    
    if os.path.exists(mentions4h_file):
        print(f"✓ 4H mentions data found: {mentions4h_file}")
    else:
        print("✗ 4H mentions data not found")
        
    if os.path.exists(mentions14d_file):
        print(f"✓ 14D mentions data found: {mentions14d_file}")
    else:
        print("✗ 14D mentions data not found")
    
    # Check for parquet files
    import glob
    parquet_files = glob.glob(os.path.join(GDRIVE_DIR, "*.parquet"))
    print(f"\n✓ Found {len(parquet_files)} parquet files in {GDRIVE_DIR}")
    
    if parquet_files:
        # Show sample files
        print("  Sample files:")
        for f in parquet_files[:5]:
            print(f"    - {os.path.basename(f)}")


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("ENHANCED CRYPTO PREDICTION PIPELINE WITH WHALE AND SOCIAL DATA")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    
    # Install required packages
    print("\nChecking required packages...")
    install_required_packages()
    
    # Check data availability
    check_data_availability()
    
    # Ask user what to do
    print("\n" + "="*80)
    print("PIPELINE OPTIONS")
    print("="*80)
    print("1. Run smoke test (quick test with limited data)")
    print("2. Run full pipeline (train on all data)")
    print("3. Load models and make predictions")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == '1':
        run_smoke_test()
    elif choice == '2':
        run_full_pipeline()
    elif choice == '3':
        demonstrate_prediction()
    elif choice == '4':
        print("Exiting...")
        return
    else:
        print("Invalid choice")
        return
    
    print(f"\nEnd time: {datetime.now()}")
    print("Pipeline execution complete!")


if __name__ == "__main__":
    # For Google Colab compatibility
    if 'google.colab' in sys.modules:
        # If running in Colab, automatically run the full pipeline
        print("Detected Google Colab environment")
        print("Installing packages and running full pipeline...")
        install_required_packages()
        check_data_availability()
        run_full_pipeline()
    else:
        # If running locally, show interactive menu
        main()