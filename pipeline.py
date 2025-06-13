"""
Main pipeline class that orchestrates the entire process with whale and social data
"""
import os
import gc
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import *
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from evaluation import ModelEvaluator

class EnhancedCryptoPipeline:
    def __init__(self, data_path: str, use_sqlite: bool = False):
        self.data_path = data_path
        self.use_sqlite = use_sqlite
        self.data_loader = DataLoader(data_path)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        # Storage for models and results
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.selected_features = {}
        self.fill_values = {}
        
    def run_enhanced_pipeline(self, symbols: Optional[List[str]] = None, 
                            horizons: Optional[List[int]] = None,
                            max_files: Optional[int] = None,
                            optimize_params: bool = False, 
                            batch_size: int = DEFAULT_BATCH_SIZE,
                            include_whale_data: bool = True,
                            include_social_data: bool = True) -> Dict:
        """Run the enhanced pipeline with batch processing, whale, and social data"""
        
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
        if horizons is None:
            horizons = DEFAULT_HORIZONS
            
        print("\n" + "="*80)
        print("ENHANCED CRYPTO PREDICTION PIPELINE")
        print("="*80)
        print(f"Symbols: {symbols}")
        print(f"Horizons: {[h*5 for h in horizons]} minutes")
        print(f"Include whale data: {include_whale_data}")
        print(f"Include social data: {include_social_data}")
        print("="*80 + "\n")
        
        # Load and aggregate data
        if self.use_sqlite and os.path.exists(SQLITE_DB_PATH):
            print("Loading data from SQLite database...")
            df = self.data_loader.load_from_sqlite(symbols)
        else:
            print("Loading and aggregating data from parquet files...")
            df = self.data_loader.load_and_aggregate_batched(
                symbols, 
                max_files=max_files, 
                batch_size=batch_size,
                include_whale_data=include_whale_data,
                include_social_data=include_social_data
            )
        
        if len(df) == 0:
            print("No data loaded.")
            return {}
        
        # Display data summary
        print("\nData loaded successfully!")
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df['time_bucket'].min()} to {df['time_bucket'].max()}")
        
        # Check for whale and social features
        whale_cols = [col for col in df.columns if 'whale_' in col or 'inst_' in col or 'retail_' in col]
        social_cols = [col for col in df.columns if 'mention' in col or 'sentiment' in col or 'social' in col]
        
        if whale_cols:
            print(f"Whale features found: {len(whale_cols)}")
        else:
            print("No whale features found in data")
            include_whale_data = False
            
        if social_cols:
            print(f"Social features found: {len(social_cols)}")
        else:
            print("No social features found in data")
            include_social_data = False
        
        # Create advanced features (no fill_values yet - first run)
        df_features = self.feature_engineer.create_advanced_features(
            df, 
            include_whale_data=include_whale_data,
            include_social_data=include_social_data
        )
        del df
        gc.collect()
        
        # Prepare targets
        df_features = self.feature_engineer.prepare_advanced_targets(df_features, horizons)
        
        # CRITICAL FIX: Drop rows with NaN targets BEFORE split
        # This prevents future data from leaking into test set
        print("\nDropping rows with NaN targets before split...")
        original_len = len(df_features)
        
        # Drop rows where ANY target is NaN (last h rows of each symbol)
        target_cols = [col for col in df_features.columns if col.startswith('target_')]
        df_features = df_features.dropna(subset=target_cols)
        
        print(f"Dropped {original_len - len(df_features)} rows with NaN targets")
        
        # Walk-forward split (more realistic than random split)
        df_features = df_features.sort_values('time_bucket')
        n = len(df_features)
        # To:
        # To:
        train_end = int(n * 0.75)  # 75% for training
        val_end = int(n * 0.9)     # 15% for validation, 10% for test
        
        train_df = df_features.iloc[:train_end].copy()
        val_df = df_features.iloc[train_end:val_end].copy()
        test_df = df_features.iloc[val_end:].copy()
        
        print(f"\nWalk-forward split:")
        print(f"→ Train: {len(train_df)} rows ({train_df['time_bucket'].min()} to {train_df['time_bucket'].max()})")
        print(f"→ Val:   {len(val_df)} rows ({val_df['time_bucket'].min()} to {val_df['time_bucket'].max()})")
        print(f"→ Test:  {len(test_df)} rows ({test_df['time_bucket'].min()} to {test_df['time_bucket'].max()})")
        
        # Calculate fill values from training set for each symbol
        print("\nCalculating fill values from training set...")
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for symbol in train_df['symbol'].unique():
            self.fill_values[symbol] = {}
            symbol_data = train_df[train_df['symbol'] == symbol]
            for col in numeric_cols:
                if col not in ['symbol', 'time_bucket']:
                    # Use median for most features, 0 for flags
                    if 'flag' in col or col.startswith('is_') or col.startswith('near_'):
                        self.fill_values[symbol][col] = 0
                    else:
                        self.fill_values[symbol][col] = symbol_data[col].median()
        
        # Store fill values in feature engineer
        self.feature_engineer.fill_values = self.fill_values
        
        # Display feature summary
        feature_cols = [col for col in train_df.columns if col not in ['symbol', 'time_bucket'] and not col.startswith('target_')]
        print(f"\nTotal features: {len(feature_cols)}")
        
        # Count feature types
        feature_types = {
            'Orderbook': len([col for col in feature_cols if any(x in col for x in ['bid', 'ask', 'spread', 'book', 'imbalance_top'])]),
            'Trades': len([col for col in feature_cols if any(x in col for x in ['cvd', 'vwap', 'order_flow', 'volume', 'quantity'])]),
            'Technical': len([col for col in feature_cols if any(x in col for x in ['rsi', 'macd', 'bb_', 'atr'])]),
            'Momentum': len([col for col in feature_cols if any(x in col for x in ['return_', 'momentum', 'acceleration'])]),
            'Volatility': len([col for col in feature_cols if any(x in col for x in ['volatility', 'vol_of_vol', 'skew', 'kurt'])]),
            'Microstructure': len([col for col in feature_cols if any(x in col for x in ['microprice', 'vpin', 'relative_spread'])]),
            'Time': len([col for col in feature_cols if any(x in col for x in ['hour', 'day', 'session', 'weekend'])]),
            'Whale': len([col for col in feature_cols if any(x in col for x in ['whale', 'inst', 'retail', 'smart_dumb'])]),
            'Social': len([col for col in feature_cols if any(x in col for x in ['mention', 'sentiment', 'social', 'viral'])])
        }
        
        print("\nFeature breakdown:")
        for ftype, count in feature_types.items():
            if count > 0:
                print(f"  {ftype}: {count}")
        
        # Train models
        results = {}
        for symbol in train_df['symbol'].unique():
            results[symbol] = {}
            
            for h in horizons:
                print(f"\n{'='*60}")
                print(f"Training {symbol} @ {h*5}min horizon")
                print(f"{'='*60}")
                
                # Train diverse ensemble
                model_results = self.model_trainer.train_ensemble_model(
                    train_df, val_df, symbol, h, optimize=optimize_params
                )
                
                if model_results[0] is None:
                    continue
                
                models, scalers, feature_sets, importance = model_results
                
                # Train meta-learner
                meta_learner = self.model_trainer.train_meta_learner(
                    models, scalers, feature_sets, train_df, val_df, symbol, h
                )
                
                # Test evaluation
                ts = test_df[test_df['symbol'] == symbol].copy()
                if len(ts) == 0:
                    continue
                
                # Get test features and target
                y_test = ts[f'target_return_{h}']
                time_index = ts['time_bucket']
                
                print("\nTest Set Performance:")
                print("-" * 60)
                
                # Get predictions from all models
                predictions_dict = self.model_trainer.get_predictions(
                    models, scalers, feature_sets, ts
                )
                
                # Evaluate all models and ensemble
                eval_results = self.evaluator.evaluate_ensemble(
                    predictions_dict, meta_learner, y_test, time_index
                )
                
                # Store results
                for model_name, metrics in eval_results.items():
                    results[symbol][f'{model_name}_{h}'] = metrics
                
                # Store models
                safe_sym = symbol.replace('/', '_').replace(':', '_')
                self.models[f"{safe_sym}_models_{h}"] = models
                self.models[f"{safe_sym}_meta_{h}"] = meta_learner
                self.scalers[f"{safe_sym}_scalers_{h}"] = scalers
                self.selected_features[f"{safe_sym}_features_{h}"] = feature_sets
                self.feature_importances[f"{safe_sym}_importance_{h}"] = importance
                
                # Display top features if whale/social features are included
                if importance is not None and not importance.empty:
                    top_features = importance.nlargest(20, 'importance')
                    
                    # Check for whale/social features in top features
                    whale_features = [f for f in top_features['feature'] if any(x in f for x in ['whale', 'inst', 'retail', 'smart_dumb'])]
                    social_features = [f for f in top_features['feature'] if any(x in f for x in ['mention', 'sentiment', 'social', 'viral'])]
                    
                    if whale_features or social_features:
                        print("\nTop whale/social features:")
                        if whale_features:
                            print(f"  Whale: {whale_features[:5]}")
                        if social_features:
                            print(f"  Social: {social_features[:5]}")
        
        return results
    
    def save_enhanced_models(self, save_path: str = None):
        """Save models, scalers, and metadata"""
        if save_path is None:
            save_path = MODEL_SAVE_DIR
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save models
        for k, model in self.models.items():
            joblib.dump(model, os.path.join(save_path, f"model_{k}.pkl"))
        
        # Save scalers
        for k, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(save_path, f"scaler_{k}.pkl"))
        
        # Save feature sets
        joblib.dump(self.selected_features, os.path.join(save_path, "selected_features.pkl"))
        
        # Save feature importances
        joblib.dump(self.feature_importances, os.path.join(save_path, "feature_importances.pkl"))
        
        # Save fill values
        joblib.dump(self.fill_values, os.path.join(save_path, "fill_values.pkl"))
        
        print(f"\nSaved {len(self.models)} models and metadata to {save_path}")
    
    def load_enhanced_models(self, load_path: str = None):
        """Load models, scalers, and metadata"""
        if load_path is None:
            load_path = MODEL_SAVE_DIR
            
        # Load models
        model_files = [f for f in os.listdir(load_path) if f.startswith('model_')]
        for model_file in model_files:
            key = model_file.replace('model_', '').replace('.pkl', '')
            self.models[key] = joblib.load(os.path.join(load_path, model_file))
        
        # Load scalers
        scaler_files = [f for f in os.listdir(load_path) if f.startswith('scaler_')]
        for scaler_file in scaler_files:
            key = scaler_file.replace('scaler_', '').replace('.pkl', '')
            self.scalers[key] = joblib.load(os.path.join(load_path, scaler_file))
        
        # Load feature sets
        if os.path.exists(os.path.join(load_path, "selected_features.pkl")):
            self.selected_features = joblib.load(os.path.join(load_path, "selected_features.pkl"))
        
        # Load feature importances
        if os.path.exists(os.path.join(load_path, "feature_importances.pkl")):
            self.feature_importances = joblib.load(os.path.join(load_path, "feature_importances.pkl"))
        
        # Load fill values
        if os.path.exists(os.path.join(load_path, "fill_values.pkl")):
            self.fill_values = joblib.load(os.path.join(load_path, "fill_values.pkl"))
            self.feature_engineer.fill_values = self.fill_values
        
        print(f"\nLoaded {len(self.models)} models and metadata from {load_path}")
    
    def predict_new_data(self, new_data: pd.DataFrame, symbol: str, horizon: int, 
                        include_whale_data: bool = True,
                        include_social_data: bool = True) -> pd.DataFrame:
        """Make predictions on new data"""
        
        # Create features
        features_df = self.feature_engineer.create_advanced_features(
            new_data, 
            fill_values=self.fill_values,
            use_cache=False,
            include_whale_data=include_whale_data,
            include_social_data=include_social_data
        )
        
        # Get model components
        safe_sym = symbol.replace('/', '_')
        models = self.models.get(f"{safe_sym}_models_{horizon}")
        scalers = self.scalers.get(f"{safe_sym}_scalers_{horizon}")
        feature_sets = self.selected_features.get(f"{safe_sym}_features_{horizon}")
        meta_learner = self.models.get(f"{safe_sym}_meta_{horizon}")
        
        if not all([models, scalers, feature_sets, meta_learner]):
            raise ValueError(f"No trained model found for {symbol} @ {horizon*5}min")
        
        # Filter for symbol
        symbol_data = features_df[features_df['symbol'] == symbol].copy()
        
        # Get predictions
        predictions_dict = self.model_trainer.get_predictions(
            models, scalers, feature_sets, symbol_data
        )
        
        # Create ensemble prediction
        meta_features = pd.DataFrame(predictions_dict)
        meta_features['xgb_lgb'] = meta_features['xgb'] * meta_features['lgb']
        meta_features['tree_linear'] = (meta_features['xgb'] + meta_features['lgb']) * meta_features['ridge']
        
        ensemble_proba = meta_learner.predict_proba(meta_features)[:, 1]
        ensemble_preds = (ensemble_proba - 0.5) * 0.002
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'time': symbol_data['time_bucket'],
            'symbol': symbol_data['symbol'],
            'xgb_pred': predictions_dict['xgb'],
            'lgb_pred': predictions_dict['lgb'],
            'ridge_pred': predictions_dict['ridge'],
            'mlp_pred': predictions_dict['mlp'],
            'ensemble_pred': ensemble_preds,
            'direction': ['UP' if p > 0 else 'DOWN' for p in ensemble_preds],
            'confidence': np.abs(ensemble_preds)
        })
        
        return results_df