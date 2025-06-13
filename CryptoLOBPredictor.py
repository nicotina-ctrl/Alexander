import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import warnings
import pickle
from datetime import datetime, timedelta
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

class LOBFeatureExtractor:
    """Extract microstructural features from LOB snapshots"""
    
    def __init__(self, k_levels=5, window_size=20):
        self.k_levels = k_levels
        self.window_size = window_size
        
    def extract_price_size_features(self, df):
        """Extract top-k bid/ask prices and volumes"""
        features = {}
        
        # Top-k bid/ask prices and volumes
        for i in range(self.k_levels):
            features[f'bid_price_{i+1}'] = df[f'bid_price_{i+1}'] if f'bid_price_{i+1}' in df.columns else 0
            features[f'ask_price_{i+1}'] = df[f'ask_price_{i+1}'] if f'ask_price_{i+1}' in df.columns else 0
            features[f'bid_size_{i+1}'] = df[f'bid_size_{i+1}'] if f'bid_size_{i+1}' in df.columns else 0
            features[f'ask_size_{i+1}'] = df[f'ask_size_{i+1}'] if f'ask_size_{i+1}' in df.columns else 0
        
        # Bid-ask spread
        features['bid_ask_spread'] = df['ask_price_1'] - df['bid_price_1'] if 'ask_price_1' in df.columns and 'bid_price_1' in df.columns else 0
        features['relative_spread'] = features['bid_ask_spread'] / ((df['ask_price_1'] + df['bid_price_1']) / 2) if 'ask_price_1' in df.columns and 'bid_price_1' in df.columns else 0
        
        # Mid-price
        features['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2 if 'ask_price_1' in df.columns and 'bid_price_1' in df.columns else 0
        
        return features
    
    def calculate_order_imbalance(self, df):
        """Calculate order imbalance metrics"""
        features = {}
        
        # First-level imbalance
        bid_vol_1 = df['bid_size_1'] if 'bid_size_1' in df.columns else 0
        ask_vol_1 = df['ask_size_1'] if 'ask_size_1' in df.columns else 0
        features['imbalance_level_1'] = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1 + 1e-10)
        
        # Five-level aggregate imbalance
        total_bid_vol = sum([df[f'bid_size_{i+1}'] if f'bid_size_{i+1}' in df.columns else 0 for i in range(5)])
        total_ask_vol = sum([df[f'ask_size_{i+1}'] if f'ask_size_{i+1}' in df.columns else 0 for i in range(5)])
        features['imbalance_level_5'] = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-10)
        
        # Weighted imbalance
        weighted_bid = 0
        weighted_ask = 0
        for i in range(self.k_levels):
            weight = 1 / (i + 1)  # Decreasing weight for deeper levels
            weighted_bid += weight * (df[f'bid_size_{i+1}'] if f'bid_size_{i+1}' in df.columns else 0)
            weighted_ask += weight * (df[f'ask_size_{i+1}'] if f'ask_size_{i+1}' in df.columns else 0)
        
        features['weighted_imbalance'] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
        
        return features
    
    def calculate_weighted_mid_price_change(self, df, lookback=5):
        """Calculate depth-weighted sum of level-by-level mid-price deltas"""
        if isinstance(df, pd.Series):
            return {'weighted_mid_price_change': 0}
        
        features = {}
        weighted_change = 0
        
        for i in range(min(self.k_levels, 5)):
            if f'bid_price_{i+1}' in df.columns and f'ask_price_{i+1}' in df.columns:
                mid_price_level = (df[f'bid_price_{i+1}'] + df[f'ask_price_{i+1}']) / 2
                if len(df) > lookback:
                    mid_price_prev = (df[f'bid_price_{i+1}'].shift(lookback) + df[f'ask_price_{i+1}'].shift(lookback)) / 2
                    change = (mid_price_level - mid_price_prev) / mid_price_prev
                    
                    # Weight by volume
                    volume_weight = (df[f'bid_size_{i+1}'] + df[f'ask_size_{i+1}']) / (
                        sum([df[f'bid_size_{j+1}'] + df[f'ask_size_{j+1}'] for j in range(self.k_levels) 
                             if f'bid_size_{j+1}' in df.columns and f'ask_size_{j+1}' in df.columns]) + 1e-10)
                    
                    weighted_change += change * volume_weight
        
        features['weighted_mid_price_change'] = weighted_change.iloc[-1] if hasattr(weighted_change, 'iloc') else weighted_change
        
        return features
    
    def calculate_statistical_summaries(self, df_full, current_idx):
        """Calculate rolling window statistics and PCA summaries"""
        features = {}
        
        # Get window of data
        start_idx = max(0, current_idx - self.window_size)
        window_data = df_full.iloc[start_idx:current_idx+1]
        
        if len(window_data) > 1:
            # Mid-price returns
            mid_prices = (window_data['bid_price_1'] + window_data['ask_price_1']) / 2
            returns = mid_prices.pct_change().dropna()
            
            # Rolling volatility
            features['rolling_volatility'] = returns.std() if len(returns) > 0 else 0
            features['rolling_mean_return'] = returns.mean() if len(returns) > 0 else 0
            
            # Price momentum
            if len(mid_prices) >= 10:
                features['momentum_10'] = (mid_prices.iloc[-1] - mid_prices.iloc[-10]) / mid_prices.iloc[-10]
            else:
                features['momentum_10'] = 0
                
            # Volume statistics
            total_volumes = window_data['bid_size_1'] + window_data['ask_size_1']
            features['volume_mean'] = total_volumes.mean()
            features['volume_std'] = total_volumes.std()
            
            # Spread statistics
            spreads = window_data['ask_price_1'] - window_data['bid_price_1']
            features['spread_mean'] = spreads.mean()
            features['spread_std'] = spreads.std()
            
        else:
            # Default values for insufficient data
            features['rolling_volatility'] = 0
            features['rolling_mean_return'] = 0
            features['momentum_10'] = 0
            features['volume_mean'] = 0
            features['volume_std'] = 0
            features['spread_mean'] = 0
            features['spread_std'] = 0
        
        return features
    
    def extract_all_features(self, df, df_full=None, current_idx=None):
        """Extract all features from a single LOB snapshot"""
        if isinstance(df, pd.DataFrame) and len(df) == 1:
            df = df.iloc[0]
        
        features = {}
        
        # Price and size features
        features.update(self.extract_price_size_features(df))
        
        # Order imbalance
        features.update(self.calculate_order_imbalance(df))
        
        # Weighted mid-price change (needs full dataframe)
        if df_full is not None and current_idx is not None:
            features.update(self.calculate_weighted_mid_price_change(df_full.iloc[max(0, current_idx-10):current_idx+1]))
        else:
            features['weighted_mid_price_change'] = 0
        
        # Statistical summaries (needs full dataframe and current index)
        if df_full is not None and current_idx is not None:
            features.update(self.calculate_statistical_summaries(df_full, current_idx))
        else:
            # Add default values
            features.update({
                'rolling_volatility': 0,
                'rolling_mean_return': 0,
                'momentum_10': 0,
                'volume_mean': 0,
                'volume_std': 0,
                'spread_mean': 0,
                'spread_std': 0
            })
        
        return features


class DataSmoother:
    """Apply Savitzky-Golay or Kalman filtering to features"""
    
    @staticmethod
    def savgol_smooth(data, window_length=11, polyorder=3):
        """Apply Savitzky-Golay filter"""
        if len(data) < window_length:
            return data
        return savgol_filter(data, window_length, polyorder)
    
    @staticmethod
    def kalman_smooth(data):
        """Apply Kalman filter"""
        kf = KalmanFilter(
            initial_state_mean=data[0],
            n_dim_obs=1,
            n_dim_state=1,
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )
        
        state_means, _ = kf.filter(data.reshape(-1, 1))
        return state_means.flatten()


class LOBPredictor:
    """Main prediction pipeline"""
    
    def __init__(self, db_path, symbol, prediction_horizon_minutes=5):
        self.db_path = db_path
        self.symbol = symbol
        self.prediction_horizon = prediction_horizon_minutes
        self.feature_extractor = LOBFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_data(self):
        """Load data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        # First, let's check what tables are available
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print(f"Available tables: {tables['name'].tolist()}")
        
        # Try to load orderbook data
        # Adjust this query based on your actual table structure
        query = f"""
        SELECT * FROM orderbook 
        WHERE symbol = '{self.symbol}' 
        ORDER BY timestamp
        """
        
        try:
            df = pd.read_sql_query(query, conn)
        except:
            # If the table structure is different, try a more general query
            print("Trying alternative query...")
            query = f"SELECT * FROM orderbook LIMIT 100000"
            df = pd.read_sql_query(query, conn)
            
            # Filter for symbol if column exists
            if 'symbol' in df.columns:
                df = df[df['symbol'] == self.symbol]
        
        conn.close()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        print(f"Loaded {len(df)} records for {self.symbol}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def prepare_features_and_labels(self, df):
        """Prepare features and labels for training"""
        features_list = []
        labels_binary = []
        labels_ternary = []
        
        print("Extracting features...")
        
        # Calculate future returns for labels
        mid_prices = (df['bid_price_1'] + df['ask_price_1']) / 2
        future_returns = mid_prices.shift(-self.prediction_horizon).pct_change(self.prediction_horizon)
        
        # Define thresholds for ternary classification
        threshold = 0.0001  # 0.01% threshold
        
        for i in tqdm(range(self.prediction_horizon, len(df) - self.prediction_horizon)):
            # Extract features
            features = self.feature_extractor.extract_all_features(df.iloc[i], df, i)
            features_list.append(features)
            
            # Create labels
            ret = future_returns.iloc[i]
            if pd.isna(ret):
                continue
                
            # Binary label (up/down)
            labels_binary.append(1 if ret > 0 else 0)
            
            # Ternary label (up/flat/down)
            if ret > threshold:
                labels_ternary.append(2)  # Up
            elif ret < -threshold:
                labels_ternary.append(0)  # Down
            else:
                labels_ternary.append(1)  # Flat
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Remove any NaN values
        valid_idx = ~(features_df.isna().any(axis=1) | pd.isna(labels_binary[:len(features_df)]))
        features_df = features_df[valid_idx]
        labels_binary = [labels_binary[i] for i in range(len(valid_idx)) if valid_idx.iloc[i]]
        labels_ternary = [labels_ternary[i] for i in range(len(valid_idx)) if valid_idx.iloc[i]]
        
        return features_df, np.array(labels_binary), np.array(labels_ternary)
    
    def apply_smoothing(self, features_df, method='savgol'):
        """Apply smoothing to features"""
        smoothed_features = features_df.copy()
        
        for col in features_df.columns:
            if method == 'savgol':
                smoothed_features[col] = DataSmoother.savgol_smooth(features_df[col].values)
            elif method == 'kalman':
                smoothed_features[col] = DataSmoother.kalman_smooth(features_df[col].values)
        
        return smoothed_features
    
    def train_models(self, X_train, y_train, X_val, y_val, task='binary'):
        """Train multiple models"""
        results = {}
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_val)
        results['logistic_regression'] = {
            'model': lr,
            'accuracy': accuracy_score(y_val, y_pred_lr),
            'predictions': y_pred_lr
        }
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_val)
        results['xgboost'] = {
            'model': xgb_model,
            'accuracy': accuracy_score(y_val, y_pred_xgb),
            'predictions': y_pred_xgb
        }
        
        # CatBoost
        print("Training CatBoost...")
        cb_model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        cb_model.fit(X_train, y_train)
        y_pred_cb = cb_model.predict(X_val)
        results['catboost'] = {
            'model': cb_model,
            'accuracy': accuracy_score(y_val, y_pred_cb),
            'predictions': y_pred_cb
        }
        
        return results
    
    def calculate_trading_metrics(self, y_true, y_pred, returns):
        """Calculate trading performance metrics"""
        # Direction accuracy
        direction_accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate returns based on predictions
        strategy_returns = returns * (2 * y_pred - 1)  # Convert 0/1 to -1/1
        
        # Sharpe Ratio (annualized)
        sharpe_ratio = np.sqrt(252 * 12) * strategy_returns.mean() / (strategy_returns.std() + 1e-10)
        
        # Maximum Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        winning_trades = strategy_returns > 0
        win_rate = winning_trades.sum() / len(winning_trades)
        
        # Profit Factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profits / (gross_losses + 1e-10)
        
        # Information Ratio
        excess_returns = strategy_returns - returns.mean()
        information_ratio = np.sqrt(252 * 12) * excess_returns.mean() / (excess_returns.std() + 1e-10)
        
        # Regression metrics for returns
        mae = mean_absolute_error(returns, strategy_returns)
        rmse = np.sqrt(mean_squared_error(returns, strategy_returns))
        r2 = r2_score(returns, strategy_returns)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Direction_Accuracy': direction_accuracy,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Information_Ratio': information_ratio
        }
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        # Load data
        print(f"Loading data for {self.symbol}...")
        df = self.load_data()
        
        # Prepare features and labels
        print("Preparing features and labels...")
        features_df, labels_binary, labels_ternary = self.prepare_features_and_labels(df)
        
        # Apply PCA for dimensionality reduction
        print("Applying PCA...")
        pca = PCA(n_components=min(20, features_df.shape[1]))
        features_pca = pca.fit_transform(features_df)
        
        # Split data
        X_train, X_temp, y_train_binary, y_temp_binary = train_test_split(
            features_pca, labels_binary, test_size=0.3, random_state=42, stratify=labels_binary
        )
        X_val, X_test, y_val_binary, y_test_binary = train_test_split(
            X_temp, y_temp_binary, test_size=0.5, random_state=42, stratify=y_temp_binary
        )
        
        # Also split for ternary
        _, _, y_train_ternary, y_temp_ternary = train_test_split(
            features_pca, labels_ternary, test_size=0.3, random_state=42, stratify=labels_ternary
        )
        _, _, y_val_ternary, y_test_ternary = train_test_split(
            X_temp, y_temp_ternary, test_size=0.5, random_state=42, stratify=y_temp_ternary
        )
        
        # Apply different smoothing methods
        results_summary = {}
        
        for smoothing_method in ['none', 'savgol', 'kalman']:
            print(f"\n{'='*50}")
            print(f"Testing with {smoothing_method} smoothing...")
            print(f"{'='*50}")
            
            if smoothing_method == 'none':
                features_smoothed = features_df
            else:
                features_smoothed = self.apply_smoothing(features_df, smoothing_method)
            
            # Re-apply PCA on smoothed features
            features_smoothed_pca = pca.fit_transform(features_smoothed)
            
            # Re-split with smoothed features
            X_train_s, X_temp_s, _, _ = train_test_split(
                features_smoothed_pca, labels_binary, test_size=0.3, random_state=42, stratify=labels_binary
            )
            X_val_s, X_test_s, _, _ = train_test_split(
                X_temp_s, y_temp_binary, test_size=0.5, random_state=42, stratify=y_temp_binary
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_s)
            X_val_scaled = self.scaler.transform(X_val_s)
            X_test_scaled = self.scaler.transform(X_test_s)
            
            # Train models for binary classification
            print("\nBinary Classification Results:")
            binary_results = self.train_models(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, 'binary')
            
            # Train models for ternary classification
            print("\nTernary Classification Results:")
            ternary_results = self.train_models(X_train_scaled, y_train_ternary, X_val_scaled, y_val_ternary, 'ternary')
            
            # Store results
            results_summary[smoothing_method] = {
                'binary': binary_results,
                'ternary': ternary_results
            }
            
            # Print accuracies
            print(f"\nAccuracies with {smoothing_method} smoothing:")
            print("Binary Classification:")
            for model_name, result in binary_results.items():
                print(f"  {model_name}: {result['accuracy']:.4f}")
            print("Ternary Classification:")
            for model_name, result in ternary_results.items():
                print(f"  {model_name}: {result['accuracy']:.4f}")
        
        # Find best model and configuration
        best_accuracy = 0
        best_config = None
        
        for smoothing_method, results in results_summary.items():
            for task, task_results in results.items():
                for model_name, result in task_results.items():
                    if result['accuracy'] > best_accuracy:
                        best_accuracy = result['accuracy']
                        best_config = {
                            'smoothing': smoothing_method,
                            'task': task,
                            'model_name': model_name,
                            'model': result['model'],
                            'accuracy': result['accuracy']
                        }
        
        print(f"\n{'='*50}")
        print(f"Best Configuration:")
        print(f"Smoothing: {best_config['smoothing']}")
        print(f"Task: {best_config['task']}")
        print(f"Model: {best_config['model_name']}")
        print(f"Validation Accuracy: {best_config['accuracy']:.4f}")
        print(f"{'='*50}")
        
        # Test on test set and calculate trading metrics
        if best_config['smoothing'] == 'none':
            features_final = features_df
        else:
            features_final = self.apply_smoothing(features_df, best_config['smoothing'])
        
        features_final_pca = pca.transform(features_final)
        
        # Get test set indices
        test_start_idx = len(features_final) - len(X_test)
        test_returns = pd.Series(labels_binary[test_start_idx:])
        
        # Make predictions on test set
        X_test_final = self.scaler.transform(features_final_pca[test_start_idx:])
        y_test_pred = best_config['model'].predict(X_test_final)
        
        # Calculate all metrics
        metrics = self.calculate_trading_metrics(y_test_binary, y_test_pred, test_returns)
        
        print("\nTest Set Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        # Save the best model
        model_info = {
            'model': best_config['model'],
            'scaler': self.scaler,
            'pca': pca,
            'feature_extractor': self.feature_extractor,
            'config': best_config,
            'metrics': metrics,
            'symbol': self.symbol,
            'prediction_horizon': self.prediction_horizon
        }
        
        filename = f"lob_model_{self.symbol}_{self.prediction_horizon}min_{best_config['model_name']}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"\nModel saved to {filename}")
        
        return model_info


def main():
    """Main execution function"""
    db_path = "/content/drive/MyDrive/crypto_pipeline_whale/crypto_data_RAW_FULL.db"
    
    # Run for different symbols and time horizons
    symbols = ['ETHUSDT', 'BTCUSDT']
    horizons = [5, 30, 60]  # minutes
    
    all_results = {}
    
    for symbol in symbols:
        for horizon in horizons:
            print(f"\n{'#'*60}")
            print(f"Training model for {symbol} with {horizon}-minute prediction horizon")
            print(f"{'#'*60}")
            
            predictor = LOBPredictor(db_path, symbol, horizon)
            try:
                model_info = predictor.run_pipeline()
                all_results[f"{symbol}_{horizon}min"] = model_info
            except Exception as e:
                print(f"Error processing {symbol} with {horizon}-minute horizon: {str(e)}")
                continue
    
    # Summary of all results
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL MODELS")
    print(f"{'='*60}")
    
    for config_name, model_info in all_results.items():
        print(f"\n{config_name}:")
        print(f"  Best Model: {model_info['config']['model_name']}")
        print(f"  Smoothing: {model_info['config']['smoothing']}")
        print(f"  Metrics:")
        for metric, value in model_info['metrics'].items():
            print(f"    {metric}: {value:.6f}")


if __name__ == "__main__":
    main()