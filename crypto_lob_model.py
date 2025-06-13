import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.signal import savgol_filter
import pickle
import warnings
warnings.filterwarnings('ignore')

class CryptoLOBPredictor:
    """
    Cryptocurrency Limit Order Book predictor based on microstructural features
    Implements hand-crafted features and Savitzky-Golay filtering as per research
    """
    
    def __init__(self, db_path='/content/drive/MyDrive/crypto_pipeline_whale/crypto_data_RAW_FULL.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
    def load_data(self, symbol='BTCUSDT'):
        """Load orderbook and trade data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load orderbook data
        orderbook_query = f"""
        SELECT timestamp, symbol, mid_price, spread, book_imbalance, 
               bid_depth_10, ask_depth_10
        FROM orderbook_20250603_170215
        WHERE symbol = '{symbol}'
        ORDER BY timestamp
        """
        orderbook_df = pd.read_sql_query(orderbook_query, conn)
        
        # Load trade data
        trades_query = f"""
        SELECT timestamp, symbol, price, quantity, side, cvd
        FROM trades_20250603_170215
        WHERE symbol = '{symbol}'
        ORDER BY timestamp
        """
        trades_df = pd.read_sql_query(trades_query, conn)
        
        conn.close()
        
        return orderbook_df, trades_df
    
    def resample_to_5min(self, orderbook_df, trades_df):
        """Resample data to 5-minute intervals"""
        # Convert timestamps to datetime
        orderbook_df['datetime'] = pd.to_datetime(orderbook_df['timestamp'], unit='ms')
        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
        
        # Set datetime as index
        orderbook_df.set_index('datetime', inplace=True)
        trades_df.set_index('datetime', inplace=True)
        
        # Resample orderbook data
        orderbook_5min = orderbook_df.resample('5min').agg({
            'mid_price': ['first', 'last', 'mean', 'std'],
            'spread': ['mean', 'std', 'max'],
            'book_imbalance': ['mean', 'std'],
            'bid_depth_10': ['mean', 'std'],
            'ask_depth_10': ['mean', 'std']
        })
        
        # Flatten column names
        orderbook_5min.columns = ['_'.join(col).strip() for col in orderbook_5min.columns.values]
        
        # Resample trade data
        buy_trades = trades_df[trades_df['side'] == 'buy']
        sell_trades = trades_df[trades_df['side'] == 'sell']
        
        trades_5min = pd.DataFrame(index=orderbook_5min.index)
        
        # Aggregate trade features
        trades_5min['trade_count'] = trades_df.resample('5min').size()
        trades_5min['buy_volume'] = buy_trades.groupby(pd.Grouper(freq='5min'))['quantity'].sum()
        trades_5min['sell_volume'] = sell_trades.groupby(pd.Grouper(freq='5min'))['quantity'].sum()
        trades_5min['net_volume'] = trades_5min['buy_volume'] - trades_5min['sell_volume']
        trades_5min['volume_imbalance'] = (trades_5min['buy_volume'] - trades_5min['sell_volume']) / \
                                         (trades_5min['buy_volume'] + trades_5min['sell_volume'] + 1e-10)
        trades_5min['cvd_change'] = trades_df['cvd'].resample('5min').last() - \
                                   trades_df['cvd'].resample('5min').first()
        
        # Fill NaN values
        trades_5min.fillna(0, inplace=True)
        
        # Combine orderbook and trade data
        combined_df = pd.concat([orderbook_5min, trades_5min], axis=1)
        
        return combined_df
    
    def engineer_features(self, df):
        """Create hand-crafted microstructural features as described in the paper"""
        features = df.copy()
        
        # Price-based features
        features['price_change'] = features['mid_price_last'] - features['mid_price_first']
        features['price_return'] = features['price_change'] / (features['mid_price_first'] + 1e-10)
        
        # Weighted mid-price features (approximation since we don't have full depth)
        features['weighted_spread'] = features['spread_mean'] * features['book_imbalance_mean']
        
        # Order flow features
        features['order_flow_imbalance'] = features['net_volume'] / (features['trade_count'] + 1)
        
        # Volatility features
        features['price_volatility'] = features['mid_price_std'] / (features['mid_price_mean'] + 1e-10)
        
        # Depth imbalance features
        features['depth_imbalance'] = (features['bid_depth_10_mean'] - features['ask_depth_10_mean']) / \
                                     (features['bid_depth_10_mean'] + features['ask_depth_10_mean'] + 1e-10)
        
        # Rolling window features
        for window in [3, 6, 12]:  # 15min, 30min, 60min windows
            features[f'price_return_ma_{window}'] = features['price_return'].rolling(window).mean()
            features[f'volume_imbalance_ma_{window}'] = features['volume_imbalance'].rolling(window).mean()
            features[f'volatility_ma_{window}'] = features['price_volatility'].rolling(window).mean()
            features[f'book_imbalance_ma_{window}'] = features['book_imbalance_mean'].rolling(window).mean()
        
        # Momentum features
        features['price_momentum_15'] = features['mid_price_last'].pct_change(3)
        features['price_momentum_30'] = features['mid_price_last'].pct_change(6)
        features['price_momentum_60'] = features['mid_price_last'].pct_change(12)
        
        # Microstructure features
        features['spread_volatility'] = features['spread_std'] / (features['spread_mean'] + 1e-10)
        features['depth_ratio'] = features['bid_depth_10_mean'] / (features['ask_depth_10_mean'] + 1e-10)
        
        # Statistical features
        for col in ['price_return', 'volume_imbalance', 'book_imbalance_mean']:
            features[f'{col}_skew'] = features[col].rolling(12).skew()
            features[f'{col}_kurt'] = features[col].rolling(12).kurt()
        
        # Drop rows with NaN values from rolling calculations
        features = features.dropna()
        
        return features
    
    def apply_savitzky_golay_filter(self, features, window_length=21, polyorder=3):
        """Apply Savitzky-Golay filter to smooth noisy features"""
        smoothed_features = features.copy()
        
        # List of features to smooth
        features_to_smooth = [
            'mid_price_mean', 'spread_mean', 'book_imbalance_mean',
            'volume_imbalance', 'cvd_change', 'price_volatility',
            'depth_imbalance', 'order_flow_imbalance'
        ]
        
        for feature in features_to_smooth:
            if feature in smoothed_features.columns and len(smoothed_features) > window_length:
                try:
                    smoothed_features[f'{feature}_smoothed'] = savgol_filter(
                        smoothed_features[feature].values,
                        window_length=min(window_length, len(smoothed_features)),
                        polyorder=polyorder
                    )
                except:
                    smoothed_features[f'{feature}_smoothed'] = smoothed_features[feature]
        
        return smoothed_features
    
    def create_targets(self, df, horizons=[1, 6, 12]):  # 5min, 30min, 60min
        """Create target variables for different prediction horizons"""
        targets = {}
        
        for horizon in horizons:
            # Future price
            future_price = df['mid_price_last'].shift(-horizon)
            
            # Price change
            price_change = (future_price - df['mid_price_last']) / df['mid_price_last']
            
            # Binary classification (up/down)
            targets[f'target_binary_{horizon*5}min'] = (price_change > 0).astype(int)
            
            # Regression target (actual return)
            targets[f'target_return_{horizon*5}min'] = price_change
            
        targets_df = pd.DataFrame(targets, index=df.index)
        
        # Remove rows where targets are NaN
        valid_idx = targets_df.notna().all(axis=1)
        
        return targets_df[valid_idx], valid_idx
    
    def prepare_data(self, symbol='BTCUSDT'):
        """Complete data preparation pipeline"""
        print(f"Loading data for {symbol}...")
        orderbook_df, trades_df = self.load_data(symbol)
        
        print("Resampling to 5-minute intervals...")
        combined_df = self.resample_to_5min(orderbook_df, trades_df)
        
        print("Engineering features...")
        features_df = self.engineer_features(combined_df)
        
        print("Applying Savitzky-Golay filter...")
        features_df = self.apply_savitzky_golay_filter(features_df)
        
        print("Creating targets...")
        targets_df, valid_idx = self.create_targets(features_df)
        features_df = features_df[valid_idx]
        
        # Select feature columns (exclude target-like columns)
        exclude_cols = ['mid_price_first', 'mid_price_last', 'symbol', 'timestamp']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        return features_df[feature_cols], targets_df
    
    def train_models(self, X, y, symbol='BTCUSDT'):
        """Train XGBoost models for different prediction horizons"""
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, shuffle=False  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[symbol] = scaler
        
        results = {}
        
        # Train models for each horizon
        for horizon in ['5min', '30min', '60min']:
            print(f"\nTraining model for {horizon} horizon...")
            
            # Binary classification model
            binary_target = f'target_binary_{horizon}'
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(
                X_train_scaled, y_train[binary_target],
                eval_set=[(X_val_scaled, y_val[binary_target])],
                early_stopping_rounds=10,
                verbose=False
            )
            
            self.models[f'{symbol}_{horizon}'] = model
            
            # Evaluate model
            metrics = self.evaluate_model(
                model, X_test_scaled, y_test,
                binary_target, f'target_return_{horizon}',
                X_test.index
            )
            
            results[horizon] = metrics
            
        return results, (X_train, X_val, X_test, y_train, y_val, y_test)
    
    def evaluate_model(self, model, X_test, y_test, binary_target, return_target, test_index):
        """Calculate comprehensive evaluation metrics"""
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_binary = model.predict(X_test)
        
        # Get actual values
        y_true_binary = y_test[binary_target].values
        y_true_returns = y_test[return_target].values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y_true_returns)
        y_true_returns = y_true_returns[valid_mask]
        y_pred_binary = y_pred_binary[valid_mask]
        y_true_binary = y_true_binary[valid_mask]
        y_pred_proba = y_pred_proba[valid_mask]
        
        # Convert predictions to returns
        pred_returns = np.where(y_pred_binary == 1, np.abs(y_true_returns), -np.abs(y_true_returns))
        
        # Basic metrics
        mae = np.mean(np.abs(y_true_returns - pred_returns))
        rmse = np.sqrt(np.mean((y_true_returns - pred_returns) ** 2))
        
        # R² for returns
        ss_res = np.sum((y_true_returns - pred_returns) ** 2)
        ss_tot = np.sum((y_true_returns - np.mean(y_true_returns)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Direction accuracy
        direction_accuracy = np.mean(y_pred_binary == y_true_binary)
        
        # Trading metrics
        # Simulate trading strategy
        strategy_returns = y_true_returns * (2 * y_pred_binary - 1)  # Long if 1, short if 0
        
        # Sharpe ratio (annualized, assuming 5-min bars)
        periods_per_year = 252 * 24 * 12  # Trading days * hours * 5-min periods
        sharpe_ratio = np.sqrt(periods_per_year) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = strategy_returns > 0
        win_rate = np.mean(winning_trades)
        
        # Profit factor
        gross_profits = np.sum(strategy_returns[strategy_returns > 0])
        gross_losses = -np.sum(strategy_returns[strategy_returns < 0])
        profit_factor = gross_profits / (gross_losses + 1e-10)
        
        # Information ratio
        benchmark_returns = y_true_returns  # Using actual returns as benchmark
        excess_returns = strategy_returns - benchmark_returns
        info_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(periods_per_year)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Direction Accuracy': direction_accuracy,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Information Ratio': info_ratio
        }
        
        return metrics
    
    def save_models(self, filepath='crypto_lob_models.pkl'):
        """Save trained models and scalers"""
        save_dict = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='crypto_lob_models.pkl'):
        """Load trained models and scalers"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.models = save_dict['models']
        self.scalers = save_dict['scalers']
        self.feature_names = save_dict['feature_names']
        
        print(f"Models loaded from {filepath}")


def main():
    """Main training pipeline"""
    # Initialize predictor
    predictor = CryptoLOBPredictor()
    
    # Train for both symbols
    all_results = {}
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print('='*50)
        
        try:
            # Prepare data
            X, y = predictor.prepare_data(symbol)
            
            # Train models
            results, data_splits = predictor.train_models(X, y, symbol)
            
            all_results[symbol] = results
            
            # Print results
            print(f"\nResults for {symbol}:")
            for horizon, metrics in results.items():
                print(f"\n{horizon} Prediction Horizon:")
                for metric, value in metrics.items():
                    if metric in ['MAE', 'RMSE']:
                        print(f"  {metric}: {value:.6f}")
                    elif metric in ['R²', 'Direction Accuracy', 'Win Rate']:
                        print(f"  {metric}: {value:.4f}")
                    elif metric in ['Sharpe Ratio', 'Information Ratio']:
                        print(f"  {metric}: {value:.3f}")
                    elif metric == 'Max Drawdown':
                        print(f"  {metric}: {value:.2%}")
                    elif metric == 'Profit Factor':
                        print(f"  {metric}: {value:.2f}")
        
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Save models
    predictor.save_models('crypto_lob_models.pkl')
    
    return predictor, all_results


if __name__ == "__main__":
    predictor, results = main()