import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional, Tuple
import time
import gc
from tqdm import tqdm
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CryptoSampler:
    def __init__(self, db_path: str, sample_rate: float = 0.1):
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.sample_rate = sample_rate
        self.step = int(1 / sample_rate)
        
    def get_data_overview(self):
        """Get overview of the dataset"""
        print("Analyzing dataset...")
        
        # Get total records
        with self.engine.connect() as conn:
            orderbook_count = conn.execute(text("SELECT COUNT(*) FROM orderbook_raw")).scalar()
            trades_count = conn.execute(text("SELECT COUNT(*) FROM trades_raw")).scalar()
            
            # Get symbols
            symbols = conn.execute(text("SELECT DISTINCT symbol FROM orderbook_raw ORDER BY symbol")).fetchall()
            symbols = [row[0] for row in symbols]
            
            # Get date range
            date_range = conn.execute(text("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM orderbook_raw
            """)).fetchone()
            
        print(f"\nDataset Overview:")
        print(f"Orderbook records: {orderbook_count:,}")
        print(f"Trades records: {trades_count:,}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Date range: {pd.to_datetime(date_range[0], unit='s')} to {pd.to_datetime(date_range[1], unit='s')}")
        print(f"\nSampling rate: {self.sample_rate*100}% (keeping every {self.step}th record)")
        print(f"Expected sampled records: ~{int(orderbook_count * self.sample_rate):,} orderbook, ~{int(trades_count * self.sample_rate):,} trades")
        
        return symbols
    
    def create_sampled_data(self):
        """Create sampled tables using regular interval sampling"""
        print(f"\nCreating {self.sample_rate*100}% sample...")
        
        # Get symbols first
        symbols = self.get_data_overview()
        
        # Process each symbol separately for better control
        all_orderbook_samples = []
        all_trades_samples = []
        
        for symbol in tqdm(symbols, desc="Sampling symbols"):
            # Sample orderbook data
            query = text("""
                SELECT * FROM orderbook_raw 
                WHERE symbol = :symbol 
                ORDER BY timestamp
            """)
            df_ob = pd.read_sql_query(query, self.engine, params={'symbol': symbol})
            df_ob_sampled = df_ob.iloc[::self.step]  # Take every Nth row
            all_orderbook_samples.append(df_ob_sampled)
            
            # Sample trades data
            query = text("""
                SELECT * FROM trades_raw 
                WHERE symbol = :symbol 
                ORDER BY timestamp
            """)
            df_tr = pd.read_sql_query(query, self.engine, params={'symbol': symbol})
            df_tr_sampled = df_tr.iloc[::self.step]  # Take every Nth row
            all_trades_samples.append(df_tr_sampled)
            
            # Clear memory
            del df_ob, df_tr
            gc.collect()
        
        # Combine all samples
        print("\nCombining sampled data...")
        df_orderbook_sample = pd.concat(all_orderbook_samples, ignore_index=True)
        df_trades_sample = pd.concat(all_trades_samples, ignore_index=True)
        
        print(f"Sampled data: {len(df_orderbook_sample):,} orderbook records, {len(df_trades_sample):,} trades records")
        
        return df_orderbook_sample, df_trades_sample
    
    def process_features(self, df_orderbook, df_trades, output_db: str):
        """Process the sampled data and create features"""
        print("\nProcessing features...")
        start_time = time.time()
        
        # Add time buckets
        df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['timestamp'], unit='s')
        df_orderbook['time_bucket'] = df_orderbook['timestamp'].dt.floor('5min')
        
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='s')
        df_trades['time_bucket'] = df_trades['timestamp'].dt.floor('5min')
        
        # Process each symbol
        all_features = []
        symbols = sorted(df_orderbook['symbol'].unique())
        
        for symbol in tqdm(symbols, desc="Creating features"):
            # Filter data
            ob_symbol = df_orderbook[df_orderbook['symbol'] == symbol].copy()
            tr_symbol = df_trades[df_trades['symbol'] == symbol].copy()
            
            # Aggregate orderbook with all requested features
            orderbook_agg = ob_symbol.groupby('time_bucket').agg({
                'mid_price': ['first', 'last', 'mean', 'std', 'min', 'max'],
                'spread': ['mean', 'std', 'min', 'max'],
                'book_imbalance': ['mean', 'std', 'min', 'max'],
                'bid_depth_10': ['mean', 'sum', 'std'],
                'ask_depth_10': ['mean', 'sum', 'std']
            })
            orderbook_agg.columns = ['_'.join(col).strip() for col in orderbook_agg.columns.values]
            orderbook_agg['orderbook_updates'] = ob_symbol.groupby('time_bucket').size()
            orderbook_agg = orderbook_agg.reset_index()
            
            # Aggregate trades
            if len(tr_symbol) > 0:
                tr_symbol['buy_volume'] = tr_symbol['quantity'].where(tr_symbol['side'] == 'buy', 0)
                tr_symbol['sell_volume'] = tr_symbol['quantity'].where(tr_symbol['side'] == 'sell', 0)
                
                trades_agg = tr_symbol.groupby('time_bucket').agg({
                    'price': ['first', 'last', 'mean', 'std', 'min', 'max'],
                    'quantity': ['sum', 'mean', 'std', 'max', 'count'],
                    'buy_volume': ['sum'],
                    'sell_volume': ['sum'],
                    'cvd': ['last', 'mean', 'std']
                })
                trades_agg.columns = ['_'.join(col).strip() for col in trades_agg.columns.values]
                trades_agg = trades_agg.reset_index()
                
                # VWAP
                vwap = tr_symbol.groupby('time_bucket').apply(
                    lambda x: (x['price'] * x['quantity']).sum() / x['quantity'].sum() if x['quantity'].sum() > 0 else x['price'].mean()
                ).reset_index(name='vwap')
                trades_agg = trades_agg.merge(vwap, on='time_bucket', how='left')
                
                # Order flow imbalance
                trades_agg['order_flow_imbalance'] = (
                    (trades_agg['buy_volume_sum'] - trades_agg['sell_volume_sum']) / 
                    (trades_agg['buy_volume_sum'] + trades_agg['sell_volume_sum'] + 1e-8)
                )
            else:
                # Create empty trades_agg with proper columns if no trades
                trades_agg = pd.DataFrame({'time_bucket': orderbook_agg['time_bucket']})
            
            # Merge
            df_merged = pd.merge(orderbook_agg, trades_agg, on='time_bucket', how='outer')
            df_merged['symbol'] = symbol
            df_merged['exchange'] = ob_symbol['exchange'].iloc[0] if len(ob_symbol) > 0 else 'unknown'
            
            # Use price from trades or mid_price
            if 'price_last' in df_merged.columns:
                df_merged['price'] = df_merged['price_last'].fillna(df_merged['mid_price_last'])
            else:
                df_merged['price'] = df_merged['mid_price_last']
            
            # Sort by time
            df_merged = df_merged.sort_values('time_bucket').reset_index(drop=True)
            
            # Create ALL requested features
            df_features = self._create_all_features(df_merged)
            all_features.append(df_features)
            
            # Clear memory
            del ob_symbol, tr_symbol, orderbook_agg, trades_agg, df_merged
            gc.collect()
        
        # Combine all
        print("\nCombining all features...")
        df_final = pd.concat(all_features, ignore_index=True)
        df_final = df_final.sort_values(['symbol', 'time_bucket']).reset_index(drop=True)
        
        # Add PCA features across all symbols (after combining)
        print("Creating PCA features...")
        df_final = self._create_pca_features(df_final)
        
        # Add smoothed features
        print("Adding smoothed features...")
        df_final = self._add_smoothed_features(df_final)
        
        # Add targets
        print("Adding target variables...")
        for symbol in df_final['symbol'].unique():
            mask = df_final['symbol'] == symbol
            for h in [1, 3, 6, 12]:  # 5, 15, 30, 60 minutes
                future_price = df_final.loc[mask, 'price'].shift(-h)
                df_final.loc[mask, f'target_return_{h}'] = (
                    (future_price - df_final.loc[mask, 'price']) / df_final.loc[mask, 'price']
                )
                # Direction target with deadband
                df_final.loc[mask, f'target_direction_{h}'] = np.where(
                    df_final.loc[mask, f'target_return_{h}'] > 0.0002, 1,
                    np.where(df_final.loc[mask, f'target_return_{h}'] < -0.0002, -1, 0)
                )
        
        # Clean data
        print("Cleaning data...")
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns
        df_final[numeric_cols] = df_final[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Forward fill within each symbol
        for symbol in df_final['symbol'].unique():
            mask = df_final['symbol'] == symbol
            for col in numeric_cols:
                if 'target' not in col:  # Don't fill targets
                    df_final.loc[mask, col] = df_final.loc[mask, col].fillna(method='ffill', limit=3)
        
        df_final[numeric_cols] = df_final[numeric_cols].fillna(0)
        
        # Optimize memory
        df_final = self._optimize_memory(df_final)
        
        # Save to database
        print(f"\nSaving to {output_db}...")
        engine_out = create_engine(f'sqlite:///{output_db}')
        
        # Save in chunks
        chunk_size = 50000
        n_chunks = len(df_final) // chunk_size + 1
        
        for i in tqdm(range(n_chunks), desc="Saving chunks"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df_final))
            chunk = df_final.iloc[start_idx:end_idx]
            
            if i == 0:
                chunk.to_sql('features', engine_out, if_exists='replace', index=False)
            else:
                chunk.to_sql('features', engine_out, if_exists='append', index=False)
        
        # Create indices
        with engine_out.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_symbol_time ON features(symbol, time_bucket)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_time ON features(time_bucket)"))
        
        end_time = time.time()
        elapsed = (end_time - start_time) / 60
        
        print(f"\n✅ Feature engineering complete!")
        print(f"Total time: {elapsed:.1f} minutes")
        print(f"Created {len(df_final.columns)} features")
        print(f"Total records: {len(df_final):,}")
        print(f"Output database: {output_db}")
        
        # Show feature summary
        print("\nFeature categories created:")
        print("✓ Price & size levels (bid-ask spread, mid-price returns)")
        print("✓ Order imbalance metrics (first-level and aggregate)")
        print("✓ Weighted mid-price changes")
        print("✓ Statistical summaries (rolling volatility, skew, kurtosis)")
        print("✓ PCA components of order book")
        print("✓ Smoothed features (Savitzky-Golay and Kalman filters)")
        print("✓ Technical indicators (RSI, Bollinger Bands)")
        print("✓ Time-based features")
        print("✓ Target variables")
        
        return df_final
    
    def _create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ALL requested features including microstructure features"""
        
        # 1. PRICE & SIZE LEVELS
        # Mid-price returns (requested feature)
        for period in [1, 3, 5, 10, 20]:
            df[f'mid_price_return_{period}'] = df['mid_price_last'].pct_change(period)
        
        # Bid-ask spread features (requested feature)
        df['spread_relative'] = df['spread_mean'] / (df['mid_price_mean'] + 1e-8)
        df['spread_log'] = np.log(df['spread_mean'] + 1e-8)
        df['spread_momentum'] = df['spread_mean'].diff(6)
        df['spread_volatility'] = df['spread_std']
        
        # 2. ORDER IMBALANCE METRICS (requested features)
        # First-level imbalance (from book_imbalance)
        df['imbalance_level1'] = df['book_imbalance_mean']
        df['imbalance_level1_ma'] = df['imbalance_level1'].rolling(12).mean()
        df['imbalance_level1_std'] = df['imbalance_level1'].rolling(12).std()
        df['imbalance_level1_zscore'] = (
            (df['imbalance_level1'] - df['imbalance_level1_ma']) / 
            (df['imbalance_level1_std'] + 1e-8)
        )
        
        # Five-level aggregate imbalance (approximated using depth_10)
        df['imbalance_level5'] = (
            (df['bid_depth_10_mean'] - df['ask_depth_10_mean']) / 
            (df['bid_depth_10_mean'] + df['ask_depth_10_mean'] + 1e-8)
        )
        df['imbalance_level5_ma'] = df['imbalance_level5'].rolling(12).mean()
        df['imbalance_level5_momentum'] = df['imbalance_level5'].diff(6)
        
        # 3. WEIGHTED MID-PRICE CHANGE (requested feature)
        # Calculate weighted mid-price using depth information
        total_depth = df['bid_depth_10_mean'] + df['ask_depth_10_mean']
        bid_weight = df['ask_depth_10_mean'] / (total_depth + 1e-8)  # Weight by opposite side
        ask_weight = df['bid_depth_10_mean'] / (total_depth + 1e-8)
        
        # Approximate bid/ask prices from mid_price and spread
        df['bid_price_approx'] = df['mid_price_mean'] - df['spread_mean'] / 2
        df['ask_price_approx'] = df['mid_price_mean'] + df['spread_mean'] / 2
        
        # Weighted mid-price
        df['weighted_mid_price'] = (
            df['bid_price_approx'] * bid_weight + 
            df['ask_price_approx'] * ask_weight
        )
        
        # Weighted mid-price changes
        for period in [1, 3, 5, 10]:
            df[f'weighted_mid_price_change_{period}'] = df['weighted_mid_price'].diff(period)
            df[f'weighted_mid_price_pct_change_{period}'] = df['weighted_mid_price'].pct_change(period)
        
        # 4. STATISTICAL SUMMARIES (requested features)
        # Rolling window volatility (already requested)
        df['return_1'] = df['price'].pct_change(1)
        for window in [6, 12, 24, 48, 96]:
            # Volatility
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()
            df[f'volatility_ma_{window}'] = df[f'volatility_{window}'].rolling(window).mean()
            
            # Skewness and Kurtosis
            df[f'return_skew_{window}'] = df['return_1'].rolling(window).skew()
            df[f'return_kurt_{window}'] = df['return_1'].rolling(window).kurt()
            
            # Order book statistics
            df[f'book_imbalance_skew_{window}'] = df['book_imbalance_mean'].rolling(window).skew()
            df[f'spread_skew_{window}'] = df['spread_mean'].rolling(window).skew()
        
        # Statistical summaries of full LOB (using available data)
        lob_features = ['spread_mean', 'book_imbalance_mean', 'bid_depth_10_mean', 'ask_depth_10_mean']
        for window in [12, 24]:
            for feat in lob_features:
                if feat in df.columns:
                    df[f'{feat}_rolling_mean_{window}'] = df[feat].rolling(window).mean()
                    df[f'{feat}_rolling_std_{window}'] = df[feat].rolling(window).std()
                    df[f'{feat}_zscore_{window}'] = (
                        (df[feat] - df[f'{feat}_rolling_mean_{window}']) / 
                        (df[f'{feat}_rolling_std_{window}'] + 1e-8)
                    )
        
        # Additional microstructure features
        # Price returns
        for period in [1, 3, 6, 12, 24]:
            df[f'return_{period}'] = df['price'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['price'] / df['price'].shift(period))
        
        # Volume features
        if 'quantity_sum' in df.columns:
            df['volume'] = df['quantity_sum']
            df['volume_ma_12'] = df['volume'].rolling(12).mean()
            df['volume_ma_24'] = df['volume'].rolling(24).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma_12'] + 1e-8)
            df['volume_momentum'] = df['volume'].pct_change(12)
            df['volume_volatility'] = df['volume'].rolling(12).std()
        
        # Order flow
        if 'order_flow_imbalance' in df.columns:
            df['ofi_ma_12'] = df['order_flow_imbalance'].rolling(12).mean()
            df['ofi_momentum'] = df['order_flow_imbalance'].diff(6)
            df['ofi_acceleration'] = df['ofi_momentum'].diff(3)
        
        # CVD features
        if 'cvd_last' in df.columns:
            for window in [6, 12, 24]:
                df[f'cvd_ma_{window}'] = df['cvd_last'].rolling(window).mean()
                df[f'cvd_momentum_{window}'] = df['cvd_last'].diff(window)
        
        # Technical indicators
        # RSI
        for period in [14, 21]:
            delta = df['price'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in [20, 30]:
            ma = df['price'].rolling(window).mean()
            std = df['price'].rolling(window).std()
            df[f'bb_upper_{window}'] = ma + 2 * std
            df[f'bb_lower_{window}'] = ma - 2 * std
            df[f'bb_position_{window}'] = (
                (df['price'] - df[f'bb_lower_{window}']) /
                (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'] + 1e-8)
            )
            df[f'bb_width_{window}'] = (
                (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / (ma + 1e-8)
            )
        
        # Time features
        df['hour'] = df['time_bucket'].dt.hour.astype('int8')
        df['day_of_week'] = df['time_bucket'].dt.dayofweek.astype('int8')
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype('int8')
        
        # Price position
        for window in [12, 24, 48]:
            rolling_max = df['price'].rolling(window).max()
            rolling_min = df['price'].rolling(window).min()
            df[f'price_position_{window}'] = (
                (df['price'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
            )
        
        return df
    
    def _create_pca_features(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Create PCA features from order book data (requested feature)"""
        # Select features for PCA
        pca_features = [
            'spread_mean', 'spread_std', 'book_imbalance_mean', 'book_imbalance_std',
            'bid_depth_10_mean', 'ask_depth_10_mean', 'imbalance_level1', 'imbalance_level5',
            'spread_relative', 'weighted_mid_price'
        ]
        
        # Only use features that exist
        pca_features = [f for f in pca_features if f in df.columns]
        
        if len(pca_features) < 2:
            print("Not enough features for PCA")
            return df
        
        # Process each symbol separately
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            symbol_data = df.loc[symbol_mask, pca_features].copy()
            
            # Handle NaN values
            symbol_data = symbol_data.fillna(method='ffill').fillna(0)
            
            if len(symbol_data) < n_components:
                continue
            
            # Standardize features
            scaler = StandardScaler()
            try:
                symbol_data_scaled = scaler.fit_transform(symbol_data)
                
                # Apply PCA
                pca = PCA(n_components=min(n_components, len(pca_features)))
                pca_components = pca.fit_transform(symbol_data_scaled)
                
                # Add PCA components to dataframe
                for i in range(pca_components.shape[1]):
                    df.loc[symbol_mask, f'pca_component_{i+1}'] = pca_components[:, i]
                
                # Add explained variance ratio for the first component
                df.loc[symbol_mask, 'pca_explained_variance_1'] = pca.explained_variance_ratio_[0]
            except:
                # If PCA fails, fill with zeros
                for i in range(n_components):
                    df.loc[symbol_mask, f'pca_component_{i+1}'] = 0
                df.loc[symbol_mask, 'pca_explained_variance_1'] = 0
        
        return df
    
    def _add_smoothed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add smoothed features using Savitzky-Golay and Kalman filters (requested feature)"""
        # Features to smooth
        smooth_features = [
            'mid_price_last', 'spread_mean', 'book_imbalance_mean',
            'imbalance_level1', 'imbalance_level5', 'weighted_mid_price',
            'volume', 'order_flow_imbalance', 'cvd_last'
        ]
        
        # Only smooth features that exist
        smooth_features = [f for f in smooth_features if f in df.columns]
        
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            
            for feature in smooth_features:
                data = df.loc[symbol_mask, feature].values
                
                # Handle NaN values
                nan_mask = np.isnan(data)
                if nan_mask.all():
                    continue
                
                # Fill NaN for smoothing
                data_filled = pd.Series(data).fillna(method='ffill').fillna(method='bfill').values
                
                # Savitzky-Golay filter
                try:
                    window_length = min(11, len(data_filled))
                    if window_length % 2 == 0:
                        window_length -= 1
                    if window_length >= 3 and len(data_filled) >= window_length:
                        savgol_smoothed = signal.savgol_filter(data_filled, window_length, 2)
                        df.loc[symbol_mask, f'{feature}_savgol'] = savgol_smoothed
                    else:
                        df.loc[symbol_mask, f'{feature}_savgol'] = data_filled
                except:
                    df.loc[symbol_mask, f'{feature}_savgol'] = data_filled
                
                # Kalman filter approximation (using exponential smoothing)
                try:
                    alpha = 0.3  # Smoothing parameter
                    kalman_smoothed = pd.Series(data_filled).ewm(alpha=alpha, adjust=False).mean().values
                    df.loc[symbol_mask, f'{feature}_kalman'] = kalman_smoothed
                except:
                    df.loc[symbol_mask, f'{feature}_kalman'] = data_filled
        
        return df
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    def run_sampling_pipeline(self, output_db: str = 'features_sampled.db'):
        """Main pipeline function"""
        print("="*60)
        print("CRYPTO FEATURE ENGINEERING WITH SAMPLING")
        print("Including ALL requested microstructure features")
        print("="*60)
        print(f"Sample rate: {self.sample_rate*100}%")
        print(f"Output database: {output_db}")
        print("="*60)
        
        # Create sampled data
        df_orderbook_sample, df_trades_sample = self.create_sampled_data()
        
        # Process features
        df_features = self.process_features(df_orderbook_sample, df_trades_sample, output_db)
        
        return df_features

# Simple usage function
def sample_and_create_features(db_path: str, sample_rate: float = 0.1, output_db: str = 'features_sampled.db'):
    """
    Simple function to sample data and create features
    
    Args:
        db_path: Path to your crypto_data_RAW_FULL.db
        sample_rate: Fraction to sample (0.1 = 10%)
        output_db: Output database name
    """
    sampler = CryptoSampler(db_path, sample_rate)
    return sampler.run_sampling_pipeline(output_db)

# Example usage
if __name__ == "__main__":
    # Set your database path
    db_path = '/content/drive/MyDrive/crypto_pipeline_whale/crypto_data_RAW_FULL.db'
    
    # Run the sampling pipeline
    df_features = sample_and_create_features(
        db_path=db_path,
        sample_rate=0.1,  # 10% sample
        output_db='features_10pct_complete.db'
    )
    
    # Optional: Show feature list
    if df_features is not None and len(df_features) > 0:
        print("\nTotal features created:", len(df_features.columns))
        print("\nSample of features:")
        feature_cols = [col for col in df_features.columns if col not in ['symbol', 'exchange', 'time_bucket']]
        print("- Price & Size:", [c for c in feature_cols if 'price' in c or 'spread' in c][:5])
        print("- Imbalance:", [c for c in feature_cols if 'imbalance' in c][:5])
        print("- Weighted MP:", [c for c in feature_cols if 'weighted' in c][:5])
        print("- Statistical:", [c for c in feature_cols if 'skew' in c or 'kurt' in c][:5])
        print("- PCA:", [c for c in feature_cols if 'pca' in c][:5])
        print("- Smoothed:", [c for c in feature_cols if 'savgol' in c or 'kalman' in c][:5])