"""
Enhanced Feature Engineering Pipeline with Whale Transaction and Social Mention Data Integration
"""
import pandas as pd
import numpy as np
import sqlite3
import gc
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime, timedelta
import os

from config import *
from utils import optimize_memory_usage, save_cache_data, load_cache_data, bool_to_int8


class WhaleTransactionClassifier:
    """Classify whale transactions as institutional or retail"""
    
    @staticmethod
    def parse_market_cap(mc_string):
        """Convert market cap string to numeric value"""
        if pd.isna(mc_string) or not isinstance(mc_string, str):
            return 0
        
        # Remove $ and commas
        cleaned = mc_string.replace('$', '').replace(',', '')
        
        # Handle millions (M) and billions (B)
        if 'M' in cleaned:
            return float(cleaned.replace('M', '')) * 1_000_000
        elif 'B' in cleaned:
            return float(cleaned.replace('B', '')) * 1_000_000_000
        else:
            try:
                return float(cleaned)
            except:
                return 0

    @staticmethod
    def classify_transaction(row):
        """
        Classify a transaction as Institutional, Retail, or Mixed based on:
        - Market cap size
        - Transaction amount
        - Token characteristics
        - Network
        """
        
        # Parse values
        market_cap = WhaleTransactionClassifier.parse_market_cap(row['market_cap'])
        amount_usd = float(row['amount_usd']) if pd.notna(row['amount_usd']) else 0
        token = str(row['token']).lower() if pd.notna(row['token']) else ''
        network = str(row['network']).lower() if pd.notna(row['network']) else ''
        
        # Calculate institutional score (0-10)
        inst_score = 0
        
        # Market cap scoring
        if market_cap >= 1_000_000_000:  # $1B+
            inst_score += 4
        elif market_cap >= 500_000_000:  # $500M+
            inst_score += 3
        elif market_cap >= 100_000_000:  # $100M+
            inst_score += 2
        elif market_cap >= 50_000_000:   # $50M+
            inst_score += 1
        
        # Transaction amount scoring
        if amount_usd >= 250_000:        # $250k+
            inst_score += 4
        elif amount_usd >= 100_000:      # $100k+
            inst_score += 3
        elif amount_usd >= 50_000:       # $50k+
            inst_score += 2
        elif amount_usd >= 25_000:       # $25k+
            inst_score += 1
        
        # Network bonus (established networks)
        if network in ['ethereum', 'bitcoin']:
            inst_score += 1
        
        # Token type penalty (likely retail tokens)
        retail_indicators = ['doge', 'shib', 'pepe', 'meme', 'moon', 'safe', 
                            'inu', 'floki', 'baby', 'mini', 'micro']
        if any(indicator in token for indicator in retail_indicators):
            inst_score -= 2
        
        # Classification logic
        classification = ''
        confidence = ''
        
        if inst_score >= 7:
            classification = 'Institutional'
            confidence = 'High'
        elif inst_score >= 5:
            classification = 'Institutional'
            confidence = 'Medium'
        elif inst_score >= 3:
            classification = 'Mixed'
            confidence = 'Medium'
        else:
            classification = 'Retail'
            confidence = 'High' if inst_score <= 1 else 'Medium'
        
        # Special cases override
        if market_cap >= 1_000_000_000 and amount_usd >= 100_000:
            classification = 'Institutional'
            confidence = 'Very High'
        elif market_cap < 50_000_000 and amount_usd < 10_000:
            classification = 'Retail'
            confidence = 'Very High'
        
        return pd.Series({
            'classification': classification,
            'confidence': confidence,
            'institutional_score': inst_score,
            'market_cap_numeric': market_cap
        })


class DataIntegrator:
    """Integrate orderbook, trades, whale, and social mention data"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "crypto_integrated_data.db")
    
    def load_and_prepare_whale_data(self, whale_df: pd.DataFrame) -> pd.DataFrame:
        """Load and prepare whale transaction data"""
        print("Preparing whale transaction data...")
        
        # Apply classification
        classifications = whale_df.apply(WhaleTransactionClassifier.classify_transaction, axis=1)
        whale_df = pd.concat([whale_df, classifications], axis=1)
        
        # Parse timestamp
        whale_df['timestamp'] = pd.to_datetime(whale_df['timestamp'])
        
        # Parse numeric fields
        whale_df['amount_usd'] = pd.to_numeric(whale_df['amount_usd'], errors='coerce')
        whale_df['quantity'] = whale_df['quantity'].str.replace(',', '').astype(float)
        whale_df['price'] = whale_df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Create time bucket (5-minute intervals to match orderbook data)
        whale_df['time_bucket'] = whale_df['timestamp'].dt.floor('5min')
        
        # Create binary features
        whale_df['is_institutional'] = (whale_df['classification'] == 'Institutional').astype(int)
        whale_df['is_retail'] = (whale_df['classification'] == 'Retail').astype(int)
        whale_df['is_buy'] = (whale_df['transaction_type'] == 'BUY').astype(int)
        whale_df['is_sell'] = (whale_df['transaction_type'] == 'SELL').astype(int)
        
        return whale_df
    
    def aggregate_whale_features(self, whale_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate whale features by token and time bucket"""
        print("Aggregating whale features...")
        
        # Group by token and time bucket
        agg_features = []
        
        for (token, time_bucket), group in whale_df.groupby(['token', 'time_bucket']):
            features = {
                'symbol': token.upper(),
                'time_bucket': time_bucket,
                
                # Transaction counts
                'whale_amount_usd_sum': group['amount_usd'].sum(),
                'whale_amount_usd_mean': group['amount_usd'].mean(),
                'whale_amount_usd_max': group['amount_usd'].max(),
                'whale_amount_usd_count': len(group),
                
                # Institutional vs Retail
                'inst_amount_usd_sum': group[group['is_institutional'] == 1]['amount_usd'].sum(),
                'retail_amount_usd_sum': group[group['is_retail'] == 1]['amount_usd'].sum(),
                'inst_count': group['is_institutional'].sum(),
                'retail_count': group['is_retail'].sum(),
                
                # Buy vs Sell
                'whale_buy_amount': group[group['is_buy'] == 1]['amount_usd'].sum(),
                'whale_sell_amount': group[group['is_sell'] == 1]['amount_usd'].sum(),
                'whale_buy_count': group['is_buy'].sum(),
                'whale_sell_count': group['is_sell'].sum(),
                
                # Market cap stats
                'avg_market_cap': group['market_cap_numeric'].mean(),
                'max_market_cap': group['market_cap_numeric'].max(),
            }
            
            # Derived features
            total_amount = features['inst_amount_usd_sum'] + features['retail_amount_usd_sum']
            if total_amount > 0:
                features['inst_participation_rate'] = features['inst_amount_usd_sum'] / total_amount
            else:
                features['inst_participation_rate'] = 0
            
            # Buy/Sell pressure
            total_whale = features['whale_buy_amount'] + features['whale_sell_amount']
            if total_whale > 0:
                features['whale_buy_pressure'] = features['whale_buy_amount'] / total_whale
                features['whale_sell_pressure'] = features['whale_sell_amount'] / total_whale
            else:
                features['whale_buy_pressure'] = 0.5
                features['whale_sell_pressure'] = 0.5
            
            # Flow imbalance
            features['whale_flow_imbalance'] = (
                (features['whale_buy_amount'] - features['whale_sell_amount']) / 
                (features['whale_buy_amount'] + features['whale_sell_amount'] + 1e-8)
            )
            
            # Retail selling pressure (negative correlation with price)
            features['retail_sell_pressure'] = (
                features['retail_count'] * features['whale_sell_pressure']
            )
            
            # Smart money divergence
            features['smart_dumb_divergence'] = (
                features['inst_participation_rate'] - (1 - features['inst_participation_rate'])
            )
            
            # Average trade size
            if features['whale_amount_usd_count'] > 0:
                features['avg_trade_size'] = (
                    features['whale_amount_usd_sum'] / features['whale_amount_usd_count']
                )
            else:
                features['avg_trade_size'] = 0
            
            agg_features.append(features)
        
        return pd.DataFrame(agg_features)
    
    def load_and_prepare_social_data(self, mentions4h_df: pd.DataFrame, mentions14d_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare social mention data"""
        print("Preparing social mention data...")
        
        # Process 4H mentions data
        mentions4h_df['date'] = pd.to_datetime(mentions4h_df['date'])
        mentions4h_df['time_bucket'] = mentions4h_df['date'].dt.floor('5min')
        
        # Melt the dataframe to get token-wise data
        token_columns = ['ethereum', 'bitcoin', 'BTC', 'ETH', 'Solana']
        momentum_columns = {
            '4H': ['ethereum_4H', 'bitcoin_4H', 'BTC_4H', 'ETH_4H', 'Solana_4H'],
            '7d': ['ethereum_7d', 'bitcoin_7d', 'BTC_7d', 'ETH_7d', 'Solana_7d'],
            '1m': ['ethereum_1m', 'bitcoin_1m', 'BTC_1m', 'ETH_1m', 'Solana_1m']
        }
        
        social_features = []
        
        for idx, row in mentions4h_df.iterrows():
            for i, token in enumerate(token_columns):
                # Map token names to symbols
                symbol = token.upper()
                if token.lower() == 'ethereum':
                    symbol = 'ETH'
                elif token.lower() == 'bitcoin':
                    symbol = 'BTC'
                elif token.lower() == 'solana':
                    symbol = 'SOL'
                
                features = {
                    'symbol': symbol,
                    'time_bucket': row['time_bucket'],
                    'mention_count': row[token],
                    'mention_momentum_4h': row[momentum_columns['4H'][i]],
                    'mention_momentum_7d': row[momentum_columns['7d'][i]],
                    'mention_momentum_1m': row[momentum_columns['1m'][i]]
                }
                social_features.append(features)
        
        social_4h_df = pd.DataFrame(social_features)
        
        # Process 14D mentions data
        mentions14d_df['timestamp'] = pd.to_datetime(mentions14d_df['timestamp'])
        mentions14d_df['time_bucket'] = mentions14d_df['timestamp'].dt.floor('5min')
        
        # Aggregate by ticker and time bucket
        social_14d_agg = mentions14d_df.groupby(['ticker', 'time_bucket']).agg({
            'mention_count': 'sum',
            'mention_change': 'mean',
            'sentiment': 'mean',
            'price_change_1d': 'mean'
        }).reset_index()
        
        social_14d_agg.rename(columns={'ticker': 'symbol'}, inplace=True)
        
        return social_4h_df, social_14d_agg
    
    def aggregate_social_features(self, social_4h_df: pd.DataFrame, social_14d_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate social features"""
        print("Aggregating social features...")
        
        # Merge the two social dataframes
        social_merged = pd.merge(
            social_4h_df,
            social_14d_df,
            on=['symbol', 'time_bucket'],
            how='outer',
            suffixes=('_4h', '_14d')
        )
        
        # Fill missing values
        social_merged['mention_count_4h'] = social_merged['mention_count'].fillna(0)
        social_merged['mention_count_14d'] = social_merged['mention_count_14d'].fillna(0)
        social_merged['sentiment'] = social_merged['sentiment'].fillna(0)
        
        # Create composite social features
        social_merged['total_mentions'] = (
            social_merged['mention_count_4h'] + social_merged['mention_count_14d']
        )
        
        # Social momentum score
        social_merged['social_momentum_score'] = (
            social_merged['mention_momentum_4h'].fillna(0) * 0.5 +
            social_merged['mention_momentum_7d'].fillna(0) * 0.3 +
            social_merged['mention_momentum_1m'].fillna(0) * 0.2
        )
        
        # Sentiment-weighted mentions
        social_merged['sentiment_weighted_mentions'] = (
            social_merged['total_mentions'] * (1 + social_merged['sentiment'])
        )
        
        return social_merged
    
    def merge_all_data(self, orderbook_df: pd.DataFrame, trades_df: pd.DataFrame, 
                      whale_agg_df: pd.DataFrame, social_agg_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources"""
        print("Merging all data sources...")
        
        # First merge orderbook and trades
        merged_df = pd.merge(
            orderbook_df,
            trades_df,
            on=['symbol', 'time_bucket'],
            how='outer',
            suffixes=('_orderbook', '_trades')
        )
        
        # Merge with whale data
        merged_df = pd.merge(
            merged_df,
            whale_agg_df,
            on=['symbol', 'time_bucket'],
            how='left'
        )
        
        # Merge with social data
        merged_df = pd.merge(
            merged_df,
            social_agg_df,
            on=['symbol', 'time_bucket'],
            how='left'
        )
        
        # Fill missing whale and social features with 0
        whale_cols = [col for col in merged_df.columns if 'whale_' in col or 'inst_' in col or 'retail_' in col]
        social_cols = [col for col in merged_df.columns if 'mention' in col or 'sentiment' in col or 'social' in col]
        
        merged_df[whale_cols] = merged_df[whale_cols].fillna(0)
        merged_df[social_cols] = merged_df[social_cols].fillna(0)
        
        return merged_df
    
    def save_to_sqlite(self, dataframes: Dict[str, pd.DataFrame]):
        """Save all dataframes to SQLite database"""
        print(f"Saving data to SQLite database: {self.db_path}")
        
        with sqlite3.connect(self.db_path) as conn:
            for table_name, df in dataframes.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"  - Saved {table_name}: {len(df)} rows")
    
    def create_market_cap_features(self, whale_agg_df: pd.DataFrame) -> pd.DataFrame:
        """Create market cap stratified features"""
        # Define market cap tiers
        whale_agg_df['is_megacap'] = (whale_agg_df['max_market_cap'] >= 10_000_000_000).astype(int)
        whale_agg_df['is_largecap'] = (
            (whale_agg_df['max_market_cap'] >= 1_000_000_000) & 
            (whale_agg_df['max_market_cap'] < 10_000_000_000)
        ).astype(int)
        whale_agg_df['is_midcap'] = (
            (whale_agg_df['max_market_cap'] >= 100_000_000) & 
            (whale_agg_df['max_market_cap'] < 1_000_000_000)
        ).astype(int)
        whale_agg_df['is_smallcap'] = (whale_agg_df['max_market_cap'] < 100_000_000).astype(int)
        
        # Create tier-specific flow features
        whale_agg_df['megacap_flow'] = whale_agg_df['is_megacap'] * whale_agg_df['whale_flow_imbalance']
        whale_agg_df['smallcap_speculation'] = whale_agg_df['is_smallcap'] * whale_agg_df['retail_participation_rate']
        
        return whale_agg_df


class EnhancedFeatureEngineer(FeatureEngineer):
    """Enhanced feature engineer that includes whale and social features"""
    
    def __init__(self):
        super().__init__()
        self.data_integrator = DataIntegrator()
    
    def create_advanced_features(self, df: pd.DataFrame, 
                               fill_values: Optional[Dict] = None, 
                               use_cache: bool = True,
                               include_whale_data: bool = True,
                               include_social_data: bool = True,
                               whale_df: Optional[pd.DataFrame] = None,
                               mentions4h_df: Optional[pd.DataFrame] = None,
                               mentions14d_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create advanced features with whale and social data integration"""
        
        # Try to load cached features first
        cache_suffix = ""
        if include_whale_data:
            cache_suffix += "_whale"
        if include_social_data:
            cache_suffix += "_social"
            
        if use_cache:
            cached_features = load_cache_data(CACHE_DIR, f"features_latest{cache_suffix}.parquet")
            if cached_features is not None:
                user_input = input(f"\nUse cached features{cache_suffix}? (y/n): ").lower()
                if user_input == 'y':
                    print("Using cached features!")
                    return cached_features
                else:
                    print("Recalculating features...")
        
        # If we need whale or social data, process them first
        if include_whale_data and whale_df is not None:
            # Process whale data
            whale_processed = self.data_integrator.load_and_prepare_whale_data(whale_df)
            whale_agg = self.data_integrator.aggregate_whale_features(whale_processed)
            whale_agg = self.data_integrator.create_market_cap_features(whale_agg)
            
            # Merge whale features with main dataframe
            df = pd.merge(df, whale_agg, on=['symbol', 'time_bucket'], how='left')
            
            # Fill missing whale features
            whale_cols = [col for col in df.columns if 'whale_' in col or 'inst_' in col or 'retail_' in col]
            df[whale_cols] = df[whale_cols].fillna(0)
        
        if include_social_data and mentions4h_df is not None and mentions14d_df is not None:
            # Process social data
            social_4h, social_14d = self.data_integrator.load_and_prepare_social_data(mentions4h_df, mentions14d_df)
            social_agg = self.data_integrator.aggregate_social_features(social_4h, social_14d)
            
            # Merge social features with main dataframe
            df = pd.merge(df, social_agg, on=['symbol', 'time_bucket'], how='left')
            
            # Fill missing social features
            social_cols = [col for col in df.columns if 'mention' in col or 'sentiment' in col or 'social' in col]
            df[social_cols] = df[social_cols].fillna(0)
        
        # Now call the parent class method to create all features
        return super().create_advanced_features(df, fill_values, use_cache, include_whale_data)
    
    def save_integrated_data(self, orderbook_df: pd.DataFrame, trades_df: pd.DataFrame,
                           whale_df: pd.DataFrame, mentions4h_df: pd.DataFrame, 
                           mentions14d_df: pd.DataFrame):
        """Save all integrated data to SQLite database"""
        
        # Process all data
        whale_processed = self.data_integrator.load_and_prepare_whale_data(whale_df)
        whale_agg = self.data_integrator.aggregate_whale_features(whale_processed)
        
        social_4h, social_14d = self.data_integrator.load_and_prepare_social_data(mentions4h_df, mentions14d_df)
        social_agg = self.data_integrator.aggregate_social_features(social_4h, social_14d)
        
        # Merge all data
        integrated_df = self.data_integrator.merge_all_data(orderbook_df, trades_df, whale_agg, social_agg)
        
        # Create feature-engineered version
        features_df = self.create_advanced_features(
            integrated_df, 
            include_whale_data=True, 
            include_social_data=True
        )
        
        # Save to SQLite
        dataframes = {
            'orderbook_raw': orderbook_df,
            'trades_raw': trades_df,
            'whale_transactions': whale_processed,
            'whale_aggregated': whale_agg,
            'social_4h': social_4h,
            'social_14d': social_14d,
            'social_aggregated': social_agg,
            'integrated_raw': integrated_df,
            'features_engineered': features_df
        }
        
        self.data_integrator.save_to_sqlite(dataframes)
        
        print("\nData integration complete!")
        print(f"SQLite database saved to: {self.data_integrator.db_path}")
        
        # Print summary statistics
        print("\nIntegrated data summary:")
        print(f"  - Date range: {integrated_df['time_bucket'].min()} to {integrated_df['time_bucket'].max()}")
        print(f"  - Total rows: {len(integrated_df)}")
        print(f"  - Unique symbols: {integrated_df['symbol'].nunique()}")
        print(f"  - Total features: {len(features_df.columns)}")
        
        return integrated_df, features_df


# Example usage
if __name__ == "__main__":
    # Load your data files
    print("Loading data files...")
    
    # Assuming you have these dataframes loaded
    # orderbook_df = pd.read_parquet('orderbook_data.parquet')
    # trades_df = pd.read_parquet('trades_data.parquet')
    whale_df = pd.read_csv('whale.csv')
    mentions4h_df = pd.read_csv('mentions4h.csv')
    mentions14d_df = pd.read_csv('mentions14d.csv')
    
    # Create enhanced feature engineer
    feature_engineer = EnhancedFeatureEngineer()
    
    # For demonstration, create dummy orderbook and trades data
    # In real usage, load your actual orderbook and trades data
    date_range = pd.date_range('2025-06-05 00:00:00', '2025-06-05 23:55:00', freq='5min')
    symbols = ['BTC', 'ETH', 'SOL']
    
    orderbook_df = pd.DataFrame({
        'symbol': np.repeat(symbols, len(date_range)),
        'time_bucket': np.tile(date_range, len(symbols)),
        'bid_price_1_mean': np.random.randn(len(symbols) * len(date_range)) * 100 + 50000,
        'ask_price_1_mean': np.random.randn(len(symbols) * len(date_range)) * 100 + 50000,
        'bid_size_1_mean': np.random.exponential(10, len(symbols) * len(date_range)),
        'ask_size_1_mean': np.random.exponential(10, len(symbols) * len(date_range)),
    })
    
    trades_df = pd.DataFrame({
        'symbol': np.repeat(symbols, len(date_range)),
        'time_bucket': np.tile(date_range, len(symbols)),
        'price_last': np.random.randn(len(symbols) * len(date_range)) * 100 + 50000,
        'quantity_sum': np.random.exponential(100, len(symbols) * len(date_range)),
        'quantity_count': np.random.poisson(50, len(symbols) * len(date_range)),
    })
    
    # Save integrated data
    integrated_df, features_df = feature_engineer.save_integrated_data(
        orderbook_df, trades_df, whale_df, mentions4h_df, mentions14d_df
    )
    
    print("\nFeature engineering with whale and social data complete!")