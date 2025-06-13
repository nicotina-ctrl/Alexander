"""
Integrated Data Loader with Enhanced Whale Parquet, Trends, and Social Data Support
"""
import os
import glob
import re
import gc
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from config import *
from utils import optimize_memory_usage, save_cache_data, load_cache_data


class TrendsDataLoader:
    """Load and manage trends data from multiple sources"""
    
    def __init__(self, trends_data_path: str = "/content/drive/MyDrive/crypto_pipeline_whale/trends_data"):
        self.trends_data_path = trends_data_path
        self.file_inventory = {}
        self.date_coverage = {}
        
    def scan_directory(self) -> Dict[str, List[str]]:
        """Scan trends directory for all CSV and Parquet files"""
        print(f"Scanning directory: {self.trends_data_path}")
        
        if not os.path.exists(self.trends_data_path):
            print(f"Directory not found: {self.trends_data_path}")
            return {}
        
        # Find all CSV and Parquet files
        csv_files = glob.glob(os.path.join(self.trends_data_path, "**/*.csv"), recursive=True)
        parquet_files = glob.glob(os.path.join(self.trends_data_path, "**/*.parquet"), recursive=True)
        
        print(f"Found {len(csv_files)} CSV files and {len(parquet_files)} Parquet files")
        
        self.file_inventory = {
            'csv': csv_files,
            'parquet': parquet_files,
            'all': csv_files + parquet_files
        }
        
        return self.file_inventory
    
    def extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from filename using multiple patterns"""
        basename = os.path.basename(filename)
        
        # Common date patterns
        patterns = [
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'(\d{4}_\d{2}_\d{2})', '%Y_%m_%d'),
            (r'(\d{8})', '%Y%m%d'),
            (r'(\d{2}-\d{2}-\d{4})', '%d-%m-%Y'),
            (r'(\d{2}_\d{2}_\d{4})', '%d_%m_%Y'),
        ]
        
        for pattern, date_format in patterns:
            match = re.search(pattern, basename)
            if match:
                try:
                    return datetime.strptime(match.group(1), date_format)
                except:
                    continue
        
        return None
    
    def load_trends_data(self, 
                        start_date: Optional[Union[str, datetime]] = None,
                        end_date: Optional[Union[str, datetime]] = None,
                        symbols: Optional[List[str]] = None,
                        use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load trends data with date filtering"""
        
        # Try to load cached trends features first
        if use_cache:
            cached_trends_features = load_cache_data(CACHE_DIR, "trends_features_latest.parquet")
            if cached_trends_features is not None:
                user_input = input("\nUse cached trends features? (y/n): ").lower()
                if user_input == 'y':
                    print("Using cached trends features!")
                    return cached_trends_features
                else:
                    print("Recalculating trends features...")
        
        # Convert dates if provided
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Scan directory if not already done
        if not self.file_inventory:
            self.scan_directory()
        
        if not self.file_inventory.get('all'):
            print("No trends files found")
            return None
        
        # Load data from files
        print(f"\nLoading trends data...")
        
        all_data = []
        for file_path in tqdm(self.file_inventory['all'][:10], desc="Loading trends files"):  # Limit to first 10 for now
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_parquet(file_path)
                
                # Find and parse date column
                date_cols = [col for col in df.columns 
                           if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]
                
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    # Filter by date range if specified
                    if start_date:
                        df = df[df[date_col] >= start_date]
                    if end_date:
                        df = df[df[date_col] <= end_date]
                    
                    # Rename date column to standard name
                    df = df.rename(columns={date_col: 'timestamp'})
                
                # Add source file information
                df['source_file'] = os.path.basename(file_path)
                
                all_data.append(df)
                
            except Exception as e:
                print(f"  Error loading {os.path.basename(file_path)}: {e}")
        
        if not all_data:
            print("No trends data loaded")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create time bucket for aggregation
        if 'timestamp' in combined_df.columns:
            combined_df['time_bucket'] = combined_df['timestamp'].dt.floor('5min')
        
        # Process and aggregate trends features
        trends_features = self._process_trends_features(combined_df, symbols)
        
        # Save cached trends features
        if use_cache and trends_features is not None and len(trends_features) > 0:
            save_cache_data(trends_features, CACHE_DIR, "trends_features_latest.parquet")
        
        return trends_features
    
    def _process_trends_features(self, df: pd.DataFrame, symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Process raw trends data into features"""
        
        if df.empty or 'time_bucket' not in df.columns:
            return None
        
        # Example processing - adjust based on your actual trends data structure
        # This assumes trends data has columns like: token, metric, value, etc.
        
        features_list = []
        
        # If we have token/symbol column, group by it
        if 'token' in df.columns or 'symbol' in df.columns:
            group_col = 'token' if 'token' in df.columns else 'symbol'
            
            # Map tokens to standard symbols if needed
            if group_col == 'token' and symbols:
                # Filter for relevant tokens
                allowed_tokens = []
                for symbol in symbols:
                    if symbol in SYMBOL_MAPPING:
                        allowed_tokens.extend(SYMBOL_MAPPING[symbol])
                
                pattern = '|'.join(allowed_tokens)
                df = df[df[group_col].str.upper().str.contains(pattern, na=False)]
                
                # Map to standard symbol
                df['symbol'] = df[group_col].str.upper().map(TOKEN_TO_SYMBOL)
                df = df.dropna(subset=['symbol'])
                df['symbol'] = df['symbol'] + '/USDT'
            
            # Aggregate by symbol and time bucket
            grouped = df.groupby(['symbol', 'time_bucket'])
            
            # Create aggregated features based on available columns
            agg_dict = {}
            
            # Common metrics in trends data
            metric_cols = ['volume', 'price', 'market_cap', 'rank', 'mentions', 'sentiment']
            for col in metric_cols:
                if col in df.columns:
                    agg_dict[col] = ['mean', 'std', 'min', 'max']
            
            if agg_dict:
                trends_agg = grouped.agg(agg_dict)
                trends_agg.columns = ['trend_' + '_'.join(col).strip() for col in trends_agg.columns]
                trends_agg = trends_agg.reset_index()
                
                features_list.append(trends_agg)
        
        if not features_list:
            return None
        
        # Combine all features
        trends_features = pd.concat(features_list, ignore_index=True) if len(features_list) > 1 else features_list[0]
        
        # Optimize memory
        trends_features = optimize_memory_usage(trends_features, verbose=False)
        
        print(f"Created {len(trends_features)} trends feature rows")
        
        return trends_features


class WhaleDataParquetLoader:
    """Specialized loader for whale transaction data from parquet files"""
    
    def __init__(self, whale_data_path: str = "/content/drive/MyDrive/crypto_pipeline_whale/data/whale_data"):
        self.whale_data_path = whale_data_path
        
    def find_whale_parquet_files(self, start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> List[str]:
        """Find all whale parquet files in the directory, optionally filtered by date"""
        
        # Find all parquet files
        parquet_pattern = os.path.join(self.whale_data_path, "*.parquet")
        all_files = sorted(glob.glob(parquet_pattern))
        
        if not all_files:
            print(f"No parquet files found in {self.whale_data_path}")
            return []
        
        print(f"Found {len(all_files)} parquet files in whale_data directory")
        
        # If no date filtering, return all files
        if not start_date and not end_date:
            return all_files
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        # Try to extract dates from filenames
        filtered_files = []
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{8})',              # YYYYMMDD
            r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
        ]
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            file_date = None
            
            # Try each date pattern
            for pattern in date_patterns:
                match = re.search(pattern, filename)
                if match:
                    date_str = match.group(1).replace('_', '-')
                    try:
                        # Handle YYYYMMDD format
                        if len(date_str) == 8 and '-' not in date_str:
                            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        file_date = pd.to_datetime(date_str)
                        break
                    except:
                        continue
            
            # If we couldn't extract date from filename, include file by default
            if file_date is None:
                print(f"  Warning: Could not extract date from {filename}, including by default")
                filtered_files.append(file_path)
            else:
                # Check if file date is within range
                include = True
                if start_dt and file_date < start_dt:
                    include = False
                if end_dt and file_date > end_dt + timedelta(days=1):
                    include = False
                
                if include:
                    filtered_files.append(file_path)
                    print(f"  Including {filename} (date: {file_date.date()})")
        
        print(f"Filtered to {len(filtered_files)} files within date range")
        return filtered_files
    
    def load_whale_parquet_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Load and combine multiple whale parquet files"""
        
        if not file_paths:
            return pd.DataFrame()
        
        whale_dfs = []
        
        for file_path in tqdm(file_paths, desc="Loading whale parquet files"):
            try:
                df = pd.read_parquet(file_path)
                
                # Check for essential columns
                print(f"\n  File: {os.path.basename(file_path)}")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
                
                # Standardize column names if needed
                df = self._standardize_whale_columns(df)
                
                whale_dfs.append(df)
                
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue
        
        if not whale_dfs:
            print("No whale data successfully loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        print(f"\nCombining {len(whale_dfs)} whale dataframes...")
        combined_df = pd.concat(whale_dfs, ignore_index=True)
        
        # Remove duplicates if any
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        after_dedup = len(combined_df)
        if before_dedup > after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate rows")
        
        print(f"Combined whale data shape: {combined_df.shape}")
        
        return combined_df
    
    def _standardize_whale_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize whale data column names across different formats"""
        
        # Common column mappings
        column_mappings = {
            # Timestamp variations
            'date': 'timestamp',
            'time': 'timestamp',
            'datetime': 'timestamp',
            'created_at': 'timestamp',
            'transaction_time': 'timestamp',
            
            # Token variations
            'coin': 'token',
            'symbol': 'token',
            'ticker': 'token',
            'asset': 'token',
            
            # Amount variations
            'amount': 'amount_usd',
            'value': 'amount_usd',
            'usd_value': 'amount_usd',
            'value_usd': 'amount_usd',
            'transaction_value': 'amount_usd',
            
            # Transaction type variations
            'type': 'transaction_type',
            'side': 'transaction_type',
            'direction': 'transaction_type',
            'tx_type': 'transaction_type',
            
            # Market cap variations
            'mcap': 'market_cap',
            'marketcap': 'market_cap',
            'market_capitalization': 'market_cap',
            
            # Network variations
            'chain': 'network',
            'blockchain': 'network',
        }
        
        # Apply mappings
        df = df.rename(columns=column_mappings)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Standardize token names
        if 'token' in df.columns:
            df['token'] = df['token'].str.upper()
        
        # Standardize transaction types
        if 'transaction_type' in df.columns:
            df['transaction_type'] = df['transaction_type'].str.upper()
            # Map common variations
            type_mappings = {
                'BUY': 'BUY',
                'BOUGHT': 'BUY',
                'PURCHASE': 'BUY',
                'LONG': 'BUY',
                'SELL': 'SELL',
                'SOLD': 'SELL',
                'SHORT': 'SELL',
            }
            df['transaction_type'] = df['transaction_type'].map(
                lambda x: type_mappings.get(x, x) if pd.notna(x) else x
            )
        
        return df
    
    def load_and_aggregate_whale_data(self, symbols: Optional[List[str]] = None,
                                     start_date: Optional[Union[str, pd.Timestamp]] = None,
                                     end_date: Optional[Union[str, pd.Timestamp]] = None,
                                     use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load whale transaction data from parquet files and create aggregated features"""
        
        # Convert string dates to timestamps if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Try to load cached whale features first
        if use_cache:
            cached_whale_features = load_cache_data(CACHE_DIR, "whale_features_latest.parquet")
            if cached_whale_features is not None:
                user_input = input("\nUse cached whale features? (y/n): ").lower()
                if user_input == 'y':
                    print("Using cached whale features!")
                    return cached_whale_features
                else:
                    print("Recalculating whale features...")
        
        # Find whale parquet files
        whale_files = self.find_whale_parquet_files(
            start_date=start_date.strftime('%Y-%m-%d') if start_date else None,
            end_date=end_date.strftime('%Y-%m-%d') if end_date else None
        )
        
        if not whale_files:
            print("No whale parquet files found in the specified date range")
            return None
        
        # Load whale data from parquet files
        whale_df = self.load_whale_parquet_files(whale_files)
        
        if whale_df.empty:
            print("No whale data loaded")
            return None
        
        print(f"\nLoaded {len(whale_df)} whale transactions")
        
        # Additional date filtering on loaded data
        if 'timestamp' in whale_df.columns:
            if start_date is not None:
                whale_df = whale_df[whale_df['timestamp'] >= start_date]
            if end_date is not None:
                whale_df = whale_df[whale_df['timestamp'] <= end_date]
            
            print(f"After date filtering: {len(whale_df)} transactions")
        
        # Filter by symbols if specified
        if symbols and 'token' in whale_df.columns:
            # Create token filter based on symbol mapping
            allowed_tokens = []
            for symbol in symbols:
                if symbol in SYMBOL_MAPPING:
                    allowed_tokens.extend(SYMBOL_MAPPING[symbol])
            
            # Filter whale data
            pattern = '|'.join(allowed_tokens)
            whale_df = whale_df[whale_df['token'].str.upper().str.contains(pattern, na=False)]
            
            print(f"After symbol filtering: {len(whale_df)} transactions")
        
        if len(whale_df) == 0:
            print("No whale data after filtering")
            return None
        
        # Map token to standard symbol
        whale_df['symbol'] = whale_df['token'].str.upper().map(TOKEN_TO_SYMBOL)
        whale_df = whale_df.dropna(subset=['symbol'])
        
        # Convert symbol back to exchange format
        whale_df['symbol'] = whale_df['symbol'] + '/USDT'
        
        print(f"Mapped to symbols: {whale_df['symbol'].unique()}")
        
        # Classify transactions if not already done
        if 'institutional_score' not in whale_df.columns:
            whale_df = self._classify_transactions(whale_df)
        
        # Create time bucket
        whale_df['time_bucket'] = whale_df['timestamp'].dt.floor('5min')
        
        # Aggregate features
        whale_features = self._create_whale_features(whale_df)
        
        # Save cached whale features
        if use_cache and len(whale_features) > 0:
            save_cache_data(whale_features, CACHE_DIR, "whale_features_latest.parquet")
        
        return whale_features
    
    def _classify_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify whale transactions as institutional vs retail"""
        
        def classify_transaction(row):
            """Classify a single transaction"""
            # Parse market cap
            market_cap = 0
            if pd.notna(row.get('market_cap')):
                mc_str = str(row['market_cap']).replace('$', '').replace(',', '')
                if 'M' in mc_str:
                    market_cap = float(mc_str.replace('M', '')) * 1_000_000
                elif 'B' in mc_str:
                    market_cap = float(mc_str.replace('B', '')) * 1_000_000_000
                else:
                    try:
                        market_cap = float(mc_str)
                    except:
                        market_cap = 0
            
            # Get amount
            amount_usd = float(row.get('amount_usd', 0))
            
            # Calculate institutional score
            inst_score = 0
            
            # Market cap scoring
            for threshold, points in [(1_000_000_000, 4), (500_000_000, 3), 
                                     (100_000_000, 2), (50_000_000, 1)]:
                if market_cap >= threshold:
                    inst_score += points
                    break
            
            # Transaction amount scoring
            for threshold, points in [(250_000, 4), (100_000, 3), 
                                     (50_000, 2), (25_000, 1)]:
                if amount_usd >= threshold:
                    inst_score += points
                    break
            
            # Network bonus
            network = str(row.get('network', '')).lower()
            if network in ['ethereum', 'bitcoin']:
                inst_score += 1
            
            # Token type penalty (likely retail tokens)
            token = str(row.get('token', '')).lower()
            retail_indicators = ['doge', 'shib', 'pepe', 'meme', 'moon', 'safe', 
                               'inu', 'floki', 'baby', 'mini', 'micro']
            if any(indicator in token for indicator in retail_indicators):
                inst_score -= 2
            
            # Classification
            if inst_score >= 7:
                classification = 'Institutional'
            elif inst_score >= 3:
                classification = 'Mixed'
            else:
                classification = 'Retail'
            
            return pd.Series({
                'classification': classification,
                'institutional_score': inst_score,
                'market_cap_numeric': market_cap
            })
        
        # Apply classification
        print("Classifying whale transactions...")
        classifications = df.apply(classify_transaction, axis=1)
        
        # Add to dataframe
        for col in classifications.columns:
            df[col] = classifications[col]
        
        # Create binary indicators
        df['is_institutional'] = (df['classification'] == 'Institutional').astype(int)
        df['is_retail'] = (df['classification'] == 'Retail').astype(int)
        
        # Handle transaction type if present
        if 'transaction_type' in df.columns:
            df['is_buy'] = (df['transaction_type'] == 'BUY').astype(int)
            df['is_sell'] = (df['transaction_type'] == 'SELL').astype(int)
        else:
            # If no transaction type, assume 50/50 for now
            print("Warning: No transaction_type column found, assuming equal buy/sell")
            df['is_buy'] = 0
            df['is_sell'] = 0
        
        return df
    
    def _create_whale_features(self, whale_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated whale features for model training"""
        
        # Group by time_bucket and symbol
        grouped = whale_df.groupby(['symbol', 'time_bucket'])
        
        # Aggregate features
        agg_dict = {
            # Volume metrics
            'amount_usd': ['sum', 'mean', 'max', 'count', 'std'],
            'institutional_score': ['mean', 'max'],
            'market_cap_numeric': ['mean', 'max'],
            # Binary indicators
            'is_institutional': 'sum',
            'is_retail': 'sum',
            'is_buy': 'sum',
            'is_sell': 'sum'
        }
        
        print("Aggregating whale features...")
        whale_features = grouped.agg(agg_dict)
        
        # Flatten column names
        whale_features.columns = ['whale_' + '_'.join(col).strip() for col in whale_features.columns]
        whale_features = whale_features.reset_index()
        
        # Rename some columns for clarity
        rename_dict = {
            'whale_is_institutional_sum': 'inst_count',
            'whale_is_retail_sum': 'retail_count',
            'whale_is_buy_sum': 'whale_buy_count',
            'whale_is_sell_sum': 'whale_sell_count'
        }
        whale_features = whale_features.rename(columns=rename_dict)
        
        # Calculate directional volumes
        buy_volume = whale_df[whale_df['is_buy'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        sell_volume = whale_df[whale_df['is_sell'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
        whale_features = whale_features.set_index(['symbol', 'time_bucket'])
        whale_features['whale_buy_amount'] = buy_volume
        whale_features['whale_sell_amount'] = sell_volume
        whale_features = whale_features.fillna(0).reset_index()
        
        # Calculate derived features
        print("Calculating derived whale features...")
        
        # Flow imbalance
        whale_features['whale_flow_imbalance'] = (
            (whale_features['whale_buy_amount'] - whale_features['whale_sell_amount']) /
            (whale_features['whale_buy_amount'] + whale_features['whale_sell_amount'] + 1e-8)
        )
        
        # Institutional vs retail volumes
        inst_volume = whale_df[whale_df['is_institutional'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        retail_volume = whale_df[whale_df['is_retail'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
        whale_features = whale_features.set_index(['symbol', 'time_bucket'])
        whale_features['inst_amount_usd_sum'] = inst_volume
        whale_features['retail_amount_usd_sum'] = retail_volume
        whale_features = whale_features.fillna(0).reset_index()
        
        # Institutional participation rate
        whale_features['inst_participation_rate'] = (
            whale_features['inst_amount_usd_sum'] / 
            (whale_features['whale_amount_usd_sum'] + 1e-8)
        )
        
        # Retail selling pressure
        retail_sell = whale_df[(whale_df['is_retail'] == 1) & 
                              (whale_df['is_sell'] == 1)].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
        whale_features = whale_features.set_index(['symbol', 'time_bucket'])
        whale_features['retail_sell_volume'] = retail_sell
        whale_features = whale_features.fillna(0).reset_index()
        
        whale_features['retail_sell_pressure'] = (
            whale_features['retail_sell_volume'] / 
            (whale_features['whale_amount_usd_sum'] + 1e-8)
        )
        
        # Smart money divergence
        whale_features['smart_dumb_divergence'] = (
            whale_features['inst_participation_rate'] - 
            (1 - whale_features['inst_participation_rate'])
        )
        
        # Average trade size
        whale_features['avg_trade_size'] = (
            whale_features['whale_amount_usd_sum'] / 
            (whale_features['whale_amount_usd_count'] + 1e-8)
        )
        
        # Market cap stratified features
        whale_features['is_megacap'] = (whale_features['whale_market_cap_numeric_max'] >= 10_000_000_000).astype(int)
        whale_features['is_smallcap'] = (whale_features['whale_market_cap_numeric_max'] < 100_000_000).astype(int)
        
        whale_features['megacap_flow'] = (
            whale_features['is_megacap'] * whale_features['whale_flow_imbalance']
        )
        
        whale_features['smallcap_speculation'] = (
            whale_features['is_smallcap'] * whale_features['retail_amount_usd_sum']
        )
        
        # Clean up columns
        columns_to_keep = [
            'symbol', 'time_bucket',
            'whale_amount_usd_sum', 'whale_amount_usd_mean', 'whale_amount_usd_max', 
            'whale_amount_usd_count', 'whale_amount_usd_std',
            'whale_institutional_score_mean', 'whale_institutional_score_max',
            'whale_market_cap_numeric_mean', 'whale_market_cap_numeric_max',
            'inst_count', 'retail_count',
            'whale_buy_count', 'whale_sell_count',
            'whale_buy_amount', 'whale_sell_amount',
            'whale_flow_imbalance',
            'inst_amount_usd_sum', 'retail_amount_usd_sum',
            'inst_participation_rate', 'retail_sell_pressure',
            'smart_dumb_divergence', 'avg_trade_size',
            'megacap_flow', 'smallcap_speculation'
        ]
        
        # Keep only columns that exist
        columns_to_keep = [col for col in columns_to_keep if col in whale_features.columns]
        whale_features = whale_features[columns_to_keep]
        
        # Optimize memory
        whale_features = optimize_memory_usage(whale_features, verbose=False)
        
        print(f"Created {len(whale_features)} whale feature rows")
        print(f"Whale features shape: {whale_features.shape}")
        
        return whale_features


class WhaleDataLoader:
    """Specialized loader for whale transaction data - now supports both CSV and Parquet"""
    
    def __init__(self, whale_data_path: str):
        self.whale_data_path = whale_data_path
        self.parquet_loader = WhaleDataParquetLoader(whale_data_path)
        
    def load_and_aggregate_whale_data(self, symbols: Optional[List[str]] = None,
                                     start_date: Optional[pd.Timestamp] = None,
                                     end_date: Optional[pd.Timestamp] = None,
                                     use_cache: bool = True,
                                     prefer_parquet: bool = True) -> Optional[pd.DataFrame]:
        """Load whale transaction data and create aggregated features"""
        
        # Check if parquet files exist
        whale_data_parquet_path = os.path.join(self.whale_data_path, "whale_data")
        has_parquet = os.path.exists(whale_data_parquet_path) and len(glob.glob(os.path.join(whale_data_parquet_path, "*.parquet"))) > 0
        
        if has_parquet and prefer_parquet:
            print("Using parquet loader for whale data...")
            return self.parquet_loader.load_and_aggregate_whale_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache
            )
        
        # Fall back to CSV loader
        print("Using CSV loader for whale data...")
        
        # Try to load cached whale features first
        if use_cache:
            cached_whale_features = load_cache_data(CACHE_DIR, "whale_features_latest.parquet")
            if cached_whale_features is not None:
                user_input = input("\nUse cached whale features? (y/n): ").lower()
                if user_input == 'y':
                    print("Using cached whale features!")
                    return cached_whale_features
                else:
                    print("Recalculating whale features...")
        
        # Load whale CSV data
        whale_file = os.path.join(self.whale_data_path, "whale.csv")
        if not os.path.exists(whale_file):
            print(f"Whale data file not found: {whale_file}")
            return None
        
        print(f"Loading whale data from {whale_file}")
        whale_df = pd.read_csv(whale_file)
        print(f"Loaded {len(whale_df)} whale transactions")
        
        # Parse timestamp
        whale_df['timestamp'] = pd.to_datetime(whale_df['timestamp'])
        
        # Filter by date range if specified
        if start_date is not None:
            whale_df = whale_df[whale_df['timestamp'] >= start_date]
        if end_date is not None:
            whale_df = whale_df[whale_df['timestamp'] <= end_date]
        
        # Map tokens to symbols
        if symbols:
            # Create token filter based on symbol mapping
            allowed_tokens = []
            for symbol in symbols:
                if symbol in SYMBOL_MAPPING:
                    allowed_tokens.extend(SYMBOL_MAPPING[symbol])
            
            # Filter whale data
            pattern = '|'.join(allowed_tokens)
            whale_df = whale_df[whale_df['token'].str.upper().str.contains(pattern, na=False)]
        
        # Map token to standard symbol
        whale_df['symbol'] = whale_df['token'].str.upper().map(TOKEN_TO_SYMBOL)
        whale_df = whale_df.dropna(subset=['symbol'])
        
        # Convert symbol back to exchange format
        whale_df['symbol'] = whale_df['symbol'] + '/USDT'
        
        print(f"Filtered to {len(whale_df)} transactions for symbols: {whale_df['symbol'].unique()}")
        
        if len(whale_df) == 0:
            return None
        
        # Calculate institutional score if not present
        whale_df = self._classify_transactions(whale_df)
        
        # Create time bucket
        whale_df['time_bucket'] = whale_df['timestamp'].dt.floor('5min')
        
        # Aggregate features
        whale_features = self._create_whale_features(whale_df)
        
        # Save cached whale features
        if use_cache and len(whale_features) > 0:
            save_cache_data(whale_features, CACHE_DIR, "whale_features_latest.parquet")
        
        return whale_features
    
    def _classify_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify whale transactions as institutional vs retail"""
        # Use the same classification logic as parquet loader
        return self.parquet_loader._classify_transactions(df)
    
    def _create_whale_features(self, whale_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated whale features for model training"""
        # Use the same feature creation logic as parquet loader
        return self.parquet_loader._create_whale_features(whale_df)


class SocialDataLoader:
    """Loader for social mention data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_and_aggregate_social_data(self, symbols: Optional[List[str]] = None,
                                      start_date: Optional[pd.Timestamp] = None,
                                      end_date: Optional[pd.Timestamp] = None,
                                      use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load and aggregate social mention data"""
        
        # Try to load cached social features first
        if use_cache:
            cached_social_features = load_cache_data(CACHE_DIR, "social_features_latest.parquet")
            if cached_social_features is not None:
                user_input = input("\nUse cached social features? (y/n): ").lower()
                if user_input == 'y':
                    print("Using cached social features!")
                    return cached_social_features
                else:
                    print("Recalculating social features...")
        
        # Load mentions data
        mentions4h_file = os.path.join(self.data_path, "mentions4h.csv")
        mentions14d_file = os.path.join(self.data_path, "mentions14d.csv")
        
        social_features_list = []
        
        # Process 4H mentions data
        if os.path.exists(mentions4h_file):
            print(f"Loading 4H mentions data from {mentions4h_file}")
            mentions4h_df = pd.read_csv(mentions4h_file)
            mentions4h_df['date'] = pd.to_datetime(mentions4h_df['date'])
            
            # Filter by date range
            if start_date:
                mentions4h_df = mentions4h_df[mentions4h_df['date'] >= start_date]
            if end_date:
                mentions4h_df = mentions4h_df[mentions4h_df['date'] <= end_date]
            
            # Process each tracked token
            features_4h = self._process_mentions_4h(mentions4h_df, symbols)
            if features_4h is not None:
                social_features_list.append(features_4h)
        
        # Process 14D mentions data
        if os.path.exists(mentions14d_file):
            print(f"Loading 14D mentions data from {mentions14d_file}")
            mentions14d_df = pd.read_csv(mentions14d_file)
            mentions14d_df['timestamp'] = pd.to_datetime(mentions14d_df['timestamp'])
            
            # Filter by date range
            if start_date:
                mentions14d_df = mentions14d_df[mentions14d_df['timestamp'] >= start_date]
            if end_date:
                mentions14d_df = mentions14d_df[mentions14d_df['timestamp'] <= end_date]
            
            # Process 14D data
            features_14d = self._process_mentions_14d(mentions14d_df, symbols)
            if features_14d is not None:
                social_features_list.append(features_14d)
        
        if not social_features_list:
            return None
        
        # Merge all social features
        if len(social_features_list) == 1:
            social_features = social_features_list[0]
        else:
            social_features = social_features_list[0]
            for df in social_features_list[1:]:
                social_features = pd.merge(
                    social_features, df,
                    on=['symbol', 'time_bucket'],
                    how='outer'
                )
        
        # Create composite features
        social_features = self._create_composite_social_features(social_features)
        
        # Optimize memory
        social_features = optimize_memory_usage(social_features, verbose=False)
        
        # Save cached social features
        if use_cache and len(social_features) > 0:
            save_cache_data(social_features, CACHE_DIR, "social_features_latest.parquet")
        
        return social_features
    
    def _process_mentions_4h(self, mentions_df: pd.DataFrame, symbols: Optional[List[str]]) -> pd.DataFrame:
        """Process 4H mentions data"""
        
        # Token columns and their momentum columns
        token_mappings = {
            'ethereum': ('ETH/USDT', ['ethereum_4H', 'ethereum_7d', 'ethereum_1m']),
            'bitcoin': ('BTC/USDT', ['bitcoin_4H', 'bitcoin_7d', 'bitcoin_1m']),
            'BTC': ('BTC/USDT', ['BTC_4H', 'BTC_7d', 'BTC_1m']),
            'ETH': ('ETH/USDT', ['ETH_4H', 'ETH_7d', 'ETH_1m']),
            'Solana': ('SOL/USDT', ['Solana_4H', 'Solana_7d', 'Solana_1m'])
        }
        
        features_list = []
        
        for token, (symbol, momentum_cols) in token_mappings.items():
            if symbols and symbol not in symbols:
                continue
            
            # Check if token column exists
            if token not in mentions_df.columns:
                continue
            
            # Check which momentum columns exist
            existing_momentum_cols = [col for col in momentum_cols if col in mentions_df.columns]
            if not existing_momentum_cols:
                existing_momentum_cols = momentum_cols  # Keep original list for later handling
            
            # Create time bucket
            cols_to_select = ['date', token] + existing_momentum_cols
            cols_to_select = [col for col in cols_to_select if col in mentions_df.columns]
            
            token_df = mentions_df[cols_to_select].copy()
            token_df['time_bucket'] = token_df['date'].dt.floor('5min')
            token_df['symbol'] = symbol
            
            # Aggregate by time bucket
            agg_dict = {
                token: 'mean',  # Average mention count
            }
            
            # Add momentum columns that exist
            for i, col in enumerate(momentum_cols):
                if col in token_df.columns:
                    agg_dict[col] = 'mean'
            
            token_agg = token_df.groupby(['symbol', 'time_bucket']).agg(agg_dict).reset_index()
            
            # Rename columns
            rename_dict = {
                token: 'mention_count_4h',
            }
            
            # Handle momentum column renaming based on what exists
            momentum_names = ['mention_momentum_4h', 'mention_momentum_7d', 'mention_momentum_1m']
            for i, col in enumerate(momentum_cols):
                if col in token_agg.columns:
                    rename_dict[col] = momentum_names[i]
            
            token_agg = token_agg.rename(columns=rename_dict)
            
            features_list.append(token_agg)
        
        if not features_list:
            return None
        
        # Combine all tokens
        social_features = pd.concat(features_list, ignore_index=True)
        
        return social_features
    
    def _process_mentions_14d(self, mentions_df: pd.DataFrame, symbols: Optional[List[str]]) -> pd.DataFrame:
        """Process 14D mentions data"""
        
        # Map tickers to our symbols
        ticker_mapping = {
            'BTC': 'BTC/USDT',
            'ETH': 'ETH/USDT',
            'SOL': 'SOL/USDT'
        }
        
        # Filter relevant tickers
        relevant_tickers = [t for t, s in ticker_mapping.items() if not symbols or s in symbols]
        mentions_df = mentions_df[mentions_df['ticker'].isin(relevant_tickers)]
        
        if len(mentions_df) == 0:
            return None
        
        # Map to standard symbols
        mentions_df['symbol'] = mentions_df['ticker'].map(ticker_mapping)
        
        # Create time bucket
        mentions_df['time_bucket'] = mentions_df['timestamp'].dt.floor('5min')
        
        # Aggregate by symbol and time bucket
        agg_dict = {
            'mention_count': 'sum',
            'mention_change': 'mean',
            'sentiment': 'mean',
            'price_change_1d': 'mean'
        }
        
        social_agg = mentions_df.groupby(['symbol', 'time_bucket']).agg(agg_dict).reset_index()
        
        # Rename columns
        rename_dict = {
            'mention_count': 'mention_count_14d',
            'mention_change': 'mention_change_14d',
            'sentiment': 'sentiment_14d',
            'price_change_1d': 'social_price_change_1d'
        }
        social_agg = social_agg.rename(columns=rename_dict)
        
        return social_agg
    
    def _create_composite_social_features(self, social_df: pd.DataFrame) -> pd.DataFrame:
        """Create composite social features"""
        
        # Total mentions across timeframes
        mention_cols = [col for col in social_df.columns if col.startswith('mention_count_')]
        if mention_cols:
            social_df['total_mentions'] = social_df[mention_cols].sum(axis=1)
        
        # Social momentum score (weighted average of momentum indicators)
        momentum_cols = {
            'mention_momentum_4h': 0.5,
            'mention_momentum_7d': 0.3,
            'mention_momentum_1m': 0.2
        }
        
        social_df['social_momentum_score'] = 0
        for col, weight in momentum_cols.items():
            if col in social_df.columns:
                social_df['social_momentum_score'] += social_df[col].fillna(0) * weight
        
        # Sentiment-weighted mentions
        if 'sentiment_14d' in social_df.columns and 'total_mentions' in social_df.columns:
            social_df['sentiment_weighted_mentions'] = (
                social_df['total_mentions'] * (1 + social_df['sentiment_14d'].fillna(0))
            )
        
        # Sentiment categories
        if 'sentiment_14d' in social_df.columns:
            social_df['sentiment_very_positive'] = (
                social_df['sentiment_14d'] > SENTIMENT_THRESHOLDS['very_positive']
            ).astype(int)
            social_df['sentiment_positive'] = (
                social_df['sentiment_14d'] > SENTIMENT_THRESHOLDS['positive']
            ).astype(int)
            social_df['sentiment_negative'] = (
                social_df['sentiment_14d'] < SENTIMENT_THRESHOLDS['neutral']
            ).astype(int)
            social_df['sentiment_very_negative'] = (
                social_df['sentiment_14d'] < SENTIMENT_THRESHOLDS['negative']
            ).astype(int)
        
        # Fill missing values
        social_df = social_df.fillna(0)
        
        return social_df


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.whale_loader = WhaleDataLoader(WHALE_DATA_DIR)
        self.social_loader = SocialDataLoader(WHALE_DATA_DIR)
        self.trends_loader = TrendsDataLoader(os.path.join(os.path.dirname(WHALE_DATA_DIR), "trends_data"))
        
    def process_file(self, fpath: str, symbols: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Process a single file with proper type detection and error handling"""
        try:
            # Read the parquet file
            df = pd.read_parquet(fpath)
            
            # Check for required columns first
            if 'timestamp' not in df.columns:
                print(f"Warning: No 'timestamp' column in {os.path.basename(fpath)}, checking for alternatives...")
                
                # Try common timestamp column names
                timestamp_alternatives = ['time', 'datetime', 'date', 'ts', 'created_at', 'updated_at']
                timestamp_col = None
                
                for alt in timestamp_alternatives:
                    if alt in df.columns:
                        timestamp_col = alt
                        print(f"  Found alternative timestamp column: '{alt}'")
                        break
                
                if timestamp_col is None:
                    print(f"  No timestamp column found in {os.path.basename(fpath)}, skipping...")
                    return None, None
                else:
                    # Rename to standard 'timestamp' column
                    df = df.rename(columns={timestamp_col: 'timestamp'})
            
            # Ensure timestamp is numeric (unix timestamp in seconds)
            if df['timestamp'].dtype == 'object' or df['timestamp'].dtype.name.startswith('datetime'):
                try:
                    # If it's already datetime, convert to unix timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
                except:
                    print(f"Warning: Could not convert timestamp in {os.path.basename(fpath)}, skipping...")
                    return None, None
            
            # Memory optimization for large columns only
            large_float_cols = ['mid_price', 'spread', 'book_imbalance', 'bid_depth_10',
                               'ask_depth_10', 'price', 'quantity', 'cvd']
            
            for col in large_float_cols:
                if col in df.columns:
                    # Check memory usage safely
                    try:
                        if df[col].memory_usage(deep=True) > MEMORY_OPTIMIZATION_THRESHOLD:
                            df[col] = df[col].astype('float32')
                    except:
                        # If conversion fails, continue with original dtype
                        pass
            
            # Convert symbol to category if present
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype('category')
            else:
                print(f"Warning: No 'symbol' column in {os.path.basename(fpath)}")
            
            # Create datetime column
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            
            # Check if datetime conversion was successful
            if df['datetime'].isna().all():
                print(f"Warning: Could not convert timestamps to datetime in {os.path.basename(fpath)}, skipping...")
                return None, None
            
            # Filter symbols with proper regex handling
            if symbols and 'symbol' in df.columns:
                # Ensure symbols is a list
                if isinstance(symbols, str):
                    symbols = [symbols]
                
                # Create pattern with proper escaping
                pattern = '|'.join(map(re.escape, symbols))
                
                # Apply filter
                mask = df['symbol'].astype(str).str.contains(pattern, regex=True, na=False)
                df = df[mask]
                
                if len(df) == 0:
                    print(f"  No matching symbols found in {os.path.basename(fpath)}")
                    return None, None
            
            # Ensure we have data after filtering
            if len(df) == 0:
                return None, None
            
            # Improved file type detection
            filename = os.path.basename(fpath).lower()
            
            # First try filename-based detection
            if 'orderbook' in filename or 'order_book' in filename or 'ob' in filename:
                file_type = 'orderbook'
            elif 'trades' in filename or 'trade' in filename or 'transactions' in filename:
                file_type = 'trades'
            else:
                # Try to detect by columns
                orderbook_indicators = ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1', 
                                       'bid_price', 'ask_price', 'bids', 'asks']
                trades_indicators = ['side', 'cvd', 'trade_id', 'buyer', 'seller', 'taker', 'maker']
                
                # Check for orderbook columns
                if any(col in df.columns for col in orderbook_indicators):
                    file_type = 'orderbook'
                # Check for trades columns
                elif any(col in df.columns for col in trades_indicators):
                    file_type = 'trades'
                else:
                    # Last resort: check data patterns
                    if 'price' in df.columns and 'quantity' in df.columns:
                        # Could be either, but if we have many repeated timestamps, likely orderbook
                        timestamp_counts = df['timestamp'].value_counts()
                        if timestamp_counts.mean() > 2:  # Multiple entries per timestamp
                            file_type = 'orderbook'
                        else:
                            file_type = 'trades'
                    else:
                        print(f"Warning: Could not determine file type for {filename}")
                        print(f"  Available columns: {', '.join(df.columns[:10])}...")
                        return None, None
            
            # Final validation based on file type
            if file_type == 'orderbook':
                # Ensure we have at least basic orderbook columns
                required_cols = ['mid_price', 'spread', 'book_imbalance']
                if not any(col in df.columns for col in required_cols):
                    # Try to calculate mid_price if we have bid/ask
                    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
                        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
                        df['spread'] = df['ask_price_1'] - df['bid_price_1']
                    elif 'bid_price' in df.columns and 'ask_price' in df.columns:
                        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
                        df['spread'] = df['ask_price'] - df['bid_price']
            
            elif file_type == 'trades':
                # Ensure we have basic trades columns
                if 'price' not in df.columns:
                    print(f"Warning: No 'price' column in trades file {filename}, skipping...")
                    return None, None
                if 'quantity' not in df.columns and 'volume' in df.columns:
                    df['quantity'] = df['volume']
            
            print(f"  Processed {os.path.basename(fpath)}: {file_type} with {len(df)} rows")
            return df, file_type
            
        except Exception as e:
            print(f"Error processing file {os.path.basename(fpath)}: {type(e).__name__}: {e}")
            return None, None
    
    def load_and_aggregate_batched(self, symbols: Optional[List[str]] = None, 
                                   start_date: Optional[str] = None, 
                                   end_date: Optional[str] = None,
                                   max_files: Optional[int] = None, 
                                   batch_size: int = 50, 
                                   test_first_batch: bool = True, 
                                   use_cache: bool = True,
                                   include_whale_data: bool = True,
                                   include_social_data: bool = True,
                                   include_trends_data: bool = True) -> pd.DataFrame:
        """Enhanced data loading with batch processing, whale, social, and trends data integration"""
        
        # Try to load cached data first
        cache_suffix = ""
        if include_whale_data:
            cache_suffix += "_whale"
        if include_social_data:
            cache_suffix += "_social"
        if include_trends_data:
            cache_suffix += "_trends"
            
        if use_cache:
            cached_df = load_cache_data(CACHE_DIR, f"aggregated_data_latest{cache_suffix}.parquet")
            if cached_df is not None:
                user_input = input(f"\nUse cached aggregated data{cache_suffix}? (y/n): ").lower()
                if user_input == 'y':
                    print("Using cached data!")
                    return cached_df
                else:
                    print("Reprocessing data...")
        
        # Load orderbook and trades data
        all_files = sorted(glob.glob(os.path.join(self.data_path, "*.parquet")))
        
        if not all_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        print(f"Found {len(all_files)} parquet files")
        
        if max_files:
            all_files = all_files[:max_files]
        
        print(f"Will process {len(all_files)} files in batches of {batch_size}")
        
        # Process in batches
        n_batches = (len(all_files) + batch_size - 1) // batch_size
        
        final_ob_agg = []
        final_tr_agg = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_files))
            batch_files = all_files[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{n_batches} (files {start_idx}-{end_idx})...")
            
            orderbook_data = []
            trades_data = []
            
            # Process current batch
            for fpath in batch_files:
                df, file_type = self.process_file(fpath, symbols)
                
                if df is None or file_type is None:
                    continue
                
                if file_type == 'orderbook':
                    orderbook_data.append(df)
                elif file_type == 'trades':
                    trades_data.append(df)
                
                del df
                gc.collect()
            
            # Aggregate current batch for orderbook data
            if orderbook_data:
                print(f"  Aggregating {len(orderbook_data)} orderbook files...")
                df_ob = pd.concat(orderbook_data, ignore_index=True)
                df_ob['time_bucket'] = df_ob['datetime'].dt.floor(TIME_BUCKET)
                
                # Enhanced aggregation with multi-level orderbook data
                agg_dict = {
                    'mid_price': ['first', 'last', 'mean', 'std'],
                    'spread': ['mean', 'std', 'min', 'max'],
                    'book_imbalance': ['mean', 'std', 'min', 'max'],
                    'bid_depth_10': ['mean', 'std'],
                    'ask_depth_10': ['mean', 'std']
                }
                
                # Add multi-level aggregations if available
                for level in range(1, 6):
                    for side in ['bid', 'ask']:
                        price_col = f'{side}_price_{level}'
                        size_col = f'{side}_size_{level}'
                        if price_col in df_ob.columns:
                            agg_dict[price_col] = ['first', 'last', 'mean']
                        if size_col in df_ob.columns:
                            agg_dict[size_col] = ['mean', 'std', 'max']
                
                ob_agg = df_ob.groupby(['symbol', 'time_bucket'], observed=True).agg(agg_dict)
                ob_agg.columns = ['_'.join(col).strip() for col in ob_agg.columns]
                ob_agg = ob_agg.reset_index()
                
                ob_agg = optimize_memory_usage(ob_agg, verbose=False)
                final_ob_agg.append(ob_agg)
                
                del df_ob, orderbook_data, ob_agg
                gc.collect()
            
            # Aggregate current batch for trades data
            if trades_data:
                print(f"  Aggregating {len(trades_data)} trades files...")
                df_tr = pd.concat(trades_data, ignore_index=True)
                df_tr['time_bucket'] = df_tr['datetime'].dt.floor(TIME_BUCKET)
                
                # Calculate buy/sell volumes
                df_tr.loc[df_tr['side'] == 'buy', 'buy_volume'] = df_tr.loc[df_tr['side'] == 'buy', 'quantity']
                df_tr.loc[df_tr['side'] == 'sell', 'sell_volume'] = df_tr.loc[df_tr['side'] == 'sell', 'quantity']
                df_tr['buy_volume'] = df_tr['buy_volume'].fillna(0)
                df_tr['sell_volume'] = df_tr['sell_volume'].fillna(0)
                df_tr['buy_value'] = df_tr['buy_volume'] * df_tr['price']
                df_tr['sell_value'] = df_tr['sell_volume'] * df_tr['price']
                
                tr_agg = df_tr.groupby(['symbol', 'time_bucket'], observed=True).agg({
                    'price': ['first', 'last', 'min', 'max', 'mean', 'std'],
                    'quantity': ['sum', 'mean', 'std', 'count', 'max'],
                    'buy_volume': 'sum',
                    'sell_volume': 'sum',
                    'buy_value': 'sum',
                    'sell_value': 'sum',
                    'cvd': 'last'
                })
                tr_agg.columns = ['_'.join(col).strip() for col in tr_agg.columns]
                tr_agg = tr_agg.reset_index()
                
                # Calculate derived features
                tr_agg['order_flow_imbalance'] = (
                    (tr_agg['buy_volume_sum'] - tr_agg['sell_volume_sum']) /
                    (tr_agg['buy_volume_sum'] + tr_agg['sell_volume_sum'] + 1e-8)
                ).astype('float32')
                
                total_value = tr_agg['buy_value_sum'] + tr_agg['sell_value_sum']
                total_volume = tr_agg['buy_volume_sum'] + tr_agg['sell_volume_sum']
                tr_agg['vwap'] = (total_value / (total_volume + 1e-8)).astype('float32')
                
                tr_agg = optimize_memory_usage(tr_agg, verbose=False)
                final_tr_agg.append(tr_agg)
                
                del df_tr, trades_data, tr_agg
                gc.collect()
            
            # Early validation after first batch
            if test_first_batch and batch_idx == 0:
                print("\n" + "="*70)
                print("EARLY VALIDATION TEST - Checking first batch results")
                print("="*70)
                
                validation_passed = True
                
                if not final_ob_agg and not final_tr_agg:
                    print("❌ ERROR: No data collected in first batch!")
                    validation_passed = False
                else:
                    print("✓ Data collection successful")
                
                if final_ob_agg:
                    ob_test = final_ob_agg[0]
                    print(f"\nOrderbook data check:")
                    print(f"  - Shape: {ob_test.shape}")
                    print(f"  - Symbols: {ob_test['symbol'].unique()}")
                    print(f"  - Date range: {ob_test['time_bucket'].min()} to {ob_test['time_bucket'].max()}")
                
                if final_tr_agg:
                    tr_test = final_tr_agg[0]
                    print(f"\nTrades data check:")
                    print(f"  - Shape: {tr_test.shape}")
                    print(f"  - Symbols: {tr_test['symbol'].unique()}")
                    print(f"  - Date range: {tr_test['time_bucket'].min()} to {tr_test['time_bucket'].max()}")
                
                print("\n" + "="*70)
                
                if not validation_passed:
                    raise ValueError("Early validation failed!")
                else:
                    print("✅ All validation checks passed! Continuing...")
                    print("="*70 + "\n")
        
        # Combine all batches with proper aggregation
        print("\nCombining all batches...")
        
        if final_ob_agg:
            print("  Optimizing orderbook data before concatenation...")
            # Optimize memory BEFORE concatenation to avoid spike
            final_ob_agg = [optimize_memory_usage(df, verbose=False) for df in final_ob_agg]
            
            print("  Combining orderbook data...")
            all_ob = pd.concat(final_ob_agg, ignore_index=True)
            
            # Define proper aggregation for each column type
            ob_agg_dict = {}
            for col in all_ob.columns:
                if col in ['symbol', 'time_bucket']:
                    continue
                elif '_first' in col:
                    ob_agg_dict[col] = 'first'
                elif '_last' in col:
                    ob_agg_dict[col] = 'last'
                elif '_min' in col:
                    ob_agg_dict[col] = 'min'
                elif '_max' in col:
                    ob_agg_dict[col] = 'max'
                elif '_sum' in col:
                    ob_agg_dict[col] = 'sum'
                elif '_count' in col:
                    ob_agg_dict[col] = 'sum'
                else:  # means and stds
                    ob_agg_dict[col] = 'mean'
            
            all_ob = all_ob.groupby(['symbol', 'time_bucket'], observed=True).agg(ob_agg_dict).reset_index()
            del final_ob_agg
            gc.collect()
        else:
            all_ob = pd.DataFrame()
        
        if final_tr_agg:
            print("  Optimizing trades data before concatenation...")
            # Optimize memory BEFORE concatenation
            final_tr_agg = [optimize_memory_usage(df, verbose=False) for df in final_tr_agg]
            
            print("  Combining trades data...")
            all_tr = pd.concat(final_tr_agg, ignore_index=True)
            
            # Define proper aggregation for each column type
            tr_agg_dict = {}
            for col in all_tr.columns:
                if col in ['symbol', 'time_bucket']:
                    continue
                elif '_first' in col:
                    tr_agg_dict[col] = 'first'
                elif '_last' in col:
                    tr_agg_dict[col] = 'last'
                elif '_min' in col:
                    tr_agg_dict[col] = 'min'
                elif '_max' in col:
                    tr_agg_dict[col] = 'max'
                elif '_sum' in col:
                    tr_agg_dict[col] = 'sum'
                elif '_count' in col:
                    tr_agg_dict[col] = 'sum'
                elif col in ['order_flow_imbalance', 'vwap']:
                    tr_agg_dict[col] = 'mean'
                else:  # means and stds
                    tr_agg_dict[col] = 'mean'
            
            all_tr = all_tr.groupby(['symbol', 'time_bucket'], observed=True).agg(tr_agg_dict).reset_index()
            
            # Recalculate order flow imbalance after aggregation
            all_tr['order_flow_imbalance'] = (
                (all_tr['buy_volume_sum'] - all_tr['sell_volume_sum']) /
                (all_tr['buy_volume_sum'] + all_tr['sell_volume_sum'] + 1e-8)
            ).astype('float32')
            
            # Recalculate VWAP
            total_value = all_tr['buy_value_sum'] + all_tr['sell_value_sum']
            total_volume = all_tr['buy_volume_sum'] + all_tr['sell_volume_sum']
            all_tr['vwap'] = (total_value / (total_volume + 1e-8)).astype('float32')
            
            del final_tr_agg
            gc.collect()
        else:
            all_tr = pd.DataFrame()
        
        # Final merge
        print("  Merging orderbook and trades data...")
        if not all_ob.empty and not all_tr.empty:
            df_merged = pd.merge(all_ob, all_tr, on=['symbol', 'time_bucket'], how='outer')
        elif not all_ob.empty:
            df_merged = all_ob
        elif not all_tr.empty:
            df_merged = all_tr
        else:
            raise ValueError("No data to process")
        
        # Sort by symbol and time - CRITICAL for proper forward fill
        df_merged = df_merged.sort_values(['symbol', 'time_bucket'])
        
        # Load and integrate whale data if requested
        if include_whale_data:
            print("\nLoading and integrating whale data...")
            # Get date range from orderbook/trades data
            data_start = df_merged['time_bucket'].min()
            data_end = df_merged['time_bucket'].max()
            
            whale_features = self.whale_loader.load_and_aggregate_whale_data(
                symbols=symbols,
                start_date=data_start,
                end_date=data_end,
                use_cache=use_cache
            )
            
            if whale_features is not None and not whale_features.empty:
                print(f"Merging {len(whale_features)} whale feature rows...")
                df_merged = pd.merge(
                    df_merged,
                    whale_features,
                    on=['symbol', 'time_bucket'],
                    how='left'
                )
                
                # Fill missing whale features with 0
                whale_cols = [col for col in whale_features.columns if col not in ['symbol', 'time_bucket']]
                df_merged[whale_cols] = df_merged[whale_cols].fillna(0)
            else:
                print("No whale data available for the specified date range")
        
        # Load and integrate social data if requested
        if include_social_data:
            print("\nLoading and integrating social data...")
            social_features = self.social_loader.load_and_aggregate_social_data(
                symbols=symbols,
                start_date=df_merged['time_bucket'].min(),
                end_date=df_merged['time_bucket'].max(),
                use_cache=use_cache
            )
            
            if social_features is not None and not social_features.empty:
                print(f"Merging {len(social_features)} social feature rows...")
                df_merged = pd.merge(
                    df_merged,
                    social_features,
                    on=['symbol', 'time_bucket'],
                    how='left'
                )
                
                # Fill missing social features with 0
                social_cols = [col for col in social_features.columns if col not in ['symbol', 'time_bucket']]
                df_merged[social_cols] = df_merged[social_cols].fillna(0)
            else:
                print("No social data available for the specified date range")
        
        # Load and integrate trends data if requested
        if include_trends_data:
            print("\nLoading and integrating trends data...")
            trends_features = self.trends_loader.load_trends_data(
                symbols=symbols,
                start_date=df_merged['time_bucket'].min(),
                end_date=df_merged['time_bucket'].max(),
                use_cache=use_cache
            )
            
            if trends_features is not None and not trends_features.empty:
                print(f"Merging {len(trends_features)} trends feature rows...")
                df_merged = pd.merge(
                    df_merged,
                    trends_features,
                    on=['symbol', 'time_bucket'],
                    how='left'
                )
                
                # Fill missing trends features with 0
                trends_cols = [col for col in trends_features.columns if col not in ['symbol', 'time_bucket']]
                df_merged[trends_cols] = df_merged[trends_cols].fillna(0)
            else:
                print("No trends data available for the specified date range")
        
        # FIXED: Forward fill only (no backward fill to prevent data leakage)
        # Group by symbol and forward fill missing values
        numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
        df_merged[numeric_cols] = df_merged.groupby('symbol', observed=True)[numeric_cols].transform(lambda x: x.ffill())
        
        # Final memory optimization
        df_merged = optimize_memory_usage(df_merged)
        
        print(f"\nData summary:")
        print(f"→ Total 5-min bars: {len(df_merged)}")
        print(f"→ Symbols: {df_merged['symbol'].value_counts().to_dict()}")
        print(f"→ Date range: {df_merged['time_bucket'].min()} → {df_merged['time_bucket'].max()}")
        print(f"→ Columns: {len(df_merged.columns)}")
        print(f"→ Memory usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # List whale, social, and trends features if included
        if include_whale_data:
            whale_feature_cols = [col for col in df_merged.columns if 'whale_' in col or 'inst_' in col or 'retail_' in col]
            print(f"→ Whale features: {len(whale_feature_cols)}")
        
        if include_social_data:
            social_feature_cols = [col for col in df_merged.columns if 'mention' in col or 'sentiment' in col or 'social' in col]
            print(f"→ Social features: {len(social_feature_cols)}")
        
        if include_trends_data:
            trends_feature_cols = [col for col in df_merged.columns if 'trend_' in col]
            print(f"→ Trends features: {len(trends_feature_cols)}")
        
        # Save the aggregated data before returning (CACHE AMENDMENT)
        if use_cache:
            save_cache_data(df_merged, CACHE_DIR, f"aggregated_data_latest{cache_suffix}.parquet")
        
        return df_merged
    
    def load_from_sqlite(self, symbols: Optional[List[str]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Load integrated data from SQLite database"""
        print(f"Loading data from SQLite database: {SQLITE_DB_PATH}")
        
        # Build query
        query = "SELECT * FROM integrated_raw"
        conditions = []
        
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        
        if start_date:
            conditions.append(f"time_bucket >= '{start_date}'")
        
        if end_date:
            conditions.append(f"time_bucket <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY symbol, time_bucket"
        
        # Load data
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            df = pd.read_sql_query(query, conn)
        
        # Convert time_bucket to datetime
        df['time_bucket'] = pd.to_datetime(df['time_bucket'])
        
        # Convert symbol to category
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype('category')
        
        print(f"Loaded {len(df)} rows from SQLite")
        return df


# """
# Data loading and aggregation module with whale and social data support
# """
# import os
# import glob
# import re
# import gc
# import pandas as pd
# import numpy as np
# import sqlite3
# from typing import Dict, List, Tuple, Optional
# from tqdm import tqdm

# from config import *
# from utils import optimize_memory_usage, save_cache_data, load_cache_data

# """
# Enhanced Data Loader that handles parquet files in whale_data directory
# """
# import os
# import glob
# import re
# import gc
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Dict, List, Tuple, Optional, Union
# from tqdm import tqdm

# from config import *
# from utils import optimize_memory_usage, save_cache_data, load_cache_data


# class WhaleDataParquetLoader:
#     """Specialized loader for whale transaction data from parquet files"""
    
#     def __init__(self, whale_data_path: str = "/content/drive/MyDrive/crypto_pipeline_whale/data/whale_data"):
#         self.whale_data_path = whale_data_path
        
#     def find_whale_parquet_files(self, start_date: Optional[str] = None, 
#                                 end_date: Optional[str] = None) -> List[str]:
#         """Find all whale parquet files in the directory, optionally filtered by date"""
        
#         # Find all parquet files
#         parquet_pattern = os.path.join(self.whale_data_path, "*.parquet")
#         all_files = sorted(glob.glob(parquet_pattern))
        
#         if not all_files:
#             print(f"No parquet files found in {self.whale_data_path}")
#             return []
        
#         print(f"Found {len(all_files)} parquet files in whale_data directory")
        
#         # If no date filtering, return all files
#         if not start_date and not end_date:
#             return all_files
        
#         # Convert dates to datetime
#         start_dt = pd.to_datetime(start_date) if start_date else None
#         end_dt = pd.to_datetime(end_date) if end_date else None
        
#         # Try to extract dates from filenames
#         filtered_files = []
#         date_patterns = [
#             r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
#             r'(\d{8})',              # YYYYMMDD
#             r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
#         ]
        
#         for file_path in all_files:
#             filename = os.path.basename(file_path)
#             file_date = None
            
#             # Try each date pattern
#             for pattern in date_patterns:
#                 match = re.search(pattern, filename)
#                 if match:
#                     date_str = match.group(1).replace('_', '-')
#                     try:
#                         # Handle YYYYMMDD format
#                         if len(date_str) == 8 and '-' not in date_str:
#                             date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
#                         file_date = pd.to_datetime(date_str)
#                         break
#                     except:
#                         continue
            
#             # If we couldn't extract date from filename, include file by default
#             if file_date is None:
#                 print(f"  Warning: Could not extract date from {filename}, including by default")
#                 filtered_files.append(file_path)
#             else:
#                 # Check if file date is within range
#                 include = True
#                 if start_dt and file_date < start_dt:
#                     include = False
#                 if end_dt and file_date > end_dt + timedelta(days=1):
#                     include = False
                
#                 if include:
#                     filtered_files.append(file_path)
#                     print(f"  Including {filename} (date: {file_date.date()})")
        
#         print(f"Filtered to {len(filtered_files)} files within date range")
#         return filtered_files
    
#     def load_whale_parquet_files(self, file_paths: List[str]) -> pd.DataFrame:
#         """Load and combine multiple whale parquet files"""
        
#         if not file_paths:
#             return pd.DataFrame()
        
#         whale_dfs = []
        
#         for file_path in tqdm(file_paths, desc="Loading whale parquet files"):
#             try:
#                 df = pd.read_parquet(file_path)
                
#                 # Check for essential columns
#                 print(f"\n  File: {os.path.basename(file_path)}")
#                 print(f"  Shape: {df.shape}")
#                 print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
                
#                 # Standardize column names if needed
#                 df = self._standardize_whale_columns(df)
                
#                 whale_dfs.append(df)
                
#             except Exception as e:
#                 print(f"  Error loading {file_path}: {e}")
#                 continue
        
#         if not whale_dfs:
#             print("No whale data successfully loaded")
#             return pd.DataFrame()
        
#         # Combine all dataframes
#         print(f"\nCombining {len(whale_dfs)} whale dataframes...")
#         combined_df = pd.concat(whale_dfs, ignore_index=True)
        
#         # Remove duplicates if any
#         before_dedup = len(combined_df)
#         combined_df = combined_df.drop_duplicates()
#         after_dedup = len(combined_df)
#         if before_dedup > after_dedup:
#             print(f"Removed {before_dedup - after_dedup} duplicate rows")
        
#         print(f"Combined whale data shape: {combined_df.shape}")
        
#         return combined_df
    
#     def _standardize_whale_columns(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Standardize whale data column names across different formats"""
        
#         # Common column mappings
#         column_mappings = {
#             # Timestamp variations
#             'date': 'timestamp',
#             'time': 'timestamp',
#             'datetime': 'timestamp',
#             'created_at': 'timestamp',
#             'transaction_time': 'timestamp',
            
#             # Token variations
#             'coin': 'token',
#             'symbol': 'token',
#             'ticker': 'token',
#             'asset': 'token',
            
#             # Amount variations
#             'amount': 'amount_usd',
#             'value': 'amount_usd',
#             'usd_value': 'amount_usd',
#             'value_usd': 'amount_usd',
#             'transaction_value': 'amount_usd',
            
#             # Transaction type variations
#             'type': 'transaction_type',
#             'side': 'transaction_type',
#             'direction': 'transaction_type',
#             'tx_type': 'transaction_type',
            
#             # Market cap variations
#             'mcap': 'market_cap',
#             'marketcap': 'market_cap',
#             'market_capitalization': 'market_cap',
            
#             # Network variations
#             'chain': 'network',
#             'blockchain': 'network',
#         }
        
#         # Apply mappings
#         df = df.rename(columns=column_mappings)
        
#         # Ensure timestamp is datetime
#         if 'timestamp' in df.columns:
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
        
#         # Standardize token names
#         if 'token' in df.columns:
#             df['token'] = df['token'].str.upper()
        
#         # Standardize transaction types
#         if 'transaction_type' in df.columns:
#             df['transaction_type'] = df['transaction_type'].str.upper()
#             # Map common variations
#             type_mappings = {
#                 'BUY': 'BUY',
#                 'BOUGHT': 'BUY',
#                 'PURCHASE': 'BUY',
#                 'LONG': 'BUY',
#                 'SELL': 'SELL',
#                 'SOLD': 'SELL',
#                 'SHORT': 'SELL',
#             }
#             df['transaction_type'] = df['transaction_type'].map(
#                 lambda x: type_mappings.get(x, x) if pd.notna(x) else x
#             )
        
#         return df
    
#     def load_and_aggregate_whale_data(self, symbols: Optional[List[str]] = None,
#                                      start_date: Optional[Union[str, pd.Timestamp]] = None,
#                                      end_date: Optional[Union[str, pd.Timestamp]] = None,
#                                      use_cache: bool = True) -> Optional[pd.DataFrame]:
#         """Load whale transaction data from parquet files and create aggregated features"""
        
#         # Convert string dates to timestamps if needed
#         if isinstance(start_date, str):
#             start_date = pd.to_datetime(start_date)
#         if isinstance(end_date, str):
#             end_date = pd.to_datetime(end_date)
        
#         # Try to load cached whale features first
#         if use_cache:
#             cached_whale_features = load_cache_data(CACHE_DIR, "whale_features_latest.parquet")
#             if cached_whale_features is not None:
#                 user_input = input("\nUse cached whale features? (y/n): ").lower()
#                 if user_input == 'y':
#                     print("Using cached whale features!")
#                     return cached_whale_features
#                 else:
#                     print("Recalculating whale features...")
        
#         # Find whale parquet files
#         whale_files = self.find_whale_parquet_files(
#             start_date=start_date.strftime('%Y-%m-%d') if start_date else None,
#             end_date=end_date.strftime('%Y-%m-%d') if end_date else None
#         )
        
#         if not whale_files:
#             print("No whale parquet files found in the specified date range")
#             return None
        
#         # Load whale data from parquet files
#         whale_df = self.load_whale_parquet_files(whale_files)
        
#         if whale_df.empty:
#             print("No whale data loaded")
#             return None
        
#         print(f"\nLoaded {len(whale_df)} whale transactions")
        
#         # Additional date filtering on loaded data
#         if 'timestamp' in whale_df.columns:
#             if start_date is not None:
#                 whale_df = whale_df[whale_df['timestamp'] >= start_date]
#             if end_date is not None:
#                 whale_df = whale_df[whale_df['timestamp'] <= end_date]
            
#             print(f"After date filtering: {len(whale_df)} transactions")
        
#         # Filter by symbols if specified
#         if symbols and 'token' in whale_df.columns:
#             # Create token filter based on symbol mapping
#             allowed_tokens = []
#             for symbol in symbols:
#                 if symbol in SYMBOL_MAPPING:
#                     allowed_tokens.extend(SYMBOL_MAPPING[symbol])
            
#             # Filter whale data
#             pattern = '|'.join(allowed_tokens)
#             whale_df = whale_df[whale_df['token'].str.upper().str.contains(pattern, na=False)]
            
#             print(f"After symbol filtering: {len(whale_df)} transactions")
        
#         if len(whale_df) == 0:
#             print("No whale data after filtering")
#             return None
        
#         # Map token to standard symbol
#         whale_df['symbol'] = whale_df['token'].str.upper().map(TOKEN_TO_SYMBOL)
#         whale_df = whale_df.dropna(subset=['symbol'])
        
#         # Convert symbol back to exchange format
#         whale_df['symbol'] = whale_df['symbol'] + '/USDT'
        
#         print(f"Mapped to symbols: {whale_df['symbol'].unique()}")
        
#         # Classify transactions if not already done
#         if 'institutional_score' not in whale_df.columns:
#             whale_df = self._classify_transactions(whale_df)
        
#         # Create time bucket
#         whale_df['time_bucket'] = whale_df['timestamp'].dt.floor('5min')
        
#         # Aggregate features
#         whale_features = self._create_whale_features(whale_df)
        
#         # Save cached whale features
#         if use_cache and len(whale_features) > 0:
#             save_cache_data(whale_features, CACHE_DIR, "whale_features_latest.parquet")
        
#         return whale_features
    
#     def _classify_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Classify whale transactions as institutional vs retail"""
        
#         def classify_transaction(row):
#             """Classify a single transaction"""
#             # Parse market cap
#             market_cap = 0
#             if pd.notna(row.get('market_cap')):
#                 mc_str = str(row['market_cap']).replace('$', '').replace(',', '')
#                 if 'M' in mc_str:
#                     market_cap = float(mc_str.replace('M', '')) * 1_000_000
#                 elif 'B' in mc_str:
#                     market_cap = float(mc_str.replace('B', '')) * 1_000_000_000
#                 else:
#                     try:
#                         market_cap = float(mc_str)
#                     except:
#                         market_cap = 0
            
#             # Get amount
#             amount_usd = float(row.get('amount_usd', 0))
            
#             # Calculate institutional score
#             inst_score = 0
            
#             # Market cap scoring
#             for threshold, points in [(1_000_000_000, 4), (500_000_000, 3), 
#                                      (100_000_000, 2), (50_000_000, 1)]:
#                 if market_cap >= threshold:
#                     inst_score += points
#                     break
            
#             # Transaction amount scoring
#             for threshold, points in [(250_000, 4), (100_000, 3), 
#                                      (50_000, 2), (25_000, 1)]:
#                 if amount_usd >= threshold:
#                     inst_score += points
#                     break
            
#             # Network bonus
#             network = str(row.get('network', '')).lower()
#             if network in ['ethereum', 'bitcoin']:
#                 inst_score += 1
            
#             # Token type penalty (likely retail tokens)
#             token = str(row.get('token', '')).lower()
#             retail_indicators = ['doge', 'shib', 'pepe', 'meme', 'moon', 'safe', 
#                                'inu', 'floki', 'baby', 'mini', 'micro']
#             if any(indicator in token for indicator in retail_indicators):
#                 inst_score -= 2
            
#             # Classification
#             if inst_score >= 7:
#                 classification = 'Institutional'
#             elif inst_score >= 3:
#                 classification = 'Mixed'
#             else:
#                 classification = 'Retail'
            
#             return pd.Series({
#                 'classification': classification,
#                 'institutional_score': inst_score,
#                 'market_cap_numeric': market_cap
#             })
        
#         # Apply classification
#         print("Classifying whale transactions...")
#         classifications = df.apply(classify_transaction, axis=1)
        
#         # Add to dataframe
#         for col in classifications.columns:
#             df[col] = classifications[col]
        
#         # Create binary indicators
#         df['is_institutional'] = (df['classification'] == 'Institutional').astype(int)
#         df['is_retail'] = (df['classification'] == 'Retail').astype(int)
        
#         # Handle transaction type if present
#         if 'transaction_type' in df.columns:
#             df['is_buy'] = (df['transaction_type'] == 'BUY').astype(int)
#             df['is_sell'] = (df['transaction_type'] == 'SELL').astype(int)
#         else:
#             # If no transaction type, assume 50/50 for now
#             print("Warning: No transaction_type column found, assuming equal buy/sell")
#             df['is_buy'] = 0
#             df['is_sell'] = 0
        
#         return df
    
#     def _create_whale_features(self, whale_df: pd.DataFrame) -> pd.DataFrame:
#         """Create aggregated whale features for model training"""
        
#         # Group by time_bucket and symbol
#         grouped = whale_df.groupby(['symbol', 'time_bucket'])
        
#         # Aggregate features
#         agg_dict = {
#             # Volume metrics
#             'amount_usd': ['sum', 'mean', 'max', 'count', 'std'],
#             'institutional_score': ['mean', 'max'],
#             'market_cap_numeric': ['mean', 'max'],
#             # Binary indicators
#             'is_institutional': 'sum',
#             'is_retail': 'sum',
#             'is_buy': 'sum',
#             'is_sell': 'sum'
#         }
        
#         print("Aggregating whale features...")
#         whale_features = grouped.agg(agg_dict)
        
#         # Flatten column names
#         whale_features.columns = ['whale_' + '_'.join(col).strip() for col in whale_features.columns]
#         whale_features = whale_features.reset_index()
        
#         # Rename some columns for clarity
#         rename_dict = {
#             'whale_is_institutional_sum': 'inst_count',
#             'whale_is_retail_sum': 'retail_count',
#             'whale_is_buy_sum': 'whale_buy_count',
#             'whale_is_sell_sum': 'whale_sell_count'
#         }
#         whale_features = whale_features.rename(columns=rename_dict)
        
#         # Calculate directional volumes
#         buy_volume = whale_df[whale_df['is_buy'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
#         sell_volume = whale_df[whale_df['is_sell'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
#         whale_features = whale_features.set_index(['symbol', 'time_bucket'])
#         whale_features['whale_buy_amount'] = buy_volume
#         whale_features['whale_sell_amount'] = sell_volume
#         whale_features = whale_features.fillna(0).reset_index()
        
#         # Calculate derived features
#         print("Calculating derived whale features...")
        
#         # Flow imbalance
#         whale_features['whale_flow_imbalance'] = (
#             (whale_features['whale_buy_amount'] - whale_features['whale_sell_amount']) /
#             (whale_features['whale_buy_amount'] + whale_features['whale_sell_amount'] + 1e-8)
#         )
        
#         # Institutional vs retail volumes
#         inst_volume = whale_df[whale_df['is_institutional'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
#         retail_volume = whale_df[whale_df['is_retail'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
#         whale_features = whale_features.set_index(['symbol', 'time_bucket'])
#         whale_features['inst_amount_usd_sum'] = inst_volume
#         whale_features['retail_amount_usd_sum'] = retail_volume
#         whale_features = whale_features.fillna(0).reset_index()
        
#         # Institutional participation rate
#         whale_features['inst_participation_rate'] = (
#             whale_features['inst_amount_usd_sum'] / 
#             (whale_features['whale_amount_usd_sum'] + 1e-8)
#         )
        
#         # Retail selling pressure
#         retail_sell = whale_df[(whale_df['is_retail'] == 1) & 
#                               (whale_df['is_sell'] == 1)].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
#         whale_features = whale_features.set_index(['symbol', 'time_bucket'])
#         whale_features['retail_sell_volume'] = retail_sell
#         whale_features = whale_features.fillna(0).reset_index()
        
#         whale_features['retail_sell_pressure'] = (
#             whale_features['retail_sell_volume'] / 
#             (whale_features['whale_amount_usd_sum'] + 1e-8)
#         )
        
#         # Smart money divergence
#         whale_features['smart_dumb_divergence'] = (
#             whale_features['inst_participation_rate'] - 
#             (1 - whale_features['inst_participation_rate'])
#         )
        
#         # Average trade size
#         whale_features['avg_trade_size'] = (
#             whale_features['whale_amount_usd_sum'] / 
#             (whale_features['whale_amount_usd_count'] + 1e-8)
#         )
        
#         # Market cap stratified features
#         whale_features['is_megacap'] = (whale_features['whale_market_cap_numeric_max'] >= 10_000_000_000).astype(int)
#         whale_features['is_smallcap'] = (whale_features['whale_market_cap_numeric_max'] < 100_000_000).astype(int)
        
#         whale_features['megacap_flow'] = (
#             whale_features['is_megacap'] * whale_features['whale_flow_imbalance']
#         )
        
#         whale_features['smallcap_speculation'] = (
#             whale_features['is_smallcap'] * whale_features['retail_amount_usd_sum']
#         )
        
#         # Clean up columns
#         columns_to_keep = [
#             'symbol', 'time_bucket',
#             'whale_amount_usd_sum', 'whale_amount_usd_mean', 'whale_amount_usd_max', 
#             'whale_amount_usd_count', 'whale_amount_usd_std',
#             'whale_institutional_score_mean', 'whale_institutional_score_max',
#             'whale_market_cap_numeric_mean', 'whale_market_cap_numeric_max',
#             'inst_count', 'retail_count',
#             'whale_buy_count', 'whale_sell_count',
#             'whale_buy_amount', 'whale_sell_amount',
#             'whale_flow_imbalance',
#             'inst_amount_usd_sum', 'retail_amount_usd_sum',
#             'inst_participation_rate', 'retail_sell_pressure',
#             'smart_dumb_divergence', 'avg_trade_size',
#             'megacap_flow', 'smallcap_speculation'
#         ]
        
#         # Keep only columns that exist
#         columns_to_keep = [col for col in columns_to_keep if col in whale_features.columns]
#         whale_features = whale_features[columns_to_keep]
        
#         # Optimize memory
#         whale_features = optimize_memory_usage(whale_features, verbose=False)
        
#         print(f"Created {len(whale_features)} whale feature rows")
#         print(f"Whale features shape: {whale_features.shape}")
        
#         return whale_features



# class DataLoader:
#     def __init__(self, data_path: str):
#         self.data_path = data_path
#         self.whale_loader = WhaleDataLoader(WHALE_DATA_DIR)
#         self.social_loader = SocialDataLoader(WHALE_DATA_DIR)
        
#     def process_file(self, fpath: str, symbols: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
#         """Process a single file with proper type detection and error handling"""
#         try:
#             # Read the parquet file
#             df = pd.read_parquet(fpath)
            
#             # Check for required columns first
#             if 'timestamp' not in df.columns:
#                 print(f"Warning: No 'timestamp' column in {os.path.basename(fpath)}, checking for alternatives...")
                
#                 # Try common timestamp column names
#                 timestamp_alternatives = ['time', 'datetime', 'date', 'ts', 'created_at', 'updated_at']
#                 timestamp_col = None
                
#                 for alt in timestamp_alternatives:
#                     if alt in df.columns:
#                         timestamp_col = alt
#                         print(f"  Found alternative timestamp column: '{alt}'")
#                         break
                
#                 if timestamp_col is None:
#                     print(f"  No timestamp column found in {os.path.basename(fpath)}, skipping...")
#                     return None, None
#                 else:
#                     # Rename to standard 'timestamp' column
#                     df = df.rename(columns={timestamp_col: 'timestamp'})
            
#             # Ensure timestamp is numeric (unix timestamp in seconds)
#             if df['timestamp'].dtype == 'object' or df['timestamp'].dtype.name.startswith('datetime'):
#                 try:
#                     # If it's already datetime, convert to unix timestamp
#                     df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
#                 except:
#                     print(f"Warning: Could not convert timestamp in {os.path.basename(fpath)}, skipping...")
#                     return None, None
            
#             # Memory optimization for large columns only
#             large_float_cols = ['mid_price', 'spread', 'book_imbalance', 'bid_depth_10',
#                                'ask_depth_10', 'price', 'quantity', 'cvd']
            
#             for col in large_float_cols:
#                 if col in df.columns:
#                     # Check memory usage safely
#                     try:
#                         if df[col].memory_usage(deep=True) > MEMORY_OPTIMIZATION_THRESHOLD:
#                             df[col] = df[col].astype('float32')
#                     except:
#                         # If conversion fails, continue with original dtype
#                         pass
            
#             # Convert symbol to category if present
#             if 'symbol' in df.columns:
#                 df['symbol'] = df['symbol'].astype('category')
#             else:
#                 print(f"Warning: No 'symbol' column in {os.path.basename(fpath)}")
            
#             # Create datetime column
#             df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            
#             # Check if datetime conversion was successful
#             if df['datetime'].isna().all():
#                 print(f"Warning: Could not convert timestamps to datetime in {os.path.basename(fpath)}, skipping...")
#                 return None, None
            
#             # Filter symbols with proper regex handling
#             if symbols and 'symbol' in df.columns:
#                 # Ensure symbols is a list
#                 if isinstance(symbols, str):
#                     symbols = [symbols]
                
#                 # Create pattern with proper escaping
#                 pattern = '|'.join(map(re.escape, symbols))
                
#                 # Apply filter
#                 mask = df['symbol'].astype(str).str.contains(pattern, regex=True, na=False)
#                 df = df[mask]
                
#                 if len(df) == 0:
#                     print(f"  No matching symbols found in {os.path.basename(fpath)}")
#                     return None, None
            
#             # Ensure we have data after filtering
#             if len(df) == 0:
#                 return None, None
            
#             # Improved file type detection
#             filename = os.path.basename(fpath).lower()
            
#             # First try filename-based detection
#             if 'orderbook' in filename or 'order_book' in filename or 'ob' in filename:
#                 file_type = 'orderbook'
#             elif 'trades' in filename or 'trade' in filename or 'transactions' in filename:
#                 file_type = 'trades'
#             else:
#                 # Try to detect by columns
#                 orderbook_indicators = ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1', 
#                                        'bid_price', 'ask_price', 'bids', 'asks']
#                 trades_indicators = ['side', 'cvd', 'trade_id', 'buyer', 'seller', 'taker', 'maker']
                
#                 # Check for orderbook columns
#                 if any(col in df.columns for col in orderbook_indicators):
#                     file_type = 'orderbook'
#                 # Check for trades columns
#                 elif any(col in df.columns for col in trades_indicators):
#                     file_type = 'trades'
#                 else:
#                     # Last resort: check data patterns
#                     if 'price' in df.columns and 'quantity' in df.columns:
#                         # Could be either, but if we have many repeated timestamps, likely orderbook
#                         timestamp_counts = df['timestamp'].value_counts()
#                         if timestamp_counts.mean() > 2:  # Multiple entries per timestamp
#                             file_type = 'orderbook'
#                         else:
#                             file_type = 'trades'
#                     else:
#                         print(f"Warning: Could not determine file type for {filename}")
#                         print(f"  Available columns: {', '.join(df.columns[:10])}...")
#                         return None, None
            
#             # Final validation based on file type
#             if file_type == 'orderbook':
#                 # Ensure we have at least basic orderbook columns
#                 required_cols = ['mid_price', 'spread', 'book_imbalance']
#                 if not any(col in df.columns for col in required_cols):
#                     # Try to calculate mid_price if we have bid/ask
#                     if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
#                         df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
#                         df['spread'] = df['ask_price_1'] - df['bid_price_1']
#                     elif 'bid_price' in df.columns and 'ask_price' in df.columns:
#                         df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
#                         df['spread'] = df['ask_price'] - df['bid_price']
            
#             elif file_type == 'trades':
#                 # Ensure we have basic trades columns
#                 if 'price' not in df.columns:
#                     print(f"Warning: No 'price' column in trades file {filename}, skipping...")
#                     return None, None
#                 if 'quantity' not in df.columns and 'volume' in df.columns:
#                     df['quantity'] = df['volume']
            
#             print(f"  Processed {os.path.basename(fpath)}: {file_type} with {len(df)} rows")
#             return df, file_type
            
#         except Exception as e:
#             print(f"Error processing file {os.path.basename(fpath)}: {type(e).__name__}: {e}")
#             return None, None
    
#     def load_and_aggregate_batched(self, symbols: Optional[List[str]] = None, 
#                                    start_date: Optional[str] = None, 
#                                    end_date: Optional[str] = None,
#                                    max_files: Optional[int] = None, 
#                                    batch_size: int = 50, 
#                                    test_first_batch: bool = True, 
#                                    use_cache: bool = True,
#                                    include_whale_data: bool = True,
#                                    include_social_data: bool = True) -> pd.DataFrame:
#         """Enhanced data loading with batch processing, whale, and social data integration"""
        
#         # Try to load cached data first
#         cache_suffix = ""
#         if include_whale_data:
#             cache_suffix += "_whale"
#         if include_social_data:
#             cache_suffix += "_social"
            
#         if use_cache:
#             cached_df = load_cache_data(CACHE_DIR, f"aggregated_data_latest{cache_suffix}.parquet")
#             if cached_df is not None:
#                 user_input = input(f"\nUse cached aggregated data{cache_suffix}? (y/n): ").lower()
#                 if user_input == 'y':
#                     print("Using cached data!")
#                     return cached_df
#                 else:
#                     print("Reprocessing data...")
        
#         # Load orderbook and trades data
#         all_files = sorted(glob.glob(os.path.join(self.data_path, "*.parquet")))
        
#         if not all_files:
#             raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
#         print(f"Found {len(all_files)} parquet files")
        
#         if max_files:
#             all_files = all_files[:max_files]
        
#         print(f"Will process {len(all_files)} files in batches of {batch_size}")
        
#         # Process in batches
#         n_batches = (len(all_files) + batch_size - 1) // batch_size
        
#         final_ob_agg = []
#         final_tr_agg = []
        
#         for batch_idx in range(n_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min((batch_idx + 1) * batch_size, len(all_files))
#             batch_files = all_files[start_idx:end_idx]
            
#             print(f"\nProcessing batch {batch_idx + 1}/{n_batches} (files {start_idx}-{end_idx})...")
            
#             orderbook_data = []
#             trades_data = []
            
#             # Process current batch
#             for fpath in batch_files:
#                 df, file_type = self.process_file(fpath, symbols)
                
#                 if df is None or file_type is None:
#                     continue
                
#                 if file_type == 'orderbook':
#                     orderbook_data.append(df)
#                 elif file_type == 'trades':
#                     trades_data.append(df)
                
#                 del df
#                 gc.collect()
            
#             # Aggregate current batch for orderbook data
#             if orderbook_data:
#                 print(f"  Aggregating {len(orderbook_data)} orderbook files...")
#                 df_ob = pd.concat(orderbook_data, ignore_index=True)
#                 df_ob['time_bucket'] = df_ob['datetime'].dt.floor(TIME_BUCKET)
                
#                 # Enhanced aggregation with multi-level orderbook data
#                 agg_dict = {
#                     'mid_price': ['first', 'last', 'mean', 'std'],
#                     'spread': ['mean', 'std', 'min', 'max'],
#                     'book_imbalance': ['mean', 'std', 'min', 'max'],
#                     'bid_depth_10': ['mean', 'std'],
#                     'ask_depth_10': ['mean', 'std']
#                 }
                
#                 # Add multi-level aggregations if available
#                 for level in range(1, 6):
#                     for side in ['bid', 'ask']:
#                         price_col = f'{side}_price_{level}'
#                         size_col = f'{side}_size_{level}'
#                         if price_col in df_ob.columns:
#                             agg_dict[price_col] = ['first', 'last', 'mean']
#                         if size_col in df_ob.columns:
#                             agg_dict[size_col] = ['mean', 'std', 'max']
                
#                 ob_agg = df_ob.groupby(['symbol', 'time_bucket'], observed=True).agg(agg_dict)
#                 ob_agg.columns = ['_'.join(col).strip() for col in ob_agg.columns]
#                 ob_agg = ob_agg.reset_index()
                
#                 ob_agg = optimize_memory_usage(ob_agg, verbose=False)
#                 final_ob_agg.append(ob_agg)
                
#                 del df_ob, orderbook_data, ob_agg
#                 gc.collect()
            
#             # Aggregate current batch for trades data
#             if trades_data:
#                 print(f"  Aggregating {len(trades_data)} trades files...")
#                 df_tr = pd.concat(trades_data, ignore_index=True)
#                 df_tr['time_bucket'] = df_tr['datetime'].dt.floor(TIME_BUCKET)
                
#                 # Calculate buy/sell volumes
#                 df_tr.loc[df_tr['side'] == 'buy', 'buy_volume'] = df_tr.loc[df_tr['side'] == 'buy', 'quantity']
#                 df_tr.loc[df_tr['side'] == 'sell', 'sell_volume'] = df_tr.loc[df_tr['side'] == 'sell', 'quantity']
#                 df_tr['buy_volume'] = df_tr['buy_volume'].fillna(0)
#                 df_tr['sell_volume'] = df_tr['sell_volume'].fillna(0)
#                 df_tr['buy_value'] = df_tr['buy_volume'] * df_tr['price']
#                 df_tr['sell_value'] = df_tr['sell_volume'] * df_tr['price']
                
#                 tr_agg = df_tr.groupby(['symbol', 'time_bucket'], observed=True).agg({
#                     'price': ['first', 'last', 'min', 'max', 'mean', 'std'],
#                     'quantity': ['sum', 'mean', 'std', 'count', 'max'],
#                     'buy_volume': 'sum',
#                     'sell_volume': 'sum',
#                     'buy_value': 'sum',
#                     'sell_value': 'sum',
#                     'cvd': 'last'
#                 })
#                 tr_agg.columns = ['_'.join(col).strip() for col in tr_agg.columns]
#                 tr_agg = tr_agg.reset_index()
                
#                 # Calculate derived features
#                 tr_agg['order_flow_imbalance'] = (
#                     (tr_agg['buy_volume_sum'] - tr_agg['sell_volume_sum']) /
#                     (tr_agg['buy_volume_sum'] + tr_agg['sell_volume_sum'] + 1e-8)
#                 ).astype('float32')
                
#                 total_value = tr_agg['buy_value_sum'] + tr_agg['sell_value_sum']
#                 total_volume = tr_agg['buy_volume_sum'] + tr_agg['sell_volume_sum']
#                 tr_agg['vwap'] = (total_value / (total_volume + 1e-8)).astype('float32')
                
#                 tr_agg = optimize_memory_usage(tr_agg, verbose=False)
#                 final_tr_agg.append(tr_agg)
                
#                 del df_tr, trades_data, tr_agg
#                 gc.collect()
            
#             # Early validation after first batch
#             if test_first_batch and batch_idx == 0:
#                 print("\n" + "="*70)
#                 print("EARLY VALIDATION TEST - Checking first batch results")
#                 print("="*70)
                
#                 validation_passed = True
                
#                 if not final_ob_agg and not final_tr_agg:
#                     print("❌ ERROR: No data collected in first batch!")
#                     validation_passed = False
#                 else:
#                     print("✓ Data collection successful")
                
#                 if final_ob_agg:
#                     ob_test = final_ob_agg[0]
#                     print(f"\nOrderbook data check:")
#                     print(f"  - Shape: {ob_test.shape}")
#                     print(f"  - Symbols: {ob_test['symbol'].unique()}")
#                     print(f"  - Date range: {ob_test['time_bucket'].min()} to {ob_test['time_bucket'].max()}")
                
#                 if final_tr_agg:
#                     tr_test = final_tr_agg[0]
#                     print(f"\nTrades data check:")
#                     print(f"  - Shape: {tr_test.shape}")
#                     print(f"  - Symbols: {tr_test['symbol'].unique()}")
#                     print(f"  - Date range: {tr_test['time_bucket'].min()} to {tr_test['time_bucket'].max()}")
                
#                 print("\n" + "="*70)
                
#                 if not validation_passed:
#                     raise ValueError("Early validation failed!")
#                 else:
#                     print("✅ All validation checks passed! Continuing...")
#                     print("="*70 + "\n")
        
#         # Combine all batches with proper aggregation
#         print("\nCombining all batches...")
        
#         if final_ob_agg:
#             print("  Optimizing orderbook data before concatenation...")
#             # Optimize memory BEFORE concatenation to avoid spike
#             final_ob_agg = [optimize_memory_usage(df, verbose=False) for df in final_ob_agg]
            
#             print("  Combining orderbook data...")
#             all_ob = pd.concat(final_ob_agg, ignore_index=True)
            
#             # Define proper aggregation for each column type
#             ob_agg_dict = {}
#             for col in all_ob.columns:
#                 if col in ['symbol', 'time_bucket']:
#                     continue
#                 elif '_first' in col:
#                     ob_agg_dict[col] = 'first'
#                 elif '_last' in col:
#                     ob_agg_dict[col] = 'last'
#                 elif '_min' in col:
#                     ob_agg_dict[col] = 'min'
#                 elif '_max' in col:
#                     ob_agg_dict[col] = 'max'
#                 elif '_sum' in col:
#                     ob_agg_dict[col] = 'sum'
#                 elif '_count' in col:
#                     ob_agg_dict[col] = 'sum'
#                 else:  # means and stds
#                     ob_agg_dict[col] = 'mean'
            
#             all_ob = all_ob.groupby(['symbol', 'time_bucket'], observed=True).agg(ob_agg_dict).reset_index()
#             del final_ob_agg
#             gc.collect()
#         else:
#             all_ob = pd.DataFrame()
        
#         if final_tr_agg:
#             print("  Optimizing trades data before concatenation...")
#             # Optimize memory BEFORE concatenation
#             final_tr_agg = [optimize_memory_usage(df, verbose=False) for df in final_tr_agg]
            
#             print("  Combining trades data...")
#             all_tr = pd.concat(final_tr_agg, ignore_index=True)
            
#             # Define proper aggregation for each column type
#             tr_agg_dict = {}
#             for col in all_tr.columns:
#                 if col in ['symbol', 'time_bucket']:
#                     continue
#                 elif '_first' in col:
#                     tr_agg_dict[col] = 'first'
#                 elif '_last' in col:
#                     tr_agg_dict[col] = 'last'
#                 elif '_min' in col:
#                     tr_agg_dict[col] = 'min'
#                 elif '_max' in col:
#                     tr_agg_dict[col] = 'max'
#                 elif '_sum' in col:
#                     tr_agg_dict[col] = 'sum'
#                 elif '_count' in col:
#                     tr_agg_dict[col] = 'sum'
#                 elif col in ['order_flow_imbalance', 'vwap']:
#                     tr_agg_dict[col] = 'mean'
#                 else:  # means and stds
#                     tr_agg_dict[col] = 'mean'
            
#             all_tr = all_tr.groupby(['symbol', 'time_bucket'], observed=True).agg(tr_agg_dict).reset_index()
            
#             # Recalculate order flow imbalance after aggregation
#             all_tr['order_flow_imbalance'] = (
#                 (all_tr['buy_volume_sum'] - all_tr['sell_volume_sum']) /
#                 (all_tr['buy_volume_sum'] + all_tr['sell_volume_sum'] + 1e-8)
#             ).astype('float32')
            
#             # Recalculate VWAP
#             total_value = all_tr['buy_value_sum'] + all_tr['sell_value_sum']
#             total_volume = all_tr['buy_volume_sum'] + all_tr['sell_volume_sum']
#             all_tr['vwap'] = (total_value / (total_volume + 1e-8)).astype('float32')
            
#             del final_tr_agg
#             gc.collect()
#         else:
#             all_tr = pd.DataFrame()
        
#         # Final merge
#         print("  Merging orderbook and trades data...")
#         if not all_ob.empty and not all_tr.empty:
#             df_merged = pd.merge(all_ob, all_tr, on=['symbol', 'time_bucket'], how='outer')
#         elif not all_ob.empty:
#             df_merged = all_ob
#         elif not all_tr.empty:
#             df_merged = all_tr
#         else:
#             raise ValueError("No data to process")
        
#         # Sort by symbol and time - CRITICAL for proper forward fill
#         df_merged = df_merged.sort_values(['symbol', 'time_bucket'])
        
#         # Load and integrate whale data if requested
#         if include_whale_data:
#             print("\nLoading and integrating whale data...")
#             # Get date range from orderbook/trades data
#             data_start = df_merged['time_bucket'].min()
#             data_end = df_merged['time_bucket'].max()
            
#             whale_features = self.whale_loader.load_and_aggregate_whale_data(
#                 symbols=symbols,
#                 start_date=data_start,
#                 end_date=data_end,
#                 use_cache=use_cache
#             )
            
#             if whale_features is not None and not whale_features.empty:
#                 print(f"Merging {len(whale_features)} whale feature rows...")
#                 df_merged = pd.merge(
#                     df_merged,
#                     whale_features,
#                     on=['symbol', 'time_bucket'],
#                     how='left'
#                 )
                
#                 # Fill missing whale features with 0
#                 whale_cols = [col for col in whale_features.columns if col not in ['symbol', 'time_bucket']]
#                 df_merged[whale_cols] = df_merged[whale_cols].fillna(0)
#             else:
#                 print("No whale data available for the specified date range")
        
#         # Load and integrate social data if requested
#         if include_social_data:
#             print("\nLoading and integrating social data...")
#             social_features = self.social_loader.load_and_aggregate_social_data(
#                 symbols=symbols,
#                 start_date=df_merged['time_bucket'].min(),
#                 end_date=df_merged['time_bucket'].max(),
#                 use_cache=use_cache
#             )
            
#             if social_features is not None and not social_features.empty:
#                 print(f"Merging {len(social_features)} social feature rows...")
#                 df_merged = pd.merge(
#                     df_merged,
#                     social_features,
#                     on=['symbol', 'time_bucket'],
#                     how='left'
#                 )
                
#                 # Fill missing social features with 0
#                 social_cols = [col for col in social_features.columns if col not in ['symbol', 'time_bucket']]
#                 df_merged[social_cols] = df_merged[social_cols].fillna(0)
#             else:
#                 print("No social data available for the specified date range")
        
#         # FIXED: Forward fill only (no backward fill to prevent data leakage)
#         # Group by symbol and forward fill missing values
#         numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
#         df_merged[numeric_cols] = df_merged.groupby('symbol', observed=True)[numeric_cols].transform(lambda x: x.ffill())
        
#         # Final memory optimization
#         df_merged = optimize_memory_usage(df_merged)
        
#         print(f"\nData summary:")
#         print(f"→ Total 5-min bars: {len(df_merged)}")
#         print(f"→ Symbols: {df_merged['symbol'].value_counts().to_dict()}")
#         print(f"→ Date range: {df_merged['time_bucket'].min()} → {df_merged['time_bucket'].max()}")
#         print(f"→ Columns: {len(df_merged.columns)}")
#         print(f"→ Memory usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
#         # List whale and social features if included
#         if include_whale_data:
#             whale_feature_cols = [col for col in df_merged.columns if 'whale_' in col or 'inst_' in col or 'retail_' in col]
#             print(f"→ Whale features: {len(whale_feature_cols)}")
        
#         if include_social_data:
#             social_feature_cols = [col for col in df_merged.columns if 'mention' in col or 'sentiment' in col or 'social' in col]
#             print(f"→ Social features: {len(social_feature_cols)}")
        
#         # Save the aggregated data before returning (CACHE AMENDMENT)
#         if use_cache:
#             save_cache_data(df_merged, CACHE_DIR, f"aggregated_data_latest{cache_suffix}.parquet")
        
#         return df_merged
    
#     def load_from_sqlite(self, symbols: Optional[List[str]] = None,
#                         start_date: Optional[str] = None,
#                         end_date: Optional[str] = None) -> pd.DataFrame:
#         """Load integrated data from SQLite database"""
#         print(f"Loading data from SQLite database: {SQLITE_DB_PATH}")
        
#         # Build query
#         query = "SELECT * FROM integrated_raw"
#         conditions = []
        
#         if symbols:
#             symbols_str = "', '".join(symbols)
#             conditions.append(f"symbol IN ('{symbols_str}')")
        
#         if start_date:
#             conditions.append(f"time_bucket >= '{start_date}'")
        
#         if end_date:
#             conditions.append(f"time_bucket <= '{end_date}'")
        
#         if conditions:
#             query += " WHERE " + " AND ".join(conditions)
        
#         query += " ORDER BY symbol, time_bucket"
        
#         # Load data
#         with sqlite3.connect(SQLITE_DB_PATH) as conn:
#             df = pd.read_sql_query(query, conn)
        
#         # Convert time_bucket to datetime
#         df['time_bucket'] = pd.to_datetime(df['time_bucket'])
        
#         # Convert symbol to category
#         if 'symbol' in df.columns:
#             df['symbol'] = df['symbol'].astype('category')
        
#         print(f"Loaded {len(df)} rows from SQLite")
#         return df


# class WhaleDataLoader:
#     """Specialized loader for whale transaction data"""
    
#     def __init__(self, whale_data_path: str):
#         self.whale_data_path = whale_data_path
        
#     def load_and_aggregate_whale_data(self, symbols: Optional[List[str]] = None,
#                                      start_date: Optional[pd.Timestamp] = None,
#                                      end_date: Optional[pd.Timestamp] = None,
#                                      use_cache: bool = True) -> Optional[pd.DataFrame]:
#         """Load whale transaction data and create aggregated features"""
        
#         # Try to load cached whale features first
#         if use_cache:
#             cached_whale_features = load_cache_data(CACHE_DIR, "whale_features_latest.parquet")
#             if cached_whale_features is not None:
#                 user_input = input("\nUse cached whale features? (y/n): ").lower()
#                 if user_input == 'y':
#                     print("Using cached whale features!")
#                     return cached_whale_features
#                 else:
#                     print("Recalculating whale features...")
        
#         # Load whale CSV data
#         whale_file = os.path.join(self.whale_data_path, "whale.csv")
#         if not os.path.exists(whale_file):
#             print(f"Whale data file not found: {whale_file}")
#             return None
        
#         print(f"Loading whale data from {whale_file}")
#         whale_df = pd.read_csv(whale_file)
#         print(f"Loaded {len(whale_df)} whale transactions")
        
#         # Parse timestamp
#         whale_df['timestamp'] = pd.to_datetime(whale_df['timestamp'])
        
#         # Filter by date range if specified
#         if start_date is not None:
#             whale_df = whale_df[whale_df['timestamp'] >= start_date]
#         if end_date is not None:
#             whale_df = whale_df[whale_df['timestamp'] <= end_date]
        
#         # Map tokens to symbols
#         if symbols:
#             # Create token filter based on symbol mapping
#             allowed_tokens = []
#             for symbol in symbols:
#                 if symbol in SYMBOL_MAPPING:
#                     allowed_tokens.extend(SYMBOL_MAPPING[symbol])
            
#             # Filter whale data
#             pattern = '|'.join(allowed_tokens)
#             whale_df = whale_df[whale_df['token'].str.upper().str.contains(pattern, na=False)]
        
#         # Map token to standard symbol
#         whale_df['symbol'] = whale_df['token'].str.upper().map(TOKEN_TO_SYMBOL)
#         whale_df = whale_df.dropna(subset=['symbol'])
        
#         # Convert symbol back to exchange format
#         whale_df['symbol'] = whale_df['symbol'] + '/USDT'
        
#         print(f"Filtered to {len(whale_df)} transactions for symbols: {whale_df['symbol'].unique()}")
        
#         if len(whale_df) == 0:
#             return None
        
#         # Calculate institutional score if not present
#         whale_df = self._classify_transactions(whale_df)
        
#         # Create time bucket
#         whale_df['time_bucket'] = whale_df['timestamp'].dt.floor('5min')
        
#         # Aggregate features
#         whale_features = self._create_whale_features(whale_df)
        
#         # Save cached whale features
#         if use_cache and len(whale_features) > 0:
#             save_cache_data(whale_features, CACHE_DIR, "whale_features_latest.parquet")
        
#         return whale_features
    
#     def _classify_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Classify whale transactions as institutional vs retail"""
        
#         def classify_transaction(row):
#             """Classify a single transaction"""
#             # Parse market cap
#             market_cap = 0
#             if pd.notna(row.get('market_cap')):
#                 mc_str = str(row['market_cap']).replace('$', '').replace(',', '')
#                 if 'M' in mc_str:
#                     market_cap = float(mc_str.replace('M', '')) * 1_000_000
#                 elif 'B' in mc_str:
#                     market_cap = float(mc_str.replace('B', '')) * 1_000_000_000
#                 else:
#                     try:
#                         market_cap = float(mc_str)
#                     except:
#                         market_cap = 0
            
#             # Get amount
#             amount_usd = float(row.get('amount_usd', 0))
            
#             # Calculate institutional score
#             inst_score = 0
            
#             # Market cap scoring
#             for threshold, points in [(1_000_000_000, 4), (500_000_000, 3), 
#                                      (100_000_000, 2), (50_000_000, 1)]:
#                 if market_cap >= threshold:
#                     inst_score += points
#                     break
            
#             # Transaction amount scoring
#             for threshold, points in [(250_000, 4), (100_000, 3), 
#                                      (50_000, 2), (25_000, 1)]:
#                 if amount_usd >= threshold:
#                     inst_score += points
#                     break
            
#             # Network bonus
#             network = str(row.get('network', '')).lower()
#             if network in ['ethereum', 'bitcoin']:
#                 inst_score += 1
            
#             # Token type penalty (likely retail tokens)
#             token = str(row.get('token', '')).lower()
#             retail_indicators = ['doge', 'shib', 'pepe', 'meme', 'moon', 'safe', 
#                                'inu', 'floki', 'baby', 'mini', 'micro']
#             if any(indicator in token for indicator in retail_indicators):
#                 inst_score -= 2
            
#             # Classification
#             if inst_score >= 7:
#                 classification = 'Institutional'
#             elif inst_score >= 3:
#                 classification = 'Mixed'
#             else:
#                 classification = 'Retail'
            
#             return pd.Series({
#                 'classification': classification,
#                 'institutional_score': inst_score,
#                 'market_cap_numeric': market_cap
#             })
        
#         # Apply classification
#         classifications = df.apply(classify_transaction, axis=1)
        
#         # Add to dataframe
#         for col in classifications.columns:
#             df[col] = classifications[col]
        
#         # Create binary indicators
#         df['is_institutional'] = (df['classification'] == 'Institutional').astype(int)
#         df['is_retail'] = (df['classification'] == 'Retail').astype(int)
#         df['is_buy'] = (df['transaction_type'] == 'BUY').astype(int)
#         df['is_sell'] = (df['transaction_type'] == 'SELL').astype(int)
        
#         return df
    
#     def _create_whale_features(self, whale_df: pd.DataFrame) -> pd.DataFrame:
#         """Create aggregated whale features for model training"""
        
#         # Group by time_bucket and symbol
#         grouped = whale_df.groupby(['symbol', 'time_bucket'])
        
#         # Aggregate features
#         agg_dict = {
#             # Volume metrics
#             'amount_usd': ['sum', 'mean', 'max', 'count', 'std'],
#             'institutional_score': ['mean', 'max'],
#             'market_cap_numeric': ['mean', 'max'],
#             # Binary indicators
#             'is_institutional': 'sum',
#             'is_retail': 'sum',
#             'is_buy': 'sum',
#             'is_sell': 'sum'
#         }
        
#         whale_features = grouped.agg(agg_dict)
        
#         # Flatten column names
#         whale_features.columns = ['whale_' + '_'.join(col).strip() for col in whale_features.columns]
#         whale_features = whale_features.reset_index()
        
#         # Rename some columns for clarity
#         rename_dict = {
#             'whale_is_institutional_sum': 'inst_count',
#             'whale_is_retail_sum': 'retail_count',
#             'whale_is_buy_sum': 'whale_buy_count',
#             'whale_is_sell_sum': 'whale_sell_count'
#         }
#         whale_features = whale_features.rename(columns=rename_dict)
        
#         # Calculate directional volumes
#         buy_volume = whale_df[whale_df['is_buy'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
#         sell_volume = whale_df[whale_df['is_sell'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
#         whale_features = whale_features.set_index(['symbol', 'time_bucket'])
#         whale_features['whale_buy_amount'] = buy_volume
#         whale_features['whale_sell_amount'] = sell_volume
#         whale_features = whale_features.fillna(0).reset_index()
        
#         # Calculate derived features
#         # Flow imbalance
#         whale_features['whale_flow_imbalance'] = (
#             (whale_features['whale_buy_amount'] - whale_features['whale_sell_amount']) /
#             (whale_features['whale_buy_amount'] + whale_features['whale_sell_amount'] + 1e-8)
#         )
        
#         # Institutional vs retail volumes
#         inst_volume = whale_df[whale_df['is_institutional'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
#         retail_volume = whale_df[whale_df['is_retail'] == 1].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
#         whale_features = whale_features.set_index(['symbol', 'time_bucket'])
#         whale_features['inst_amount_usd_sum'] = inst_volume
#         whale_features['retail_amount_usd_sum'] = retail_volume
#         whale_features = whale_features.fillna(0).reset_index()
        
#         # Institutional participation rate
#         whale_features['inst_participation_rate'] = (
#             whale_features['inst_amount_usd_sum'] / 
#             (whale_features['whale_amount_usd_sum'] + 1e-8)
#         )
        
#         # Retail selling pressure
#         retail_sell = whale_df[(whale_df['is_retail'] == 1) & 
#                               (whale_df['is_sell'] == 1)].groupby(['symbol', 'time_bucket'])['amount_usd'].sum()
        
#         whale_features = whale_features.set_index(['symbol', 'time_bucket'])
#         whale_features['retail_sell_volume'] = retail_sell
#         whale_features = whale_features.fillna(0).reset_index()
        
#         whale_features['retail_sell_pressure'] = (
#             whale_features['retail_sell_volume'] / 
#             (whale_features['whale_amount_usd_sum'] + 1e-8)
#         )
        
#         # Smart money divergence
#         whale_features['smart_dumb_divergence'] = (
#             whale_features['inst_participation_rate'] - 
#             (1 - whale_features['inst_participation_rate'])
#         )
        
#         # Average trade size
#         whale_features['avg_trade_size'] = (
#             whale_features['whale_amount_usd_sum'] / 
#             (whale_features['whale_amount_usd_count'] + 1e-8)
#         )
        
#         # Market cap stratified features
#         whale_features['is_megacap'] = (whale_features['whale_market_cap_numeric_max'] >= 10_000_000_000).astype(int)
#         whale_features['is_smallcap'] = (whale_features['whale_market_cap_numeric_max'] < 100_000_000).astype(int)
        
#         whale_features['megacap_flow'] = (
#             whale_features['is_megacap'] * whale_features['whale_flow_imbalance']
#         )
        
#         whale_features['smallcap_speculation'] = (
#             whale_features['is_smallcap'] * whale_features['retail_amount_usd_sum']
#         )
        
#         # Clean up columns
#         columns_to_keep = [
#             'symbol', 'time_bucket',
#             'whale_amount_usd_sum', 'whale_amount_usd_mean', 'whale_amount_usd_max', 
#             'whale_amount_usd_count', 'whale_amount_usd_std',
#             'whale_institutional_score_mean', 'whale_institutional_score_max',
#             'whale_market_cap_numeric_mean', 'whale_market_cap_numeric_max',
#             'inst_count', 'retail_count',
#             'whale_buy_count', 'whale_sell_count',
#             'whale_buy_amount', 'whale_sell_amount',
#             'whale_flow_imbalance',
#             'inst_amount_usd_sum', 'retail_amount_usd_sum',
#             'inst_participation_rate', 'retail_sell_pressure',
#             'smart_dumb_divergence', 'avg_trade_size',
#             'megacap_flow', 'smallcap_speculation'
#         ]
        
#         whale_features = whale_features[columns_to_keep]
        
#         # Optimize memory
#         whale_features = optimize_memory_usage(whale_features, verbose=False)
        
#         return whale_features


# class SocialDataLoader:
#     """Loader for social mention data"""
    
#     def __init__(self, data_path: str):
#         self.data_path = data_path
        
#     def load_and_aggregate_social_data(self, symbols: Optional[List[str]] = None,
#                                       start_date: Optional[pd.Timestamp] = None,
#                                       end_date: Optional[pd.Timestamp] = None,
#                                       use_cache: bool = True) -> Optional[pd.DataFrame]:
#         """Load and aggregate social mention data"""
        
#         # Try to load cached social features first
#         if use_cache:
#             cached_social_features = load_cache_data(CACHE_DIR, "social_features_latest.parquet")
#             if cached_social_features is not None:
#                 user_input = input("\nUse cached social features? (y/n): ").lower()
#                 if user_input == 'y':
#                     print("Using cached social features!")
#                     return cached_social_features
#                 else:
#                     print("Recalculating social features...")
        
#         # Load mentions data
#         mentions4h_file = os.path.join(self.data_path, "mentions4h.csv")
#         mentions14d_file = os.path.join(self.data_path, "mentions14d.csv")
        
#         social_features_list = []
        
#         # Process 4H mentions data
#         if os.path.exists(mentions4h_file):
#             print(f"Loading 4H mentions data from {mentions4h_file}")
#             mentions4h_df = pd.read_csv(mentions4h_file)
#             mentions4h_df['date'] = pd.to_datetime(mentions4h_df['date'])
            
#             # Filter by date range
#             if start_date:
#                 mentions4h_df = mentions4h_df[mentions4h_df['date'] >= start_date]
#             if end_date:
#                 mentions4h_df = mentions4h_df[mentions4h_df['date'] <= end_date]
            
#             # Process each tracked token
#             features_4h = self._process_mentions_4h(mentions4h_df, symbols)
#             if features_4h is not None:
#                 social_features_list.append(features_4h)
        
#         # Process 14D mentions data
#         if os.path.exists(mentions14d_file):
#             print(f"Loading 14D mentions data from {mentions14d_file}")
#             mentions14d_df = pd.read_csv(mentions14d_file)
#             mentions14d_df['timestamp'] = pd.to_datetime(mentions14d_df['timestamp'])
            
#             # Filter by date range
#             if start_date:
#                 mentions14d_df = mentions14d_df[mentions14d_df['timestamp'] >= start_date]
#             if end_date:
#                 mentions14d_df = mentions14d_df[mentions14d_df['timestamp'] <= end_date]
            
#             # Process 14D data
#             features_14d = self._process_mentions_14d(mentions14d_df, symbols)
#             if features_14d is not None:
#                 social_features_list.append(features_14d)
        
#         if not social_features_list:
#             return None
        
#         # Merge all social features
#         if len(social_features_list) == 1:
#             social_features = social_features_list[0]
#         else:
#             social_features = social_features_list[0]
#             for df in social_features_list[1:]:
#                 social_features = pd.merge(
#                     social_features, df,
#                     on=['symbol', 'time_bucket'],
#                     how='outer'
#                 )
        
#         # Create composite features
#         social_features = self._create_composite_social_features(social_features)
        
#         # Optimize memory
#         social_features = optimize_memory_usage(social_features, verbose=False)
        
#         # Save cached social features
#         if use_cache and len(social_features) > 0:
#             save_cache_data(social_features, CACHE_DIR, "social_features_latest.parquet")
        
#         return social_features
    
#     def _process_mentions_4h(self, mentions_df: pd.DataFrame, symbols: Optional[List[str]]) -> pd.DataFrame:
#         """Process 4H mentions data"""
        
#         # Token columns and their momentum columns
#         token_mappings = {
#             'ethereum': ('ETH/USDT', ['ethereum_4H', 'ethereum_7d', 'ethereum_1m']),
#             'bitcoin': ('BTC/USDT', ['bitcoin_4H', 'bitcoin_7d', 'bitcoin_1m']),
#             'BTC': ('BTC/USDT', ['BTC_4H', 'BTC_7d', 'BTC_1m']),
#             'ETH': ('ETH/USDT', ['ETH_4H', 'ETH_7d', 'ETH_1m']),
#             'Solana': ('SOL/USDT', ['Solana_4H', 'Solana_7d', 'Solana_1m'])
#         }
        
#         features_list = []
        
#         for token, (symbol, momentum_cols) in token_mappings.items():
#             if symbols and symbol not in symbols:
#                 continue
            
#             # Create time bucket
#             token_df = mentions_df[['date', token] + momentum_cols].copy()
#             token_df['time_bucket'] = token_df['date'].dt.floor('5min')
#             token_df['symbol'] = symbol
            
#             # Aggregate by time bucket
#             agg_dict = {
#                 token: 'mean',  # Average mention count
#             }
#             for col in momentum_cols:
#                 agg_dict[col] = 'mean'
            
#             token_agg = token_df.groupby(['symbol', 'time_bucket']).agg(agg_dict).reset_index()
            
#             # Rename columns
#             rename_dict = {
#                 token: 'mention_count_4h',
#                 momentum_cols[0]: 'mention_momentum_4h',
#                 momentum_cols[1]: 'mention_momentum_7d',
#                 momentum_cols[2]: 'mention_momentum_1m'
#             }
#             token_agg = token_agg.rename(columns=rename_dict)
            
#             features_list.append(token_agg)
        
#         if not features_list:
#             return None
        
#         # Combine all tokens
#         social_features = pd.concat(features_list, ignore_index=True)
        
#         return social_features
    
#     def _process_mentions_14d(self, mentions_df: pd.DataFrame, symbols: Optional[List[str]]) -> pd.DataFrame:
#         """Process 14D mentions data"""
        
#         # Map tickers to our symbols
#         ticker_mapping = {
#             'BTC': 'BTC/USDT',
#             'ETH': 'ETH/USDT',
#             'SOL': 'SOL/USDT'
#         }
        
#         # Filter relevant tickers
#         relevant_tickers = [t for t, s in ticker_mapping.items() if not symbols or s in symbols]
#         mentions_df = mentions_df[mentions_df['ticker'].isin(relevant_tickers)]
        
#         if len(mentions_df) == 0:
#             return None
        
#         # Map to standard symbols
#         mentions_df['symbol'] = mentions_df['ticker'].map(ticker_mapping)
        
#         # Create time bucket
#         mentions_df['time_bucket'] = mentions_df['timestamp'].dt.floor('5min')
        
#         # Aggregate by symbol and time bucket
#         agg_dict = {
#             'mention_count': 'sum',
#             'mention_change': 'mean',
#             'sentiment': 'mean',
#             'price_change_1d': 'mean'
#         }
        
#         social_agg = mentions_df.groupby(['symbol', 'time_bucket']).agg(agg_dict).reset_index()
        
#         # Rename columns
#         rename_dict = {
#             'mention_count': 'mention_count_14d',
#             'mention_change': 'mention_change_14d',
#             'sentiment': 'sentiment_14d',
#             'price_change_1d': 'social_price_change_1d'
#         }
#         social_agg = social_agg.rename(columns=rename_dict)
        
#         return social_agg
    
#     def _create_composite_social_features(self, social_df: pd.DataFrame) -> pd.DataFrame:
#         """Create composite social features"""
        
#         # Total mentions across timeframes
#         mention_cols = [col for col in social_df.columns if col.startswith('mention_count_')]
#         if mention_cols:
#             social_df['total_mentions'] = social_df[mention_cols].sum(axis=1)
        
#         # Social momentum score (weighted average of momentum indicators)
#         momentum_cols = {
#             'mention_momentum_4h': 0.5,
#             'mention_momentum_7d': 0.3,
#             'mention_momentum_1m': 0.2
#         }
        
#         social_df['social_momentum_score'] = 0
#         for col, weight in momentum_cols.items():
#             if col in social_df.columns:
#                 social_df['social_momentum_score'] += social_df[col].fillna(0) * weight
        
#         # Sentiment-weighted mentions
#         if 'sentiment_14d' in social_df.columns and 'total_mentions' in social_df.columns:
#             social_df['sentiment_weighted_mentions'] = (
#                 social_df['total_mentions'] * (1 + social_df['sentiment_14d'].fillna(0))
#             )
        
#         # Sentiment categories
#         if 'sentiment_14d' in social_df.columns:
#             social_df['sentiment_very_positive'] = (
#                 social_df['sentiment_14d'] > SENTIMENT_THRESHOLDS['very_positive']
#             ).astype(int)
#             social_df['sentiment_positive'] = (
#                 social_df['sentiment_14d'] > SENTIMENT_THRESHOLDS['positive']
#             ).astype(int)
#             social_df['sentiment_negative'] = (
#                 social_df['sentiment_14d'] < SENTIMENT_THRESHOLDS['neutral']
#             ).astype(int)
#             social_df['sentiment_very_negative'] = (
#                 social_df['sentiment_14d'] < SENTIMENT_THRESHOLDS['negative']
#             ).astype(int)
        
#         # Fill missing values
#         social_df = social_df.fillna(0)
        
#         return social_df