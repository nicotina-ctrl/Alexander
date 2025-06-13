"""
Utility functions for data processing and memory optimization
"""
import os
import subprocess
import sys
import importlib
import pandas as pd
import numpy as np
import gc
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def install_required_packages():
    """Install required packages in a cross-platform way"""
    required_packages = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'optuna': 'optuna'
    }
    
    for import_name, pip_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✓ {import_name} already installed")
        except ImportError:
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pip_name])
            print(f"✓ {pip_name} installed successfully")

def optimize_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting types"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    # Downcast floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Downcast integers
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert low-cardinality object columns to category
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f"Memory usage: {start_mem:.2f} MB → {end_mem:.2f} MB ({(1 - end_mem/start_mem) * 100:.1f}% reduction)")
    
    return df

def copy_files_to_local(gdrive_dir: str, local_dir: str, parquet_files: List[str], batch_size: int = 100):
    """Copy files to local storage with deduplication"""
    print(f"Copying {len(parquet_files)} files to local storage...")
    
    copied_count = 0
    skipped_count = 0
    
    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i:i+batch_size]
        for src_file in batch_files:
            dst_file = os.path.join(local_dir, os.path.basename(src_file))
            try:
                # Only copy if destination doesn't exist
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"Error copying {src_file}: {e}")
        
        print(f"Progress: {min(i+batch_size, len(parquet_files))}/{len(parquet_files)} files...")
        gc.collect()
    
    print(f"Copied {copied_count} new files, skipped {skipped_count} existing files")

def save_cache_data(df: pd.DataFrame, cache_path: str, filename: str) -> str:
    """Save DataFrame with timestamp"""
    os.makedirs(cache_path, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"{filename.split('.')[0]}_{timestamp}.parquet"
    filepath = os.path.join(cache_path, timestamped_filename)
    
    # Save data
    df.to_parquet(filepath, compression='snappy')
    
    # Also save a "latest" version for easy access
    latest_path = os.path.join(cache_path, filename)
    df.to_parquet(latest_path, compression='snappy')
    
    print(f"Saved data to: {filepath}")
    print(f"Also saved as: {latest_path}")
    print(f"Data shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return filepath

def load_cache_data(cache_path: str, filename: str, use_latest: bool = True) -> Optional[pd.DataFrame]:
    """Load previously saved cache data"""
    if use_latest:
        filepath = os.path.join(cache_path, filename)
    else:
        # List available files
        files = sorted([f for f in os.listdir(cache_path) if f.startswith(filename.split('.')[0]) and f.endswith(".parquet")])
        if not files:
            print("No saved data found!")
            return None
        
        print("Available saved datasets:")
        for i, f in enumerate(files):
            print(f"{i}: {f}")
        
        choice = int(input("Select file number: "))
        filepath = os.path.join(cache_path, files[choice])
    
    if os.path.exists(filepath):
        print(f"Loading data from: {filepath}")
        df = pd.read_parquet(filepath)
        print(f"Loaded data shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Restore categorical dtype if needed
        if 'symbol' in df.columns and df['symbol'].dtype != 'category':
            df['symbol'] = df['symbol'].astype('category')
        
        return df
    else:
        print(f"File not found: {filepath}")
        return None

def bool_to_int8(s: pd.Series) -> pd.Series:
    """Safely cast boolean/float Series to int8, filling NaN with 0 first."""
    return s.fillna(False).astype('int8')

def calculate_dynamic_sharpe(returns: np.ndarray, volatility: float, time_index: pd.DatetimeIndex = None) -> float:
    """Calculate Sharpe ratio with dynamic annualization"""
    avg_return = returns.mean()
    
    if time_index is not None and len(time_index) > 1:
        # Calculate actual periods per year from the data
        time_span = (time_index.max() - time_index.min()).total_seconds()
        n_periods = len(time_index)
        avg_period_seconds = time_span / (n_periods - 1)
        
        # For 5-minute bars: 365 * 24 * 60 / 5 = 105,120 periods per year
        periods_per_year = 365 * 24 * 60 * 60 / avg_period_seconds
        
        # Account for potential gaps (weekends, holidays, etc.)
        actual_days = (time_index.max() - time_index.min()).days
        data_density = n_periods / (actual_days * 24 * 12) if actual_days > 0 else 1
        
        # Adjust periods per year based on actual data density
        effective_periods_per_year = periods_per_year * data_density
    else:
        # Fallback: assume 5-minute bars, 24/7 trading
        effective_periods_per_year = 365 * 24 * 12
    
    sharpe = avg_return / (volatility + 1e-8) * np.sqrt(effective_periods_per_year)
    return sharpe

# """
# Utility functions for data processing and memory optimization
# """
# import os
# import subprocess
# import sys
# import importlib
# import pandas as pd
# import numpy as np
# import gc
# import shutil
# from datetime import datetime
# from typing import Dict, List, Tuple, Optional

# def install_required_packages():
#     """Install required packages in a cross-platform way"""
#     required_packages = {
#         'xgboost': 'xgboost',
#         'lightgbm': 'lightgbm',
#         'optuna': 'optuna'
#     }
    
#     for import_name, pip_name in required_packages.items():
#         try:
#             importlib.import_module(import_name)
#             print(f"✓ {import_name} already installed")
#         except ImportError:
#             print(f"Installing {pip_name}...")
#             subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pip_name])
#             print(f"✓ {pip_name} installed successfully")

# def optimize_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
#     """Optimize DataFrame memory usage by downcasting types"""
#     start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
#     # Downcast floats
#     for col in df.select_dtypes(include=['float64']).columns:
#         df[col] = pd.to_numeric(df[col], downcast='float')
    
#     # Downcast integers
#     for col in df.select_dtypes(include=['int64']).columns:
#         df[col] = pd.to_numeric(df[col], downcast='integer')
    
#     # Convert low-cardinality object columns to category
#     for col in df.select_dtypes(include=['object']).columns:
#         num_unique = df[col].nunique()
#         num_total = len(df[col])
#         if num_unique / num_total < 0.5:
#             df[col] = df[col].astype('category')
    
#     end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
#     if verbose:
#         print(f"Memory usage: {start_mem:.2f} MB → {end_mem:.2f} MB ({(1 - end_mem/start_mem) * 100:.1f}% reduction)")
    
#     return df

# def copy_files_to_local(gdrive_dir: str, local_dir: str, parquet_files: List[str], batch_size: int = 100):
#     """Copy files to local storage with deduplication"""
#     print(f"Copying {len(parquet_files)} files to local storage...")
    
#     copied_count = 0
#     skipped_count = 0
    
#     for i in range(0, len(parquet_files), batch_size):
#         batch_files = parquet_files[i:i+batch_size]
#         for src_file in batch_files:
#             dst_file = os.path.join(local_dir, os.path.basename(src_file))
#             try:
#                 # Only copy if destination doesn't exist
#                 if not os.path.exists(dst_file):
#                     shutil.copy2(src_file, dst_file)
#                     copied_count += 1
#                 else:
#                     skipped_count += 1
#             except Exception as e:
#                 print(f"Error copying {src_file}: {e}")
        
#         print(f"Progress: {min(i+batch_size, len(parquet_files))}/{len(parquet_files)} files...")
#         gc.collect()
    
#     print(f"Copied {copied_count} new files, skipped {skipped_count} existing files")

# def save_cache_data(df: pd.DataFrame, cache_path: str, filename: str) -> str:
#     """Save DataFrame with timestamp"""
#     os.makedirs(cache_path, exist_ok=True)
    
#     # Create filename with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     timestamped_filename = f"{filename.split('.')[0]}_{timestamp}.parquet"
#     filepath = os.path.join(cache_path, timestamped_filename)
    
#     # Save data
#     df.to_parquet(filepath, compression='snappy')
    
#     # Also save a "latest" version for easy access
#     latest_path = os.path.join(cache_path, filename)
#     df.to_parquet(latest_path, compression='snappy')
    
#     print(f"Saved data to: {filepath}")
#     print(f"Also saved as: {latest_path}")
#     print(f"Data shape: {df.shape}")
#     print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
#     return filepath

# def load_cache_data(cache_path: str, filename: str, use_latest: bool = True) -> Optional[pd.DataFrame]:
#     """Load previously saved cache data"""
#     if use_latest:
#         filepath = os.path.join(cache_path, filename)
#     else:
#         # List available files
#         files = sorted([f for f in os.listdir(cache_path) if f.startswith(filename.split('.')[0]) and f.endswith(".parquet")])
#         if not files:
#             print("No saved data found!")
#             return None
        
#         print("Available saved datasets:")
#         for i, f in enumerate(files):
#             print(f"{i}: {f}")
        
#         choice = int(input("Select file number: "))
#         filepath = os.path.join(cache_path, files[choice])
    
#     if os.path.exists(filepath):
#         print(f"Loading data from: {filepath}")
#         df = pd.read_parquet(filepath)
#         print(f"Loaded data shape: {df.shape}")
#         print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
#         # Restore categorical dtype if needed
#         if 'symbol' in df.columns and df['symbol'].dtype != 'category':
#             df['symbol'] = df['symbol'].astype('category')
        
#         return df
#     else:
#         print(f"File not found: {filepath}")
#         return None

# def bool_to_int8(s: pd.Series) -> pd.Series:
#     """Safely cast boolean/float Series to int8, filling NaN with 0 first."""
#     return s.fillna(False).astype('int8')

# def calculate_dynamic_sharpe(returns: np.ndarray, volatility: float, time_index: pd.DatetimeIndex = None) -> float:
#     """Calculate Sharpe ratio with dynamic annualization"""
#     avg_return = returns.mean()
    
#     if time_index is not None and len(time_index) > 1:
#         # Calculate actual periods per year from the data
#         time_span = (time_index.max() - time_index.min()).total_seconds()
#         n_periods = len(time_index)
#         avg_period_seconds = time_span / (n_periods - 1)
        
#         # For 5-minute bars: 365 * 24 * 60 / 5 = 105,120 periods per year
#         periods_per_year = 365 * 24 * 60 * 60 / avg_period_seconds
        
#         # Account for potential gaps (weekends, holidays, etc.)
#         actual_days = (time_index.max() - time_index.min()).days
#         data_density = n_periods / (actual_days * 24 * 12) if actual_days > 0 else 1
        
#         # Adjust periods per year based on actual data density
#         effective_periods_per_year = periods_per_year * data_density
#     else:
#         # Fallback: assume 5-minute bars, 24/7 trading
#         effective_periods_per_year = 365 * 24 * 12
    
#     sharpe = avg_return / (volatility + 1e-8) * np.sqrt(effective_periods_per_year)
#     return sharpe