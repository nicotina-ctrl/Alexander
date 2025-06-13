"""
Analyze crypto symbols in parquet files to find top 20 by various metrics
"""
import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
import gc

def analyze_crypto_symbols(data_path: str = "/content/drive/MyDrive/crypto_pipeline_whale/realtime_perp_data",
                          sample_files: int = None,
                          save_results: bool = True) -> pd.DataFrame:
    """
    Analyze all crypto symbols in parquet files and rank them by various metrics
    
    Args:
        data_path: Path to parquet files
        sample_files: Number of files to sample (None = all files)
        save_results: Whether to save results to CSV
        
    Returns:
        DataFrame with symbol analysis and rankings
    """
    print("="*80)
    print("CRYPTO SYMBOL ANALYZER")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Started at: {datetime.now()}")
    print("="*80 + "\n")
    
    # Get all parquet files
    all_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    print(f"Found {len(all_files)} parquet files")
    
    # Sample files if requested
    if sample_files and sample_files < len(all_files):
        # Take evenly spaced samples across all files
        indices = np.linspace(0, len(all_files)-1, sample_files, dtype=int)
        files_to_process = [all_files[i] for i in indices]
        print(f"Sampling {sample_files} files evenly across the dataset")
    else:
        files_to_process = all_files
        print(f"Processing all {len(files_to_process)} files")
    
    # Initialize symbol statistics
    symbol_stats = {}
    
    # Process files in batches
    batch_size = 50
    n_batches = (len(files_to_process) + batch_size - 1) // batch_size
    
    print("\nAnalyzing symbols across files...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(files_to_process))
        batch_files = files_to_process[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{n_batches} (files {start_idx}-{end_idx})...")
        
        for fpath in tqdm(batch_files, desc=f"Batch {batch_idx + 1}"):
            try:
                # Read file
                df = pd.read_parquet(fpath)
                
                if 'symbol' not in df.columns:
                    continue
                
                # Get unique symbols in this file
                symbols_in_file = df['symbol'].unique()
                
                # Analyze each symbol
                for symbol in symbols_in_file:
                    symbol_data = df[df['symbol'] == symbol]
                    
                    if symbol not in symbol_stats:
                        symbol_stats[symbol] = {
                            'count': 0,
                            'total_rows': 0,
                            'files_present': 0,
                            'total_volume': 0,
                            'avg_price': 0,
                            'price_sum': 0,
                            'first_seen': None,
                            'last_seen': None,
                            'avg_spread': 0,
                            'spread_sum': 0,
                            'spread_count': 0
                        }
                    
                    stats = symbol_stats[symbol]
                    stats['count'] += 1
                    stats['total_rows'] += len(symbol_data)
                    stats['files_present'] += 1
                    
                    # Calculate volume if available
                    if 'quantity_sum' in symbol_data.columns:
                        stats['total_volume'] += symbol_data['quantity_sum'].sum()
                    elif 'volume' in symbol_data.columns:
                        stats['total_volume'] += symbol_data['volume'].sum()
                    
                    # Calculate average price
                    price_col = None
                    if 'mid_price_mean' in symbol_data.columns:
                        price_col = 'mid_price_mean'
                    elif 'price_mean' in symbol_data.columns:
                        price_col = 'price_mean'
                    elif 'vwap' in symbol_data.columns:
                        price_col = 'vwap'
                    elif 'price' in symbol_data.columns:
                        price_col = 'price'
                    
                    if price_col and price_col in symbol_data.columns:
                        valid_prices = symbol_data[price_col].dropna()
                        if len(valid_prices) > 0:
                            stats['price_sum'] += valid_prices.sum()
                            stats['count'] += len(valid_prices)
                    
                    # Track first/last seen
                    if 'timestamp' in symbol_data.columns:
                        try:
                            timestamps = pd.to_datetime(symbol_data['timestamp'], unit='s')
                            if stats['first_seen'] is None or timestamps.min() < stats['first_seen']:
                                stats['first_seen'] = timestamps.min()
                            if stats['last_seen'] is None or timestamps.max() > stats['last_seen']:
                                stats['last_seen'] = timestamps.max()
                        except:
                            pass
                    
                    # Calculate spread if available
                    if 'spread_mean' in symbol_data.columns:
                        valid_spreads = symbol_data['spread_mean'].dropna()
                        if len(valid_spreads) > 0:
                            stats['spread_sum'] += valid_spreads.sum()
                            stats['spread_count'] += len(valid_spreads)
                
                del df
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing {os.path.basename(fpath)}: {e}")
                continue
    
    # Convert to DataFrame and calculate final metrics
    print("\nCalculating final metrics...")
    
    results = []
    for symbol, stats in symbol_stats.items():
        # Calculate averages
        avg_price = stats['price_sum'] / stats['count'] if stats['count'] > 0 else 0
        avg_spread = stats['spread_sum'] / stats['spread_count'] if stats['spread_count'] > 0 else 0
        
        # Calculate activity score (combination of volume and presence)
        file_coverage = stats['files_present'] / len(files_to_process)
        volume_score = np.log1p(stats['total_volume'])
        activity_score = file_coverage * volume_score
        
        # Calculate liquidity score
        liquidity_score = volume_score / (1 + avg_spread) if avg_spread > 0 else volume_score
        
        results.append({
            'symbol': symbol,
            'total_rows': stats['total_rows'],
            'files_present': stats['files_present'],
            'file_coverage': file_coverage,
            'total_volume': stats['total_volume'],
            'avg_price': avg_price,
            'avg_spread': avg_spread,
            'first_seen': stats['first_seen'],
            'last_seen': stats['last_seen'],
            'activity_score': activity_score,
            'liquidity_score': liquidity_score,
            'volume_score': volume_score
        })
    
    # Create DataFrame and sort by activity score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('activity_score', ascending=False)
    
    # Add rankings
    results_df['rank_activity'] = range(1, len(results_df) + 1)
    results_df['rank_volume'] = results_df['total_volume'].rank(ascending=False, method='min').astype(int)
    results_df['rank_liquidity'] = results_df['liquidity_score'].rank(ascending=False, method='min').astype(int)
    results_df['rank_coverage'] = results_df['file_coverage'].rank(ascending=False, method='min').astype(int)
    
    # Calculate composite rank
    results_df['composite_rank'] = (
        results_df['rank_activity'] * 0.3 +
        results_df['rank_volume'] * 0.3 +
        results_df['rank_liquidity'] * 0.2 +
        results_df['rank_coverage'] * 0.2
    )
    
    # Sort by composite rank
    results_df = results_df.sort_values('composite_rank')
    results_df['final_rank'] = range(1, len(results_df) + 1)
    
    # Display top 20
    print("\n" + "="*80)
    print("TOP 20 CRYPTO SYMBOLS BY COMPOSITE SCORE")
    print("="*80)
    
    top_20 = results_df.head(20)
    
    for idx, row in top_20.iterrows():
        print(f"\n{row['final_rank']:2d}. {row['symbol']}")
        print(f"    Activity Score: {row['activity_score']:.2f}")
        print(f"    Total Volume: {row['total_volume']:,.0f}")
        print(f"    File Coverage: {row['file_coverage']:.1%}")
        print(f"    Avg Price: ${row['avg_price']:.2f}")
        print(f"    Avg Spread: {row['avg_spread']:.4f}")
        print(f"    Data Points: {row['total_rows']:,}")
    
    # Save results if requested
    if save_results:
        output_path = "/content/drive/MyDrive/crypto_pipeline_whale/symbol_analysis.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nFull results saved to: {output_path}")
        
        # Save top 20 separately
        top_20_path = "/content/drive/MyDrive/crypto_pipeline_whale/top_20_symbols.csv"
        top_20.to_csv(top_20_path, index=False)
        print(f"Top 20 symbols saved to: {top_20_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Total symbols found: {len(results_df)}")
    print(f"Completed at: {datetime.now()}")
    print("="*80)
    
    return results_df


def get_top_symbols_list(results_df: pd.DataFrame = None, 
                        top_n: int = 20,
                        filter_usdt_only: bool = True) -> List[str]:
    """
    Get list of top N symbols from analysis results
    
    Args:
        results_df: Analysis results (if None, will load from saved file)
        top_n: Number of top symbols to return
        filter_usdt_only: Whether to filter only USDT pairs
        
    Returns:
        List of symbol strings
    """
    if results_df is None:
        # Try to load from saved file
        saved_path = "/content/drive/MyDrive/crypto_pipeline_whale/symbol_analysis.csv"
        if os.path.exists(saved_path):
            results_df = pd.read_csv(saved_path)
        else:
            raise ValueError("No results provided and no saved analysis found")
    
    # Get top symbols
    top_symbols = results_df.head(top_n)
    
    # Filter USDT pairs if requested
    if filter_usdt_only:
        top_symbols = top_symbols[top_symbols['symbol'].str.contains('USDT')]
    
    # Get symbol list
    symbol_list = top_symbols['symbol'].tolist()
    
    print(f"\nTop {len(symbol_list)} symbols:")
    for i, symbol in enumerate(symbol_list, 1):
        print(f"{i:2d}. {symbol}")
    
    return symbol_list


def quick_symbol_check(data_path: str = "/content/drive/MyDrive/crypto_pipeline_whale/realtime_perp_data",
                      n_files: int = 10):
    """
    Quick check of symbols in a sample of files
    
    Args:
        data_path: Path to parquet files
        n_files: Number of files to check
    """
    files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))[:n_files]
    
    all_symbols = set()
    
    print(f"Checking {len(files)} files for symbols...\n")
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            if 'symbol' in df.columns:
                symbols = df['symbol'].unique()
                all_symbols.update(symbols)
                print(f"{os.path.basename(f)}: {len(symbols)} unique symbols")
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")
    
    print(f"\nTotal unique symbols found: {len(all_symbols)}")
    print("\nSample symbols:")
    for symbol in sorted(list(all_symbols))[:20]:
        print(f"  - {symbol}")
    
    return sorted(list(all_symbols))


# Example usage function for Google Colab
def run_symbol_analysis():
    """
    Main function to run in Google Colab
    """
    # Quick check first
    print("Running quick symbol check...\n")
    quick_symbols = quick_symbol_check(n_files=5)
    
    # Run full analysis
    print("\n\nRunning full symbol analysis...\n")
    results = analyze_crypto_symbols(
        sample_files=100,  # Sample 100 files for faster analysis
        save_results=True
    )
    
    # Get top 20 USDT pairs
    print("\n\nExtracting top 20 USDT pairs...\n")
    top_20_symbols = get_top_symbols_list(results, top_n=20, filter_usdt_only=True)
    
    return results, top_20_symbols


if __name__ == "__main__":
    # Run the analysis
    results, top_symbols = run_symbol_analysis()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if len(results) > 0:
        print(f"\nTotal symbols analyzed: {len(results)}")
        print(f"Average file coverage: {results['file_coverage'].mean():.1%}")
        print(f"Total volume across all symbols: {results['total_volume'].sum():,.0f}")
        
        # Show volume distribution
        print("\nVolume distribution:")
        print(f"  Top 10%: {results['total_volume'].quantile(0.9):,.0f}")
        print(f"  Median:  {results['total_volume'].median():,.0f}")
        print(f"  Bottom 10%: {results['total_volume'].quantile(0.1):,.0f}")