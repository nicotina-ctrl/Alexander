"""
data_directory_explorer.py
Utility to explore and understand the structure of your crypto data directory
"""
import os
import glob
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Dict
import json


class DataDirectoryExplorer:
    """Explore the structure and contents of crypto data directories"""
    
    def __init__(self, base_path: str = "/content/drive/MyDrive/crypto_pipeline_whale"):
        self.base_path = base_path
        self.realtime_data_path = os.path.join(base_path, "realtime_perp_data")
    
    def explore_directory(self, path: str = None) -> Dict:
        """Explore directory structure and file types"""
        if path is None:
            path = self.realtime_data_path
        
        print(f"\nExploring directory: {path}")
        print("="*80)
        
        # Get all files
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                file_size = os.path.getsize(file_path)
                all_files.append({
                    'path': rel_path,
                    'name': file,
                    'size': file_size,
                    'size_mb': file_size / (1024 * 1024),
                    'extension': os.path.splitext(file)[1]
                })
        
        # Summary by extension
        extension_summary = {}
        for file in all_files:
            ext = file['extension']
            if ext not in extension_summary:
                extension_summary[ext] = {'count': 0, 'total_size_mb': 0}
            extension_summary[ext]['count'] += 1
            extension_summary[ext]['total_size_mb'] += file['size_mb']
        
        print(f"\nFound {len(all_files)} files")
        print("\nFile types summary:")
        for ext, info in sorted(extension_summary.items()):
            print(f"  {ext or 'no extension'}: {info['count']} files, {info['total_size_mb']:.2f} MB")
        
        # Look specifically for parquet files
        parquet_files = [f for f in all_files if f['extension'] == '.parquet']
        print(f"\nFound {len(parquet_files)} parquet files")
        
        return {
            'all_files': all_files,
            'extension_summary': extension_summary,
            'parquet_files': parquet_files
        }
    
    def analyze_parquet_files(self, sample_size: int = 5) -> Dict:
        """Analyze structure of parquet files"""
        print(f"\nAnalyzing parquet files in: {self.realtime_data_path}")
        print("="*80)
        
        parquet_files = glob.glob(os.path.join(self.realtime_data_path, "**/*.parquet"), recursive=True)
        
        if not parquet_files:
            print("No parquet files found!")
            return {}
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Analyze a sample of files
        file_analysis = []
        sample_files = parquet_files[:sample_size] if len(parquet_files) > sample_size else parquet_files
        
        for file_path in sample_files:
            print(f"\nAnalyzing: {os.path.basename(file_path)}")
            try:
                # Read parquet metadata
                parquet_file = pq.ParquetFile(file_path)
                metadata = parquet_file.metadata
                
                # Read small sample
                df_sample = pd.read_parquet(file_path, nrows=100)
                
                # Get time range if possible
                time_columns = ['time_bucket', 'timestamp', 'time', 'datetime', 'date', 'traded_at']
                time_col = None
                for col in time_columns:
                    if col in df_sample.columns:
                        time_col = col
                        break
                
                time_range = None
                if time_col:
                    df_sample[time_col] = pd.to_datetime(df_sample[time_col])
                    time_range = {
                        'min': df_sample[time_col].min(),
                        'max': df_sample[time_col].max()
                    }
                
                # Identify file type based on columns
                file_type = self._identify_file_type(df_sample.columns.tolist())
                
                analysis = {
                    'file_name': os.path.basename(file_path),
                    'file_path': file_path,
                    'num_rows': metadata.num_rows,
                    'num_columns': len(df_sample.columns),
                    'columns': df_sample.columns.tolist(),
                    'file_type': file_type,
                    'time_range': time_range,
                    'sample_data': df_sample.head(3).to_dict('records')
                }
                
                file_analysis.append(analysis)
                
                print(f"  Type: {file_type}")
                print(f"  Rows: {metadata.num_rows:,}")
                print(f"  Columns: {len(df_sample.columns)}")
                if time_range:
                    print(f"  Time range: {time_range['min']} to {time_range['max']}")
                print(f"  Column names: {', '.join(df_sample.columns[:10])}...")
                
            except Exception as e:
                print(f"  Error reading file: {e}")
        
        return {'file_analysis': file_analysis}
    
    def _identify_file_type(self, columns: List[str]) -> str:
        """Identify the type of data based on column names"""
        columns_lower = [col.lower() for col in columns]
        
        # Orderbook indicators
        orderbook_indicators = ['bid', 'ask', 'spread', 'order_book', 'orderbook', 'depth']
        if any(indicator in ' '.join(columns_lower) for indicator in orderbook_indicators):
            return 'orderbook'
        
        # Trades indicators
        trades_indicators = ['trade', 'trades', 'traded_at', 'side', 'taker', 'maker']
        if any(indicator in ' '.join(columns_lower) for indicator in trades_indicators):
            return 'trades'
        
        # OHLCV indicators
        if all(x in columns_lower for x in ['open', 'high', 'low', 'close']):
            return 'ohlcv'
        
        # Price/Volume data
        if 'price' in ' '.join(columns_lower) and 'volume' in ' '.join(columns_lower):
            return 'price_volume'
        
        return 'unknown'
    
    def find_files_by_date(self, target_date: str) -> List[str]:
        """Find all files containing data for a specific date"""
        print(f"\nSearching for files containing date: {target_date}")
        print("="*80)
        
        # Convert date to various formats
        dt = pd.to_datetime(target_date)
        date_formats = [
            dt.strftime("%Y%m%d"),
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%Y_%m_%d"),
            dt.strftime("%Y.%m.%d"),
            dt.strftime("%d%m%Y"),
            dt.strftime("%d-%m-%Y"),
        ]
        
        matching_files = []
        
        # Search for files with date in filename
        all_files = glob.glob(os.path.join(self.realtime_data_path, "**/*"), recursive=True)
        for file_path in all_files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                if any(date_fmt in filename for date_fmt in date_formats):
                    matching_files.append(file_path)
        
        print(f"Found {len(matching_files)} files with date in filename")
        
        # For parquet files, check content
        parquet_files = [f for f in all_files if f.endswith('.parquet')]
        print(f"\nChecking {len(parquet_files)} parquet files for date content...")
        
        content_matches = []
        for file_path in parquet_files[:20]:  # Check first 20 files
            try:
                df_sample = pd.read_parquet(file_path, nrows=1000)
                
                # Check for time columns
                time_columns = ['time_bucket', 'timestamp', 'time', 'datetime', 'date', 'traded_at']
                for col in time_columns:
                    if col in df_sample.columns:
                        df_sample[col] = pd.to_datetime(df_sample[col])
                        if ((df_sample[col] >= dt) & (df_sample[col] < dt + pd.Timedelta(days=1))).any():
                            content_matches.append(file_path)
                            print(f"  Found matching data in: {os.path.basename(file_path)}")
                            break
            except:
                pass
        
        all_matches = list(set(matching_files + content_matches))
        print(f"\nTotal files with data for {target_date}: {len(all_matches)}")
        
        return all_matches
    
    def save_exploration_report(self, output_file: str = "data_exploration_report.json"):
        """Save a comprehensive exploration report"""
        print(f"\nGenerating comprehensive exploration report...")
        
        report = {
            'exploration_date': datetime.now().isoformat(),
            'base_path': self.base_path,
            'realtime_data_path': self.realtime_data_path,
            'directory_structure': self.explore_directory(),
            'parquet_analysis': self.analyze_parquet_files(sample_size=10),
            'date_coverage': {}
        }
        
        # Check date coverage for June 3-5, 2025
        for date in ['2025-06-03', '2025-06-04', '2025-06-05']:
            report['date_coverage'][date] = len(self.find_files_by_date(date))
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nExploration report saved to: {output_file}")
        
        return report


def main():
    """Main function to explore data directory"""
    explorer = DataDirectoryExplorer("/content/drive/MyDrive/crypto_pipeline_whale")
    
    # Explore directory structure
    explorer.explore_directory()
    
    # Analyze parquet files
    explorer.analyze_parquet_files()
    
    # Find files for specific dates
    for date in ['2025-06-03', '2025-06-04', '2025-06-05']:
        explorer.find_files_by_date(date)
    
    # Save comprehensive report
    report = explorer.save_exploration_report()
    
    print("\n" + "="*80)
    print("EXPLORATION COMPLETE!")
    print("="*80)
    print("\nCheck 'data_exploration_report.json' for detailed findings.")
    print("\nNext steps:")
    print("1. Review the exploration report to understand your data structure")
    print("2. Update the CryptoDataLoader patterns if needed")
    print("3. Run the data_loader_complete.py to integrate all data")


if __name__ == "__main__":
    main()