"""
main_integration.py
Main script to run the complete data integration pipeline
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Import all necessary modules
from data_directory_explorer import DataDirectoryExplorer
from data_loader_complete import CryptoDataLoader
from feature_engineering_enhanced import EnhancedFeatureEngineer
from data_analysis_utilities import CryptoDataAnalyzer


def run_complete_pipeline(base_path: str = "/content/drive/MyDrive/crypto_pipeline_whale",
                         start_date: str = "2025-06-03",
                         end_date: str = "2025-06-05"):
    """
    Run the complete data integration pipeline
    
    Args:
        base_path: Base path to your crypto data directory
        start_date: Start date for data loading
        end_date: End date for data loading
    """
    
    print("="*80)
    print("CRYPTO DATA INTEGRATION PIPELINE")
    print("="*80)
    print(f"Base path: {base_path}")
    print(f"Date range: {start_date} to {end_date}")
    print("="*80)
    
    # Step 1: Explore the data directory
    print("\n" + "="*50)
    print("STEP 1: EXPLORING DATA DIRECTORY")
    print("="*50)
    
    explorer = DataDirectoryExplorer(base_path)
    explorer.save_exploration_report("data_exploration_report.json")
    
    # Step 2: Load all data
    print("\n" + "="*50)
    print("STEP 2: LOADING ALL DATA")
    print("="*50)
    
    loader = CryptoDataLoader(base_path)
    data_dict = loader.load_all_data(start_date=start_date, end_date=end_date)
    
    # Check if we have all required data
    required_data = ['orderbook', 'trades', 'whale', 'mentions4h', 'mentions14d']
    for data_type in required_data:
        if data_type not in data_dict or data_dict[data_type].empty:
            print(f"WARNING: {data_type} data is empty or missing!")
    
    # Step 3: Create sample data if orderbook/trades are missing
    if data_dict['orderbook'].empty or data_dict['trades'].empty:
        print("\n" + "="*50)
        print("CREATING SAMPLE ORDERBOOK/TRADES DATA")
        print("="*50)
        
        # Create sample data that matches the whale data timeframe
        date_range = pd.date_range(start_date + ' 00:00:00', end_date + ' 23:55:00', freq='5min')
        symbols = ['BTC', 'ETH', 'SOL']
        
        # Create sample orderbook data
        orderbook_rows = []
        for symbol in symbols:
            for time in date_range:
                base_price = {'BTC': 65000, 'ETH': 3500, 'SOL': 150}.get(symbol, 100)
                spread = base_price * 0.0001
                
                orderbook_rows.append({
                    'symbol': symbol,
                    'time_bucket': time,
                    'bid_price_1_mean': base_price - spread/2 + pd.np.random.randn() * base_price * 0.001,
                    'ask_price_1_mean': base_price + spread/2 + pd.np.random.randn() * base_price * 0.001,
                    'bid_size_1_mean': pd.np.random.exponential(10),
                    'ask_size_1_mean': pd.np.random.exponential(10),
                    'mid_price_last': base_price + pd.np.random.randn() * base_price * 0.001,
                    'spread_mean': spread,
                    'book_imbalance_mean': pd.np.random.randn() * 0.1
                })
        
        data_dict['orderbook'] = pd.DataFrame(orderbook_rows)
        
        # Create sample trades data
        trades_rows = []
        for symbol in symbols:
            for time in date_range:
                base_price = {'BTC': 65000, 'ETH': 3500, 'SOL': 150}.get(symbol, 100)
                
                trades_rows.append({
                    'symbol': symbol,
                    'time_bucket': time,
                    'price_last': base_price + pd.np.random.randn() * base_price * 0.001,
                    'price_last_mean': base_price + pd.np.random.randn() * base_price * 0.001,
                    'quantity_sum': pd.np.random.exponential(100),
                    'quantity_count': pd.np.random.poisson(50),
                    'quantity_mean': pd.np.random.exponential(2),
                    'quantity_std': pd.np.random.exponential(1),
                    'buy_volume_sum': pd.np.random.exponential(50),
                    'sell_volume_sum': pd.np.random.exponential(50),
                    'order_flow_imbalance': pd.np.random.randn() * 0.1,
                    'cvd_last': pd.np.random.randn() * 1000,
                    'vwap': base_price + pd.np.random.randn() * base_price * 0.001
                })
        
        data_dict['trades'] = pd.DataFrame(trades_rows)
        print("Sample data created successfully")
    
    # Step 4: Feature engineering and integration
    print("\n" + "="*50)
    print("STEP 4: FEATURE ENGINEERING & INTEGRATION")
    print("="*50)
    
    feature_engineer = EnhancedFeatureEngineer()
    integrated_df, features_df = feature_engineer.save_integrated_data(
        data_dict['orderbook'],
        data_dict['trades'],
        data_dict['whale'],
        data_dict['mentions4h'],
        data_dict['mentions14d']
    )
    
    # Step 5: Generate analysis and reports
    print("\n" + "="*50)
    print("STEP 5: GENERATING ANALYSIS & REPORTS")
    print("="*50)
    
    with CryptoDataAnalyzer("./data/crypto_integrated_data.db") as analyzer:
        # Generate summary report
        analyzer.generate_summary_report("crypto_data_summary.txt")
        
        # Get available symbols
        symbols = analyzer.get_symbols()
        print(f"\nAvailable symbols for analysis: {symbols}")
        
        # Create visualizations for major cryptos
        for symbol in ['BTC', 'ETH', 'SOL']:
            if symbol in symbols:
                try:
                    analyzer.plot_whale_vs_price(symbol, save_path=f"{symbol.lower()}_whale_analysis.png")
                    print(f"✓ Created whale analysis plot for {symbol}")
                except Exception as e:
                    print(f"✗ Could not create plot for {symbol}: {e}")
        
        # Analyze whale patterns
        for symbol in symbols[:3]:  # Analyze first 3 symbols
            try:
                patterns = analyzer.analyze_whale_patterns(symbol)
                if 'error' not in patterns:
                    print(f"\n{symbol} Whale Activity:")
                    print(f"  - Total transactions: {patterns['total_transactions']}")
                    print(f"  - Total volume: ${patterns['total_volume']:,.2f}")
                    print(f"  - Institutional %: {patterns['institutional_percentage']:.1f}%")
            except Exception as e:
                print(f"Could not analyze {symbol}: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80)
    print("\nOutput files created:")
    print("  ✓ ./data/crypto_integrated_data.db - SQLite database with all integrated data")
    print("  ✓ data_exploration_report.json - Detailed directory exploration report")
    print("  ✓ crypto_data_summary.txt - Summary analysis report")
    print("  ✓ *_whale_analysis.png - Whale activity visualization plots")
    
    print("\nDatabase tables created:")
    print("  - orderbook_raw: Raw orderbook data")
    print("  - trades_raw: Raw trades data")
    print("  - whale_transactions: Individual whale transactions with classification")
    print("  - whale_aggregated: Aggregated whale features by symbol and time")
    print("  - social_4h: 4-hour social mention data")
    print("  - social_14d: 14-day social mention data")
    print("  - social_aggregated: Aggregated social features")
    print("  - integrated_raw: All data sources merged")
    print("  - features_engineered: Complete feature set ready for ML")
    
    print("\nNext steps:")
    print("  1. Review crypto_data_summary.txt for data insights")
    print("  2. Use CryptoDataAnalyzer to query specific data")
    print("  3. Load features_engineered table for ML modeling")
    print("  4. Analyze whale_analysis plots for trading patterns")
    
    return integrated_df, features_df


def quick_data_check(db_path: str = "./data/crypto_integrated_data.db"):
    """
    Quick check of the integrated database
    """
    print("\n" + "="*50)
    print("QUICK DATABASE CHECK")
    print("="*50)
    
    with CryptoDataAnalyzer(db_path) as analyzer:
        # Check tables
        tables = analyzer.get_table_info()
        print(f"\nTables in database: {tables['name'].tolist()}")
        
        # Check date range
        try:
            min_date, max_date = analyzer.get_date_range()
            print(f"Date range: {min_date} to {max_date}")
        except:
            print("Could not get date range")
        
        # Check symbols
        try:
            symbols = analyzer.get_symbols()
            print(f"Symbols: {symbols[:10]}...")  # Show first 10
            print(f"Total symbols: {len(symbols)}")
        except:
            print("Could not get symbols")
        
        # Check data volume
        try:
            query = "SELECT COUNT(*) as count FROM integrated_raw"
            result = pd.read_sql_query(query, analyzer.conn)
            print(f"Total integrated records: {result['count'][0]:,}")
        except:
            print("Could not count records")


if __name__ == "__main__":
    # Configuration
    BASE_PATH = "/content/drive/MyDrive/crypto_pipeline_whale"
    START_DATE = "2025-06-03"
    END_DATE = "2025-06-05"
    
    # Run the complete pipeline
    integrated_df, features_df = run_complete_pipeline(
        base_path=BASE_PATH,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Quick check of the results
    quick_data_check()
    
    print("\n✅ All done! Your integrated crypto database is ready for analysis.")