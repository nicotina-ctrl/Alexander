"""
data_analysis_utilities.py
Utility functions for analyzing and querying the integrated crypto data
"""
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple


class CryptoDataAnalyzer:
    """Analyze integrated crypto data from SQLite database"""
    
    def __init__(self, db_path: str = "./data/crypto_integrated_data.db"):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def get_table_info(self) -> pd.DataFrame:
        """Get information about all tables in the database"""
        query = """
        SELECT name, sql 
        FROM sqlite_master 
        WHERE type='table'
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_date_range(self, table_name: str = 'integrated_raw') -> Tuple[datetime, datetime]:
        """Get the date range of data in a table"""
        query = f"""
        SELECT MIN(time_bucket) as min_date, MAX(time_bucket) as max_date
        FROM {table_name}
        """
        result = pd.read_sql_query(query, self.conn)
        return pd.to_datetime(result['min_date'][0]), pd.to_datetime(result['max_date'][0])
    
    def get_symbols(self, table_name: str = 'integrated_raw') -> List[str]:
        """Get all unique symbols in the database"""
        query = f"""
        SELECT DISTINCT symbol 
        FROM {table_name}
        ORDER BY symbol
        """
        result = pd.read_sql_query(query, self.conn)
        return result['symbol'].tolist()
    
    def get_whale_activity_summary(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get whale activity summary by symbol"""
        where_clause = f"WHERE symbol = '{symbol}'" if symbol else ""
        
        query = f"""
        SELECT 
            symbol,
            COUNT(*) as total_periods,
            SUM(whale_amount_usd_count) as total_whale_txns,
            SUM(whale_amount_usd_sum) as total_whale_volume,
            AVG(whale_amount_usd_mean) as avg_whale_tx_size,
            SUM(inst_count) as total_inst_txns,
            SUM(retail_count) as total_retail_txns,
            AVG(inst_participation_rate) as avg_inst_rate,
            AVG(whale_flow_imbalance) as avg_flow_imbalance
        FROM whale_aggregated
        {where_clause}
        GROUP BY symbol
        ORDER BY total_whale_volume DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_social_sentiment_summary(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get social sentiment summary by symbol"""
        where_clause = f"WHERE symbol = '{symbol}'" if symbol else ""
        
        query = f"""
        SELECT 
            symbol,
            COUNT(*) as total_periods,
            SUM(total_mentions) as total_mentions,
            AVG(sentiment) as avg_sentiment,
            AVG(social_momentum_score) as avg_social_momentum,
            MAX(total_mentions) as max_mentions_period,
            MIN(sentiment) as min_sentiment,
            MAX(sentiment) as max_sentiment
        FROM social_aggregated
        {where_clause}
        GROUP BY symbol
        ORDER BY total_mentions DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_price_whale_correlation(self, symbol: str, 
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get price and whale activity data for correlation analysis"""
        time_clause = ""
        if start_time:
            time_clause += f" AND time_bucket >= '{start_time}'"
        if end_time:
            time_clause += f" AND time_bucket <= '{end_time}'"
        
        query = f"""
        SELECT 
            time_bucket,
            price_last,
            whale_amount_usd_sum,
            whale_flow_imbalance,
            inst_participation_rate,
            retail_sell_pressure,
            smart_dumb_divergence
        FROM integrated_raw
        WHERE symbol = '{symbol}' {time_clause}
        ORDER BY time_bucket
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_features_for_modeling(self, symbols: List[str], 
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get feature-engineered data ready for modeling"""
        symbols_str = "', '".join(symbols)
        time_clause = ""
        if start_time:
            time_clause += f" AND time_bucket >= '{start_time}'"
        if end_time:
            time_clause += f" AND time_bucket <= '{end_time}'"
        
        query = f"""
        SELECT *
        FROM features_engineered
        WHERE symbol IN ('{symbols_str}') {time_clause}
        ORDER BY symbol, time_bucket
        """
        return pd.read_sql_query(query, self.conn)
    
    def analyze_whale_patterns(self, symbol: str) -> Dict:
        """Analyze whale trading patterns for a symbol"""
        # Get whale transaction data
        query = f"""
        SELECT *
        FROM whale_transactions
        WHERE token = '{symbol}'
        ORDER BY timestamp
        """
        whale_data = pd.read_sql_query(query, self.conn)
        
        if len(whale_data) == 0:
            return {"error": f"No whale data found for {symbol}"}
        
        # Analyze patterns
        analysis = {
            "symbol": symbol,
            "total_transactions": len(whale_data),
            "total_volume": whale_data['amount_usd'].sum(),
            "avg_transaction_size": whale_data['amount_usd'].mean(),
            "institutional_percentage": (whale_data['classification'] == 'Institutional').mean() * 100,
            "retail_percentage": (whale_data['classification'] == 'Retail').mean() * 100,
            "buy_sell_ratio": whale_data['is_buy'].sum() / (whale_data['is_sell'].sum() + 1),
            "top_5_transactions": whale_data.nlargest(5, 'amount_usd')[['timestamp', 'amount_usd', 'classification', 'transaction_type']].to_dict('records'),
            "hourly_distribution": whale_data.groupby(pd.to_datetime(whale_data['timestamp']).dt.hour)['amount_usd'].sum().to_dict()
        }
        
        return analysis
    
    def plot_whale_vs_price(self, symbol: str, save_path: Optional[str] = None):
        """Plot whale activity vs price movement"""
        # Get integrated data
        query = f"""
        SELECT 
            time_bucket,
            price_last,
            whale_amount_usd_sum,
            whale_flow_imbalance,
            inst_participation_rate
        FROM integrated_raw
        WHERE symbol = '{symbol}'
        ORDER BY time_bucket
        """
        data = pd.read_sql_query(query, self.conn)
        
        if len(data) == 0:
            print(f"No data found for {symbol}")
            return
        
        # Convert time_bucket to datetime
        data['time_bucket'] = pd.to_datetime(data['time_bucket'])
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # Price
        axes[0].plot(data['time_bucket'], data['price_last'], 'b-', linewidth=2)
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].set_title(f'{symbol} Price and Whale Activity Analysis', fontsize=16, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Whale volume
        axes[1].bar(data['time_bucket'], data['whale_amount_usd_sum'], alpha=0.7, color='green')
        axes[1].set_ylabel('Whale Volume ($)', fontsize=12)
        axes[1].set_title('Whale Transaction Volume', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Whale flow imbalance
        colors = ['red' if x < 0 else 'green' for x in data['whale_flow_imbalance']]
        axes[2].bar(data['time_bucket'], data['whale_flow_imbalance'], alpha=0.7, color=colors)
        axes[2].set_ylabel('Flow Imbalance', fontsize=12)
        axes[2].set_title('Whale Flow Imbalance (Buy - Sell)', fontsize=14)
        axes[2].axhline(y=0, color='black', linewidth=0.5)
        axes[2].grid(True, alpha=0.3)
        
        # Institutional participation
        axes[3].plot(data['time_bucket'], data['inst_participation_rate'] * 100, 'purple', linewidth=2)
        axes[3].set_ylabel('Inst. Rate (%)', fontsize=12)
        axes[3].set_xlabel('Time', fontsize=12)
        axes[3].set_title('Institutional Participation Rate', fontsize=14)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_correlation_matrix(self, symbol: str) -> pd.DataFrame:
        """Get correlation matrix between price and whale/social features"""
        query = f"""
        SELECT 
            price_last,
            whale_amount_usd_sum,
            whale_flow_imbalance,
            inst_participation_rate,
            retail_sell_pressure,
            smart_dumb_divergence,
            total_mentions,
            sentiment,
            social_momentum_score
        FROM integrated_raw
        WHERE symbol = '{symbol}'
        """
        data = pd.read_sql_query(query, self.conn)
        
        if len(data) == 0:
            return pd.DataFrame()
        
        # Calculate returns
        data['price_return'] = data['price_last'].pct_change()
        
        # Select features for correlation
        features = [
            'price_return',
            'whale_amount_usd_sum',
            'whale_flow_imbalance',
            'inst_participation_rate',
            'retail_sell_pressure',
            'smart_dumb_divergence',
            'total_mentions',
            'sentiment',
            'social_momentum_score'
        ]
        
        # Filter out features that don't exist
        available_features = [f for f in features if f in data.columns]
        
        # Calculate correlation matrix
        corr_matrix = data[available_features].corr()
        
        return corr_matrix
    
    def plot_correlation_heatmap(self, symbol: str, save_path: Optional[str] = None):
        """Plot correlation heatmap for a symbol"""
        corr_matrix = self.get_correlation_matrix(symbol)
        
        if corr_matrix.empty:
            print(f"No data available for correlation matrix of {symbol}")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": .8})
        
        plt.title(f'{symbol} Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_summary_report(self, output_file: str = "crypto_data_summary.txt"):
        """Generate a comprehensive summary report of the integrated data"""
        with open(output_file, 'w') as f:
            f.write("CRYPTO DATA INTEGRATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Get tables
            tables = self.get_table_info()
            f.write("DATABASE TABLES\n")
            f.write("-" * 30 + "\n")
            for table in tables['name']:
                f.write(f"  - {table}\n")
            f.write("\n")
            
            # Date range
            try:
                min_date, max_date = self.get_date_range()
                f.write(f"Data Range: {min_date} to {max_date}\n")
                f.write(f"Duration: {(max_date - min_date).days} days\n\n")
            except:
                f.write("Could not determine date range\n\n")
            
            # Symbols
            try:
                symbols = self.get_symbols()
                f.write(f"Symbols in database: {', '.join(symbols[:20])}")
                if len(symbols) > 20:
                    f.write(f"... and {len(symbols) - 20} more")
                f.write(f"\nTotal symbols: {len(symbols)}\n\n")
            except:
                f.write("Could not get symbols\n\n")
            
            # Whale activity summary
            try:
                f.write("WHALE ACTIVITY SUMMARY\n")
                f.write("-" * 30 + "\n")
                whale_summary = self.get_whale_activity_summary()
                if not whale_summary.empty:
                    f.write(whale_summary.to_string(index=False))
                else:
                    f.write("No whale activity data available")
                f.write("\n\n")
            except Exception as e:
                f.write(f"Could not generate whale summary: {e}\n\n")
            
            # Social sentiment summary
            try:
                f.write("SOCIAL SENTIMENT SUMMARY\n")
                f.write("-" * 30 + "\n")
                social_summary = self.get_social_sentiment_summary()
                if not social_summary.empty:
                    f.write(social_summary.to_string(index=False))
                else:
                    f.write("No social sentiment data available")
                f.write("\n\n")
            except Exception as e:
                f.write(f"Could not generate social summary: {e}\n\n")
            
            # Correlation analysis for major symbols
            try:
                f.write("PRICE CORRELATION ANALYSIS\n")
                f.write("-" * 30 + "\n")
                for symbol in ['BTC', 'ETH', 'SOL']:
                    if symbol in symbols:
                        f.write(f"\n{symbol}:\n")
                        corr_matrix = self.get_correlation_matrix(symbol)
                        if not corr_matrix.empty and 'price_return' in corr_matrix.columns:
                            price_corr = corr_matrix['price_return'].sort_values(ascending=False)
                            f.write("Top correlations with price returns:\n")
                            for feature, corr in price_corr.items():
                                if feature != 'price_return' and not pd.isna(corr):
                                    f.write(f"  {feature}: {corr:.3f}\n")
                        else:
                            f.write("  No correlation data available\n")
            except Exception as e:
                f.write(f"\nCould not generate correlation analysis: {e}\n")
            
            f.write(f"\n\nReport generated on: {datetime.now()}\n")
        
        print(f"Summary report saved to: {output_file}")
    
    def get_top_whale_tokens(self, limit: int = 10) -> pd.DataFrame:
        """Get top tokens by whale activity"""
        query = f"""
        SELECT 
            token as symbol,
            COUNT(*) as transaction_count,
            SUM(amount_usd) as total_volume,
            AVG(amount_usd) as avg_transaction_size,
            SUM(CASE WHEN classification = 'Institutional' THEN 1 ELSE 0 END) as institutional_txns,
            SUM(CASE WHEN classification = 'Retail' THEN 1 ELSE 0 END) as retail_txns
        FROM whale_transactions
        GROUP BY token
        ORDER BY total_volume DESC
        LIMIT {limit}
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_hourly_whale_activity(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get whale activity by hour of day"""
        where_clause = f"WHERE token = '{symbol}'" if symbol else ""
        
        query = f"""
        SELECT 
            CAST(strftime('%H', timestamp) AS INTEGER) as hour,
            COUNT(*) as transaction_count,
            SUM(amount_usd) as total_volume,
            AVG(amount_usd) as avg_size
        FROM whale_transactions
        {where_clause}
        GROUP BY hour
        ORDER BY hour
        """
        return pd.read_sql_query(query, self.conn)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    with CryptoDataAnalyzer() as analyzer:
        # Get basic information
        print("DATABASE OVERVIEW")
        print("=" * 50)
        
        # Get tables
        tables = analyzer.get_table_info()
        print(f"Tables: {tables['name'].tolist()}")
        
        # Get date range
        try:
            min_date, max_date = analyzer.get_date_range()
            print(f"Date range: {min_date} to {max_date}")
        except:
            print("Could not get date range")
        
        # Get symbols
        try:
            symbols = analyzer.get_symbols()
            print(f"Symbols: {symbols[:10]}...")
            print(f"Total symbols: {len(symbols)}")
        except:
            print("Could not get symbols")
        
        # Get top whale tokens
        print("\nTOP WHALE TOKENS")
        print("=" * 50)
        top_tokens = analyzer.get_top_whale_tokens()
        print(top_tokens)
        
        # Analyze specific symbol
        symbol = 'ETH'
        if symbol in symbols:
            print(f"\n{symbol} ANALYSIS")
            print("=" * 50)
            
            # Whale patterns
            patterns = analyzer.analyze_whale_patterns(symbol)
            if 'error' not in patterns:
                print(f"Total transactions: {patterns['total_transactions']}")
                print(f"Total volume: ${patterns['total_volume']:,.2f}")
                print(f"Institutional %: {patterns['institutional_percentage']:.1f}%")
            
            # Create visualizations
            analyzer.plot_whale_vs_price(symbol, save_path=f'{symbol.lower()}_whale_analysis.png')
            analyzer.plot_correlation_heatmap(symbol, save_path=f'{symbol.lower()}_correlation.png')
            print(f"Visualizations saved for {symbol}")
        
        # Generate summary report
        analyzer.generate_summary_report()
        print("\nSummary report generated!")