"""
Enhanced Feature engineering with deadband targets and improved whale features
"""
import pandas as pd
import numpy as np
import gc
from typing import Dict, List, Optional
from tqdm import tqdm

from config import *
from utils import optimize_memory_usage, save_cache_data, load_cache_data, bool_to_int8

class FeatureEngineer:
    def __init__(self):
        self.fill_values = {}
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def create_advanced_features(self, df: pd.DataFrame, 
                                fill_values: Optional[Dict] = None, 
                                use_cache: bool = True,
                                include_whale_data: bool = True,
                                include_social_data: bool = True) -> pd.DataFrame:
        """Create advanced features with caching - Enhanced with better whale features"""
        
        # Determine cache suffix based on included features
        cache_suffix = ""
        if include_whale_data:
            cache_suffix += "_whale"
        if include_social_data:
            cache_suffix += "_social"
            
        # Try to load cached features first
        if use_cache:
            cached_features = load_cache_data(CACHE_DIR, f"features_latest{cache_suffix}.parquet")
            if cached_features is not None:
                user_input = input(f"\nUse cached features{cache_suffix}? (y/n): ").lower()
                if user_input == 'y':
                    print("Using cached features!")
                    return cached_features
                else:
                    print("Recalculating features...")
        
        print("\nCreating advanced features...")
        
        # Check if whale and social data are present
        whale_cols = [col for col in df.columns if 'whale_' in col or 'inst_' in col or 'retail_' in col]
        social_cols = [col for col in df.columns if 'mention' in col or 'sentiment' in col or 'social' in col]
        
        has_whale_data = len(whale_cols) > 0
        has_social_data = len(social_cols) > 0
        
        if include_whale_data and not has_whale_data:
            print("Warning: Whale data requested but not found in dataframe")
            include_whale_data = False
        
        if include_social_data and not has_social_data:
            print("Warning: Social data requested but not found in dataframe")
            include_social_data = False
        
        features_list = []
        
        for symbol in tqdm(df['symbol'].unique(), desc="Advanced feature engineering"):
            sd = df[df['symbol'] == symbol].copy().sort_values('time_bucket').reset_index(drop=True)
            
            # Price selection logic
            if 'mid_price_last' in sd.columns:
                price_col = 'mid_price_last'
            elif 'price_last' in sd.columns:
                price_col = 'price_last'
            elif 'vwap' in sd.columns:
                price_col = 'vwap'
            else:
                price_cols = [col for col in sd.columns if 'price' in col and pd.api.types.is_numeric_dtype(sd[col])]
                if price_cols:
                    price_col = price_cols[0]
                else:
                    print(f"Warning: No price column found for {symbol}")
                    continue
            
            sd['price'] = sd[price_col]
            
            # Create all feature categories
            self._create_orderbook_features(sd)
            self._create_orderflow_features(sd)
            self._create_momentum_features(sd)
            self._create_volatility_features(sd)
            self._create_microstructure_features(sd)
            self._create_composite_indicators(sd)
            self._create_technical_indicators(sd)
            self._create_time_features(sd)
            self._create_interaction_features(sd)
            
            # Enhanced whale and social features
            if include_whale_data and has_whale_data:
                self._create_enhanced_whale_features(sd)
            
            if include_social_data and has_social_data:
                self._create_enhanced_social_features(sd)
            
            # Cleanup features
            sd = self._cleanup_features(sd, symbol, fill_values)
            
            features_list.append(sd)
            del sd
            gc.collect()
        
        result = pd.concat(features_list, ignore_index=True)
        result = optimize_memory_usage(result, verbose=False)
        
        print(f"Created {len(result.columns)} features")
        print(f"Feature DataFrame memory usage: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Save features cache
        if use_cache:
            save_cache_data(result, CACHE_DIR, f"features_latest{cache_suffix}.parquet")
        
        return result
    
    def _create_enhanced_whale_features(self, sd: pd.DataFrame):
        """Enhanced whale features with better predictive power"""
        
        # 1. WHALE FLOW MOMENTUM WITH ACCELERATION
        if 'whale_flow_imbalance' in sd.columns:
            for window in [6, 12, 24]:
                sd[f'whale_flow_ma_{window}'] = sd['whale_flow_imbalance'].rolling(window).mean()
                sd[f'whale_flow_momentum_{window}'] = sd['whale_flow_imbalance'].diff(window)
                sd[f'whale_flow_acceleration_{window}'] = sd[f'whale_flow_momentum_{window}'].diff(window//2 if window//2 > 0 else 1)
                
                # Z-score normalization
                whale_mean = sd[f'whale_flow_ma_{window}'].rolling(48).mean()
                whale_std = sd[f'whale_flow_ma_{window}'].rolling(48).std()
                sd[f'whale_flow_zscore_{window}'] = (sd['whale_flow_imbalance'] - whale_mean) / (whale_std + 1e-8)
        
        # 2. INSTITUTIONAL PARTICIPATION DYNAMICS
        if 'inst_participation_rate' in sd.columns:
            # Multi-scale z-scores
            for window in [12, 24, 48]:
                inst_mean = sd['inst_participation_rate'].rolling(window).mean()
                inst_std = sd['inst_participation_rate'].rolling(window).std()
                sd[f'inst_participation_zscore_{window}'] = (
                    (sd['inst_participation_rate'] - inst_mean) / (inst_std + 1e-8)
                )
            
            # Rate of change
            sd['inst_participation_change_12'] = sd['inst_participation_rate'].diff(12)
            sd['inst_participation_acceleration'] = sd['inst_participation_change_12'].diff(6)
            
            # Extreme institutional activity
            inst_95 = sd['inst_participation_rate'].rolling(96).quantile(0.95)
            sd['extreme_inst_activity'] = bool_to_int8(sd['inst_participation_rate'] > inst_95)
        
        # 3. RETAIL PANIC INDICATORS
        if 'retail_sell_pressure' in sd.columns:
            # Multi-timeframe retail behavior
            for window in [6, 12, 24]:
                sd[f'retail_sell_ma_{window}'] = sd['retail_sell_pressure'].rolling(window).mean()
                sd[f'retail_sell_momentum_{window}'] = sd['retail_sell_pressure'].diff(window)
            
            # Retail capitulation with price context
            price_drop_12 = sd['return_12'] < -0.02  # 2% drop
            retail_spike = sd['retail_sell_pressure'] > sd['retail_sell_pressure'].rolling(48).quantile(0.9)
            sd['retail_capitulation_signal'] = bool_to_int8(price_drop_12 & retail_spike)
            
            # Retail divergence from institutions
            if 'inst_participation_rate' in sd.columns:
                sd['retail_inst_divergence'] = (
                    sd['retail_sell_pressure'] - (1 - sd['inst_participation_rate'])
                )
        
        # 4. SMART MONEY DIVERGENCE PATTERNS
        if 'smart_dumb_divergence' in sd.columns:
            # Enhanced smart money signals with momentum
            sd['smart_money_strength'] = sd['smart_dumb_divergence'].rolling(12).mean()
            sd['smart_money_momentum'] = sd['smart_dumb_divergence'].diff(6)
            sd['smart_money_acceleration'] = sd['smart_money_momentum'].diff(3)
            
            # Smart accumulation during weakness
            price_weakness = sd['return_24'] < -0.01
            smart_positive = sd['smart_dumb_divergence'] > sd['smart_dumb_divergence'].rolling(48).quantile(0.7)
            sd['smart_accumulation_signal'] = bool_to_int8(price_weakness & smart_positive)
            
            # Smart distribution during strength
            price_strength = sd['return_24'] > 0.01
            smart_negative = sd['smart_dumb_divergence'] < sd['smart_dumb_divergence'].rolling(48).quantile(0.3)
            sd['smart_distribution_signal'] = bool_to_int8(price_strength & smart_negative)
        
        # 5. WHALE SIZE AND VOLUME METRICS
        if 'whale_amount_usd_mean' in sd.columns:
            # Transaction size dynamics
            sd['avg_whale_size_ma_24'] = sd['whale_amount_usd_mean'].rolling(24).mean()
            sd['whale_size_ratio'] = sd['whale_amount_usd_mean'] / (sd['avg_whale_size_ma_24'] + 1e-8)
            
            # Large whale detection
            if 'whale_amount_usd_max' in sd.columns:
                whale_99 = sd['whale_amount_usd_max'].rolling(192).quantile(0.99)
                sd['mega_whale_activity'] = bool_to_int8(sd['whale_amount_usd_max'] > whale_99)
                
                # Whale size momentum
                sd['whale_size_momentum'] = sd['whale_amount_usd_mean'].pct_change(12)
        
        # 6. WHALE-MARKET ALIGNMENT
        if 'whale_flow_imbalance' in sd.columns and 'order_flow_imbalance' in sd.columns:
            # Correlation between whale and regular flow
            sd['whale_market_alignment'] = sd['whale_flow_imbalance'] * sd['order_flow_imbalance']
            
            # Whale leading market
            whale_lead = sd['whale_flow_imbalance'].rolling(6).mean()
            market_lag = sd['order_flow_imbalance'].shift(-6).rolling(6).mean()
            sd['whale_leads_market'] = whale_lead * market_lag
            
            # Whale contra-market positioning
            sd['whale_contra_market'] = bool_to_int8(
                (sd['whale_flow_imbalance'] > 0) & (sd['order_flow_imbalance'] < 0)
            )
        
        # 7. WHALE PERSISTENCE AND STREAKS
        if 'whale_buy_count' in sd.columns and 'whale_sell_count' in sd.columns:
            # Net whale direction
            sd['whale_net_direction'] = np.sign(sd['whale_buy_count'] - sd['whale_sell_count'])
            
            # Calculate consecutive streaks
            direction_change = sd['whale_net_direction'] != sd['whale_net_direction'].shift(1)
            streak_groups = direction_change.cumsum()
            sd['whale_streak_length'] = sd.groupby(streak_groups).cumcount() + 1
            sd['whale_streak_strength'] = sd['whale_streak_length'] * sd['whale_net_direction']
        
        # 8. COMPOSITE WHALE SENTIMENT SCORE
        whale_features_for_composite = []
        
        if 'whale_flow_imbalance' in sd.columns:
            whale_features_for_composite.append(sd['whale_flow_imbalance'])
        
        if 'inst_participation_zscore_12' in sd.columns:
            whale_features_for_composite.append(sd['inst_participation_zscore_12'])
        
        if 'smart_dumb_divergence' in sd.columns:
            whale_features_for_composite.append(sd['smart_dumb_divergence'])
        
        if 'whale_size_ratio' in sd.columns:
            centered_size_ratio = sd['whale_size_ratio'] - 1  # Center around 1
            whale_features_for_composite.append(centered_size_ratio)
        
        if whale_features_for_composite:
            # Normalize each component
            normalized_components = []
            for component in whale_features_for_composite:
                comp_mean = component.rolling(96).mean()
                comp_std = component.rolling(96).std()
                normalized = (component - comp_mean) / (comp_std + 1e-8)
                normalized_components.append(normalized)
            
            # Weighted composite with emphasis on key signals
            weights = [3.0, 2.5, 2.0, 1.5][:len(normalized_components)]
            weighted_sum = sum(w * c for w, c in zip(weights, normalized_components))
            sd['whale_sentiment_composite'] = weighted_sum / sum(weights)
            
            # Extreme whale sentiment
            composite_95 = sd['whale_sentiment_composite'].rolling(192).quantile(0.95)
            composite_5 = sd['whale_sentiment_composite'].rolling(192).quantile(0.05)
            sd['whale_extreme_bullish'] = bool_to_int8(sd['whale_sentiment_composite'] > composite_95)
            sd['whale_extreme_bearish'] = bool_to_int8(sd['whale_sentiment_composite'] < composite_5)
    
    def _create_enhanced_social_features(self, sd: pd.DataFrame):
        """Enhanced social features with better predictive signals"""
        
        # 1. SOCIAL MOMENTUM WITH MULTI-SCALE
        if 'social_momentum_score' in sd.columns:
            for window in [6, 12, 24, 48]:
                sd[f'social_momentum_ma_{window}'] = sd['social_momentum_score'].rolling(window).mean()
                sd[f'social_momentum_change_{window}'] = sd['social_momentum_score'].diff(window)
                
                # Z-score normalization
                social_mean = sd[f'social_momentum_ma_{window}'].rolling(96).mean()
                social_std = sd[f'social_momentum_ma_{window}'].rolling(96).std()
                sd[f'social_momentum_zscore_{window}'] = (
                    (sd['social_momentum_score'] - social_mean) / (social_std + 1e-8)
                )
            
            # Social acceleration
            sd['social_acceleration'] = sd['social_momentum_change_12'].diff(6)
            
            # Extreme social activity
            momentum_99 = sd['social_momentum_score'].rolling(192).quantile(0.99)
            sd['social_fomo_indicator'] = bool_to_int8(sd['social_momentum_score'] > momentum_99)
        
        # 2. MENTION VELOCITY AND VIRAL DETECTION
        if 'total_mentions' in sd.columns:
            # Mention growth metrics
            sd['mention_velocity'] = sd['total_mentions'].pct_change(12)
            sd['mention_acceleration'] = sd['mention_velocity'].diff(6)
            
            # Viral detection with persistence
            viral_threshold = sd['total_mentions'].rolling(192).quantile(0.95)
            viral_condition = sd['total_mentions'] > viral_threshold
            sd['viral_indicator'] = bool_to_int8(viral_condition)
            sd['viral_persistence'] = viral_condition.rolling(12).sum() / 12
            
            # Abnormal mention spikes
            mention_mean = sd['total_mentions'].rolling(48).mean()
            mention_std = sd['total_mentions'].rolling(48).std()
            sd['mention_spike_zscore'] = (sd['total_mentions'] - mention_mean) / (mention_std + 1e-8)
        
        # 3. SENTIMENT DYNAMICS WITH REGIME DETECTION
        if 'sentiment_14d' in sd.columns:
            # Sentiment momentum at multiple scales
            for window in [6, 12, 24]:
                sd[f'sentiment_momentum_{window}'] = sd['sentiment_14d'].diff(window)
                sd[f'sentiment_ma_{window}'] = sd['sentiment_14d'].rolling(window).mean()
            
            # Sentiment volatility
            sd['sentiment_volatility'] = sd['sentiment_14d'].rolling(24).std()
            
            # Sentiment regime classification
            sentiment_ma = sd['sentiment_14d'].rolling(48).mean()
            sd['bullish_sentiment_regime'] = bool_to_int8(sentiment_ma > SENTIMENT_THRESHOLDS['positive'])
            sd['bearish_sentiment_regime'] = bool_to_int8(sentiment_ma < SENTIMENT_THRESHOLDS['neutral'])
            
            # Sentiment extremes with persistence
            if 'sentiment_very_positive' in sd.columns and 'sentiment_very_negative' in sd.columns:
                sd['persistent_euphoria'] = bool_to_int8(
                    sd['sentiment_very_positive'].rolling(12).sum() >= 8
                )
                sd['persistent_despair'] = bool_to_int8(
                    sd['sentiment_very_negative'].rolling(12).sum() >= 8
                )
        
        # 4. SOCIAL-PRICE DIVERGENCE PATTERNS
        if 'social_momentum_score' in sd.columns and 'return_12' in sd.columns:
            # Normalized divergence
            price_zscore = (sd['return_12'] - sd['return_12'].rolling(48).mean()) / (
                sd['return_12'].rolling(48).std() + 1e-8
            )
            social_zscore = sd.get('social_momentum_zscore_12', sd['social_momentum_score'])
            
            sd['social_price_divergence_score'] = social_zscore - price_zscore
            
            # Strong divergence signals
            sd['strong_bullish_divergence'] = bool_to_int8(
                (price_zscore < -1) & (social_zscore > 1)
            )
            sd['strong_bearish_divergence'] = bool_to_int8(
                (price_zscore > 1) & (social_zscore < -1)
            )
        
        # 5. COMPOSITE SOCIAL SENTIMENT
        social_components = []
        
        if 'social_momentum_zscore_12' in sd.columns:
            social_components.append(sd['social_momentum_zscore_12'])
        elif 'social_momentum_score' in sd.columns:
            social_components.append(sd['social_momentum_score'])
        
        if 'sentiment_14d' in sd.columns:
            # Normalize sentiment to [-1, 1] range
            normalized_sentiment = sd['sentiment_14d'] * 2  # Assuming sentiment is in [-0.5, 0.5]
            social_components.append(normalized_sentiment)
        
        if 'mention_spike_zscore' in sd.columns:
            social_components.append(sd['mention_spike_zscore'] * 0.5)  # Lower weight for spikes
        
        if social_components:
            # Create weighted composite
            weights = [2.0, 2.5, 1.0][:len(social_components)]
            weighted_social = sum(w * c for w, c in zip(weights, social_components))
            sd['social_sentiment_composite'] = weighted_social / sum(weights)
            
            # Social extremes
            composite_95 = sd['social_sentiment_composite'].rolling(192).quantile(0.95)
            composite_5 = sd['social_sentiment_composite'].rolling(192).quantile(0.05)
            sd['social_euphoria'] = bool_to_int8(sd['social_sentiment_composite'] > composite_95)
            sd['social_despair'] = bool_to_int8(sd['social_sentiment_composite'] < composite_5)
    
    def _create_orderbook_features(self, sd: pd.DataFrame):
        """Create order book related features"""
        # Microprice (Weighted Mid-Price)
        if 'bid_size_1_mean' in sd.columns and 'ask_size_1_mean' in sd.columns:
            if 'bid_price_1_mean' in sd.columns and 'ask_price_1_mean' in sd.columns:
                sd['microprice'] = (
                    (sd['bid_price_1_mean'] * sd['ask_size_1_mean'] + sd['ask_price_1_mean'] * sd['bid_size_1_mean']) /
                    (sd['bid_size_1_mean'] + sd['ask_size_1_mean'] + 1e-8)
                )
                
                # Microprice momentum
                sd['microprice_momentum'] = sd['microprice'].diff(1)
                sd['microprice_acceleration'] = sd['microprice_momentum'].diff(1)
        
        # Multi-level Order Book Imbalances
        for level in [1, 3, 5]:
            bid_cols = [f'bid_size_{i}_mean' for i in range(1, level+1)]
            ask_cols = [f'ask_size_{i}_mean' for i in range(1, level+1)]
            
            if all(col in sd.columns for col in bid_cols + ask_cols):
                # Top N level imbalance
                bid_sum = sum(sd[col] for col in bid_cols)
                ask_sum = sum(sd[col] for col in ask_cols)
                sd[f'imbalance_top{level}'] = (
                    (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-8)
                )
                
                # Z-score of imbalance
                for window in [12, 24, 48]:
                    rolling_mean = sd[f'imbalance_top{level}'].rolling(window).mean()
                    rolling_std = sd[f'imbalance_top{level}'].rolling(window).std()
                    sd[f'imbalance_top{level}_zscore_{window}'] = (
                        (sd[f'imbalance_top{level}'] - rolling_mean) / (rolling_std + 1e-8)
                    )
        
        # Order Book Slope/Gradient
        if all(f'bid_price_{i}_mean' in sd.columns for i in range(1, 6)):
            if all(f'bid_size_{i}_mean' in sd.columns for i in range(1, 6)):
                sd['bid_depth_slope'] = (
                    (sd['bid_size_5_mean'] - sd['bid_size_1_mean']) /
                    (sd['bid_price_1_mean'] - sd['bid_price_5_mean'] + 1e-8)
                )
                
                if all(f'ask_size_{i}_mean' in sd.columns for i in range(1, 6)):
                    sd['ask_depth_slope'] = (
                        (sd['ask_size_5_mean'] - sd['ask_size_1_mean']) /
                        (sd['ask_price_5_mean'] - sd['ask_price_1_mean'] + 1e-8)
                    )
                    
                    sd['depth_slope_ratio'] = (
                        sd['bid_depth_slope'] / (sd['ask_depth_slope'] + 1e-8)
                    )
        
        # Book Convexity/Concentration
        for side in ['bid', 'ask']:
            if f'{side}_size_1_mean' in sd.columns and f'{side}_size_5_mean' in sd.columns:
                sd[f'{side}_convexity'] = (
                    sd[f'{side}_size_1_mean'] / (sd[f'{side}_size_5_mean'] + 1e-8)
                )
        
        # Book Pressure Pivot
        if 'imbalance_top3' in sd.columns:
            price_direction = np.sign(sd['price'].diff(1))
            sd['book_pressure_pivot'] = sd['imbalance_top3'] * price_direction
    
    def _create_orderflow_features(self, sd: pd.DataFrame):
        """Create order flow and trade flow features"""
        # Enhanced CVD with multiple windows
        if 'cvd_last' in sd.columns:
            for window in [6, 12, 24, 48]:
                sd[f'cvd_ma_{window}'] = sd['cvd_last'].rolling(window).mean()
                sd[f'cvd_momentum_{window}'] = sd['cvd_last'].diff(window)
                
                # CVD acceleration
                sd[f'cvd_acceleration_{window}'] = (
                    sd[f'cvd_momentum_{window}'].diff(window//2 if window//2 > 0 else 1)
                )
        
        # Trade Count Imbalance
        if 'quantity_count' in sd.columns and 'order_flow_imbalance' in sd.columns:
            total_trades = sd['quantity_count']
            sd['buy_trade_count'] = total_trades * (1 + sd['order_flow_imbalance']) / 2
            sd['sell_trade_count'] = total_trades * (1 - sd['order_flow_imbalance']) / 2
            
            for window in [12, 24]:
                buy_sum = sd['buy_trade_count'].rolling(window).sum()
                sell_sum = sd['sell_trade_count'].rolling(window).sum()
                sd[f'trade_count_imbalance_{window}'] = (
                    (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-8)
                )
        
        # Large Trade Detection
        if 'quantity_mean' in sd.columns and 'quantity_std' in sd.columns:
            large_trade_threshold = sd['quantity_mean'] + 2 * sd['quantity_std']
            sd['large_trade_ratio'] = sd['quantity_max'] / (large_trade_threshold + 1e-8)
        
        # VWAP Deviation Features
        if 'vwap' in sd.columns:
            for window in [6, 12, 24]:
                sd[f'vwap_ma_{window}'] = sd['vwap'].rolling(window).mean()
                sd[f'vwap_deviation_{window}'] = (
                    (sd['price'] - sd[f'vwap_ma_{window}']) / (sd['price'] + 1e-8)
                )
    
    def _create_momentum_features(self, sd: pd.DataFrame):
        """Create momentum and directional features"""
        # Price momentum at multiple scales
        for period in PRICE_RETURN_PERIODS:
            sd[f'return_{period}'] = sd['price'].pct_change(period)
            sd[f'log_return_{period}'] = np.log(sd['price'] / sd['price'].shift(period))
            
            # Momentum of momentum (acceleration)
            if period <= 24:
                sd[f'return_acceleration_{period}'] = (
                    sd[f'return_{period}'].diff(period//2 if period//2 > 0 else 1)
                )
        
        # Directional Price Imbalance (Tick direction)
        sd['price_tick'] = np.sign(sd['price'].diff(1))
        for window in [12, 24, 48]:
            sd[f'tick_momentum_{window}'] = sd['price_tick'].rolling(window).sum()
            sd[f'tick_momentum_norm_{window}'] = sd[f'tick_momentum_{window}'] / window
        
        # Price Position Features
        for window in [12, 24, 48, 96]:
            rolling_max = sd['price'].rolling(window).max()
            rolling_min = sd['price'].rolling(window).min()
            rolling_median = sd['price'].rolling(window).median()
            
            # Position in range
            sd[f'price_position_{window}'] = (
                (sd['price'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
            )
            
            # Distance from median
            sd[f'price_median_distance_{window}'] = (
                (sd['price'] - rolling_median) / (rolling_median + 1e-8)
            )
            
            # Is near high/low
            sd[f'near_high_{window}'] = bool_to_int8(sd[f'price_position_{window}'] > 0.8)
            sd[f'near_low_{window}'] = bool_to_int8(sd[f'price_position_{window}'] < 0.2)
    
    def _create_volatility_features(self, sd: pd.DataFrame):
        """Create volatility and risk features"""
        # Realized volatility at multiple frequencies
        for window in ROLLING_WINDOWS:
            sd[f'volatility_{window}'] = sd['return_1'].rolling(window).std()
            sd[f'volatility_ma_{window}'] = sd[f'volatility_{window}'].rolling(window).mean()
            
            # Volatility of volatility
            sd[f'vol_of_vol_{window}'] = sd[f'volatility_{window}'].rolling(window).std()
            
            # Skew and kurtosis
            skew_vals = sd['return_1'].rolling(window).skew()
            kurt_vals = sd['return_1'].rolling(window).kurt()
            sd[f'skew_{window}'] = skew_vals
            sd[f'kurt_{window}'] = kurt_vals
        
        # Parkinson volatility
        if 'bid_price_1_mean' in sd.columns and 'ask_price_1_mean' in sd.columns:
            for window in [12, 24]:
                high = sd['ask_price_1_mean'].rolling(window).max()
                low = sd['bid_price_1_mean'].rolling(window).min()
                sd[f'parkinson_vol_{window}'] = np.sqrt(
                    1/(4*np.log(2)) * np.log((high + 1e-8)/(low + 1e-8))**2
                )
    
    def _create_microstructure_features(self, sd: pd.DataFrame):
        """Create microstructure features"""
        # Spread features
        if 'spread_mean' in sd.columns:
            sd['relative_spread'] = sd['spread_mean'] / (sd['price'] + 1e-8)
            sd['spread_volatility'] = sd['spread_mean'].rolling(12).std()
            sd['spread_momentum'] = sd['spread_mean'].diff(6)
            
            # High spread indicator
            spread_threshold = sd['spread_mean'].rolling(48).mean() + 2*sd['spread_mean'].rolling(48).std()
            sd['high_spread_flag'] = bool_to_int8(sd['spread_mean'] > spread_threshold)
        
        # Quoted Volume Ratio
        if 'bid_size_1_mean' in sd.columns and 'ask_size_1_mean' in sd.columns:
            sd['quoted_volume_ratio'] = sd['bid_size_1_mean'] / (sd['ask_size_1_mean'] + 1e-8)
            sd['log_quoted_volume_ratio'] = np.log(sd['quoted_volume_ratio'] + 1e-8)
        
        # Order book imbalance features
        if 'book_imbalance_mean' in sd.columns:
            sd['imbalance_ma_12'] = sd['book_imbalance_mean'].rolling(12).mean()
            sd['imbalance_std_12'] = sd['book_imbalance_mean'].rolling(12).std()
            sd['imbalance_momentum'] = sd['book_imbalance_mean'].diff(6)
        
        # Order flow features
        if 'order_flow_imbalance' in sd.columns:
            sd['ofi_ma_12'] = sd['order_flow_imbalance'].rolling(12).mean()
            sd['ofi_momentum'] = sd['order_flow_imbalance'].diff(6)
            sd['ofi_acceleration'] = sd['ofi_momentum'].diff(3)
        
        # Volume features
        if 'quantity_sum' in sd.columns:
            sd['volume'] = sd['quantity_sum']
            sd['volume_ma_12'] = sd['volume'].rolling(12).mean()
            sd['volume_ratio'] = sd['volume'] / (sd['volume_ma_12'] + 1e-8)
            sd['volume_momentum'] = sd['volume'].pct_change(12)
            
            # VPIN approximation
            if 'buy_volume_sum' in sd.columns and 'sell_volume_sum' in sd.columns:
                sd['vpin'] = np.abs(
                    sd['buy_volume_sum'].rolling(12).sum() - sd['sell_volume_sum'].rolling(12).sum()
                ) / (sd['volume'].rolling(12).sum() + 1e-8)
    
    def _create_composite_indicators(self, sd: pd.DataFrame):
        """Create composite indicators"""
        # Composite Momentum Score
        momentum_features = []
        if 'return_12' in sd.columns:
            momentum_features.append(sd['return_12'].rolling(12).mean())
        if 'cvd_momentum_12' in sd.columns:
            momentum_features.append(sd['cvd_momentum_12'])
        if 'tick_momentum_norm_12' in sd.columns:
            momentum_features.append(sd['tick_momentum_norm_12'])
        
        if momentum_features:
            # Normalize and combine
            for i, feat in enumerate(momentum_features):
                feat_mean = feat.rolling(48).mean()
                feat_std = feat.rolling(48).std()
                feat_norm = (feat - feat_mean) / (feat_std + 1e-8)
                momentum_features[i] = feat_norm
            
            sd['composite_momentum'] = sum(momentum_features) / len(momentum_features)
        
        # Composite Pressure Score
        pressure_features = []
        if 'imbalance_top3' in sd.columns:
            pressure_features.append(sd['imbalance_top3'])
        if 'order_flow_imbalance' in sd.columns:
            pressure_features.append(sd['order_flow_imbalance'])
        if 'quoted_volume_ratio' in sd.columns:
            pressure_features.append(np.log(sd['quoted_volume_ratio'] + 1e-8))
        
        if pressure_features:
            sd['composite_pressure'] = sum(pressure_features) / len(pressure_features)
    
    def _create_technical_indicators(self, sd: pd.DataFrame):
        """Create enhanced technical indicators"""
        # RSI variations
        for period in RSI_PERIODS:
            sd[f'rsi_{period}'] = self.calculate_rsi(sd['price'], period)
            sd[f'rsi_{period}_momentum'] = sd[f'rsi_{period}'].diff(3)
            # FIXED with safe conversion
            sd[f'rsi_{period}_oversold'] = bool_to_int8(sd[f'rsi_{period}'] < 30)
            sd[f'rsi_{period}_overbought'] = bool_to_int8(sd[f'rsi_{period}'] > 70)
        
        # Enhanced Bollinger Bands
        for window in BOLLINGER_WINDOWS:
            ma = sd['price'].rolling(window).mean()
            std = sd['price'].rolling(window).std()
            
            # Multiple standard deviations
            for n_std in [1.5, 2, 2.5]:
                sd[f'bb_upper_{window}_{n_std}'] = ma + n_std * std
                sd[f'bb_lower_{window}_{n_std}'] = ma - n_std * std
            
            # BB width (volatility proxy)
            sd[f'bb_width_{window}'] = (
                (sd[f'bb_upper_{window}_2'] - sd[f'bb_lower_{window}_2']) / (ma + 1e-8)
            )
            
            # BB position
            sd[f'bb_position_{window}'] = (
                (sd['price'] - sd[f'bb_lower_{window}_2']) /
                (sd[f'bb_upper_{window}_2'] - sd[f'bb_lower_{window}_2'] + 1e-8)
            )
            
            # BB squeeze (low volatility)
            bb_width_ma = sd[f'bb_width_{window}'].rolling(window).mean()
            bb_width_std = sd[f'bb_width_{window}'].rolling(window).std()
            sd[f'bb_squeeze_{window}'] = bool_to_int8(
                sd[f'bb_width_{window}'] < (bb_width_ma - bb_width_std)
            )
        
        # MACD variations
        for fast, slow in MACD_PAIRS:
            ema_fast = sd['price'].ewm(span=fast, adjust=False).mean()
            ema_slow = sd['price'].ewm(span=slow, adjust=False).mean()
            
            sd[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
            sd[f'macd_signal_{fast}_{slow}'] = sd[f'macd_{fast}_{slow}'].ewm(span=9, adjust=False).mean()
            sd[f'macd_hist_{fast}_{slow}'] = sd[f'macd_{fast}_{slow}'] - sd[f'macd_signal_{fast}_{slow}']
            
            # MACD cross signals
            macd_above = sd[f'macd_{fast}_{slow}'] > sd[f'macd_signal_{fast}_{slow}']
            macd_above_prev = sd[f'macd_{fast}_{slow}'].shift(1) > sd[f'macd_signal_{fast}_{slow}'].shift(1)
            sd[f'macd_cross_up_{fast}_{slow}'] = bool_to_int8(macd_above & ~macd_above_prev)
        
        # ATR (Average True Range)
        if 'price_max' in sd.columns and 'price_min' in sd.columns:
            high = sd['price_max']
            low = sd['price_min']
            close = sd['price']
            
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            for period in [14, 21]:
                sd[f'atr_{period}'] = tr.rolling(period).mean()
                sd[f'atr_ratio_{period}'] = sd[f'atr_{period}'] / (sd['price'] + 1e-8)
    
    def _create_time_features(self, sd: pd.DataFrame):
        """Create time-based features"""
        sd['hour'] = sd['time_bucket'].dt.hour.astype('int8')
        sd['day_of_week'] = sd['time_bucket'].dt.dayofweek.astype('int8')
        sd['is_weekend'] = bool_to_int8(sd['day_of_week'].isin([5, 6]))
        sd['hour_sin'] = np.sin(2 * np.pi * sd['hour'] / 24)
        sd['hour_cos'] = np.cos(2 * np.pi * sd['hour'] / 24)
        sd['dow_sin'] = np.sin(2 * np.pi * sd['day_of_week'] / 7)
        sd['dow_cos'] = np.cos(2 * np.pi * sd['day_of_week'] / 7)
        
        # Trading sessions
        sd['is_asian_session'] = bool_to_int8(sd['hour'].between(0, 9))
        sd['is_european_session'] = bool_to_int8(sd['hour'].between(7, 16))
        sd['is_us_session'] = bool_to_int8(sd['hour'].between(12, 21))
        sd['is_late_us_session'] = bool_to_int8(sd['hour'].between(20, 23))
        
        # Session transitions
        session_diff = (
            sd['is_asian_session'].diff().abs() +
            sd['is_european_session'].diff().abs() +
            sd['is_us_session'].diff().abs()
        )
        sd['session_transition'] = bool_to_int8(session_diff.clip(0, 1))
    
    def _create_interaction_features(self, sd: pd.DataFrame):
        """Create interaction features"""
        # Momentum × Volume
        if 'return_12' in sd.columns and 'volume_ratio' in sd.columns:
            sd['momentum_volume_interaction'] = sd['return_12'] * sd['volume_ratio']
        
        # Imbalance × Volatility
        if 'book_imbalance_mean' in sd.columns and 'volatility_12' in sd.columns:
            sd['imbalance_volatility_interaction'] = sd['book_imbalance_mean'] * sd['volatility_12']
        
        # Spread × Volume (liquidity pressure)
        if 'relative_spread' in sd.columns and 'volume' in sd.columns:
            volume_mean = sd['volume'].rolling(48).mean()
            volume_std = sd['volume'].rolling(48).std()
            volume_norm = (sd['volume'] - volume_mean) / (volume_std + 1e-8)
            sd['spread_volume_pressure'] = sd['relative_spread'] * volume_norm
        
        # Order flow × Price momentum
        if 'order_flow_imbalance' in sd.columns and 'return_6' in sd.columns:
            sd['ofi_momentum_interaction'] = sd['order_flow_imbalance'] * sd['return_6']
    
    def _cleanup_features(self, sd: pd.DataFrame, symbol: str, fill_values: Optional[Dict]) -> pd.DataFrame:
        """Clean up features and handle NaN values properly"""
        # Get numeric columns (exclude identifiers)
        numeric_cols = sd.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['symbol', 'time_bucket']]
        
        # Apply fill values if provided
        if fill_values and symbol in fill_values:
            symbol_fills = fill_values[symbol]
            for col in numeric_cols:
                if col in symbol_fills:
                    sd[col] = sd[col].fillna(symbol_fills[col])
        else:
            # Default filling strategy
            for col in numeric_cols:
                if 'flag' in col or col.startswith('is_') or col.startswith('near_'):
                    # Binary/flag columns -> 0
                    sd[col] = sd[col].fillna(0)
                elif 'return' in col or 'momentum' in col or 'change' in col:
                    # Return/momentum columns -> 0 
                    sd[col] = sd[col].fillna(0)
                else:
                    # Other features -> forward fill then 0
                    sd[col] = sd[col].fillna(method='ffill', limit=3).fillna(0)
        
        # Ensure no infinities
        sd[numeric_cols] = sd[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Final memory optimization
        sd = optimize_memory_usage(sd, verbose=False)
        
        return sd
    
    def prepare_advanced_targets(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """Prepare target variables with deadband for noise reduction"""
        print(f"\nPreparing targets for horizons: {[h*5 for h in horizons]} minutes")
        print(f"Using direction deadband: {DIRECTION_DEADBAND} ({DIRECTION_DEADBAND*100:.2f}%)")
        
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            symbol_data = df.loc[symbol_mask].copy()
            
            # Get price column
            if 'mid_price_last' in symbol_data.columns:
                price_col = 'mid_price_last'
            elif 'price_last' in symbol_data.columns:
                price_col = 'price_last'
            elif 'vwap' in symbol_data.columns:
                price_col = 'vwap'
            else:
                price_cols = [col for col in symbol_data.columns if 'price' in col and 
                            pd.api.types.is_numeric_dtype(symbol_data[col])]
                if price_cols:
                    price_col = price_cols[0]
                else:
                    print(f"Warning: No price column found for {symbol}")
                    continue
            
            # Create targets for each horizon
            for h in horizons:
                # Future return
                future_price = symbol_data[price_col].shift(-h)
                target_return = (future_price - symbol_data[price_col]) / symbol_data[price_col]
                
                # Assign back to main dataframe
                df.loc[symbol_mask, f'target_return_{h}'] = target_return
                
                # Enhanced direction target with deadband
                # 1 = UP (return > deadband), 0 = DOWN (return < -deadband), -1 = FLAT
                direction = np.where(target_return > DIRECTION_DEADBAND, 1,
                                   np.where(target_return < -DIRECTION_DEADBAND, 0, -1))
                df.loc[symbol_mask, f'target_direction_{h}'] = direction
                
                # Confidence level (distance from deadband)
                confidence = np.abs(target_return) - DIRECTION_DEADBAND
                df.loc[symbol_mask, f'target_confidence_{h}'] = confidence.clip(0, None)
                
                # High confidence trades only
                high_conf = confidence > DIRECTION_CONFIDENCE_THRESHOLD
                df.loc[symbol_mask, f'target_high_conf_{h}'] = bool_to_int8(high_conf)
        
        # Print target statistics
        print("\nTarget statistics:")
        for h in horizons:
            if f'target_direction_{h}' in df.columns:
                direction_counts = df[f'target_direction_{h}'].value_counts()
                total = len(df.dropna(subset=[f'target_direction_{h}']))
                print(f"\nHorizon {h*5} min:")
                if 1 in direction_counts.index:
                    print(f"  UP: {direction_counts.get(1, 0)} ({direction_counts.get(1, 0)/total*100:.1f}%)")
                if 0 in direction_counts.index:
                    print(f"  DOWN: {direction_counts.get(0, 0)} ({direction_counts.get(0, 0)/total*100:.1f}%)")
                if -1 in direction_counts.index:
                    print(f"  FLAT: {direction_counts.get(-1, 0)} ({direction_counts.get(-1, 0)/total*100:.1f}%)")
        
        return df


# """
# Feature engineering module for creating advanced features including whale and social data
# """
# import pandas as pd
# import numpy as np
# import gc
# from typing import Dict, List, Optional
# from tqdm import tqdm

# from config import *
# from utils import optimize_memory_usage, save_cache_data, load_cache_data, bool_to_int8

# class FeatureEngineer:
#     def __init__(self):
#         self.fill_values = {}
        
#     def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
#         """Calculate RSI"""
#         delta = prices.diff()
#         gain = delta.where(delta > 0, 0).rolling(period).mean()
#         loss = -delta.where(delta < 0, 0).rolling(period).mean()
#         rs = gain / (loss + 1e-8)
#         return 100 - (100 / (1 + rs))
    
#     def create_advanced_features(self, df: pd.DataFrame, 
#                                 fill_values: Optional[Dict] = None, 
#                                 use_cache: bool = True,
#                                 include_whale_data: bool = True,
#                                 include_social_data: bool = True) -> pd.DataFrame:
#         """Create advanced features with caching - Enhanced with whale and social features"""
        
#         # Determine cache suffix based on included features
#         cache_suffix = ""
#         if include_whale_data:
#             cache_suffix += "_whale"
#         if include_social_data:
#             cache_suffix += "_social"
            
#         # Try to load cached features first
#         if use_cache:
#             cached_features = load_cache_data(CACHE_DIR, f"features_latest{cache_suffix}.parquet")
#             if cached_features is not None:
#                 user_input = input(f"\nUse cached features{cache_suffix}? (y/n): ").lower()
#                 if user_input == 'y':
#                     print("Using cached features!")
#                     return cached_features
#                 else:
#                     print("Recalculating features...")
        
#         print("\nCreating advanced features...")
        
#         # Check if whale and social data are present
#         whale_cols = [col for col in df.columns if 'whale_' in col or 'inst_' in col or 'retail_' in col]
#         social_cols = [col for col in df.columns if 'mention' in col or 'sentiment' in col or 'social' in col]
        
#         has_whale_data = len(whale_cols) > 0
#         has_social_data = len(social_cols) > 0
        
#         if include_whale_data and not has_whale_data:
#             print("Warning: Whale data requested but not found in dataframe")
#             include_whale_data = False
        
#         if include_social_data and not has_social_data:
#             print("Warning: Social data requested but not found in dataframe")
#             include_social_data = False
        
#         features_list = []
        
#         for symbol in tqdm(df['symbol'].unique(), desc="Advanced feature engineering"):
#             sd = df[df['symbol'] == symbol].copy().sort_values('time_bucket').reset_index(drop=True)
            
#             # Price selection logic
#             if 'mid_price_last' in sd.columns:
#                 price_col = 'mid_price_last'
#             elif 'price_last' in sd.columns:
#                 price_col = 'price_last'
#             elif 'vwap' in sd.columns:
#                 price_col = 'vwap'
#             else:
#                 price_cols = [col for col in sd.columns if 'price' in col and pd.api.types.is_numeric_dtype(sd[col])]
#                 if price_cols:
#                     price_col = price_cols[0]
#                 else:
#                     print(f"Warning: No price column found for {symbol}")
#                     continue
            
#             sd['price'] = sd[price_col]
            
#             # 1. ENHANCED ORDER BOOK FEATURES
#             self._create_orderbook_features(sd)
            
#             # 2. ORDER FLOW AND TRADE FLOW FEATURES
#             self._create_orderflow_features(sd)
            
#             # 3. MOMENTUM AND DIRECTIONAL FEATURES
#             self._create_momentum_features(sd)
            
#             # 4. VOLATILITY AND RISK FEATURES
#             self._create_volatility_features(sd)
            
#             # 5. MICROSTRUCTURE FEATURES
#             self._create_microstructure_features(sd)
            
#             # 6. COMPOSITE INDICATORS
#             self._create_composite_indicators(sd)
            
#             # 7. ENHANCED TECHNICAL INDICATORS
#             self._create_technical_indicators(sd)
            
#             # 8. TIME-BASED FEATURES
#             self._create_time_features(sd)
            
#             # 9. INTERACTION FEATURES
#             self._create_interaction_features(sd)
            
#             # 10. WHALE FEATURES (if data available)
#             if include_whale_data and has_whale_data:
#                 self._create_whale_features(sd)
            
#             # 11. SOCIAL FEATURES (if data available)
#             if include_social_data and has_social_data:
#                 self._create_social_features(sd)
            
#             # FINAL CLEANUP - Handle NaN values properly
#             sd = self._cleanup_features(sd, symbol, fill_values)
            
#             features_list.append(sd)
#             del sd
#             gc.collect()
        
#         result = pd.concat(features_list, ignore_index=True)
#         result = optimize_memory_usage(result, verbose=False)
        
#         print(f"Created {len(result.columns)} features")
#         print(f"Feature DataFrame memory usage: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
#         # Save the features before returning (CACHE AMENDMENT)
#         if use_cache:
#             save_cache_data(result, CACHE_DIR, f"features_latest{cache_suffix}.parquet")
        
#         return result
    
#     def _create_orderbook_features(self, sd: pd.DataFrame):
#         """Create order book related features"""
#         # Microprice (Weighted Mid-Price)
#         if 'bid_size_1_mean' in sd.columns and 'ask_size_1_mean' in sd.columns:
#             if 'bid_price_1_mean' in sd.columns and 'ask_price_1_mean' in sd.columns:
#                 sd['microprice'] = (
#                     (sd['bid_price_1_mean'] * sd['ask_size_1_mean'] + sd['ask_price_1_mean'] * sd['bid_size_1_mean']) /
#                     (sd['bid_size_1_mean'] + sd['ask_size_1_mean'] + 1e-8)
#                 )
                
#                 # Microprice momentum
#                 sd['microprice_momentum'] = sd['microprice'].diff(1)
#                 sd['microprice_acceleration'] = sd['microprice_momentum'].diff(1)
        
#         # Multi-level Order Book Imbalances
#         for level in [1, 3, 5]:
#             bid_cols = [f'bid_size_{i}_mean' for i in range(1, level+1)]
#             ask_cols = [f'ask_size_{i}_mean' for i in range(1, level+1)]
            
#             if all(col in sd.columns for col in bid_cols + ask_cols):
#                 # Top N level imbalance
#                 bid_sum = sum(sd[col] for col in bid_cols)
#                 ask_sum = sum(sd[col] for col in ask_cols)
#                 sd[f'imbalance_top{level}'] = (
#                     (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-8)
#                 )
                
#                 # Z-score of imbalance
#                 for window in [12, 24, 48]:
#                     rolling_mean = sd[f'imbalance_top{level}'].rolling(window).mean()
#                     rolling_std = sd[f'imbalance_top{level}'].rolling(window).std()
#                     sd[f'imbalance_top{level}_zscore_{window}'] = (
#                         (sd[f'imbalance_top{level}'] - rolling_mean) / (rolling_std + 1e-8)
#                     )
        
#         # Order Book Slope/Gradient
#         if all(f'bid_price_{i}_mean' in sd.columns for i in range(1, 6)):
#             if all(f'bid_size_{i}_mean' in sd.columns for i in range(1, 6)):
#                 sd['bid_depth_slope'] = (
#                     (sd['bid_size_5_mean'] - sd['bid_size_1_mean']) /
#                     (sd['bid_price_1_mean'] - sd['bid_price_5_mean'] + 1e-8)
#                 )
                
#                 if all(f'ask_size_{i}_mean' in sd.columns for i in range(1, 6)):
#                     sd['ask_depth_slope'] = (
#                         (sd['ask_size_5_mean'] - sd['ask_size_1_mean']) /
#                         (sd['ask_price_5_mean'] - sd['ask_price_1_mean'] + 1e-8)
#                     )
                    
#                     sd['depth_slope_ratio'] = (
#                         sd['bid_depth_slope'] / (sd['ask_depth_slope'] + 1e-8)
#                     )
        
#         # Book Convexity/Concentration
#         for side in ['bid', 'ask']:
#             if f'{side}_size_1_mean' in sd.columns and f'{side}_size_5_mean' in sd.columns:
#                 sd[f'{side}_convexity'] = (
#                     sd[f'{side}_size_1_mean'] / (sd[f'{side}_size_5_mean'] + 1e-8)
#                 )
        
#         # Book Pressure Pivot
#         if 'imbalance_top3' in sd.columns:
#             price_direction = np.sign(sd['price'].diff(1))
#             sd['book_pressure_pivot'] = sd['imbalance_top3'] * price_direction
    
#     def _create_orderflow_features(self, sd: pd.DataFrame):
#         """Create order flow and trade flow features"""
#         # Enhanced CVD with multiple windows
#         if 'cvd_last' in sd.columns:
#             for window in [6, 12, 24, 48]:
#                 sd[f'cvd_ma_{window}'] = sd['cvd_last'].rolling(window).mean()
#                 sd[f'cvd_momentum_{window}'] = sd['cvd_last'].diff(window)
                
#                 # CVD acceleration
#                 sd[f'cvd_acceleration_{window}'] = (
#                     sd[f'cvd_momentum_{window}'].diff(window//2 if window//2 > 0 else 1)
#                 )
        
#         # Trade Count Imbalance
#         if 'quantity_count' in sd.columns and 'order_flow_imbalance' in sd.columns:
#             total_trades = sd['quantity_count']
#             sd['buy_trade_count'] = total_trades * (1 + sd['order_flow_imbalance']) / 2
#             sd['sell_trade_count'] = total_trades * (1 - sd['order_flow_imbalance']) / 2
            
#             for window in [12, 24]:
#                 buy_sum = sd['buy_trade_count'].rolling(window).sum()
#                 sell_sum = sd['sell_trade_count'].rolling(window).sum()
#                 sd[f'trade_count_imbalance_{window}'] = (
#                     (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-8)
#                 )
        
#         # Large Trade Detection
#         if 'quantity_mean' in sd.columns and 'quantity_std' in sd.columns:
#             large_trade_threshold = sd['quantity_mean'] + 2 * sd['quantity_std']
#             sd['large_trade_ratio'] = sd['quantity_max'] / (large_trade_threshold + 1e-8)
        
#         # VWAP Deviation Features
#         if 'vwap' in sd.columns:
#             for window in [6, 12, 24]:
#                 sd[f'vwap_ma_{window}'] = sd['vwap'].rolling(window).mean()
#                 sd[f'vwap_deviation_{window}'] = (
#                     (sd['price'] - sd[f'vwap_ma_{window}']) / (sd['price'] + 1e-8)
#                 )
    
#     def _create_momentum_features(self, sd: pd.DataFrame):
#         """Create momentum and directional features"""
#         # Price momentum at multiple scales
#         for period in PRICE_RETURN_PERIODS:
#             sd[f'return_{period}'] = sd['price'].pct_change(period)
#             sd[f'log_return_{period}'] = np.log(sd['price'] / sd['price'].shift(period))
            
#             # Momentum of momentum (acceleration)
#             if period <= 24:
#                 sd[f'return_acceleration_{period}'] = (
#                     sd[f'return_{period}'].diff(period//2 if period//2 > 0 else 1)
#                 )
        
#         # Directional Price Imbalance (Tick direction)
#         sd['price_tick'] = np.sign(sd['price'].diff(1))
#         for window in [12, 24, 48]:
#             sd[f'tick_momentum_{window}'] = sd['price_tick'].rolling(window).sum()
#             sd[f'tick_momentum_norm_{window}'] = sd[f'tick_momentum_{window}'] / window
        
#         # Price Position Features
#         for window in [12, 24, 48, 96]:
#             rolling_max = sd['price'].rolling(window).max()
#             rolling_min = sd['price'].rolling(window).min()
#             rolling_median = sd['price'].rolling(window).median()
            
#             # Position in range
#             sd[f'price_position_{window}'] = (
#                 (sd['price'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
#             )
            
#             # Distance from median
#             sd[f'price_median_distance_{window}'] = (
#                 (sd['price'] - rolling_median) / (rolling_median + 1e-8)
#             )
            
#             # Is near high/low - FIXED with safe conversion
#             sd[f'near_high_{window}'] = bool_to_int8(sd[f'price_position_{window}'] > 0.8)
#             sd[f'near_low_{window}'] = bool_to_int8(sd[f'price_position_{window}'] < 0.2)
    
#     def _create_volatility_features(self, sd: pd.DataFrame):
#         """Create volatility and risk features"""
#         # Realized volatility at multiple frequencies
#         for window in ROLLING_WINDOWS:
#             sd[f'volatility_{window}'] = sd['return_1'].rolling(window).std()
#             sd[f'volatility_ma_{window}'] = sd[f'volatility_{window}'].rolling(window).mean()
            
#             # Volatility of volatility
#             sd[f'vol_of_vol_{window}'] = sd[f'volatility_{window}'].rolling(window).std()
            
#             # FIXED: skew() and kurt() return float64 - handle once
#             skew_vals = sd['return_1'].rolling(window).skew()
#             kurt_vals = sd['return_1'].rolling(window).kurt()
#             sd[f'skew_{window}'] = skew_vals
#             sd[f'kurt_{window}'] = kurt_vals
        
#         # Parkinson volatility
#         if 'bid_price_1_mean' in sd.columns and 'ask_price_1_mean' in sd.columns:
#             for window in [12, 24]:
#                 high = sd['ask_price_1_mean'].rolling(window).max()
#                 low = sd['bid_price_1_mean'].rolling(window).min()
#                 sd[f'parkinson_vol_{window}'] = np.sqrt(
#                     1/(4*np.log(2)) * np.log((high + 1e-8)/(low + 1e-8))**2
#                 )
    
#     def _create_microstructure_features(self, sd: pd.DataFrame):
#         """Create microstructure features"""
#         # Spread features
#         if 'spread_mean' in sd.columns:
#             sd['relative_spread'] = sd['spread_mean'] / (sd['price'] + 1e-8)
#             sd['spread_volatility'] = sd['spread_mean'].rolling(12).std()
#             sd['spread_momentum'] = sd['spread_mean'].diff(6)
            
#             # High spread indicator - FIXED with safe conversion
#             spread_threshold = sd['spread_mean'].rolling(48).mean() + 2*sd['spread_mean'].rolling(48).std()
#             sd['high_spread_flag'] = bool_to_int8(sd['spread_mean'] > spread_threshold)
        
#         # Quoted Volume Ratio
#         if 'bid_size_1_mean' in sd.columns and 'ask_size_1_mean' in sd.columns:
#             sd['quoted_volume_ratio'] = sd['bid_size_1_mean'] / (sd['ask_size_1_mean'] + 1e-8)
#             sd['log_quoted_volume_ratio'] = np.log(sd['quoted_volume_ratio'] + 1e-8)
        
#         # Order book imbalance features
#         if 'book_imbalance_mean' in sd.columns:
#             sd['imbalance_ma_12'] = sd['book_imbalance_mean'].rolling(12).mean()
#             sd['imbalance_std_12'] = sd['book_imbalance_mean'].rolling(12).std()
#             sd['imbalance_momentum'] = sd['book_imbalance_mean'].diff(6)
        
#         # Order flow features
#         if 'order_flow_imbalance' in sd.columns:
#             sd['ofi_ma_12'] = sd['order_flow_imbalance'].rolling(12).mean()
#             sd['ofi_momentum'] = sd['order_flow_imbalance'].diff(6)
#             sd['ofi_acceleration'] = sd['ofi_momentum'].diff(3)
        
#         # Volume features
#         if 'quantity_sum' in sd.columns:
#             sd['volume'] = sd['quantity_sum']
#             sd['volume_ma_12'] = sd['volume'].rolling(12).mean()
#             sd['volume_ratio'] = sd['volume'] / (sd['volume_ma_12'] + 1e-8)
#             sd['volume_momentum'] = sd['volume'].pct_change(12)
            
#             # VPIN approximation
#             if 'buy_volume_sum' in sd.columns and 'sell_volume_sum' in sd.columns:
#                 sd['vpin'] = np.abs(
#                     sd['buy_volume_sum'].rolling(12).sum() - sd['sell_volume_sum'].rolling(12).sum()
#                 ) / (sd['volume'].rolling(12).sum() + 1e-8)
    
#     def _create_composite_indicators(self, sd: pd.DataFrame):
#         """Create composite indicators"""
#         # Composite Momentum Score
#         momentum_features = []
#         if 'return_12' in sd.columns:
#             momentum_features.append(sd['return_12'].rolling(12).mean())
#         if 'cvd_momentum_12' in sd.columns:
#             momentum_features.append(sd['cvd_momentum_12'])
#         if 'tick_momentum_norm_12' in sd.columns:
#             momentum_features.append(sd['tick_momentum_norm_12'])
        
#         if momentum_features:
#             # Normalize and combine
#             for i, feat in enumerate(momentum_features):
#                 feat_mean = feat.rolling(48).mean()
#                 feat_std = feat.rolling(48).std()
#                 feat_norm = (feat - feat_mean) / (feat_std + 1e-8)
#                 momentum_features[i] = feat_norm
            
#             sd['composite_momentum'] = sum(momentum_features) / len(momentum_features)
        
#         # Composite Pressure Score
#         pressure_features = []
#         if 'imbalance_top3' in sd.columns:
#             pressure_features.append(sd['imbalance_top3'])
#         if 'order_flow_imbalance' in sd.columns:
#             pressure_features.append(sd['order_flow_imbalance'])
#         if 'quoted_volume_ratio' in sd.columns:
#             pressure_features.append(np.log(sd['quoted_volume_ratio'] + 1e-8))
        
#         if pressure_features:
#             sd['composite_pressure'] = sum(pressure_features) / len(pressure_features)
    
#     def _create_technical_indicators(self, sd: pd.DataFrame):
#         """Create enhanced technical indicators"""
#         # RSI variations
#         for period in RSI_PERIODS:
#             sd[f'rsi_{period}'] = self.calculate_rsi(sd['price'], period)
#             sd[f'rsi_{period}_momentum'] = sd[f'rsi_{period}'].diff(3)
#             # FIXED with safe conversion
#             sd[f'rsi_{period}_oversold'] = bool_to_int8(sd[f'rsi_{period}'] < 30)
#             sd[f'rsi_{period}_overbought'] = bool_to_int8(sd[f'rsi_{period}'] > 70)
        
#         # Enhanced Bollinger Bands
#         for window in BOLLINGER_WINDOWS:
#             ma = sd['price'].rolling(window).mean()
#             std = sd['price'].rolling(window).std()
            
#             # Multiple standard deviations
#             for n_std in [1.5, 2, 2.5]:
#                 sd[f'bb_upper_{window}_{n_std}'] = ma + n_std * std
#                 sd[f'bb_lower_{window}_{n_std}'] = ma - n_std * std
            
#             # BB width (volatility proxy)
#             sd[f'bb_width_{window}'] = (
#                 (sd[f'bb_upper_{window}_2'] - sd[f'bb_lower_{window}_2']) / (ma + 1e-8)
#             )
            
#             # BB position
#             sd[f'bb_position_{window}'] = (
#                 (sd['price'] - sd[f'bb_lower_{window}_2']) /
#                 (sd[f'bb_upper_{window}_2'] - sd[f'bb_lower_{window}_2'] + 1e-8)
#             )
            
#             # BB squeeze (low volatility) - FIXED with safe conversion
#             bb_width_ma = sd[f'bb_width_{window}'].rolling(window).mean()
#             bb_width_std = sd[f'bb_width_{window}'].rolling(window).std()
#             sd[f'bb_squeeze_{window}'] = bool_to_int8(
#                 sd[f'bb_width_{window}'] < (bb_width_ma - bb_width_std)
#             )
        
#         # MACD variations
#         for fast, slow in MACD_PAIRS:
#             ema_fast = sd['price'].ewm(span=fast, adjust=False).mean()
#             ema_slow = sd['price'].ewm(span=slow, adjust=False).mean()
            
#             sd[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
#             sd[f'macd_signal_{fast}_{slow}'] = sd[f'macd_{fast}_{slow}'].ewm(span=9, adjust=False).mean()
#             sd[f'macd_hist_{fast}_{slow}'] = sd[f'macd_{fast}_{slow}'] - sd[f'macd_signal_{fast}_{slow}']
            
#             # MACD cross signals - FIXED with safe conversion
#             macd_above = sd[f'macd_{fast}_{slow}'] > sd[f'macd_signal_{fast}_{slow}']
#             macd_above_prev = sd[f'macd_{fast}_{slow}'].shift(1) > sd[f'macd_signal_{fast}_{slow}'].shift(1)
#             sd[f'macd_cross_up_{fast}_{slow}'] = bool_to_int8(macd_above & ~macd_above_prev)
        
#         # ATR (Average True Range)
#         if 'price_max' in sd.columns and 'price_min' in sd.columns:
#             high = sd['price_max']
#             low = sd['price_min']
#             close = sd['price']
            
#             tr1 = high - low
#             tr2 = np.abs(high - close.shift(1))
#             tr3 = np.abs(low - close.shift(1))
#             tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
#             for period in [14, 21]:
#                 sd[f'atr_{period}'] = tr.rolling(period).mean()
#                 sd[f'atr_ratio_{period}'] = sd[f'atr_{period}'] / (sd['price'] + 1e-8)
    
#     def _create_time_features(self, sd: pd.DataFrame):
#         """Create time-based features"""
#         sd['hour'] = sd['time_bucket'].dt.hour.astype('int8')
#         sd['day_of_week'] = sd['time_bucket'].dt.dayofweek.astype('int8')
#         sd['is_weekend'] = bool_to_int8(sd['day_of_week'].isin([5, 6]))
#         sd['hour_sin'] = np.sin(2 * np.pi * sd['hour'] / 24)
#         sd['hour_cos'] = np.cos(2 * np.pi * sd['hour'] / 24)
#         sd['dow_sin'] = np.sin(2 * np.pi * sd['day_of_week'] / 7)
#         sd['dow_cos'] = np.cos(2 * np.pi * sd['day_of_week'] / 7)
        
#         # Trading sessions - FIXED with safe conversion
#         sd['is_asian_session'] = bool_to_int8(sd['hour'].between(0, 9))
#         sd['is_european_session'] = bool_to_int8(sd['hour'].between(7, 16))
#         sd['is_us_session'] = bool_to_int8(sd['hour'].between(12, 21))
#         sd['is_late_us_session'] = bool_to_int8(sd['hour'].between(20, 23))
        
#         # Session transitions - FIXED with safe conversion
#         session_diff = (
#             sd['is_asian_session'].diff().abs() +
#             sd['is_european_session'].diff().abs() +
#             sd['is_us_session'].diff().abs()
#         )
#         sd['session_transition'] = bool_to_int8(session_diff.clip(0, 1))
    
#     def _create_interaction_features(self, sd: pd.DataFrame):
#         """Create interaction features"""
#         # Momentum × Volume
#         if 'return_12' in sd.columns and 'volume_ratio' in sd.columns:
#             sd['momentum_volume_interaction'] = sd['return_12'] * sd['volume_ratio']
        
#         # Imbalance × Volatility
#         if 'book_imbalance_mean' in sd.columns and 'volatility_12' in sd.columns:
#             sd['imbalance_volatility_interaction'] = sd['book_imbalance_mean'] * sd['volatility_12']
        
#         # Spread × Volume (liquidity pressure)
#         if 'relative_spread' in sd.columns and 'volume' in sd.columns:
#             volume_mean = sd['volume'].rolling(48).mean()
#             volume_std = sd['volume'].rolling(48).std()
#             volume_norm = (sd['volume'] - volume_mean) / (volume_std + 1e-8)
#             sd['spread_volume_pressure'] = sd['relative_spread'] * volume_norm
        
#         # Order flow × Price momentum
#         if 'order_flow_imbalance' in sd.columns and 'return_6' in sd.columns:
#             sd['ofi_momentum_interaction'] = sd['order_flow_imbalance'] * sd['return_6']
    
#     def _create_whale_features(self, sd: pd.DataFrame):
#         """Create whale-based features leveraging strong correlations found in analysis"""
        
#         # 1. WHALE FLOW MOMENTUM
#         if 'whale_flow_imbalance' in sd.columns:
#             for window in [6, 12, 24]:
#                 sd[f'whale_flow_ma_{window}'] = sd['whale_flow_imbalance'].rolling(window).mean()
#                 sd[f'whale_flow_momentum_{window}'] = sd['whale_flow_imbalance'].diff(window)
                
#                 # Whale flow acceleration
#                 sd[f'whale_flow_acceleration_{window}'] = (
#                     sd[f'whale_flow_momentum_{window}'].diff(window//2 if window//2 > 0 else 1)
#                 )
        
#         # 2. INSTITUTIONAL PARTICIPATION DYNAMICS
#         if 'inst_participation_rate' in sd.columns:
#             # Z-score of institutional participation
#             for window in [12, 24, 48]:
#                 inst_mean = sd['inst_participation_rate'].rolling(window).mean()
#                 inst_std = sd['inst_participation_rate'].rolling(window).std()
#                 sd[f'inst_participation_zscore_{window}'] = (
#                     (sd['inst_participation_rate'] - inst_mean) / (inst_std + 1e-8)
#                 )
            
#             # Institutional concentration changes
#             sd['inst_participation_change_12'] = sd['inst_participation_rate'].diff(12)
#             sd['inst_participation_acceleration'] = sd['inst_participation_change_12'].diff(6)
        
#         # 3. RETAIL COUNTER-TRADING SIGNALS
#         if 'retail_sell_pressure' in sd.columns:
#             # Retail panic indicator (high retail selling + price drop)
#             price_drop_12 = sd['return_12'] < -0.01  # Price down more than 1%
#             high_retail_sell = sd['retail_sell_pressure'] > sd['retail_sell_pressure'].rolling(48).quantile(0.8)
#             sd['retail_panic_indicator'] = bool_to_int8(price_drop_12 & high_retail_sell)
            
#             # Retail capitulation momentum
#             sd['retail_sell_momentum'] = sd['retail_sell_pressure'].diff(6)
#             sd['retail_capitulation_signal'] = bool_to_int8(
#                 sd['retail_sell_momentum'] > sd['retail_sell_momentum'].rolling(48).quantile(0.9)
#             )
        
#         # 4. SMART MONEY DIVERGENCE
#         if 'smart_dumb_divergence' in sd.columns:
#             # Enhanced smart money signals
#             sd['smart_money_strength'] = sd['smart_dumb_divergence'].rolling(12).mean()
#             sd['smart_money_momentum'] = sd['smart_dumb_divergence'].diff(6)
            
#             # Smart money accumulation (positive divergence + price weakness)
#             price_weakness = sd['return_24'] < 0
#             smart_accumulation = sd['smart_dumb_divergence'] > 0.5
#             sd['smart_accumulation_signal'] = bool_to_int8(price_weakness & smart_accumulation)
        
#         # 5. WHALE TRANSACTION CHARACTERISTICS
#         if 'whale_amount_usd_mean' in sd.columns:
#             # Average transaction size trends
#             sd['avg_whale_size_ma_24'] = sd['whale_amount_usd_mean'].rolling(24).mean()
#             sd['whale_size_ratio'] = sd['whale_amount_usd_mean'] / (sd['avg_whale_size_ma_24'] + 1e-8)
            
#             # Large whale activity spikes
#             if 'whale_amount_usd_max' in sd.columns:
#                 max_percentile_95 = sd['whale_amount_usd_max'].rolling(96).quantile(0.95)
#                 sd['mega_whale_spike'] = bool_to_int8(sd['whale_amount_usd_max'] > max_percentile_95)
        
#         # 6. WHALE-MARKET MICROSTRUCTURE INTERACTION
#         if 'whale_flow_imbalance' in sd.columns and 'order_flow_imbalance' in sd.columns:
#             # Whale vs regular order flow alignment
#             sd['whale_market_alignment'] = sd['whale_flow_imbalance'] * sd['order_flow_imbalance']
            
#             # Whale leading indicator (whale flow leads market flow)
#             sd['whale_leads_market'] = (
#                 sd['whale_flow_imbalance'].rolling(6).mean() * 
#                 sd['order_flow_imbalance'].shift(-6).rolling(6).mean()
#             )
        
#         # 7. MARKET CAP STRATIFIED FEATURES
#         if 'megacap_flow' in sd.columns:
#             # Institutional preference shifts
#             sd['megacap_momentum'] = sd['megacap_flow'].diff(12)
            
#             # Flight to quality indicator
#             if 'smallcap_speculation' in sd.columns:
#                 sd['flight_to_quality'] = sd['megacap_flow'] - sd['smallcap_speculation']
#                 sd['quality_rotation_signal'] = bool_to_int8(sd['flight_to_quality'].diff(6) > 0)
        
#         # 8. WHALE PERSISTENCE PATTERNS
#         if 'whale_buy_count' in sd.columns and 'whale_sell_count' in sd.columns:
#             # Consecutive buying/selling streaks
#             buy_signal = sd['whale_buy_count'] > sd['whale_sell_count']
#             sell_signal = sd['whale_sell_count'] > sd['whale_buy_count']
            
#             # Calculate streaks
#             buy_streak = buy_signal.groupby((buy_signal != buy_signal.shift()).cumsum()).cumsum()
#             sell_streak = sell_signal.groupby((sell_signal != sell_signal.shift()).cumsum()).cumsum()
            
#             sd['whale_buy_streak'] = buy_streak
#             sd['whale_sell_streak'] = sell_streak
            
#             # Persistence strength
#             sd['whale_persistence'] = np.where(buy_streak > 0, buy_streak, -sell_streak)
        
#         # 9. VOLUME-WEIGHTED WHALE METRICS
#         if 'whale_amount_usd_sum' in sd.columns and 'volume' in sd.columns:
#             # Whale share of total volume
#             sd['whale_volume_share'] = sd['whale_amount_usd_sum'] / (sd['volume'] * sd['price'] + 1e-8)
            
#             # Abnormal whale activity
#             whale_share_ma = sd['whale_volume_share'].rolling(48).mean()
#             whale_share_std = sd['whale_volume_share'].rolling(48).std()
#             sd['abnormal_whale_activity'] = (
#                 (sd['whale_volume_share'] - whale_share_ma) / (whale_share_std + 1e-8)
#             )
        
#         # 10. COMPOSITE WHALE INDICATORS
#         # Whale Sentiment Score (combines multiple whale signals)
#         whale_signals = []
#         if 'whale_flow_imbalance' in sd.columns:
#             whale_signals.append(sd['whale_flow_imbalance'])
#         if 'inst_participation_rate' in sd.columns:
#             whale_signals.append(sd['inst_participation_rate'] - 0.5)  # Center around 0.5
#         if 'smart_dumb_divergence' in sd.columns:
#             whale_signals.append(sd['smart_dumb_divergence'])
        
#         if whale_signals:
#             # Normalize and combine
#             normalized_signals = []
#             for signal in whale_signals:
#                 signal_mean = signal.rolling(48).mean()
#                 signal_std = signal.rolling(48).std()
#                 normalized_signals.append((signal - signal_mean) / (signal_std + 1e-8))
            
#             sd['whale_sentiment_composite'] = sum(normalized_signals) / len(normalized_signals)
            
#             # Whale sentiment extremes
#             sd['whale_extreme_bullish'] = bool_to_int8(sd['whale_sentiment_composite'] > 2)
#             sd['whale_extreme_bearish'] = bool_to_int8(sd['whale_sentiment_composite'] < -2)
    
#     def _create_social_features(self, sd: pd.DataFrame):
#         """Create social sentiment features from mentions and trend data"""
        
#         # 1. SOCIAL MOMENTUM FEATURES
#         if 'social_momentum_score' in sd.columns:
#             # Multi-timeframe momentum
#             for window in [6, 12, 24]:
#                 sd[f'social_momentum_ma_{window}'] = sd['social_momentum_score'].rolling(window).mean()
#                 sd[f'social_momentum_change_{window}'] = sd['social_momentum_score'].diff(window)
            
#             # Social momentum acceleration
#             sd['social_acceleration'] = sd['social_momentum_change_12'].diff(6)
            
#             # Extreme social momentum
#             momentum_95 = sd['social_momentum_score'].rolling(96).quantile(0.95)
#             sd['social_momentum_extreme'] = bool_to_int8(sd['social_momentum_score'] > momentum_95)
        
#         # 2. MENTION VELOCITY FEATURES
#         if 'total_mentions' in sd.columns:
#             # Mention growth rates
#             sd['mention_velocity'] = sd['total_mentions'].pct_change(12)
#             sd['mention_acceleration'] = sd['mention_velocity'].diff(6)
            
#             # Viral threshold detection
#             viral_threshold = sd['total_mentions'].rolling(96).quantile(0.9)
#             sd['viral_indicator'] = bool_to_int8(sd['total_mentions'] > viral_threshold)
            
#             # Sustained attention
#             high_mentions = sd['total_mentions'] > sd['total_mentions'].rolling(48).median()
#             sd['sustained_attention'] = high_mentions.rolling(12).sum() / 12
        
#         # 3. SENTIMENT DYNAMICS
#         if 'sentiment_14d' in sd.columns:
#             # Sentiment momentum
#             sd['sentiment_momentum'] = sd['sentiment_14d'].diff(6)
#             sd['sentiment_reversal'] = sd['sentiment_momentum'].diff(3)
            
#             # Sentiment extremes with persistence
#             very_positive = sd['sentiment_very_positive'].rolling(6).sum()
#             very_negative = sd['sentiment_very_negative'].rolling(6).sum()
#             sd['persistent_positive_sentiment'] = bool_to_int8(very_positive >= 4)
#             sd['persistent_negative_sentiment'] = bool_to_int8(very_negative >= 4)
            
#             # Sentiment volatility
#             sd['sentiment_volatility'] = sd['sentiment_14d'].rolling(24).std()
        
#         # 4. SENTIMENT-WEIGHTED ACTIVITY
#         if 'sentiment_weighted_mentions' in sd.columns:
#             # Weighted mention momentum
#             sd['weighted_mention_ma_12'] = sd['sentiment_weighted_mentions'].rolling(12).mean()
#             sd['weighted_mention_momentum'] = sd['sentiment_weighted_mentions'].diff(12)
            
#             # Positive vs negative mention flow
#             positive_mentions = sd['total_mentions'] * sd['sentiment_positive']
#             negative_mentions = sd['total_mentions'] * sd['sentiment_negative']
#             sd['mention_sentiment_flow'] = (positive_mentions - negative_mentions).rolling(12).sum()
        
#         # 5. SOCIAL DIVERGENCE INDICATORS
#         if 'social_momentum_score' in sd.columns and 'return_12' in sd.columns:
#             # Social-price divergence
#             price_direction = np.sign(sd['return_12'])
#             social_direction = np.sign(sd['social_momentum_score'])
#             sd['social_price_divergence'] = bool_to_int8(price_direction != social_direction)
            
#             # Bearish divergence (price up, social down)
#             sd['social_bearish_divergence'] = bool_to_int8(
#                 (sd['return_12'] > 0.005) & (sd['social_momentum_score'] < -0.1)
#             )
            
#             # Bullish divergence (price down, social up)  
#             sd['social_bullish_divergence'] = bool_to_int8(
#                 (sd['return_12'] < -0.005) & (sd['social_momentum_score'] > 0.1)
#             )
        
#         # 6. CROSS-TIMEFRAME SOCIAL FEATURES
#         mention_windows = ['mention_momentum_4h', 'mention_momentum_7d', 'mention_momentum_1m']
#         available_windows = [col for col in mention_windows if col in sd.columns]
        
#         if len(available_windows) >= 2:
#             # Short vs long term momentum
#             if 'mention_momentum_4h' in sd.columns and 'mention_momentum_1m' in sd.columns:
#                 sd['social_momentum_divergence'] = (
#                     sd['mention_momentum_4h'] - sd['mention_momentum_1m']
#                 )
                
#                 # Momentum crossover signals
#                 short_above_long = sd['mention_momentum_4h'] > sd['mention_momentum_1m']
#                 short_above_long_prev = (
#                     sd['mention_momentum_4h'].shift(1) > sd['mention_momentum_1m'].shift(1)
#                 )
#                 sd['social_momentum_crossover'] = bool_to_int8(
#                     short_above_long & ~short_above_long_prev
#                 )
        
#         # 7. SOCIAL VOLUME INTERACTION
#         if 'total_mentions' in sd.columns and 'volume' in sd.columns:
#             # Normalize both series
#             mention_norm = (sd['total_mentions'] - sd['total_mentions'].rolling(48).mean()) / (
#                 sd['total_mentions'].rolling(48).std() + 1e-8
#             )
#             volume_norm = (sd['volume'] - sd['volume'].rolling(48).mean()) / (
#                 sd['volume'].rolling(48).std() + 1e-8
#             )
            
#             # Social-volume correlation
#             sd['social_volume_alignment'] = mention_norm * volume_norm
            
#             # High social activity with low volume (potential breakout)
#             high_social = mention_norm > 1
#             low_volume = volume_norm < -0.5
#             sd['social_volume_divergence'] = bool_to_int8(high_social & low_volume)
        
#         # 8. SENTIMENT REGIME DETECTION
#         if 'sentiment_14d' in sd.columns:
#             # Define sentiment regimes
#             sentiment_ma = sd['sentiment_14d'].rolling(24).mean()
            
#             # Regime classification
#             sd['bullish_regime'] = bool_to_int8(sentiment_ma > SENTIMENT_THRESHOLDS['positive'])
#             sd['bearish_regime'] = bool_to_int8(sentiment_ma < SENTIMENT_THRESHOLDS['neutral'])
#             sd['neutral_regime'] = bool_to_int8(
#                 ~sd['bullish_regime'] & ~sd['bearish_regime']
#             )
            
#             # Regime transitions
#             regime = np.where(sd['bullish_regime'], 1, 
#                             np.where(sd['bearish_regime'], -1, 0))
#             sd['regime_change'] = bool_to_int8(regime != np.roll(regime, 1))
        
#         # 9. COMPOSITE SOCIAL INDICATORS
#         social_features = []
        
#         # Add normalized social signals
#         if 'social_momentum_score' in sd.columns:
#             social_features.append(sd['social_momentum_score'])
#         if 'sentiment_14d' in sd.columns:
#             social_features.append(sd['sentiment_14d'])
#         if 'mention_velocity' in sd.columns:
#             vel_norm = (sd['mention_velocity'] - sd['mention_velocity'].rolling(48).mean()) / (
#                 sd['mention_velocity'].rolling(48).std() + 1e-8
#             )
#             social_features.append(vel_norm)
        
#         if social_features:
#             # Create composite social sentiment
#             sd['social_sentiment_composite'] = sum(social_features) / len(social_features)
            
#             # Social extremes
#             composite_95 = sd['social_sentiment_composite'].rolling(96).quantile(0.95)
#             composite_5 = sd['social_sentiment_composite'].rolling(96).quantile(0.05)
            
#             sd['social_euphoria'] = bool_to_int8(sd['social_sentiment_composite'] > composite_95)
#             sd['social_despair'] = bool_to_int8(sd['social_sentiment_composite'] < composite_5)
    
#     def _cleanup_features(self, sd: pd.DataFrame, symbol: str, fill_values: Optional[Dict]) -> pd.DataFrame:
#         """Clean up features and handle NaN values properly"""
#         # Get numeric columns (exclude identifiers)
#         numeric_cols = sd.select_dtypes(include=[np.number]).columns
#         numeric_cols = [col for col in numeric_cols if col not in ['symbol', 'time_bucket']]
        
#         # Apply fill values if provided
#         if fill_values and symbol in fill_values:
#             symbol_fills = fill_values[symbol]
#             for col in numeric_cols:
#                 if col in symbol_fills:
#                     sd[col] = sd[col].fillna(symbol_fills[col])
#         else:
#             # Default filling strategy
#             for col in numeric_cols:
#                 if 'flag' in col or col.startswith('is_') or col.startswith('near_'):
#                     # Binary/flag columns -> 0
#                     sd[col] = sd[col].fillna(0)
#                 elif 'return' in col or 'momentum' in col or 'change' in col:
#                     # Return/momentum columns -> 0 
#                     sd[col] = sd[col].fillna(0)
#                 else:
#                     # Other features -> forward fill then 0
#                     sd[col] = sd[col].fillna(method='ffill', limit=3).fillna(0)
        
#         # Ensure no infinities
#         sd[numeric_cols] = sd[numeric_cols].replace([np.inf, -np.inf], 0)
        
#         # Final memory optimization
#         sd = optimize_memory_usage(sd, verbose=False)
        
#         return sd
    
#     def prepare_advanced_targets(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
#         """Prepare target variables for multiple prediction horizons"""
#         print(f"\nPreparing targets for horizons: {[h*5 for h in horizons]} minutes")
        
#         for symbol in df['symbol'].unique():
#             symbol_mask = df['symbol'] == symbol
#             symbol_data = df.loc[symbol_mask].copy()
            
#             # Get price column
#             if 'mid_price_last' in symbol_data.columns:
#                 price_col = 'mid_price_last'
#             elif 'price_last' in symbol_data.columns:
#                 price_col = 'price_last'
#             elif 'vwap' in symbol_data.columns:
#                 price_col = 'vwap'
#             else:
#                 price_cols = [col for col in symbol_data.columns if 'price' in col and 
#                             pd.api.types.is_numeric_dtype(symbol_data[col])]
#                 if price_cols:
#                     price_col = price_cols[0]
#                 else:
#                     print(f"Warning: No price column found for {symbol}")
#                     continue
            
#             # Create targets for each horizon
#             for h in horizons:
#                 # Future return
#                 future_price = symbol_data[price_col].shift(-h)
#                 target_return = (future_price - symbol_data[price_col]) / symbol_data[price_col]
                
#                 # Assign back to main dataframe
#                 df.loc[symbol_mask, f'target_return_{h}'] = target_return
                
#                 # Binary direction target
#                 df.loc[symbol_mask, f'target_direction_{h}'] = (target_return > 0).astype(int)
        
#         return df