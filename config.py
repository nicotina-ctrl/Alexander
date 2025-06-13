"""
Configuration file for Enhanced Crypto Prediction Pipeline with Whale and Social Data
"""
import os

# Paths
GDRIVE_DIR = "/content/drive/MyDrive/crypto_pipeline_whale/realtime_perp_data"
WHALE_DATA_DIR = "/content/drive/MyDrive/crypto_pipeline_whale"
LOCAL_DIR = "/tmp/data"
MODEL_SAVE_DIR = "/content/drive/MyDrive/crypto_pipeline_whale/models_enhanced"
CACHE_DIR = "/content/drive/MyDrive/crypto_pipeline_whale/cache"
SQLITE_DB_PATH = "/content/drive/MyDrive/crypto_pipeline_whale/data/crypto_integrated_data.db"

# Create directories
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

# Model Parameters
DEFAULT_HORIZONS = [6, 12, 24]  # 30min, 60min, 120min
DEFAULT_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

# Processing Parameters
DEFAULT_BATCH_SIZE = 50
MAX_FILES_SMOKE_TEST = 20
FEATURE_SELECTION_MAX_SAMPLES = 500000  # Increased for better feature selection

# Feature Selection Limits - Reduced to prevent overfitting
MAX_FEATURES_PER_CATEGORY = {
    'technical': 15,      # Reduced from 30
    'microstructure': 10, # Reduced from 30
    'whale': 25,          # Increased from 20 - whale features are more important
    'social': 10,         # Kept moderate
    'total': 60           # Reduced from 100
}

# Direction Prediction Thresholds - Add deadband to reduce noise
DIRECTION_DEADBAND = 0.0001  # 0.01% threshold for direction classification
DIRECTION_CONFIDENCE_THRESHOLD = 0.0005  # 0.05% for high confidence trades
DIRECTION_THRESHOLD_MULTIPLIER = 1.0  # Multiplier for direction thresholds (ADDED THIS)

# Updated Model Hyperparameters with better defaults
XGB_PARAMS = {
    'n_estimators': 200,      # Increased from 50
    'max_depth': 5,           # Increased from 3
    'learning_rate': 0.05,    # Increased from 0.01
    'subsample': 0.8,         # Increased from 0.6
    'colsample_bytree': 0.8,  # Increased from 0.6
    'reg_alpha': 0.5,         # Reduced from 1.0
    'reg_lambda': 0.5,        # Reduced from 1.0
    'min_child_weight': 30,   # Reduced from 50
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1,             # Use all cores
    'tree_method': 'hist',    # Faster training
    'predictor': 'cpu_predictor'
}

LGB_PARAMS = {
    'n_estimators': 200,      # Increased from 50
    'num_leaves': 31,         # Increased from 10
    'learning_rate': 0.05,    # Increased from 0.01
    'feature_fraction': 0.8,  # Increased from 0.6
    'bagging_fraction': 0.8,  # Increased from 0.6
    'bagging_freq': 5,
    'min_child_samples': 30,  # Reduced from 50
    'reg_alpha': 0.5,         # Reduced from 1.0
    'reg_lambda': 0.5,        # Reduced from 1.0
    'min_gain_to_split': 0.01,# Reduced from 0.1
    'random_state': 42,
    'n_jobs': -1,
    'force_col_wise': True,
    'verbosity': -1,
    'metric': 'rmse'
}

# Ridge parameters for grid search
RIDGE_PARAMS = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'fit_intercept': True,
    'random_state': 42
}

# MLP parameters for grid search
MLP_PARAMS = {
    'hidden_layer_sizes': [(100, 50), (200, 100), (150, 75, 25)],
    'activation': 'relu',
    'solver': 'adam',
    'alpha': [0.001, 0.01, 0.1],
    'batch_size': 64,
    'learning_rate': 'adaptive',
    'learning_rate_init': [0.001, 0.01],
    'max_iter': 200,  # Increased from 100
    'early_stopping': True,
    'validation_fraction': 0.2,
    'n_iter_no_change': 20,
    'random_state': 42
}

# Ensemble weights - prioritize whale-aware models
ENSEMBLE_WEIGHTS = {
    'whale_priority': {
        'xgb': 0.25,
        'lgb': 0.25,
        'ridge': 0.20,
        'mlp': 0.30  # MLP can capture non-linear whale patterns
    },
    'balanced': {
        'xgb': 0.25,
        'lgb': 0.25,
        'ridge': 0.25,
        'mlp': 0.25
    }
}

# Feature Engineering Parameters
ROLLING_WINDOWS = [6, 12, 24, 48, 96]
PRICE_RETURN_PERIODS = [1, 3, 6, 12, 24, 48]
RSI_PERIODS = [7, 14, 21]
BOLLINGER_WINDOWS = [12, 24, 48]
MACD_PAIRS = [(12, 26), (5, 20), (8, 21)]

# Data Processing
MEMORY_OPTIMIZATION_THRESHOLD = 1024 * 1024  # 1MB
TIME_BUCKET = '5min'  # 5-minute bars

# Trading Parameters
CRYPTO_PERIODS_PER_YEAR = 365 * 24 * 12  # For 5-minute bars
MIN_SAMPLES_FOR_TRAINING = 100  # Minimum samples per symbol/horizon

# Whale Data Parameters - Enhanced
WHALE_TRANSACTION_THRESHOLD = 10000  # Minimum USD value
WHALE_FEATURE_WEIGHTS = {
    'whale_flow_imbalance': 2.0,
    'inst_participation_rate': 2.5,
    'smart_dumb_divergence': 2.0,
    'whale_sentiment_composite': 3.0,  # Highest weight for composite
    'inst_participation_zscore_12': 2.5,
    'inst_participation_change_12': 2.0,
    'whale_size_ratio': 1.5,
    'retail_capitulation_signal': 1.8
}

INSTITUTIONAL_THRESHOLDS = {
    'market_cap': {
        'megacap': 10_000_000_000,    # $10B+
        'largecap': 1_000_000_000,    # $1B+
        'midcap': 100_000_000,        # $100M+
        'smallcap': 50_000_000        # $50M+
    },
    'transaction_amount': {
        'very_large': 250_000,         # $250k+
        'large': 100_000,              # $100k+
        'medium': 50_000,              # $50k+
        'small': 25_000                # $25k+
    }
}

# Social Data Parameters
SOCIAL_MENTION_WINDOWS = ['4H', '7d', '1m']
SENTIMENT_THRESHOLDS = {
    'very_positive': 0.5,
    'positive': 0.2,
    'neutral': -0.2,
    'negative': -0.5
}

# Symbol Mapping for Data Integration
SYMBOL_MAPPING = {
    'BTC/USDT': ['BTC', 'BITCOIN', 'WBTC'],
    'ETH/USDT': ['ETH', 'ETHEREUM', 'WETH'],
    'SOL/USDT': ['SOL', 'SOLANA', 'WSOL']
}

# Reverse mapping for easy lookup
TOKEN_TO_SYMBOL = {}
for symbol, tokens in SYMBOL_MAPPING.items():
    for token in tokens:
        TOKEN_TO_SYMBOL[token] = symbol.split('/')[0]

# Correlation threshold for feature pruning
CORRELATION_THRESHOLD = 0.95

# Validation parameters
WALK_FORWARD_TRAIN_RATIO = 0.7  # 70% for train in each walk-forward window
WALK_FORWARD_VAL_RATIO = 0.15   # 15% for validation
WALK_FORWARD_TEST_RATIO = 0.15  # 15% for test

# """
# Configuration file for Enhanced Crypto Prediction Pipeline with Whale and Social Data
# """
# import os

# # Paths
# GDRIVE_DIR = "/content/drive/MyDrive/crypto_pipeline_whale/realtime_perp_data"
# WHALE_DATA_DIR = "/content/drive/MyDrive/crypto_pipeline_whale/data"
# LOCAL_DIR = "/tmp/data"
# MODEL_SAVE_DIR = "/content/drive/MyDrive/crypto_pipeline_whale/models_enhanced"
# CACHE_DIR = "/content/drive/MyDrive/crypto_pipeline_whale/cache"
# SQLITE_DB_PATH = "/content/drive/MyDrive/crypto_pipeline_whale/data/crypto_integrated_data.db"

# # Create directories
# os.makedirs(LOCAL_DIR, exist_ok=True)
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# os.makedirs(CACHE_DIR, exist_ok=True)
# os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

# # Model Parameters
# DEFAULT_HORIZONS = [6, 12, 24]  # 30min, 60min, 120min
# DEFAULT_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

# # Processing Parameters
# DEFAULT_BATCH_SIZE = 50
# MAX_FILES_SMOKE_TEST = 20
# FEATURE_SELECTION_MAX_SAMPLES = 500000  # Increased for better feature selection

# # Feature Selection Limits - Reduced to prevent overfitting
# MAX_FEATURES_PER_CATEGORY = {
#     'technical': 15,      # Reduced from 30
#     'microstructure': 10, # Reduced from 30
#     'whale': 25,          # Increased from 20 - whale features are more important
#     'social': 10,         # Kept moderate
#     'total': 60           # Reduced from 100
# }

# # Direction Prediction Thresholds - Add deadband to reduce noise
# DIRECTION_DEADBAND = 0.0001  # 0.01% threshold for direction classification
# DIRECTION_CONFIDENCE_THRESHOLD = 0.0005  # 0.05% for high confidence trades

# # Updated Model Hyperparameters with better defaults
# XGB_PARAMS = {
#     'n_estimators': 200,      # Increased from 50
#     'max_depth': 5,           # Increased from 3
#     'learning_rate': 0.05,    # Increased from 0.01
#     'subsample': 0.8,         # Increased from 0.6
#     'colsample_bytree': 0.8,  # Increased from 0.6
#     'reg_alpha': 0.5,         # Reduced from 1.0
#     'reg_lambda': 0.5,        # Reduced from 1.0
#     'min_child_weight': 30,   # Reduced from 50
#     'gamma': 0.1,
#     'random_state': 42,
#     'n_jobs': -1,             # Use all cores
#     'tree_method': 'hist',    # Faster training
#     'predictor': 'cpu_predictor'
# }

# LGB_PARAMS = {
#     'n_estimators': 200,      # Increased from 50
#     'num_leaves': 31,         # Increased from 10
#     'learning_rate': 0.05,    # Increased from 0.01
#     'feature_fraction': 0.8,  # Increased from 0.6
#     'bagging_fraction': 0.8,  # Increased from 0.6
#     'bagging_freq': 5,
#     'min_child_samples': 30,  # Reduced from 50
#     'reg_alpha': 0.5,         # Reduced from 1.0
#     'reg_lambda': 0.5,        # Reduced from 1.0
#     'min_gain_to_split': 0.01,# Reduced from 0.1
#     'random_state': 42,
#     'n_jobs': -1,
#     'force_col_wise': True,
#     'verbosity': -1,
#     'metric': 'rmse'
# }

# # Ridge parameters for grid search
# RIDGE_PARAMS = {
#     'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
#     'fit_intercept': True,
#     'random_state': 42
# }

# # MLP parameters for grid search
# MLP_PARAMS = {
#     'hidden_layer_sizes': [(100, 50), (200, 100), (150, 75, 25)],
#     'activation': 'relu',
#     'solver': 'adam',
#     'alpha': [0.001, 0.01, 0.1],
#     'batch_size': 64,
#     'learning_rate': 'adaptive',
#     'learning_rate_init': [0.001, 0.01],
#     'max_iter': 200,  # Increased from 100
#     'early_stopping': True,
#     'validation_fraction': 0.2,
#     'n_iter_no_change': 20,
#     'random_state': 42
# }

# # Ensemble weights - prioritize whale-aware models
# ENSEMBLE_WEIGHTS = {
#     'whale_priority': {
#         'xgb': 0.25,
#         'lgb': 0.25,
#         'ridge': 0.20,
#         'mlp': 0.30  # MLP can capture non-linear whale patterns
#     },
#     'balanced': {
#         'xgb': 0.25,
#         'lgb': 0.25,
#         'ridge': 0.25,
#         'mlp': 0.25
#     }
# }

# # Feature Engineering Parameters
# ROLLING_WINDOWS = [6, 12, 24, 48, 96]
# PRICE_RETURN_PERIODS = [1, 3, 6, 12, 24, 48]
# RSI_PERIODS = [7, 14, 21]
# BOLLINGER_WINDOWS = [12, 24, 48]
# MACD_PAIRS = [(12, 26), (5, 20), (8, 21)]

# # Data Processing
# MEMORY_OPTIMIZATION_THRESHOLD = 1024 * 1024  # 1MB
# TIME_BUCKET = '5min'  # 5-minute bars

# # Trading Parameters
# CRYPTO_PERIODS_PER_YEAR = 365 * 24 * 12  # For 5-minute bars
# MIN_SAMPLES_FOR_TRAINING = 500  # Minimum samples per symbol/horizon

# # Whale Data Parameters - Enhanced
# WHALE_TRANSACTION_THRESHOLD = 10000  # Minimum USD value
# WHALE_FEATURE_WEIGHTS = {
#     'whale_flow_imbalance': 2.0,
#     'inst_participation_rate': 2.5,
#     'smart_dumb_divergence': 2.0,
#     'whale_sentiment_composite': 3.0,  # Highest weight for composite
#     'inst_participation_zscore_12': 2.5,
#     'inst_participation_change_12': 2.0,
#     'whale_size_ratio': 1.5,
#     'retail_capitulation_signal': 1.8
# }

# INSTITUTIONAL_THRESHOLDS = {
#     'market_cap': {
#         'megacap': 10_000_000_000,    # $10B+
#         'largecap': 1_000_000_000,    # $1B+
#         'midcap': 100_000_000,        # $100M+
#         'smallcap': 50_000_000        # $50M+
#     },
#     'transaction_amount': {
#         'very_large': 250_000,         # $250k+
#         'large': 100_000,              # $100k+
#         'medium': 50_000,              # $50k+
#         'small': 25_000                # $25k+
#     }
# }

# # Social Data Parameters
# SOCIAL_MENTION_WINDOWS = ['4H', '7d', '1m']
# SENTIMENT_THRESHOLDS = {
#     'very_positive': 0.5,
#     'positive': 0.2,
#     'neutral': -0.2,
#     'negative': -0.5
# }

# # Symbol Mapping for Data Integration
# SYMBOL_MAPPING = {
#     'BTC/USDT': ['BTC', 'BITCOIN', 'WBTC'],
#     'ETH/USDT': ['ETH', 'ETHEREUM', 'WETH'],
#     'SOL/USDT': ['SOL', 'SOLANA', 'WSOL']
# }

# # Reverse mapping for easy lookup
# TOKEN_TO_SYMBOL = {}
# for symbol, tokens in SYMBOL_MAPPING.items():
#     for token in tokens:
#         TOKEN_TO_SYMBOL[token] = symbol.split('/')[0]

# # Correlation threshold for feature pruning
# CORRELATION_THRESHOLD = 0.95

# # Validation parameters
# WALK_FORWARD_TRAIN_RATIO = 0.7  # 70% for train in each walk-forward window
# WALK_FORWARD_VAL_RATIO = 0.15   # 15% for validation
# WALK_FORWARD_TEST_RATIO = 0.15  # 15% for test