# **Comprehensive Feature Table for Crypto Prediction Algorithm**

## **Feature Categories and Descriptions**

### **1\. Order Book Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `microprice` | Orderbook (bid/ask prices & sizes) | Weighted mid-price that better reflects true market value by considering order book imbalance |
| `microprice_momentum` | Derived from microprice | Captures directional movement of weighted price |
| `microprice_acceleration` | Derived from microprice | Detects changes in price momentum speed |
| `imbalance_top{1,3,5}` | Orderbook (multi-level sizes) | Measures buy/sell pressure at different order book depths |
| `imbalance_top{level}_zscore_{window}` | Derived from imbalances | Normalized imbalance relative to recent history |
| `bid_depth_slope` | Orderbook (prices & sizes) | Measures how quickly bid liquidity drops off with price |
| `ask_depth_slope` | Orderbook (prices & sizes) | Measures how quickly ask liquidity drops off with price |
| `depth_slope_ratio` | Derived from depth slopes | Compares bid vs ask liquidity profiles |
| `{bid,ask}_convexity` | Orderbook (sizes) | Measures concentration of liquidity at top levels |
| `book_pressure_pivot` | Orderbook \+ price direction | Combines order book pressure with price movement direction |
| `spread_mean` | Orderbook | Average bid-ask spread |
| `spread_std` | Orderbook | Volatility of spread |
| `relative_spread` | Orderbook \+ price | Spread normalized by price level |
| `spread_volatility` | Derived from spread | Measures stability of market making |
| `spread_momentum` | Derived from spread | Changes in spread over time |
| `high_spread_flag` | Derived from spread | Binary indicator of abnormally wide spreads |
| `quoted_volume_ratio` | Orderbook (bid/ask sizes) | Ratio of bid to ask liquidity |
| `log_quoted_volume_ratio` | Derived from volume ratio | Log-transformed for better distribution |

### **2\. Order Flow & Trade Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `cvd_last` | Trades (cumulative volume delta) | Net buying vs selling pressure |
| `cvd_ma_{window}` | Derived from CVD | Smoothed cumulative volume delta |
| `cvd_momentum_{window}` | Derived from CVD | Rate of change in net buying pressure |
| `cvd_acceleration_{window}` | Derived from CVD momentum | Acceleration of buying/selling pressure |
| `order_flow_imbalance` | Trades | Buy vs sell volume imbalance |
| `ofi_ma_{window}` | Derived from OFI | Smoothed order flow imbalance |
| `ofi_momentum` | Derived from OFI | Changes in order flow direction |
| `ofi_acceleration` | Derived from OFI momentum | Speed of order flow changes |
| `buy_trade_count` | Trades | Number of buy trades |
| `sell_trade_count` | Trades | Number of sell trades |
| `trade_count_imbalance_{window}` | Derived from trade counts | Imbalance in trade counts over time |
| `large_trade_ratio` | Trades (quantity stats) | Presence of unusually large trades |
| `vwap` | Trades | Volume-weighted average price |
| `vwap_ma_{window}` | Derived from VWAP | Smoothed VWAP |
| `vwap_deviation_{window}` | Price vs VWAP | How far price deviates from VWAP |
| `vpin` | Trades (buy/sell volumes) | Volume-synchronized probability of informed trading |

### **3\. Price Momentum Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `return_{period}` | Price | Simple returns over various periods |
| `log_return_{period}` | Price | Log returns for better statistical properties |
| `return_acceleration_{period}` | Derived from returns | Momentum of momentum |
| `price_tick` | Price | Direction of price movement (+1/-1) |
| `tick_momentum_{window}` | Derived from price ticks | Cumulative directional movements |
| `tick_momentum_norm_{window}` | Derived from tick momentum | Normalized tick momentum |
| `price_position_{window}` | Price | Where price sits in recent range (0-1) |
| `price_median_distance_{window}` | Price | Distance from rolling median |
| `near_high_{window}` | Derived from price position | Binary flag for price near recent high |
| `near_low_{window}` | Derived from price position | Binary flag for price near recent low |

### **4\. Volatility & Risk Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `volatility_{window}` | Price returns | Realized volatility |
| `volatility_ma_{window}` | Derived from volatility | Smoothed volatility |
| `vol_of_vol_{window}` | Derived from volatility | Volatility of volatility |
| `skew_{window}` | Price returns | Return distribution skewness |
| `kurt_{window}` | Price returns | Return distribution kurtosis |
| `parkinson_vol_{window}` | High/low prices | Range-based volatility estimator |

### **5\. Technical Indicators**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `rsi_{period}` | Price | Relative Strength Index |
| `rsi_{period}_momentum` | Derived from RSI | Changes in RSI |
| `rsi_{period}_oversold` | Derived from RSI | Binary flag for oversold condition |
| `rsi_{period}_overbought` | Derived from RSI | Binary flag for overbought condition |
| `bb_upper_{window}_{n_std}` | Price | Bollinger Band upper levels |
| `bb_lower_{window}_{n_std}` | Price | Bollinger Band lower levels |
| `bb_width_{window}` | Derived from BB | Bollinger Band width (volatility proxy) |
| `bb_position_{window}` | Price vs BB | Position within Bollinger Bands |
| `bb_squeeze_{window}` | Derived from BB width | Low volatility squeeze indicator |
| `macd_{fast}_{slow}` | Price | MACD line |
| `macd_signal_{fast}_{slow}` | Derived from MACD | MACD signal line |
| `macd_hist_{fast}_{slow}` | Derived from MACD | MACD histogram |
| `macd_cross_up_{fast}_{slow}` | Derived from MACD | Bullish MACD crossover |
| `atr_{period}` | High/low/close | Average True Range |
| `atr_ratio_{period}` | ATR vs price | Normalized ATR |

### **6\. Whale Transaction Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `whale_flow_imbalance` | Whale transactions | Net whale buying vs selling |
| `whale_flow_ma_{window}` | Derived from whale flow | Smoothed whale flow |
| `whale_flow_momentum_{window}` | Derived from whale flow | Changes in whale activity |
| `whale_flow_acceleration_{window}` | Derived from whale momentum | Acceleration of whale flows |
| `whale_flow_zscore_{window}` | Derived from whale flow | Normalized whale activity |
| `inst_participation_rate` | Whale classifications | Percentage of institutional vs retail |
| `inst_participation_zscore_{window}` | Derived from inst rate | Normalized institutional participation |
| `inst_participation_change_{period}` | Derived from inst rate | Changes in institutional activity |
| `inst_participation_acceleration` | Derived from inst changes | Speed of institutional changes |
| `extreme_inst_activity` | Derived from inst rate | Binary flag for high institutional activity |
| `retail_sell_pressure` | Whale classifications | Retail selling intensity |
| `retail_sell_ma_{window}` | Derived from retail pressure | Smoothed retail selling |
| `retail_sell_momentum_{window}` | Derived from retail pressure | Changes in retail selling |
| `retail_capitulation_signal` | Price \+ retail activity | Retail panic selling indicator |
| `retail_inst_divergence` | Whale classifications | Difference between retail and institutional behavior |
| `smart_dumb_divergence` | Whale classifications | Smart money vs dumb money divergence |
| `smart_money_strength` | Derived from divergence | Smoothed smart money signal |
| `smart_money_momentum` | Derived from divergence | Changes in smart money positioning |
| `smart_money_acceleration` | Derived from momentum | Speed of smart money changes |
| `smart_accumulation_signal` | Price \+ smart money | Smart money buying weakness |
| `smart_distribution_signal` | Price \+ smart money | Smart money selling strength |
| `whale_amount_usd_mean` | Whale transactions | Average whale transaction size |
| `whale_amount_usd_max` | Whale transactions | Largest whale transaction |
| `avg_whale_size_ma_{window}` | Derived from whale sizes | Smoothed average transaction size |
| `whale_size_ratio` | Derived from whale sizes | Current vs average transaction size |
| `mega_whale_activity` | Derived from max amounts | Extremely large whale transactions |
| `whale_size_momentum` | Derived from whale sizes | Changes in transaction sizes |
| `whale_market_alignment` | Whale \+ regular flow | Correlation between whale and market |
| `whale_leads_market` | Whale \+ regular flow | Whale activity leading market |
| `whale_contra_market` | Whale \+ regular flow | Whales trading against market |
| `whale_net_direction` | Whale buy/sell counts | Net whale direction |
| `whale_streak_length` | Derived from direction | Consecutive whale buying/selling |
| `whale_streak_strength` | Derived from streaks | Strength of whale persistence |
| `whale_sentiment_composite` | Multiple whale features | Combined whale sentiment score |
| `whale_extreme_bullish` | Derived from composite | Extreme bullish whale sentiment |
| `whale_extreme_bearish` | Derived from composite | Extreme bearish whale sentiment |
| `whale_buy_count` | Whale transactions | Number of whale buys |
| `whale_sell_count` | Whale transactions | Number of whale sells |
| `whale_buy_amount` | Whale transactions | Total whale buy volume |
| `whale_sell_amount` | Whale transactions | Total whale sell volume |
| `whale_volume_share` | Whale \+ regular volume | Whale percentage of total volume |
| `abnormal_whale_activity` | Derived from volume share | Unusually high whale participation |
| `megacap_flow` | Market cap \+ whale flow | Large cap token whale flows |
| `smallcap_speculation` | Market cap \+ retail | Small cap retail speculation |
| `flight_to_quality` | Market cap flows | Money moving to safer assets |

### **7\. Social Sentiment Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `social_momentum_score` | Social mentions | Combined social activity momentum |
| `social_momentum_ma_{window}` | Derived from momentum | Smoothed social momentum |
| `social_momentum_change_{window}` | Derived from momentum | Changes in social activity |
| `social_momentum_zscore_{window}` | Derived from momentum | Normalized social momentum |
| `social_acceleration` | Derived from changes | Speed of social momentum changes |
| `social_fomo_indicator` | Derived from momentum | Extreme social activity (FOMO) |
| `total_mentions` | Social data | Total mention count |
| `mention_velocity` | Derived from mentions | Rate of mention growth |
| `mention_acceleration` | Derived from velocity | Acceleration of mentions |
| `viral_indicator` | Derived from mentions | Viral threshold detection |
| `viral_persistence` | Derived from viral indicator | How long viral activity lasts |
| `mention_spike_zscore` | Derived from mentions | Abnormal mention spikes |
| `sentiment_14d` | Social sentiment data | 14-day sentiment score |
| `sentiment_momentum_{window}` | Derived from sentiment | Changes in sentiment |
| `sentiment_ma_{window}` | Derived from sentiment | Smoothed sentiment |
| `sentiment_volatility` | Derived from sentiment | Sentiment stability |
| `bullish_sentiment_regime` | Derived from sentiment | Persistent bullish sentiment |
| `bearish_sentiment_regime` | Derived from sentiment | Persistent bearish sentiment |
| `persistent_euphoria` | Sentiment indicators | Extended very positive sentiment |
| `persistent_despair` | Sentiment indicators | Extended very negative sentiment |
| `social_price_divergence_score` | Social \+ price | Divergence between social and price |
| `strong_bullish_divergence` | Social \+ price | Price down, sentiment up strongly |
| `strong_bearish_divergence` | Social \+ price | Price up, sentiment down strongly |
| `social_sentiment_composite` | Multiple social features | Combined social sentiment |
| `social_euphoria` | Derived from composite | Extreme positive social sentiment |
| `social_despair` | Derived from composite | Extreme negative social sentiment |
| `sentiment_weighted_mentions` | Mentions \+ sentiment | Sentiment-adjusted mention volume |
| `mention_sentiment_flow` | Mentions \+ sentiment | Net positive vs negative mentions |
| `social_volume_alignment` | Social \+ trading volume | Correlation between social and trading |
| `social_volume_divergence` | Social \+ trading volume | High social, low trading volume |
| `mention_momentum_{period}` | Social mentions | Mention momentum over different periods |

### **8\. Time-Based Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `hour` | Timestamp | Hour of day (0-23) |
| `day_of_week` | Timestamp | Day of week (0-6) |
| `is_weekend` | Timestamp | Weekend binary flag |
| `hour_sin` | Derived from hour | Sine encoding of hour |
| `hour_cos` | Derived from hour | Cosine encoding of hour |
| `dow_sin` | Derived from day | Sine encoding of day |
| `dow_cos` | Derived from day | Cosine encoding of day |
| `is_asian_session` | Derived from hour | Asian trading hours |
| `is_european_session` | Derived from hour | European trading hours |
| `is_us_session` | Derived from hour | US trading hours |
| `is_late_us_session` | Derived from hour | Late US trading hours |
| `session_transition` | Derived from sessions | Trading session changes |

### **9\. Composite & Interaction Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `composite_momentum` | Multiple momentum indicators | Combined momentum signal |
| `composite_pressure` | Multiple pressure indicators | Combined buying/selling pressure |
| `momentum_volume_interaction` | Momentum \+ volume | Price movement with volume confirmation |
| `imbalance_volatility_interaction` | Imbalance \+ volatility | Order book pressure in volatile markets |
| `spread_volume_pressure` | Spread \+ volume | Liquidity pressure indicator |
| `ofi_momentum_interaction` | Order flow \+ momentum | Order flow aligned with price movement |

### **10\. Microstructure Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `book_imbalance_mean` | Orderbook | Average order book imbalance |
| `book_imbalance_std` | Orderbook | Volatility of imbalance |
| `imbalance_ma_{window}` | Derived from imbalance | Smoothed imbalance |
| `imbalance_std_{window}` | Derived from imbalance | Imbalance volatility |
| `imbalance_momentum` | Derived from imbalance | Changes in imbalance |
| `volume` | Trades | Total trading volume |
| `volume_ma_{window}` | Derived from volume | Smoothed volume |
| `volume_ratio` | Volume vs MA | Current vs average volume |
| `volume_momentum` | Derived from volume | Volume growth rate |
| `volume_volatility` | Derived from volume | Volume stability |

### **11\. Target Variables**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `target_return_{horizon}` | Future price | Actual future returns (training target) |
| `target_direction_{horizon}` | Future price | Direction with deadband (-1, 0, 1\) |
| `target_confidence_{horizon}` | Future price | Distance from deadband threshold |
| `target_high_conf_{horizon}` | Future price | High confidence trade indicator |

### **12\. PCA & Smoothed Features**

| Feature Name | Data Source | Purpose in Prediction |
| ----- | ----- | ----- |
| `pca_component_{n}` | Multiple orderbook features | Principal components of order book |
| `pca_explained_variance_1` | PCA analysis | Variance explained by first component |
| `{feature}_savgol` | Various features | Savitzky-Golay filtered features |
| `{feature}_kalman` | Various features | Kalman filtered features |

## **Feature Summary by Data Source**

* **Orderbook Data**: 50+ features capturing market microstructure, liquidity, and order book dynamics  
* **Trade Data**: 30+ features capturing actual trading activity, volume patterns, and order flow  
* **Whale Transaction Data**: 40+ features capturing large player behavior and institutional vs retail dynamics  
* **Social Mention Data**: 30+ features capturing market sentiment and social momentum  
* **Derived/Calculated**: 100+ features created through transformations, interactions, and technical analysis  
* **Time-Based**: 12 features capturing temporal patterns and trading sessions

**Total Features**: \~300+ unique features designed to capture market dynamics from multiple angles for comprehensive price prediction

**Will these \~300 features be predictive?**

| Horizon | Likely High-Signal Blocks | Why | Notes / Caveats |
| ----- | ----- | ----- | ----- |
| **15 min** | Order-book microstructure (microprice, depth slopes, imbalance), short-window OFI/CVD, volume ratios | Reaction time of HFT & liquidity makers is measured in seconds; microstructure edge decays fastest | 10 days ≈ 2880 obs; that’s fine for tree models but too small for deep nets given 300 features. Regularise heavily. |
| **20 min** | Same as 15 min plus very short RSI/Tick momentum | Price discovery window is still microstructure-driven; technicals over ≤ 20 bars sometimes help | Expect feature importance rankings to look almost identical to 15 min if there’s real edge. |
| **60 min** | VWAP deviations, volatility regime flags, bird’s-eye OB imbalance (top 5 levels), whale\_flow\_momentum, smart/dumb divergence | Large whales can influence one-hour drift; wider spread & volume patterns matter more than single-level depth | Social-sentiment “spikes” begin to matter (FOMO, despair flags). |
| **120 min** | Social & sentiment composites, PCA order-book factors, volatility-of-vol, retail capitulation, session transition flags | Two hours lets information in Twitter/Reddit propagate; order-book edge partly decays | With only 10 days ⇒ 120 min horizon gives you \< 1200 rows – prune feature count or risk over-fitting. |

**Feature–predictive tests you should run**

| Test / Tool | What it tells you | How to run quickly |
| ----- | ----- | ----- |
| **Information Coefficient (IC)** | Spearman rank corr(feature, fwd return). Significance → raw predictive power | `df.groupby(date)[feature].rank()` then `.corr(target_return)` aggregate mean ± SEM |
| **Permutation Importance** | Drop-in-out impact on model performance | `sklearn.inspection.permutation_importance(model, X_test, y_test, n_repeats=30)` |
| **SHAP Values** | Local \+ global feature impact with directionality | `shap.TreeExplainer(model).shap_values(X)` ; plot summary |
| **Gain & Split Importance (LightGBM)** | Classic tree-model importance | `model.feature_importance('gain')` |
| **Boruta / RFECV** | Statistically test whether feature outperforms shadow features | `BorutaPy` on RandomForest or `RFECV` on LightGBM |

### 

**Models to Run**

1. **Start with Elastic Net & Random Forest** as sanity checks—fast, interpretable.\\  
2. **Graduate to XGBoost / LightGBM**—they usually deliver the first genuine edge on crypto microstructure.  
3. **Try CatBoost or Stacking** if LightGBM plateaus and you have time for extra tuning.  
4. **Only move to LSTM/TCN/TFT** once you collect more than \~30 days of 5-minute bars (≥ 9 k rows per asset) or augment with historical archives.  
5. **Hybrid Autoencoder \+ GBM** is a strong way to keep model capacity while respecting your smallish sample size.

| Algorithm / Approach | Strengths for Your 5-min Crypto Features | Typical Weaknesses & Pitfalls | Python Library / API | Rough Data Need\* |
| ----- | ----- | ----- | ----- | ----- |
| **Elastic Net / Ridge / Lasso** | • Fast baseline • Built-in feature shrinkage—helps when *p ≫ n* (300 features / 3 k rows) • Coefficients stay interpretable | • Linear↔can’t capture deep interactions • Sensitive to feature scaling | `sklearn.linear_model.ElasticNet` | **Small** (≤ 5 k rows) |
| **Random Forest** | • Handles non-linearities & feature interactions out-of-the-box • Native permutation importance | • Can over-fit on noisy micro-features • Slower inference than GBMs | `sklearn.ensemble.RandomForest*` | **Moderate** (5-50 k) |
| **XGBoost (GBDT)** | • State-of-the-art tabular accuracy • Missing-value handling; regularisation; SHAP built-in | • Hyper-param tuning critical • Can over-react to regime shifts | `xgboost.XGBClassifier/XGBRegressor` | **Moderate** (≥ 10 k ideal, still OK at 3 k with strong regularisation) |
| **LightGBM (GBDT)** | • Extremely fast training even with hundreds of features • Leaf-wise growth captures sharp effects in order-book data | • Needs careful `num_leaves` & learning-rate tuning on small samples | `lightgbm.LGBM*` | **Moderate** |
| **CatBoost (GBDT)** | • Handles categorical vars natively (handy if you one-hot sessions) • Out-of-the-box good defaults | • Slightly slower than LightGBM • Larger RAM footprint | `catboost.CatBoost*` | **Moderate** |
| **Stacking / Blending Ensemble** | • Combines linear \+ tree \+ NN viewpoints → higher robustness • Easy “meta” SHAP importance | • Risk of data-leakage—must use purged CV • Longer training pipeline | `sklearn.ensemble.StackingClassifier/Regressor` | **Moderate** |
| **LSTM / GRU RNN** | • Learns temporal dependencies directly (good for regime volatility) | • Over-fits with \< 50 k rows • Harder to interpret; slower | `tensorflow.keras`, `torch.nn` | **Large** (≥ 50 k sequences) |
| **Temporal Convolutional Network (TCN)** | • Parallel convolutions → faster than RNN, still captures long context | • Needs fixed-length windows; memory heavy with many channels | `pytorch-tcn`, `tensorflow-addons.tfa.layers.TCN` | **Large** |
| **Temporal Fusion Transformer (TFT)** | • Multi-horizon attention \+ static/volatility covariates \= perfect for 15-→120 min targets • Gives interpretable variable importances | •Parameter-hungry; needs \> 20 k rows to shine • GPU strongly recommended | `pytorch-forecasting.TemporalFusionTransformer` | **Large** (20 k \+ rows) |
| **Autoencoder ➜ LightGBM (Hybrid)** | • Non-linear dimensionality reduction cuts 300→\~40 dense latent factors, then GBM exploits them • Mitigates *p ≫ n* risk | • Two-stage training; latent code tuning can be fiddly | `keras` for AE \+ `lightgbm` | **Moderate-Large** (≥ 10 k rows ideal) |
| **Statistical Baselines (ARIMA / Prophet)** | • Quick sanity benchmark; highlights seasonality/day-of-week effects • Almost no data-hunger | • Ignores your rich order-book/whale features • Lags react slowly to microstructure shocks | `statsmodels.tsa.ARIMA` / `prophet` | **Small** (hundreds \+) |

### **Quick comparison table (focused on your new scale)**

| Model | Strengths on *millions* of rows | Weaknesses | Typical train time (10 M rows, single GPU) | Core Python pkg |
| ----- | ----- | ----- | ----- | ----- |
| ElasticNet | Audit-ready coefficients, quick sanity baseline | Can’t learn non-linear microstructure quirks | \< 2 min (CPU) | `sklearn` |
| **LightGBM (GPU)** | **State-of-the-art on dense tabular data; robust to outliers; built-in SHAP** | **Needs careful `num_leaves` on noisy order-book spikes** | **5–10 min** | **`lightgbm`** |
| CatBoost | Good defaults, handles categorical session features automatically | RAM-heavy on \>10 M rows | 10–15 min | `catboost` |
| Temporal Convolutional Network | Fast convolutions capture sub-minute patterns; parallelisable | Fixed input window; limited global context | 15 min | `torch`, `pytorch-tcn` |
| **Temporal Fusion Transformer** | **Multi-horizon, attention-based; interpretable variable selection** | **Parameter-hungry; longer tuning cycle** | **25-40 min** | **`pytorch-forecasting`** |
| Online models (River / Vowpal Wabbit) | Learn incrementally – perfect for live streaming | Lower asymptotic accuracy than GBMs / TFT | Continuous | `river`, `vowpalwabbit` |

### **TL;DR**

* **Elastic Net** remains your go-to explainer.  
* **LightGBM → TFT** is your best shot at raw predictive edge now that dataset size is no longer a constraint.  
* Treat LightGBM as both a production model *and* a feature-selection front-end for any deeper architecture you try next.

*I have 88GB of data with 5min orderbook, trades, social trends, whale transactions for BTCUSDT and ETHUSDT. I want to run LightGBM → TFT. How should I set this up? my scraper has pulled 10 days of 5min data into parquet files, saving each parquet every 5min to a drive directory*

### **0\. Prerequisites**

| Item | Minimum spec |
| ----- | ----- |
| **GPU** | 1× NVIDIA 16 GB+ (e.g. RTX 3090 / A6000). LightGBM GPU ≈ 6 GB; TFT ≈ 12 GB with 300 features, batch \= 128\. |
| **Disk** | SSD/NVMe with ≥ 200 GB free (raw 88 GB \+ feature Parquet \+ model checkpoints). |
| **Packages** | `lightgbm>=4.0` (built with `GPU=ON`), `pytorch>=2.2`, `pytorch-forecasting>=1.0`, `polars`, `duckdb`, `shap`, `scikit-learn`, `optuna` |

### **1\. Raw-to-feature parquet pipeline**

---

`data/`  
`└── raw/`  
    `├── symbol=BTC/date=2025-06-01/ob_…parq`  
    `└── symbol=ETH/…`  
`└── feats/`  
    `├── symbol=BTC/day=2025-06-01.parquet   # 5-min bar × 300 cols`  
    `└── symbol=ETH/…`  
---

1. **Lazy scan** raw files with **Polars** or **DuckDB** so you stream only one day at a time.  
2. Aggregate to 5-min bar, compute all 300 features, write *one* wide Parquet per symbol-day (`float32`).  
3. Append new files nightly; your training code always does  
   btcds \= pl.scan\_parquet("data/feats/symbol=BTC/\*.parquet").collect()  
   ethds \= pl.scan\_parquet("data/feats/symbol=ETH/\*.parquet").collect()  
   df \= pl.concat(\[btcds, ethdss\]).to\_pandas()  
   

### **2\. LightGBM (stage-1 edge finder)**

### ---

`import lightgbm as lgb, shap, numpy as np, pandas as pd`  
`from sklearn.model_selection import TimeSeriesSplit`

`features = [c for c in df.columns if not c.startswith("target")]`  
`X, y = df[features], df["target_direction_60"]  # e.g. 60-min direction`  
`tscv = TimeSeriesSplit(n_splits=5, test_size=288, gap=6)  # purged CV`

`oof_pred = np.zeros(len(df))`  
`for train_idx, test_idx in tscv.split(X):`  
    `lgb_train = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])`  
    `lgb_valid = lgb.Dataset(X.iloc[test_idx],  y.iloc[test_idx])`  
    `params = dict(`  
        `objective="binary", metric="auc",`  
        `device="gpu", gpu_platform_id=0, gpu_device_id=0,`  
        `num_leaves=1024, learning_rate=0.03,`  
        `feature_fraction=0.4, bagging_fraction=0.8,`  
        `reg_lambda=0.1`  
    `)`  
    `booster = lgb.train(params, lgb_train, 5000,`  
                        `valid_sets=[lgb_valid],`  
                        `early_stopping_rounds=300)`  
    `oof_pred[test_idx] = booster.predict(X.iloc[test_idx],`  
                                         `num_iteration=booster.best_iteration)`  
    `# keep model, SHAP for later`  
    `shap_values = shap.TreeExplainer(booster).shap_values(X.iloc[test_idx])[1]`

---

* LightGBM GPU install & tuning guide [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html?utm_source=chatgpt.com)  
* Simple example code [github.com](https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py?utm_source=chatgpt.com)

**Outputs saved**

* `booster.txt` – best model  
* `oof_pred.npy` – out-of-fold predictions (becomes a new feature)  
* `shap_rank.csv` – mean |SHAP| per feature (keep top-40 only)

Add `oof_pred` to every row:

`df["lgb_edge_60m"] = oof_pred`  
`df.to_parquet("train_w_lgb.parquet")`

### **3\. Prepare data for TFT**

1. **Normalise** continuous variables (including `lgb_edge_60m`) with a `StandardScaler` **per symbol**; save the scaler.  
2. Create an **integer time index**:  
   `df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 300`  
3. **Group identifiers**: `symbol`  
4. **Known-future covariates**: session flags (`is_us_session`…)  
5. **Observed covariates**: everything else, inc. `lgb_edge_60m`

### **4\. TFT (stage-2 sequence model)**

---

`from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer`  
`from pytorch_lightning import Trainer`

`max_encoder = 48   # 4 h history`  
`max_decoder = 12   # predict next 1 h (12×5 min)`

`ts_dataset = TimeSeriesDataSet(`  
    `df,`  
    `time_idx="time_idx",`  
    `target="target_return_60",`  
    `group_ids=["symbol"],`  
    `max_encoder_length=max_encoder,`  
    `max_prediction_length=max_decoder,`  
    `static_categoricals=["symbol"],`  
    `time_varying_known_categoricals=[],`  
    `time_varying_known_reals=["time_idx", "lgb_edge_60m"],`  
    `time_varying_unknown_reals=[c for c in df.columns`  
                                `if c not in ("target_return_60", "time_idx", "symbol")],`  
    `target_normalizer=None,`  
    `add_relative_time_idx=True,`  
    `add_target_scales=True,`  
    `add_encoder_length=True,`  
`)`  
`train, val = ts_dataset.split_before(0.8)`

`tft = TemporalFusionTransformer.from_dataset(`  
    `train,`  
    `learning_rate=1e-3,`  
    `hidden_size=64,`  
    `attention_head_size=4,`  
    `dropout=0.1,`  
    `output_size=1,`  
    `loss="QuantileLoss",`  
`)`

`trainer = Trainer(max_epochs=30, gradient_clip_val=0.1, accelerator="gpu")`  
`trainer.fit(tft, train_dataloader=train.to_dataloader(batch_size=128, num_workers=4),`  
                 `val_dataloaders=val.to_dataloader(batch_size=128, num_workers=4))`  
---

* TFT / TimeSeriesDataSet API docs [pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html?utm_source=chatgpt.com)[pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer._tft.TemporalFusionTransformer.html?utm_source=chatgpt.com)  
* Full “Stallion” TFT tutorial (setup & interpretation) [pytorch-forecasting.readthedocs.io](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html?utm_source=chatgpt.com)

**Interpretation**

`raw, x, y = tft.predict(val, mode="raw", return_x=True)`  
`tft.plot_interpretation(raw)      # variable importances over entire hold-out`  
`tft.plot_prediction(val, idx=0)   # attention heat-map for one sequence`

### **5\. Keep the training loop repeatable**

---

`repo/`  
`├── notebooks/`  
`├── data/            # raw + feature parquet`  
`├── models/`  
`│   ├── lightgbm/`  
`│   │   ├── booster.txt`  
`│   │   └── shap_rank.csv`  
`│   └── tft/`  
`│       ├── checkpoints/`  
`│       └── scaler.pkl`  
`└── pipelines/`  
    `├── 01_feature_build.py`  
    `├── 02_lightgbm_train.py`  
    `├── 03_tft_train.py`  
    `└── 04_backtest.py`  
---

Use **`make`** or **`prefect`** to chain scripts so nightly new data trigger:

1. Incremental feature build  
2. Incremental LightGBM retrain (warm start)  
3. TFT fine-tune last checkpoint on new window  
     
     
   

### **6\. Key hyper-parameters to watch**

| Component | Parameter | Good starting value | Why |
| ----- | ----- | ----- | ----- |
| LightGBM | `num_leaves` | 1024 (BTC/ETH) | Deep trees capture order-book spikes |
|  | `feature_fraction` | 0.4 | Random sub-features fights over-fit |
|  | `early_stopping_rounds` | 300 | Stops after \~5 min on GPU |
| TFT | `max_encoder_length` | 48 (4 h) | Covers two market sessions overlaps |
|  | `hidden_size` | 64 | Enough given 40–60 retained features |
|  | `batch_size` | 128 | Fits in 12 GB at fp32; halve if OOM |
| Both | **Walk-forward CV** | Purge gap ≥ 30 min | Removes overlap leakage |

## **TL;DR**

1. **Wide Parquet per symbol-day** → fastest I/O.  
2. **LightGBM (GPU)** → feature selection, SHAP, first set of edge predictions.  
3. Feed those predictions \+ top-40 raw features into a **Temporal Fusion Transformer** built with **PyTorch-Forecasting**.  
4. Interpret global importance from SHAP (LightGBM) and variable encoder weights (TFT) to track regime shifts.  
5. Automate the three-script pipeline so every new 5-min parquet can be turned into refreshed LightGBM → TFT models in \< 1 hour.

---

Looking at your approach, it's quite sophisticated and generally well-thought-out. The LightGBM → TFT pipeline is solid for crypto prediction, but I have several suggestions to improve accuracy and feasibility:

## **Strengths of Your Approach**

1. **Feature engineering is comprehensive** \- You're capturing microstructure, whale behavior, and social signals which are all crucial for crypto  
2. **Two-stage modeling** makes sense \- LightGBM for feature selection, then TFT for temporal patterns  
3. **Proper time series validation** with purged walk-forward CV

## **Key Improvements to Consider**

### **1\. Feature Engineering Refinements**

*\# Add these critical features that are missing:*  
\- Funding rates (perpetual futures)  
\- Open interest changes   
\- Exchange-specific order flow (Binance vs others)  
\- Stablecoin flows (USDT/USDC inflows)  
\- Cross-asset correlations (BTC-ETH rolling correlation)  
\- Options flow (if available) \- gamma exposure, put/call ratios

\- Network fees / mempool congestion

### **2\. Data Pipeline Optimization**

Instead of processing all 300 features upfront:

*\# Three-tier feature pipeline*  
tier1\_features \= \['microprice', 'imbalance\_top1', 'cvd\_last', ...\]  *\# \~50 fast features*  
tier2\_features \= \['whale\_flow\_momentum', 'social\_momentum', ...\]    *\# \~100 slower features*    
tier3\_features \= \['complex\_interactions', 'pca\_components', ...\]     *\# \~150 expensive features*

*\# Compute tier1 in real-time, tier2/3 async*

### **3\. Model Architecture Improvements**

**Replace single TFT with ensemble approach:**

*\# Ensemble of specialized models*  
models \= {  
    'microstructure': LightGBM(features=orderbook\_features, horizon='15min'),  
    'whale\_tracker': XGBoost(features=whale\_features, horizon='60min'),  
    'sentiment': LSTM(features=social\_features, horizon='120min'),  
    'technical': LightGBM(features=technical\_features, horizon='60min')  
}

*\# Meta-learner combines predictions*

meta\_model \= LightGBM(stacked\_predictions \+ market\_regime\_features)

### **4\. Critical Missing Elements**

**a) Market Regime Detection**

*\# Add regime classifier first*  
regimes \= \['trending', 'ranging', 'volatile', 'manipulated'\]  
regime\_model \= train\_regime\_classifier(volatility, volume\_profile, whale\_activity)

*\# Use different models per regime*

predictions \= regime\_specific\_models\[current\_regime\].predict(features)

**b) Anomaly Detection**

*\# Flag unusual market conditions*  
anomaly\_detector \= IsolationForest(contamination=0.01)  
anomaly\_score \= anomaly\_detector.decision\_function(current\_features)

*\# Reduce position size or skip trades when anomalous*

confidence \*= sigmoid(-anomaly\_score)

### **5\. Alternative Architecture: Hybrid CNN-LSTM**

For your data scale, consider this alternative:

*\# Process orderbook snapshots as 2D images*  
orderbook\_cnn \= Conv2D(filters\=32, kernel\_size\=(3,3))  
*\# Process time series with LSTM*  
temporal\_lstm \= LSTM(units\=128, return\_sequences\=True)  
*\# Combine with attention mechanism*

combined \= MultiHeadAttention()(cnn\_features, lstm\_features)

### **6\. Practical Implementation Tips**

1. **Start simpler**: Test with just top 50 features first  
2. **Use incremental learning**: Update models with new data rather than full retraining  
3. **Implement proper backtesting**: Include realistic slippage, fees, and market impact  
4. **Monitor feature drift**: Features that worked last month may not work now

python  
*\# Feature importance tracking*  
feature\_importance\_history \= track\_feature\_importance\_over\_time()

drop\_features\_with\_declining\_importance(threshold\=0.01)

### **7\. Data Quality Issues to Address**

*\# Critical preprocessing steps often missed:*  
\- Remove crossed quotes (bid \> ask)  
\- Filter flash crashes/wicks  
\- Normalize for tick size changes  
\- Account for exchange maintenance windows

\- Detect and handle wash trading

## **Most Important Changes**

If I had to pick the top 3 improvements:

1. **Add market regime detection** \- Different features matter in different market conditions  
2. **Include funding rates and stablecoin flows** \- These drive significant price movements in crypto  
3. **Use ensemble of specialized models** rather than one mega-model trying to learn everything

Your approach is solid, but crypto markets are adversarial \- what works today may be arbitraged away tomorrow. Build in mechanisms to detect when your edge is declining and adapt quickly.

