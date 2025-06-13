crypto_lgbm_tft/                 ←  root folder on Drive
│
├── data/
│   ├── raw/                     ← scraper drops 5-min shards here
│   │   ├── BTC/20250601/*.parquet
│   │   └── ETH/20250601/*.parquet
│   ├── feature_db/             ← DuckDB EXPORT of aggregated 5-min bars
│   │   └── *.parquet           ← one parquet per DuckDB stripe (auto-named)
│   └── scratch/                ← temporary files Colab can delete
│
├── notebooks/                  ← interactive exploration
│   ├── 00_quick_eda.ipynb
│   └── 99_demo_prediction.ipynb
│
├── pipelines/                  ← **pure-Python** (CLI) or shell scripts
│   ├── 01_build_features.py    ← Polars/DuckDB stream-aggregate
│   ├── 02_train_lgbm.py        ← LightGBM external-memory training
│   ├── 03_add_lgb_edge.py      ← write LightGBM edge back to DB
│   ├── 04_train_tft.py         ← PyTorch-Forecasting fine-tune
│   └── 05_backtest.py          ← walk-forward PnL evaluation
│
├── models/
│   ├── lightgbm/
│   │   ├── booster.bin
│   │   ├── shap_rank.csv
│   │   └── lgbm_params.json
│   └── tft/
│       ├── last.ckpt
│       ├── hparams.yaml
│       └── scaler.pkl
│
├── logs/
│   ├── lgbm_train_20250611.txt
│   └── tft_pl_logs/            ← lightning logs & TensorBoard files
│
├── configs/
│   ├── feature_config.yml      ← list of 300 feature expressions
│   └── train_config.yml        ← CV splits, target horizon, batch sizes
│
└── README.md
