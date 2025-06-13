"""
Enhanced Model training module with better feature selection and whale feature prioritization
"""
import pandas as pd
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from config import *

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.selected_features = {}
        
    def remove_correlated_features(self, X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> List[str]:
        """Remove highly correlated features to reduce multicollinearity"""
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                   if any(upper_triangle[column] > threshold)]
        
        print(f"  Removing {len(to_drop)} highly correlated features (|corr| > {threshold})")
        return [col for col in X.columns if col not in to_drop]
        
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       method: str = 'importance', top_k: int = 50, 
                       max_samples: int = None,
                       feature_weights: Optional[Dict[str, float]] = None) -> List[str]:
        """Enhanced feature selection with correlation removal and whale feature weighting"""
        if max_samples is None:
            max_samples = FEATURE_SELECTION_MAX_SAMPLES
            
        print(f"\nSelecting top {top_k} features using {method} method...")
        
        # First remove highly correlated features
        non_correlated_features = self.remove_correlated_features(X_train)
        X_train_filtered = X_train[non_correlated_features]
        
        # Subsample if dataset is too large
        if len(X_train_filtered) > max_samples:
            print(f"  Subsampling from {len(X_train_filtered)} to {max_samples} rows...")
            idx = np.random.choice(len(X_train_filtered), max_samples, replace=False)
            X_sample = X_train_filtered.iloc[idx]
            y_sample = y_train.iloc[idx]
        else:
            X_sample = X_train_filtered
            y_sample = y_train
        
        # Fill any NaN values
        X_sample = X_sample.fillna(0)
        
        if method == 'importance':
            # Use ExtraTreesRegressor for feature importance
            rf = ExtraTreesRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=20,
                random_state=42, 
                n_jobs=-1
            )
            rf.fit(X_sample, y_sample)
            
            importances = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': rf.feature_importances_
            })
            
            # Apply whale feature weights if provided
            if feature_weights:
                for feature, weight in feature_weights.items():
                    mask = importances['feature'].str.contains(feature, na=False)
                    importances.loc[mask, 'importance'] *= weight
            
            importances = importances.sort_values('importance', ascending=False)
            selected = importances.head(top_k)['feature'].tolist()
            
        elif method == 'mutual_info':
            # Mutual information with whale feature weighting
            mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
            mi_df = pd.DataFrame({
                'feature': X_sample.columns,
                'mi_score': mi_scores
            })
            
            # Apply whale feature weights
            if feature_weights:
                for feature, weight in feature_weights.items():
                    mask = mi_df['feature'].str.contains(feature, na=False)
                    mi_df.loc[mask, 'mi_score'] *= weight
            
            mi_df = mi_df.sort_values('mi_score', ascending=False)
            selected = mi_df.head(top_k)['feature'].tolist()
            
        else:
            # F-statistic based selection
            selector = SelectKBest(f_regression, k=min(top_k, len(X_sample.columns)))
            selector.fit(X_sample, y_sample)
            selected = X_sample.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(selected)} features after correlation removal")
        return selected
    
    def optimize_hyperparameters(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Optimize hyperparameters using grid search with time series split"""
        print(f"  Optimizing {model_type} hyperparameters...")
        
        if model_type == 'xgb':
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'n_estimators': [100, 200]
            }
            base_model = xgb.XGBRegressor(**{k: v for k, v in XGB_PARAMS.items() 
                                           if k not in param_grid})
        
        elif model_type == 'lgb':
            param_grid = {
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'n_estimators': [100, 200]
            }
            base_model = lgb.LGBMRegressor(**{k: v for k, v in LGB_PARAMS.items() 
                                            if k not in param_grid})
        
        elif model_type == 'ridge':
            param_grid = {'alpha': RIDGE_PARAMS['alpha']}
            base_model = RidgeClassifier(random_state=42)
        
        elif model_type == 'mlp':
            param_grid = {
                'hidden_layer_sizes': MLP_PARAMS['hidden_layer_sizes'],
                'alpha': MLP_PARAMS['alpha'],
                'learning_rate_init': MLP_PARAMS['learning_rate_init']
            }
            base_model = MLPClassifier(
                activation='relu',
                solver='adam',
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            )
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Limit grid search for efficiency
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=tscv,
            scoring='neg_mean_squared_error' if model_type in ['xgb', 'lgb'] else 'accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit on combined train+val for grid search
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val]) if model_type in ['xgb', 'lgb'] else \
                     np.hstack([y_train, y_val])
        
        grid_search.fit(X_combined, y_combined)
        
        print(f"    Best params: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def train_ensemble_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           symbol: str, horizon: int = 12, 
                           optimize: bool = True) -> Tuple[Optional[Dict], Optional[Dict], 
                                                          Optional[Dict], Optional[pd.DataFrame]]:
        """Train ensemble model with enhanced feature selection and whale feature prioritization"""
        # Prepare features
        exclude = ['symbol', 'time_bucket', 'datetime', 'price'] + \
                 [c for c in train_df.columns if c.startswith('target_') or
                  c.startswith('mid_price') or c.startswith('price_') or
                  c.endswith('_first') or c.endswith('_last')]
        
        features = [c for c in train_df.columns if c not in exclude and
                   pd.api.types.is_numeric_dtype(train_df[c])]
        
        # Get train/val data for symbol
        tr = train_df[train_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
        va = val_df[val_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
        
        if len(tr) < MIN_SAMPLES_FOR_TRAINING or len(va) < 50:
            print(f"Insufficient data for {symbol} @ horizon {horizon}")
            return None, None, None, None
        
        X_tr = tr[features].fillna(0)
        y_tr = tr[f'target_return_{horizon}']
        X_va = va[features].fillna(0)
        y_va = va[f'target_return_{horizon}']
        
        # Calculate scaling factors from training data
        return_std = y_tr.std()
        ridge_scale_factor = return_std * 0.5
        mlp_scale_factor = return_std * 2
        
        # Feature categorization with proper limits
        technical_features = [f for f in features if any(ind in f for ind in
                            ['rsi_', 'macd_', 'bb_', 'volatility_', 'return_', 'momentum'])]
        
        microstructure_features = [f for f in features if any(ms in f for ms in
                                 ['spread', 'imbalance', 'volume', 'cvd', 'vwap', 'ofi'])]
        
        whale_features = [f for f in features if any(w in f for w in
                         ['whale', 'inst_', 'retail_', 'smart_dumb', 'megacap', 'smallcap'])]
        
        social_features = [f for f in features if any(s in f for s in
                          ['mention', 'sentiment', 'social', 'viral', 'euphoria'])]
        
        # Select features with category limits and whale weighting
        selected_technical = self.select_features(
            X_tr[technical_features] if technical_features else X_tr,
            y_tr,
            method='importance',
            top_k=min(MAX_FEATURES_PER_CATEGORY['technical'], len(technical_features))
        )
        
        selected_microstructure = self.select_features(
            X_tr[microstructure_features] if microstructure_features else X_tr,
            y_tr,
            method='importance',
            top_k=min(MAX_FEATURES_PER_CATEGORY['microstructure'], len(microstructure_features))
        )
        
        # Whale features with enhanced weighting
        selected_whale = []
        if whale_features:
            selected_whale = self.select_features(
                X_tr[whale_features],
                y_tr,
                method='importance',
                top_k=min(MAX_FEATURES_PER_CATEGORY['whale'], len(whale_features)),
                feature_weights=WHALE_FEATURE_WEIGHTS
            )
        
        selected_social = []
        if social_features:
            selected_social = self.select_features(
                X_tr[social_features],
                y_tr,
                method='importance',
                top_k=min(MAX_FEATURES_PER_CATEGORY['social'], len(social_features))
            )
        
        # Combine all selected features with total limit
        all_selected = list(set(selected_technical + selected_microstructure + 
                              selected_whale + selected_social))
        
        if len(all_selected) > MAX_FEATURES_PER_CATEGORY['total']:
            # Prioritize whale features in final selection
            whale_in_selected = [f for f in all_selected if f in selected_whale]
            other_features = [f for f in all_selected if f not in selected_whale]
            
            # Keep all whale features and fill rest with others
            max_other = MAX_FEATURES_PER_CATEGORY['total'] - len(whale_in_selected)
            all_selected = whale_in_selected + other_features[:max_other]
        
        print(f"\nFeature selection summary:")
        print(f"  Technical: {len([f for f in all_selected if f in selected_technical])}")
        print(f"  Microstructure: {len([f for f in all_selected if f in selected_microstructure])}")
        print(f"  Whale: {len([f for f in all_selected if f in selected_whale])}")
        print(f"  Social: {len([f for f in all_selected if f in selected_social])}")
        print(f"  Total selected: {len(all_selected)}")
        
        # Prepare feature sets
        X_tr_all = X_tr[all_selected]
        X_va_all = X_va[all_selected]
        
        # Create specialized feature sets
        whale_heavy_features = (selected_whale + 
                              selected_technical[:5] + 
                              selected_microstructure[:5])[:30]
        technical_features = selected_technical[:20]
        
        X_tr_whale = X_tr[whale_heavy_features] if len(whale_heavy_features) > 0 else X_tr_all
        X_va_whale = X_va[whale_heavy_features] if len(whale_heavy_features) > 0 else X_va_all
        
        X_tr_tech = X_tr[technical_features] if len(technical_features) > 0 else X_tr_all
        X_va_tech = X_va[technical_features] if len(technical_features) > 0 else X_va_all
        
        # Scale features
        scaler_all = RobustScaler()
        X_tr_all_scaled = scaler_all.fit_transform(X_tr_all)
        X_va_all_scaled = scaler_all.transform(X_va_all)
        
        scaler_whale = StandardScaler()
        X_tr_whale_scaled = scaler_whale.fit_transform(X_tr_whale)
        X_va_whale_scaled = scaler_whale.transform(X_va_whale)
        
        scaler_tech = RobustScaler()
        X_tr_tech_scaled = scaler_tech.fit_transform(X_tr_tech)
        X_va_tech_scaled = scaler_tech.transform(X_va_tech)
        
        # Convert to classification with deadband
        y_tr_class = np.where(y_tr > DIRECTION_DEADBAND, 1,
                            np.where(y_tr < -DIRECTION_DEADBAND, 0, -1))
        y_va_class = np.where(y_va > DIRECTION_DEADBAND, 1,
                            np.where(y_va < -DIRECTION_DEADBAND, 0, -1))
        
        # Remove no-trade samples for classification
        train_mask = y_tr_class != -1
        val_mask = y_va_class != -1
        
        print(f"Training diverse ensemble for {symbol} @ horizon {horizon}: {sum(train_mask)} samples")
        print(f"  Removed {sum(~train_mask)} no-trade samples (deadband: {DIRECTION_DEADBAND})")
        
        # Optimize hyperparameters if requested
        if optimize:
            xgb_params = self.optimize_hyperparameters(
                'xgb', X_tr_tech_scaled, y_tr, X_va_tech_scaled, y_va
            )
            lgb_params = self.optimize_hyperparameters(
                'lgb', X_tr_all_scaled, y_tr, X_va_all_scaled, y_va
            )
        else:
            xgb_params = {}
            lgb_params = {}
        
        # 1. XGBoost on technical features
        xgb_final_params = {**XGB_PARAMS, **xgb_params}
        xgbm = xgb.XGBRegressor(**xgb_final_params)
        xgbm.fit(X_tr_tech_scaled, y_tr,
                eval_set=[(X_va_tech_scaled, y_va)],
                verbose=False)
        
        # 2. LightGBM on all features (good with many features)
        lgb_final_params = {**LGB_PARAMS, **lgb_params}
        lgbm = lgb.LGBMRegressor(**lgb_final_params)
        lgbm.fit(
            X_tr_all_scaled,
            y_tr,
            eval_set=[(X_va_all_scaled, y_va)],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # 3. Ridge Classifier on whale-heavy features
        if optimize:
            ridge_params = self.optimize_hyperparameters(
                'ridge', X_tr_whale_scaled[train_mask], y_tr_class[train_mask],
                X_va_whale_scaled[val_mask], y_va_class[val_mask]
            )
            ridge = RidgeClassifier(**ridge_params)
        else:
            ridge = RidgeClassifier(alpha=1.0, random_state=42)
        ridge.fit(X_tr_whale_scaled[train_mask], y_tr_class[train_mask])
        
        # 4. Neural Network for non-linear patterns (especially whale)
        if optimize:
            mlp_params = self.optimize_hyperparameters(
                'mlp', X_tr_whale_scaled[train_mask], y_tr_class[train_mask],
                X_va_whale_scaled[val_mask], y_va_class[val_mask]
            )
            mlp = MLPClassifier(**{**MLP_PARAMS, **mlp_params})
        else:
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size=64,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42,
                verbose=False
            )
        mlp.fit(X_tr_whale_scaled[train_mask], y_tr_class[train_mask])
        
        # Store models
        models = {
            'xgb': xgbm,
            'lgb': lgbm,
            'ridge': (ridge, ridge_scale_factor),
            'mlp': (mlp, mlp_scale_factor)
        }
        
        scalers = {
            'tech': scaler_tech,
            'whale': scaler_whale,
            'all': scaler_all
        }
        
        feature_sets = {
            'tech': technical_features,
            'whale': whale_heavy_features,
            'all': all_selected
        }
        
        # Calculate feature importance with whale weighting
        feature_importance = self._calculate_weighted_importance(
            xgbm, lgbm, all_selected, selected_whale
        )
        
        return models, scalers, feature_sets, feature_importance
    
    def _calculate_weighted_importance(self, xgb_model, lgb_model, 
                                     all_features: List[str], 
                                     whale_features: List[str]) -> pd.DataFrame:
        """Calculate feature importance with whale feature boosting"""
        # Get importances from tree models
        xgb_importance = xgb_model.feature_importances_ if hasattr(xgb_model, 'feature_importances_') else np.zeros(len(all_features))
        lgb_importance = lgb_model.feature_importances_ if hasattr(lgb_model, 'feature_importances_') else np.zeros(len(all_features))
        
        # Average importances
        avg_importance = (xgb_importance[:len(all_features)] + lgb_importance[:len(all_features)]) / 2
        
        # Apply whale feature boost
        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': avg_importance
        })
        
        # Boost whale features
        for idx, feature in enumerate(all_features):
            if feature in whale_features:
                for whale_pattern, weight in WHALE_FEATURE_WEIGHTS.items():
                    if whale_pattern in feature:
                        importance_df.loc[idx, 'importance'] *= weight
                        break
        
        return importance_df.sort_values('importance', ascending=False)
    
    def train_meta_learner(self, models: Dict, scalers: Dict, feature_sets: Dict,
                          train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          symbol: str, horizon: int) -> Optional[Any]:
        """Enhanced meta-learner with better regularization"""
        # Get data
        tr = train_df[train_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
        va = val_df[val_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
        
        if len(tr) < MIN_SAMPLES_FOR_TRAINING or len(va) < 50:
            return None
        
        y_tr = tr[f'target_return_{horizon}']
        
        # Calculate return statistics
        overall_return_std = y_tr.std()
        
        # Get out-of-fold predictions
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        oof_preds = np.zeros((len(tr), 4))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(tr)):
            fold_tr = tr.iloc[train_idx]
            fold_va = tr.iloc[val_idx]
            
            y_fold_tr = fold_tr[f'target_return_{horizon}']
            y_fold_tr_class = np.where(y_fold_tr > DIRECTION_DEADBAND, 1,
                                      np.where(y_fold_tr < -DIRECTION_DEADBAND, 0, -1))
            
            # Remove no-trade samples
            mask = y_fold_tr_class != -1
            
            fold_return_std = y_fold_tr.std()
            
            # Prepare features
            X_fold_tr_tech = fold_tr[feature_sets['tech']].fillna(0)
            X_fold_va_tech = fold_va[feature_sets['tech']].fillna(0)
            
            X_fold_tr_whale = fold_tr[feature_sets['whale']].fillna(0)
            X_fold_va_whale = fold_va[feature_sets['whale']].fillna(0)
            
            X_fold_tr_all = fold_tr[feature_sets['all']].fillna(0)
            X_fold_va_all = fold_va[feature_sets['all']].fillna(0)
            
            # Scale features
            scaler_tech_fold = RobustScaler()
            X_fold_tr_tech_scaled = scaler_tech_fold.fit_transform(X_fold_tr_tech)
            X_fold_va_tech_scaled = scaler_tech_fold.transform(X_fold_va_tech)
            
            scaler_whale_fold = StandardScaler()
            X_fold_tr_whale_scaled = scaler_whale_fold.fit_transform(X_fold_tr_whale)
            X_fold_va_whale_scaled = scaler_whale_fold.transform(X_fold_va_whale)
            
            scaler_all_fold = RobustScaler()
            X_fold_tr_all_scaled = scaler_all_fold.fit_transform(X_fold_tr_all)
            X_fold_va_all_scaled = scaler_all_fold.transform(X_fold_va_all)
            
            # Train fold models
            xgb_fold = xgb.XGBRegressor(**XGB_PARAMS)
            xgb_fold.fit(X_fold_tr_tech_scaled, y_fold_tr, verbose=False)
            oof_preds[val_idx, 0] = xgb_fold.predict(X_fold_va_tech_scaled)
            
            lgb_fold = lgb.LGBMRegressor(**LGB_PARAMS)
            lgb_fold.fit(X_fold_tr_all_scaled, y_fold_tr, callbacks=[lgb.log_evaluation(0)])
            oof_preds[val_idx, 1] = lgb_fold.predict(X_fold_va_all_scaled)
            
            ridge_fold = RidgeClassifier(alpha=1.0, random_state=42)
            ridge_fold.fit(X_fold_tr_whale_scaled[mask], y_fold_tr_class[mask])
            ridge_proba = ridge_fold.decision_function(X_fold_va_whale_scaled)
            ridge_scale_factor = fold_return_std * 0.5
            oof_preds[val_idx, 2] = ridge_proba * ridge_scale_factor
            
            mlp_fold = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=100,
                random_state=42,
                verbose=False
            )
            mlp_fold.fit(X_fold_tr_whale_scaled[mask], y_fold_tr_class[mask])
            
            try:
                mlp_proba = mlp_fold.predict_proba(X_fold_va_whale_scaled)[:, 1]
            except:
                mlp_proba = np.zeros(len(X_fold_va_whale_scaled))
            
            mlp_centered = mlp_proba - 0.5
            mlp_scale_factor = fold_return_std * 2
            oof_preds[val_idx, 3] = mlp_centered * mlp_scale_factor
        
        # Analyze predictions
        print(f"\nOut-of-fold prediction statistics:")
        print(f"XGB:   mean={np.mean(oof_preds[:, 0]):.6f}, std={np.std(oof_preds[:, 0]):.6f}")
        print(f"LGB:   mean={np.mean(oof_preds[:, 1]):.6f}, std={np.std(oof_preds[:, 1]):.6f}")
        print(f"Ridge: mean={np.mean(oof_preds[:, 2]):.6f}, std={np.std(oof_preds[:, 2]):.6f}")
        print(f"MLP:   mean={np.mean(oof_preds[:, 3]):.6f}, std={np.std(oof_preds[:, 3]):.6f}")
        print(f"Target: mean={y_tr.mean():.6f}, std={overall_return_std:.6f}")
        
        # Train meta-learner with Lasso for sparsity
        meta_features = pd.DataFrame(oof_preds, columns=['xgb', 'lgb', 'ridge', 'mlp'])
        
        # Add interaction features
        meta_features['xgb_lgb'] = meta_features['xgb'] * meta_features['lgb']
        meta_features['whale_models'] = (meta_features['ridge'] + meta_features['mlp']) / 2
        
        # Use Lasso for sparse weights
        meta_learner = Lasso(alpha=0.001, random_state=42, max_iter=1000)
        meta_learner.fit(meta_features, y_tr)
        
        # Print weights
        weights = meta_learner.coef_[:4]
        print(f"Meta-learner weights: {weights}")
        
        # If weights are too sparse, use predefined whale-priority weights
        if np.sum(np.abs(weights)) < 0.1:
            print("  Using predefined whale-priority weights due to sparse solution")
            weights = np.array([
                ENSEMBLE_WEIGHTS['whale_priority']['xgb'],
                ENSEMBLE_WEIGHTS['whale_priority']['lgb'],
                ENSEMBLE_WEIGHTS['whale_priority']['ridge'],
                ENSEMBLE_WEIGHTS['whale_priority']['mlp']
            ])
            # Create simple weighted ensemble
            meta_learner = type('WeightedEnsemble', (), {
                'coef_': np.concatenate([weights, np.zeros(2)]),
                'predict': lambda self, X: np.dot(X, self.coef_)
            })()
        
        return meta_learner
    
    def get_predictions(self, models: Dict, scalers: Dict, feature_sets: Dict,
                       test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all models with proper scaling"""
        # Prepare test features
        X_test_tech = test_data[feature_sets['tech']].fillna(0)
        X_test_whale = test_data[feature_sets['whale']].fillna(0)
        X_test_all = test_data[feature_sets['all']].fillna(0)
        
        # Scale features
        X_test_tech_scaled = scalers['tech'].transform(X_test_tech)
        X_test_whale_scaled = scalers['whale'].transform(X_test_whale)
        X_test_all_scaled = scalers['all'].transform(X_test_all)
        
        # Get predictions
        xgb_preds = models['xgb'].predict(X_test_tech_scaled)
        lgb_preds = models['lgb'].predict(X_test_all_scaled)
        
        # Ridge predictions
        if isinstance(models['ridge'], tuple):
            ridge_model, ridge_scale = models['ridge']
            ridge_preds = ridge_model.decision_function(X_test_whale_scaled) * ridge_scale
        else:
            ridge_preds = models['ridge'].decision_function(X_test_whale_scaled) * 0.001
        
        # MLP predictions
        if isinstance(models['mlp'], tuple):
            mlp_model, mlp_scale = models['mlp']
            try:
                mlp_proba = mlp_model.predict_proba(X_test_whale_scaled)[:, 1]
            except:
                mlp_proba = np.zeros(len(X_test_whale_scaled))
            mlp_preds = (mlp_proba - 0.5) * mlp_scale
        else:
            mlp_proba = models['mlp'].predict_proba(X_test_whale_scaled)[:, 1]
            mlp_preds = (mlp_proba - 0.5) * 0.002
        
        return {
            'xgb': xgb_preds,
            'lgb': lgb_preds,
            'ridge': ridge_preds,
            'mlp': mlp_preds
        }

# """
# Model training module for ensemble models
# """
# import pandas as pd
# import numpy as np
# import gc
# from typing import Dict, List, Tuple, Optional, Any
# from sklearn.preprocessing import RobustScaler, StandardScaler
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.linear_model import RidgeClassifier, LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import KFold
# import xgboost as xgb
# import lightgbm as lgb

# from config import *

# class ModelTrainer:
#     def __init__(self):
#         self.models = {}
#         self.scalers = {}
#         self.feature_importances = {}
#         self.selected_features = {}
        
#     def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
#                        method: str = 'importance', top_k: int = 50, 
#                        max_samples: int = None) -> List[str]:
#         """Feature selection using various methods - OPTIMIZED with subsampling"""
#         if max_samples is None:
#             max_samples = FEATURE_SELECTION_MAX_SAMPLES
            
#         print(f"\nSelecting top {top_k} features using {method} method...")
        
#         # Subsample if dataset is too large
#         if len(X_train) > max_samples:
#             print(f"  Subsampling from {len(X_train)} to {max_samples} rows for feature selection...")
#             idx = np.random.choice(len(X_train), max_samples, replace=False)
#             X_sample = X_train.iloc[idx]
#             y_sample = y_train.iloc[idx]
#         else:
#             X_sample = X_train
#             y_sample = y_train
        
#         # Fill any NaN values defensively
#         X_sample = X_sample.fillna(0)
        
#         if method == 'importance':
#             # Use ExtraTreesRegressor for faster feature importance
#             rf = ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
#             rf.fit(X_sample, y_sample)
            
#             importances = pd.DataFrame({
#                 'feature': X_train.columns,
#                 'importance': rf.feature_importances_
#             }).sort_values('importance', ascending=False)
            
#             selected = importances.head(top_k)['feature'].tolist()
            
#         elif method == 'mutual_info':
#             # Defensive fillna for mutual info
#             mi_scores = mutual_info_regression(X_sample.fillna(0), y_sample, random_state=42)
#             mi_scores = pd.DataFrame({
#                 'feature': X_train.columns,
#                 'mi_score': mi_scores
#             }).sort_values('mi_score', ascending=False)
            
#             selected = mi_scores.head(top_k)['feature'].tolist()
            
#         else:
#             # Default: correlation-based
#             correlations = X_sample.corrwith(y_sample).abs().sort_values(ascending=False)
#             selected = correlations.head(top_k).index.tolist()
        
#         print(f"Selected {len(selected)} features")
#         return selected
    
#     def train_ensemble_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
#                            symbol: str, horizon: int = 12, 
#                            optimize: bool = False) -> Tuple[Optional[Dict], Optional[Dict], 
#                                                           Optional[Dict], Optional[pd.DataFrame]]:
#         """Train ensemble model with diverse algorithms - FIXED with scale factors"""
#         # Prepare features
#         exclude = ['symbol', 'time_bucket', 'datetime', 'price'] + \
#                  [c for c in train_df.columns if c.startswith('target_') or
#                   c.startswith('mid_price') or c.startswith('price_') or
#                   c.endswith('_first') or c.endswith('_last')]
        
#         features = [c for c in train_df.columns if c not in exclude and
#                    pd.api.types.is_numeric_dtype(train_df[c])]
        
#         # Get train/val data for symbol
#         tr = train_df[train_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
#         va = val_df[val_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
        
#         if len(tr) < 100 or len(va) < 20:
#             print(f"Insufficient data for {symbol} @ horizon {horizon}")
#             return None, None, None, None
        
#         X_tr = tr[features].fillna(0)
#         y_tr = tr[f'target_return_{horizon}']
#         X_va = va[features].fillna(0)
#         y_va = va[f'target_return_{horizon}']
        
#         # Calculate scaling factors from training data
#         return_std = y_tr.std()
#         ridge_scale_factor = return_std * 0.5  # Conservative scaling
#         mlp_scale_factor = return_std * 2      # Map full probability range to ~2 std devs
        
#         # Feature selection - split features for different models
#         technical_features = [f for f in features if any(ind in f for ind in
#                             ['rsi_', 'macd_', 'bb_', 'volatility_', 'return_', 'momentum'])]
        
#         microstructure_features = [f for f in features if any(ms in f for ms in
#                                  ['spread', 'imbalance', 'volume', 'cvd', 'vwap', 'ofi'])]
        
#         whale_features = [f for f in features if any(w in f for w in
#                          ['whale', 'inst_', 'retail_', 'smart_dumb', 'megacap', 'smallcap'])]
        
#         social_features = [f for f in features if any(s in f for s in
#                           ['mention', 'sentiment', 'social', 'viral', 'euphoria'])]
        
#         # Ensure we have features in each category
#         if not technical_features:
#             technical_features = features[:30] if len(features) >= 30 else features
        
#         if not microstructure_features:
#             microstructure_features = features[30:60] if len(features) >= 60 else features[30:]
        
#         # Select top features from each category
#         selected_technical = self.select_features(X_tr[technical_features], y_tr,
#                                                 method='importance', top_k=min(30, len(technical_features)))
#         selected_microstructure = self.select_features(X_tr[microstructure_features], y_tr,
#                                                       method='importance', top_k=min(30, len(microstructure_features)))
        
#         # Include whale and social features if available
#         selected_whale = []
#         if whale_features:
#             selected_whale = self.select_features(X_tr[whale_features], y_tr,
#                                                 method='importance', top_k=min(20, len(whale_features)))
        
#         selected_social = []
#         if social_features:
#             selected_social = self.select_features(X_tr[social_features], y_tr,
#                                                  method='importance', top_k=min(20, len(social_features)))
        
#         # Combine all selected features
#         all_selected = list(set(selected_technical + selected_microstructure + selected_whale + selected_social))
        
#         # If we still have too many features, select top overall
#         if len(all_selected) > 100:
#             all_selected = self.select_features(X_tr[all_selected], y_tr, 
#                                               method='importance', top_k=100)
        
#         print(f"Feature selection summary:")
#         print(f"  Technical: {len(selected_technical)}")
#         print(f"  Microstructure: {len(selected_microstructure)}")
#         print(f"  Whale: {len(selected_whale)}")
#         print(f"  Social: {len(selected_social)}")
#         print(f"  Total selected: {len(all_selected)}")
        
#         # Prepare different feature sets
#         X_tr_tech = X_tr[selected_technical]
#         X_va_tech = X_va[selected_technical]
        
#         X_tr_micro = X_tr[selected_microstructure]
#         X_va_micro = X_va[selected_microstructure]
        
#         X_tr_all = X_tr[all_selected]
#         X_va_all = X_va[all_selected]
        
#         # Scale features
#         scaler_tech = RobustScaler()
#         X_tr_tech_scaled = scaler_tech.fit_transform(X_tr_tech)
#         X_va_tech_scaled = scaler_tech.transform(X_va_tech)
        
#         scaler_micro = StandardScaler()
#         X_tr_micro_scaled = scaler_micro.fit_transform(X_tr_micro)
#         X_va_micro_scaled = scaler_micro.transform(X_va_micro)
        
#         scaler_all = RobustScaler()
#         X_tr_all_scaled = scaler_all.fit_transform(X_tr_all)
#         X_va_all_scaled = scaler_all.transform(X_va_all)
        
#         # Convert to classification for some models
#         y_tr_class = (y_tr > 0).astype(int)
#         y_va_class = (y_va > 0).astype(int)
        
#         # Train diverse models
#         print(f"Training diverse ensemble for {symbol} @ horizon {horizon}: {len(X_tr)} samples")
        
#         # 1. XGBoost on technical features
#         xgbm = xgb.XGBRegressor(**XGB_PARAMS)
#         xgbm.fit(X_tr_tech_scaled, y_tr,
#                 eval_set=[(X_va_tech_scaled, y_va)],
#                 verbose=False)
        
#         # 2. LightGBM on microstructure features
#         lgbm = lgb.LGBMRegressor(**LGB_PARAMS)
#         lgbm.fit(
#             X_tr_micro_scaled,
#             y_tr,
#             eval_set=[(X_va_micro_scaled, y_va)],
#             callbacks=[lgb.log_evaluation(0)]  # This suppresses output
#         )
        
#         # 3. Ridge Classifier for linear patterns
#         ridge = RidgeClassifier(alpha=1.0, random_state=42)
#         ridge.fit(X_tr_micro_scaled, y_tr_class)
        
#         # 4. Neural Network for non-linear patterns
#         mlp = MLPClassifier(
#             hidden_layer_sizes=(50, 25),
#             activation='relu',
#             solver='adam',
#             alpha=0.01,
#             batch_size=32,
#             learning_rate='adaptive',
#             learning_rate_init=0.001,
#             max_iter=100,
#             early_stopping=True,
#             validation_fraction=0.2,
#             random_state=42,
#             verbose=False
#         )
#         mlp.fit(X_tr_all_scaled, y_tr_class)
        
#         # Store models with their scaling factors
#         models = {
#             'xgb': xgbm,
#             'lgb': lgbm,
#             'ridge': (ridge, ridge_scale_factor),  # Tuple with model and scale factor
#             'mlp': (mlp, mlp_scale_factor)         # Tuple with model and scale factor
#         }
        
#         scalers = {
#             'tech': scaler_tech,
#             'micro': scaler_micro,
#             'all': scaler_all
#         }
        
#         feature_sets = {
#             'tech': selected_technical,
#             'micro': selected_microstructure,
#             'all': all_selected
#         }
        
#         # Calculate feature importance
#         xgb_importance = np.zeros(len(all_selected))
#         if hasattr(xgbm, 'feature_importances_'):
#             for i, feat in enumerate(selected_technical):
#                 if feat in all_selected:
#                     idx = all_selected.index(feat)
#                     if i < len(xgbm.feature_importances_):
#                         xgb_importance[idx] = xgbm.feature_importances_[i]
        
#         feature_importance = pd.DataFrame({
#             'feature': all_selected,
#             'importance': xgb_importance
#         })
        
#         return models, scalers, feature_sets, feature_importance
    
#     def train_meta_learner(self, models: Dict, scalers: Dict, feature_sets: Dict,
#                           train_df: pd.DataFrame, val_df: pd.DataFrame, 
#                           symbol: str, horizon: int) -> Optional[Any]:
#         """Train a meta-learner with proper K-fold to avoid leakage"""
#         # Get data
#         tr = train_df[train_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
#         va = val_df[val_df['symbol'] == symbol].dropna(subset=[f'target_return_{horizon}'])
        
#         if len(tr) < 100 or len(va) < 20:
#             return None
        
#         y_tr = tr[f'target_return_{horizon}']
        
#         # Calculate overall return statistics for scaling
#         overall_return_std = y_tr.std()
#         overall_return_mean = y_tr.mean()
        
#         # Get out-of-fold predictions for training meta-learner
#         n_splits = 5
#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
#         # Store out-of-fold predictions
#         oof_preds = np.zeros((len(tr), 4))  # 4 models
        
#         for fold, (train_idx, val_idx) in enumerate(kf.split(tr)):
#             fold_tr = tr.iloc[train_idx]
#             fold_va = tr.iloc[val_idx]
            
#             # Get fold targets
#             y_fold_tr = fold_tr[f'target_return_{horizon}']
#             y_fold_va = fold_va[f'target_return_{horizon}']
#             y_fold_tr_class = (y_fold_tr > 0).astype(int)
            
#             # Calculate fold-specific return statistics for better scaling
#             fold_return_std = y_fold_tr.std()
#             fold_return_mean = y_fold_tr.mean()
            
#             # Prepare features for each model type
#             X_fold_tr_tech = fold_tr[feature_sets['tech']].fillna(0)
#             X_fold_va_tech = fold_va[feature_sets['tech']].fillna(0)
            
#             X_fold_tr_micro = fold_tr[feature_sets['micro']].fillna(0)
#             X_fold_va_micro = fold_va[feature_sets['micro']].fillna(0)
            
#             X_fold_tr_all = fold_tr[feature_sets['all']].fillna(0)
#             X_fold_va_all = fold_va[feature_sets['all']].fillna(0)
            
#             # Create new scalers for this fold
#             scaler_tech_fold = RobustScaler()
#             X_fold_tr_tech_scaled = scaler_tech_fold.fit_transform(X_fold_tr_tech)
#             X_fold_va_tech_scaled = scaler_tech_fold.transform(X_fold_va_tech)
            
#             scaler_micro_fold = StandardScaler()
#             X_fold_tr_micro_scaled = scaler_micro_fold.fit_transform(X_fold_tr_micro)
#             X_fold_va_micro_scaled = scaler_micro_fold.transform(X_fold_va_micro)
            
#             scaler_all_fold = RobustScaler()
#             X_fold_tr_all_scaled = scaler_all_fold.fit_transform(X_fold_tr_all)
#             X_fold_va_all_scaled = scaler_all_fold.transform(X_fold_va_all)
            
#             # Train fold-specific models to avoid leakage
#             # 1. XGBoost - already outputs in return scale
#             xgb_fold = xgb.XGBRegressor(**XGB_PARAMS)
#             xgb_fold.fit(X_fold_tr_tech_scaled, y_fold_tr, verbose=False)
#             oof_preds[val_idx, 0] = xgb_fold.predict(X_fold_va_tech_scaled)
            
#             # 2. LightGBM - already outputs in return scale
#             lgb_fold = lgb.LGBMRegressor(**LGB_PARAMS)
#             lgb_fold.fit(
#                 X_fold_tr_micro_scaled,
#                 y_fold_tr,
#                 callbacks=[lgb.log_evaluation(0)]  # This suppresses output
#             )
#             oof_preds[val_idx, 1] = lgb_fold.predict(X_fold_va_micro_scaled)
            
#             # 3. Ridge - scale predictions to return scale using data-driven approach
#             ridge_fold = RidgeClassifier(alpha=1.0, random_state=42)
#             ridge_fold.fit(X_fold_tr_micro_scaled, y_fold_tr_class)
#             ridge_proba = ridge_fold.decision_function(X_fold_va_micro_scaled)
            
#             # Scale Ridge predictions based on actual return distribution
#             ridge_scale_factor = fold_return_std * 0.5  # Conservative scaling
#             oof_preds[val_idx, 2] = ridge_proba * ridge_scale_factor
            
#             # 4. MLP - scale predictions to return scale using data-driven approach
#             mlp_fold = MLPClassifier(
#                 hidden_layer_sizes=(50, 25),
#                 activation='relu',
#                 solver='adam',
#                 alpha=0.01,
#                 max_iter=50,
#                 random_state=42,
#                 verbose=False
#             )
#             mlp_fold.fit(X_fold_tr_all_scaled, y_fold_tr_class)
            
#             # Handle case where predict_proba might only have one class
#             try:
#                 mlp_proba = mlp_fold.predict_proba(X_fold_va_all_scaled)[:, 1]
#             except IndexError:
#                 # If only one class predicted, use zeros
#                 mlp_proba = np.zeros(len(X_fold_va_all_scaled))
            
#             # Center and scale MLP predictions based on return distribution
#             mlp_centered = mlp_proba - 0.5  # Now in [-0.5, 0.5]
#             mlp_scale_factor = fold_return_std * 2  # Map full probability range to ~2 std devs
#             oof_preds[val_idx, 3] = mlp_centered * mlp_scale_factor
        
#         # Analyze the prediction distributions before meta-learning
#         print(f"\nOut-of-fold prediction statistics:")
#         print(f"XGB:   mean={np.mean(oof_preds[:, 0]):.6f}, std={np.std(oof_preds[:, 0]):.6f}")
#         print(f"LGB:   mean={np.mean(oof_preds[:, 1]):.6f}, std={np.std(oof_preds[:, 1]):.6f}")
#         print(f"Ridge: mean={np.mean(oof_preds[:, 2]):.6f}, std={np.std(oof_preds[:, 2]):.6f}")
#         print(f"MLP:   mean={np.mean(oof_preds[:, 3]):.6f}, std={np.std(oof_preds[:, 3]):.6f}")
#         print(f"Target: mean={overall_return_mean:.6f}, std={overall_return_std:.6f}")
        
#         # Train meta-learner on out-of-fold predictions
#         meta_features = pd.DataFrame(oof_preds, columns=['xgb', 'lgb', 'ridge', 'mlp'])
        
#         # Add interaction features
#         meta_features['xgb_lgb'] = meta_features['xgb'] * meta_features['lgb']
#         meta_features['tree_linear'] = (meta_features['xgb'] + meta_features['lgb']) * meta_features['ridge']
        
#         # Train meta-learner (simple linear model to avoid overfitting)
#         meta_learner = LogisticRegression(C=1.0, random_state=42)
#         meta_y = (y_tr > 0).astype(int)
#         meta_learner.fit(meta_features, meta_y)
        
#         print(f"Meta-learner weights: {meta_learner.coef_[0][:4]}")
        
#         return meta_learner
    
#     def get_predictions(self, models: Dict, scalers: Dict, feature_sets: Dict,
#                        test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
#         """Get predictions from all models with proper scaling"""
#         # Prepare test features
#         X_test_tech = test_data[feature_sets['tech']].fillna(0)
#         X_test_micro = test_data[feature_sets['micro']].fillna(0)
#         X_test_all = test_data[feature_sets['all']].fillna(0)
        
#         # Scale features
#         X_test_tech_scaled = scalers['tech'].transform(X_test_tech)
#         X_test_micro_scaled = scalers['micro'].transform(X_test_micro)
#         X_test_all_scaled = scalers['all'].transform(X_test_all)
        
#         # Get predictions with proper scaling
#         xgb_preds = models['xgb'].predict(X_test_tech_scaled)
#         lgb_preds = models['lgb'].predict(X_test_micro_scaled)
        
#         # Unpack Ridge model and scale factor
#         if isinstance(models['ridge'], tuple):
#             ridge_model, ridge_scale = models['ridge']
#             ridge_preds = ridge_model.decision_function(X_test_micro_scaled) * ridge_scale
#         else:
#             # Fallback for backward compatibility
#             ridge_preds = models['ridge'].decision_function(X_test_micro_scaled) * 0.001
        
#         # Unpack MLP model and scale factor
#         if isinstance(models['mlp'], tuple):
#             mlp_model, mlp_scale = models['mlp']
#             try:
#                 mlp_proba = mlp_model.predict_proba(X_test_all_scaled)[:, 1]
#             except IndexError:
#                 mlp_proba = np.zeros(len(X_test_all_scaled))
#             mlp_preds = (mlp_proba - 0.5) * mlp_scale
#         else:
#             # Fallback for backward compatibility
#             mlp_proba = models['mlp'].predict_proba(X_test_all_scaled)[:, 1]
#             mlp_preds = (mlp_proba - 0.5) * 0.002
        
#         return {
#             'xgb': xgb_preds,
#             'lgb': lgb_preds,
#             'ridge': ridge_preds,
#             'mlp': mlp_preds
#         }