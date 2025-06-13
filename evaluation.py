"""
Model evaluation and metrics calculation
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import CRYPTO_PERIODS_PER_YEAR, DIRECTION_THRESHOLD_MULTIPLIER
from utils import calculate_dynamic_sharpe


class ModelEvaluator:
    """Evaluate model performance with crypto-specific metrics"""
    
    def __init__(self):
        self.periods_per_year = CRYPTO_PERIODS_PER_YEAR
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         time_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Direction accuracy
        direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # Threshold-based direction accuracy (considering DIRECTION_THRESHOLD_MULTIPLIER)
        threshold = np.std(y_true) * 0.1 * DIRECTION_THRESHOLD_MULTIPLIER
        significant_moves = np.abs(y_true) > threshold
        if significant_moves.sum() > 0:
            direction_correct_significant = np.mean(
                np.sign(y_true[significant_moves]) == np.sign(y_pred[significant_moves])
            )
        else:
            direction_correct_significant = 0.0
        
        # Trading metrics
        trading_returns = y_pred * y_true  # If we trade based on prediction
        avg_return = np.mean(trading_returns)
        volatility = np.std(trading_returns)
        
        # Dynamic Sharpe ratio calculation
        sharpe = calculate_dynamic_sharpe(trading_returns, volatility, time_index)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + trading_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(trading_returns > 0)
        
        # Profit factor
        gains = trading_returns[trading_returns > 0]
        losses = trading_returns[trading_returns < 0]
        profit_factor = np.sum(gains) / (np.abs(np.sum(losses)) + 1e-8)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_correct,
            'direction_accuracy_significant': direction_correct_significant,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def evaluate_by_symbol(self, df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Dict[str, Dict[str, float]]:
        """Evaluate metrics by symbol"""
        results = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            y_true = symbol_data[y_true_col].values
            y_pred = symbol_data[y_pred_col].values
            
            # Get time index if available
            time_index = None
            if 'timestamp' in symbol_data.columns:
                time_index = pd.to_datetime(symbol_data['timestamp'])
            
            results[symbol] = self.calculate_metrics(y_true, y_pred, time_index)
        
        return results
    
    def evaluate_by_horizon(self, df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Dict[int, Dict[str, float]]:
        """Evaluate metrics by prediction horizon"""
        results = {}
        
        if 'horizon' in df.columns:
            for horizon in df['horizon'].unique():
                horizon_data = df[df['horizon'] == horizon]
                
                y_true = horizon_data[y_true_col].values
                y_pred = horizon_data[y_pred_col].values
                
                # Get time index if available
                time_index = None
                if 'timestamp' in horizon_data.columns:
                    time_index = pd.to_datetime(horizon_data['timestamp'])
                
                results[horizon] = self.calculate_metrics(y_true, y_pred, time_index)
        
        return results
    
    def create_evaluation_report(self, predictions_df: pd.DataFrame, 
                               y_true_col: str = 'actual_return',
                               y_pred_col: str = 'predicted_return') -> pd.DataFrame:
        """Create comprehensive evaluation report"""
        
        # Overall metrics
        overall_metrics = self.calculate_metrics(
            predictions_df[y_true_col].values,
            predictions_df[y_pred_col].values
        )
        
        # By symbol
        symbol_metrics = self.evaluate_by_symbol(predictions_df, y_true_col, y_pred_col)
        
        # By horizon
        horizon_metrics = self.evaluate_by_horizon(predictions_df, y_true_col, y_pred_col)
        
        # Create report dataframe
        report_data = []
        
        # Add overall metrics
        for metric, value in overall_metrics.items():
            report_data.append({
                'category': 'overall',
                'subcategory': 'all',
                'metric': metric,
                'value': value
            })
        
        # Add symbol metrics
        for symbol, metrics in symbol_metrics.items():
            for metric, value in metrics.items():
                report_data.append({
                    'category': 'symbol',
                    'subcategory': symbol,
                    'metric': metric,
                    'value': value
                })
        
        # Add horizon metrics
        for horizon, metrics in horizon_metrics.items():
            for metric, value in metrics.items():
                report_data.append({
                    'category': 'horizon',
                    'subcategory': f'{horizon}_periods',
                    'metric': metric,
                    'value': value
                })
        
        report_df = pd.DataFrame(report_data)
        
        # Format report
        report_pivot = report_df.pivot_table(
            index=['category', 'subcategory'],
            columns='metric',
            values='value'
        )
        
        return report_pivot
    
    def plot_predictions(self, predictions_df: pd.DataFrame, 
                        symbol: str, horizon: int,
                        y_true_col: str = 'actual_return',
                        y_pred_col: str = 'predicted_return'):
        """Plot predictions vs actuals for a specific symbol and horizon"""
        import matplotlib.pyplot as plt
        
        # Filter data
        mask = (predictions_df['symbol'] == symbol)
        if 'horizon' in predictions_df.columns:
            mask &= (predictions_df['horizon'] == horizon)
        
        plot_data = predictions_df[mask].copy()
        
        if len(plot_data) == 0:
            print(f"No data found for {symbol} with horizon {horizon}")
            return
        
        # Sort by timestamp
        if 'timestamp' in plot_data.columns:
            plot_data = plot_data.sort_values('timestamp')
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Predictions vs Actuals
        ax1 = axes[0]
        ax1.plot(plot_data.index, plot_data[y_true_col], label='Actual', alpha=0.7)
        ax1.plot(plot_data.index, plot_data[y_pred_col], label='Predicted', alpha=0.7)
        ax1.set_title(f'{symbol} - Predictions vs Actuals (Horizon: {horizon})')
        ax1.set_ylabel('Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        ax2 = axes[1]
        ax2.scatter(plot_data[y_true_col], plot_data[y_pred_col], alpha=0.5)
        ax2.plot([plot_data[y_true_col].min(), plot_data[y_true_col].max()],
                [plot_data[y_true_col].min(), plot_data[y_true_col].max()],
                'r--', label='Perfect Prediction')
        ax2.set_xlabel('Actual Return')
        ax2.set_ylabel('Predicted Return')
        ax2.set_title('Prediction Scatter Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative returns
        ax3 = axes[2]
        trading_returns = plot_data[y_pred_col] * plot_data[y_true_col]
        cumulative_returns = (1 + trading_returns).cumprod()
        ax3.plot(plot_data.index, cumulative_returns, label='Strategy Returns')
        ax3.set_title('Cumulative Returns from Trading Strategy')
        ax3.set_ylabel('Cumulative Return')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        metrics = self.calculate_metrics(
            plot_data[y_true_col].values,
            plot_data[y_pred_col].values
        )
        
        print(f"\nMetrics for {symbol} (Horizon: {horizon}):")
        for metric, value in metrics.items():
            print(f"{metric:25s}: {value:10.4f}")

# """
# Model evaluation module with comprehensive metrics
# """
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Any
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from config import CRYPTO_PERIODS_PER_YEAR, DIRECTION_THRESHOLD_MULTIPLIER
# from utils import calculate_dynamic_sharpe

# class ModelEvaluator:
#     def __init__(self):
#         self.results = {}
        
#     def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
#                            time_index: pd.DatetimeIndex = None,
#                            return_scale: bool = True) -> Dict[str, float]:
#         """Evaluate predictions with multiple metrics"""
#         # Basic regression metrics
#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
        
#         # R² score with safety check
#         if len(np.unique(y_true)) > 1:
#             r2 = r2_score(y_true, y_pred)
#         else:
#             r2 = np.nan
        
#         # Directional accuracy
#         direction_true = np.sign(y_true)
#         direction_pred = np.sign(y_pred)
#         direction_accuracy = np.mean(direction_true == direction_pred)
        
#         # Profitability metrics (assuming predictions are returns)
#         if return_scale:
#             # Calculate returns from following the predictions
#             strategy_returns = y_true * np.sign(y_pred)
            
#             # Sharpe ratio
#             strategy_vol = strategy_returns.std()
#             if strategy_vol > 0 and time_index is not None:
#                 sharpe = calculate_dynamic_sharpe(strategy_returns, strategy_vol, time_index)
#             else:
#                 sharpe = 0
            
#             # Maximum drawdown
#             cumulative_returns = (1 + strategy_returns).cumprod()
#             running_max = cumulative_returns.expanding().max()
#             drawdown = (cumulative_returns - running_max) / running_max
#             max_drawdown = drawdown.min()
            
#             # Win rate
#             winning_trades = strategy_returns > 0
#             win_rate = winning_trades.mean()
            
#             # Profit factor
#             gross_profits = strategy_returns[strategy_returns > 0].sum()
#             gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
#             profit_factor = gross_profits / (gross_losses + 1e-8)
            
#         else:
#             sharpe = np.nan
#             max_drawdown = np.nan
#             win_rate = np.nan
#             profit_factor = np.nan
        
#         # Information ratio (vs buy and hold)
#         excess_returns = strategy_returns - y_true
#         tracking_error = excess_returns.std()
#         if tracking_error > 0:
#             information_ratio = excess_returns.mean() / tracking_error * np.sqrt(CRYPTO_PERIODS_PER_YEAR)
#         else:
#             information_ratio = 0
        
#         return {
#             'mae': mae,
#             'rmse': rmse,
#             'r2': r2,
#             'direction_accuracy': direction_accuracy,
#             'sharpe_ratio': sharpe,
#             'max_drawdown': max_drawdown,
#             'win_rate': win_rate,
#             'profit_factor': profit_factor,
#             'information_ratio': information_ratio
#         }
    
#     def evaluate_ensemble(self, predictions_dict: Dict[str, np.ndarray], 
#                          meta_learner: Any, y_true: np.ndarray,
#                          time_index: pd.DatetimeIndex = None) -> Dict[str, Dict[str, float]]:
#         """Evaluate all models in the ensemble"""
#         results = {}
        
#         # Evaluate individual models
#         for model_name, preds in predictions_dict.items():
#             print(f"\n{model_name.upper()} Model:")
#             metrics = self.evaluate_predictions(y_true, preds, time_index)
#             results[model_name] = metrics
#             self.print_metrics(metrics)
        
#         # Create ensemble prediction
#         if meta_learner is not None:
#             # Create meta features
#             meta_features = pd.DataFrame(predictions_dict)
#             meta_features['xgb_lgb'] = meta_features['xgb'] * meta_features['lgb']
#             meta_features['tree_linear'] = (meta_features['xgb'] + meta_features['lgb']) * meta_features['ridge']
            
#             # Get ensemble predictions
#             ensemble_proba = meta_learner.predict_proba(meta_features)[:, 1]
#             ensemble_preds = (ensemble_proba - 0.5) * 0.002  # Scale to return space
            
#             print(f"\nENSEMBLE Model:")
#             ensemble_metrics = self.evaluate_predictions(y_true, ensemble_preds, time_index)
#             results['ensemble'] = ensemble_metrics
#             self.print_metrics(ensemble_metrics)
        
#         # Simple average ensemble
#         avg_preds = np.mean(list(predictions_dict.values()), axis=0)
#         print(f"\nSIMPLE AVERAGE Ensemble:")
#         avg_metrics = self.evaluate_predictions(y_true, avg_preds, time_index)
#         results['simple_avg'] = avg_metrics
#         self.print_metrics(avg_metrics)
        
#         return results
    
#     def print_metrics(self, metrics: Dict[str, float]):
#         """Print metrics in a formatted way"""
#         print(f"  MAE: {metrics['mae']:.6f}")
#         print(f"  RMSE: {metrics['rmse']:.6f}")
#         print(f"  R²: {metrics['r2']:.4f}")
#         print(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
#         print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
#         print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
#         print(f"  Win Rate: {metrics['win_rate']:.4f}")
#         print(f"  Profit Factor: {metrics['profit_factor']:.4f}")
#         print(f"  Information Ratio: {metrics['information_ratio']:.4f}")
    
#     def print_results_summary(self, results: Dict[str, Dict[str, Dict[str, float]]]):
#         """Print comprehensive results summary"""
#         print("\n" + "="*80)
#         print("RESULTS SUMMARY")
#         print("="*80)
        
#         # Collect all metrics
#         all_metrics = []
        
#         for symbol, symbol_results in results.items():
#             for model_key, metrics in symbol_results.items():
#                 if '_' in model_key:
#                     parts = model_key.split('_')
#                     model_name = '_'.join(parts[:-1])
#                     horizon = int(parts[-1]) * 5
#                 else:
#                     model_name = model_key
#                     horizon = 'unknown'
                
#                 all_metrics.append({
#                     'symbol': symbol,
#                     'model': model_name,
#                     'horizon': horizon,
#                     **metrics
#                 })
        
#         # Create DataFrame for easy analysis
#         df_results = pd.DataFrame(all_metrics)
        
#         # Best models by Sharpe ratio
#         print("\nBest Models by Sharpe Ratio:")
#         best_sharpe = df_results.nlargest(10, 'sharpe_ratio')[
#             ['symbol', 'model', 'horizon', 'sharpe_ratio', 'direction_accuracy', 'win_rate']
#         ]
#         print(best_sharpe.to_string(index=False))
        
#         # Best models by direction accuracy
#         print("\nBest Models by Direction Accuracy:")
#         best_direction = df_results.nlargest(10, 'direction_accuracy')[
#             ['symbol', 'model', 'horizon', 'direction_accuracy', 'sharpe_ratio', 'profit_factor']
#         ]
#         print(best_direction.to_string(index=False))
        
#         # Average performance by model type
#         print("\nAverage Performance by Model Type:")
#         avg_by_model = df_results.groupby('model').agg({
#             'sharpe_ratio': 'mean',
#             'direction_accuracy': 'mean',
#             'win_rate': 'mean',
#             'profit_factor': 'mean',
#             'max_drawdown': 'mean'
#         }).round(4)
#         print(avg_by_model)
        
#         # Average performance by symbol
#         print("\nAverage Performance by Symbol:")
#         avg_by_symbol = df_results.groupby('symbol').agg({
#             'sharpe_ratio': 'mean',
#             'direction_accuracy': 'mean',
#             'win_rate': 'mean',
#             'profit_factor': 'mean'
#         }).round(4)
#         print(avg_by_symbol)