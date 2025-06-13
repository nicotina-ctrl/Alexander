"""
model_visualization.py
Comprehensive visualization and analysis of crypto prediction model results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelVisualizer:
    def __init__(self, pipeline, results):
        """
        Initialize visualizer with pipeline and results
        
        Args:
            pipeline: The trained EnhancedCryptoPipeline object
            results: The results dictionary from pipeline.run_enhanced_pipeline()
        """
        self.pipeline = pipeline
        self.results = results
        self.feature_importances = pipeline.feature_importances
        self.selected_features = pipeline.selected_features
        
    def create_full_analysis(self, save_path="/content/drive/MyDrive/crypto_pipeline_whale/analysis"):
        """Create all visualizations and save them"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("\\n" + "="*80)
        print("CREATING COMPREHENSIVE MODEL ANALYSIS")
        print("="*80)
        
        # 1. Feature Importance Analysis
        self.plot_feature_importances(save_path)
        
        # 2. Model Performance Comparison
        self.plot_model_performance_comparison(save_path)
        
        # 3. Sharpe Ratio Analysis
        self.plot_sharpe_ratio_analysis(save_path)
        
        # 4. Direction Accuracy Heatmap
        self.plot_direction_accuracy_heatmap(save_path)
        
        # 5. Feature Category Analysis
        self.plot_feature_category_analysis(save_path)
        
        # 6. Time Horizon Analysis
        self.plot_horizon_performance(save_path)
        
        # 7. Risk-Return Scatter
        self.plot_risk_return_scatter(save_path)
        
        # 8. Model Ensemble Weights
        self.plot_ensemble_weights(save_path)
        
        # 9. Performance Summary Table
        self.create_performance_summary_table(save_path)
        
        # 10. Generate HTML Report
        self.generate_html_report(save_path)
        
        print(f"\\n✅ All visualizations saved to: {save_path}")
        
    def plot_feature_importances(self, save_path):
        """Create feature importance visualizations"""
        print("\\n1. Creating feature importance plots...")
        
        # Create a figure with subplots for each symbol-horizon combination
        n_models = len(self.feature_importances)
        fig = plt.figure(figsize=(20, 5 * ((n_models + 2) // 3)))
        
        for idx, (key, importance_df) in enumerate(self.feature_importances.items()):
            if importance_df is None or importance_df.empty:
                continue
                
            # Parse key
            parts = key.split('_')
            symbol = parts[0] + '/' + parts[1]
            horizon = int(parts[-1]) * 5
            
            # Get top 20 features
            top_features = importance_df.nlargest(20, 'importance')
            
            # Create subplot
            ax = plt.subplot((n_models + 2) // 3, 3, idx + 1)
            
            # Color code by feature type
            colors = []
            for feat in top_features['feature']:
                if any(x in feat for x in ['whale', 'inst', 'retail']):
                    colors.append('darkred')
                elif any(x in feat for x in ['mention', 'sentiment', 'social']):
                    colors.append('darkblue')
                elif any(x in feat for x in ['rsi', 'macd', 'bb_']):
                    colors.append('darkgreen')
                elif any(x in feat for x in ['volume', 'cvd', 'flow']):
                    colors.append('darkorange')
                else:
                    colors.append('gray')
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'], color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'], fontsize=8)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{symbol} @ {horizon}min - Top 20 Features', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add legend for first subplot
            if idx == 0:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='darkred', label='Whale Features'),
                    Patch(facecolor='darkblue', label='Social Features'),
                    Patch(facecolor='darkgreen', label='Technical'),
                    Patch(facecolor='darkorange', label='Volume/Flow'),
                    Patch(facecolor='gray', label='Other')
                ]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        plt.suptitle('Feature Importances by Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importances.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_performance_comparison(self, save_path):
        """Compare performance across different models"""
        print("2. Creating model performance comparison...")
        
        # Collect metrics for all models
        metrics_data = []
        for symbol, symbol_results in self.results.items():
            for model_key, metrics in symbol_results.items():
                parts = model_key.split('_')
                model_name = '_'.join(parts[:-1])
                horizon = int(parts[-1]) * 5
                
                metrics_data.append({
                    'symbol': symbol,
                    'model': model_name,
                    'horizon': horizon,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'direction_accuracy': metrics['direction_accuracy'],
                    'win_rate': metrics['win_rate'],
                    'max_drawdown': metrics['max_drawdown']
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sharpe Ratio by Model
        ax = axes[0, 0]
        df_pivot = df_metrics.pivot_table(values='sharpe_ratio', index='model', columns='horizon', aggfunc='mean')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Average Sharpe Ratio by Model and Horizon', fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(title='Horizon (min)', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # Direction Accuracy by Model
        ax = axes[0, 1]
        df_pivot = df_metrics.pivot_table(values='direction_accuracy', index='model', columns='symbol', aggfunc='mean')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Average Direction Accuracy by Model and Symbol', fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Direction Accuracy')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # Win Rate Distribution
        ax = axes[1, 0]
        for model in df_metrics['model'].unique():
            model_data = df_metrics[df_metrics['model'] == model]['win_rate']
            ax.hist(model_data, alpha=0.5, label=model, bins=10)
        ax.set_title('Win Rate Distribution by Model', fontweight='bold')
        ax.set_xlabel('Win Rate')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Max Drawdown Comparison
        ax = axes[1, 1]
        df_pivot = df_metrics.pivot_table(values='max_drawdown', index='model', columns='horizon', aggfunc='mean')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Average Maximum Drawdown by Model and Horizon', fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Max Drawdown')
        ax.legend(title='Horizon (min)', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_sharpe_ratio_analysis(self, save_path):
        """Detailed Sharpe ratio analysis"""
        print("3. Creating Sharpe ratio analysis...")
        
        # Extract Sharpe ratios
        sharpe_data = []
        for symbol, symbol_results in self.results.items():
            for model_key, metrics in symbol_results.items():
                parts = model_key.split('_')
                model_name = '_'.join(parts[:-1])
                horizon = int(parts[-1]) * 5
                
                sharpe_data.append({
                    'symbol': symbol,
                    'model': model_name,
                    'horizon': horizon,
                    'sharpe': metrics['sharpe_ratio']
                })
        
        df_sharpe = pd.DataFrame(sharpe_data)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sharpe by horizon
        ax = axes[0]
        for model in df_sharpe['model'].unique():
            model_data = df_sharpe[df_sharpe['model'] == model]
            avg_sharpe = model_data.groupby('horizon')['sharpe'].mean()
            ax.plot(avg_sharpe.index, avg_sharpe.values, marker='o', label=model, linewidth=2)
        
        ax.set_xlabel('Prediction Horizon (minutes)')
        ax.set_ylabel('Average Sharpe Ratio')
        ax.set_title('Sharpe Ratio vs Prediction Horizon', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Best models by Sharpe
        ax = axes[1]
        top_models = df_sharpe.nlargest(10, 'sharpe')
        
        y_pos = np.arange(len(top_models))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_models)))
        
        bars = ax.barh(y_pos, top_models['sharpe'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['symbol']} {row['model']} {row['horizon']}m" 
                            for _, row in top_models.iterrows()], fontsize=9)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_title('Top 10 Models by Sharpe Ratio', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_models['sharpe'])):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}', va='center', fontsize=8)
        
        plt.suptitle('Sharpe Ratio Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/sharpe_ratio_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_direction_accuracy_heatmap(self, save_path):
        """Create heatmap of direction accuracy"""
        print("4. Creating direction accuracy heatmap...")
        
        # Prepare data for heatmap
        accuracy_data = {}
        for symbol, symbol_results in self.results.items():
            accuracy_data[symbol] = {}
            for model_key, metrics in symbol_results.items():
                parts = model_key.split('_')
                model_name = '_'.join(parts[:-1])
                horizon = int(parts[-1])
                
                key = f"{model_name}_{horizon*5}m"
                accuracy_data[symbol][key] = metrics['direction_accuracy']
        
        df_heatmap = pd.DataFrame(accuracy_data).T
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create mask for values below 0.5 (worse than random)
        mask = df_heatmap < 0.5
        
        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, vmin=0.4, vmax=0.7,
                   cbar_kws={'label': 'Direction Accuracy'},
                   linewidths=1, linecolor='gray')
        
        plt.title('Direction Accuracy Heatmap\\n(Green = Better, Red = Worse, 0.5 = Random)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Model & Horizon')
        plt.ylabel('Symbol')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/direction_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_category_analysis(self, save_path):
        """Analyze feature importance by category"""
        print("5. Creating feature category analysis...")
        
        # Aggregate feature importance by category
        category_importance = {
            'Whale': [],
            'Social': [],
            'Technical': [],
            'Microstructure': [],
            'Volume/Flow': [],
            'Time': [],
            'Other': []
        }
        
        for key, importance_df in self.feature_importances.items():
            if importance_df is None or importance_df.empty:
                continue
                
            for _, row in importance_df.iterrows():
                feat = row['feature']
                imp = row['importance']
                
                if any(x in feat for x in ['whale', 'inst', 'retail', 'smart_dumb']):
                    category_importance['Whale'].append(imp)
                elif any(x in feat for x in ['mention', 'sentiment', 'social', 'viral']):
                    category_importance['Social'].append(imp)
                elif any(x in feat for x in ['rsi', 'macd', 'bb_', 'atr']):
                    category_importance['Technical'].append(imp)
                elif any(x in feat for x in ['spread', 'imbalance', 'microprice', 'book']):
                    category_importance['Microstructure'].append(imp)
                elif any(x in feat for x in ['volume', 'cvd', 'flow', 'vwap']):
                    category_importance['Volume/Flow'].append(imp)
                elif any(x in feat for x in ['hour', 'day', 'session']):
                    category_importance['Time'].append(imp)
                else:
                    category_importance['Other'].append(imp)
        
        # Calculate statistics
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total importance by category
        ax = axes[0]
        categories = []
        total_imp = []
        avg_imp = []
        
        for cat, imps in category_importance.items():
            if imps:  # Only include categories with features
                categories.append(cat)
                total_imp.append(sum(imps))
                avg_imp.append(np.mean(imps))
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, total_imp, width, label='Total Importance', alpha=0.8)
        ax.set_xlabel('Feature Category')
        ax.set_ylabel('Total Importance')
        ax.set_title('Total Feature Importance by Category', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Average importance by category
        ax = axes[1]
        bars2 = ax.bar(categories, avg_imp, alpha=0.8, color='orange')
        ax.set_xlabel('Feature Category')
        ax.set_ylabel('Average Importance')
        ax.set_title('Average Feature Importance by Category', fontweight='bold')
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Feature Category Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_category_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_horizon_performance(self, save_path):
        """Analyze performance across different time horizons"""
        print("6. Creating time horizon analysis...")
        
        # Collect data by horizon
        horizon_data = []
        for symbol, symbol_results in self.results.items():
            for model_key, metrics in symbol_results.items():
                parts = model_key.split('_')
                model_name = '_'.join(parts[:-1])
                horizon = int(parts[-1]) * 5
                
                horizon_data.append({
                    'horizon': horizon,
                    'model': model_name,
                    'sharpe': metrics['sharpe_ratio'],
                    'accuracy': metrics['direction_accuracy'],
                    'profit_factor': metrics['profit_factor']
                })
        
        df_horizon = pd.DataFrame(horizon_data)
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Average metrics by horizon
        metrics_to_plot = ['sharpe', 'accuracy', 'profit_factor']
        titles = ['Average Sharpe Ratio', 'Average Direction Accuracy', 'Average Profit Factor']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx]
            
            horizon_avg = df_horizon.groupby('horizon')[metric].agg(['mean', 'std'])
            
            ax.bar(horizon_avg.index, horizon_avg['mean'], 
                  yerr=horizon_avg['std'], capsize=5, alpha=0.7)
            ax.set_xlabel('Prediction Horizon (minutes)')
            ax.set_ylabel(title.split()[-1] + ' ' + title.split()[-2])
            ax.set_title(f'{title} by Time Horizon', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for x, (y, err) in zip(horizon_avg.index, zip(horizon_avg['mean'], horizon_avg['std'])):
                ax.text(x, y + err + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Performance Analysis by Time Horizon', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/horizon_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_risk_return_scatter(self, save_path):
        """Create risk-return scatter plot"""
        print("7. Creating risk-return scatter...")
        
        # Collect risk-return data
        risk_return_data = []
        for symbol, symbol_results in self.results.items():
            for model_key, metrics in symbol_results.items():
                parts = model_key.split('_')
                model_name = '_'.join(parts[:-1])
                horizon = int(parts[-1]) * 5
                
                # Use absolute max drawdown as risk measure
                risk = abs(metrics['max_drawdown'])
                # Use Sharpe ratio as return measure
                return_metric = metrics['sharpe_ratio']
                
                risk_return_data.append({
                    'model': model_name,
                    'symbol': symbol,
                    'horizon': horizon,
                    'risk': risk,
                    'return': return_metric,
                    'accuracy': metrics['direction_accuracy']
                })
        
        df_rr = pd.DataFrame(risk_return_data)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Create color map for models
        models = df_rr['model'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        model_colors = dict(zip(models, colors))
        
        # Plot each model
        for model in models:
            model_data = df_rr[df_rr['model'] == model]
            
            # Size based on accuracy (better accuracy = larger point)
            sizes = (model_data['accuracy'] - 0.4) * 1000
            
            plt.scatter(model_data['risk'], model_data['return'], 
                       c=[model_colors[model]], s=sizes, alpha=0.6, 
                       label=model, edgecolors='black', linewidth=1)
        
        plt.xlabel('Risk (Max Drawdown)', fontsize=12)
        plt.ylabel('Return (Sharpe Ratio)', fontsize=12)
        plt.title('Risk-Return Profile of Models\\n(Size = Direction Accuracy)', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        median_risk = df_rr['risk'].median()
        plt.axvline(x=median_risk, color='black', linestyle='--', alpha=0.3)
        
        # Annotate quadrants
        plt.text(0.02, 50, 'Low Risk\\nHigh Return', fontsize=10, alpha=0.5, 
                style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.2))
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/risk_return_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_ensemble_weights(self, save_path):
        """Visualize meta-learner weights if available"""
        print("8. Creating ensemble weights visualization...")
        
        # This would require access to the meta-learner weights
        # For now, create a placeholder showing model contribution
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count how often each model type performs best
        best_counts = {'xgb': 0, 'lgb': 0, 'ridge': 0, 'mlp': 0, 'ensemble': 0}
        
        for symbol, symbol_results in self.results.items():
            horizons = set()
            for model_key in symbol_results.keys():
                horizon = model_key.split('_')[-1]
                horizons.add(horizon)
            
            for h in horizons:
                best_sharpe = -float('inf')
                best_model = None
                
                for model in ['xgb', 'lgb', 'ridge', 'mlp', 'ensemble']:
                    key = f"{model}_{h}"
                    if key in symbol_results:
                        if symbol_results[key]['sharpe_ratio'] > best_sharpe:
                            best_sharpe = symbol_results[key]['sharpe_ratio']
                            best_model = model
                
                if best_model:
                    best_counts[best_model] += 1
        
        # Create bar plot
        models = list(best_counts.keys())
        counts = list(best_counts.values())
        
        bars = ax.bar(models, counts, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Times Best Performer')
        ax.set_title('Model Performance Leadership Count\\n(How often each model had highest Sharpe ratio)', 
                    fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Color code bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/ensemble_weights.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_performance_summary_table(self, save_path):
        """Create a summary table of all results"""
        print("9. Creating performance summary table...")
        
        # Collect all results
        summary_data = []
        for symbol, symbol_results in self.results.items():
            for model_key, metrics in symbol_results.items():
                parts = model_key.split('_')
                model_name = '_'.join(parts[:-1])
                horizon = int(parts[-1]) * 5
                
                summary_data.append({
                    'Symbol': symbol,
                    'Model': model_name.upper(),
                    'Horizon': f"{horizon}min",
                    'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                    'Accuracy': f"{metrics['direction_accuracy']:.3f}",
                    'Win Rate': f"{metrics['win_rate']:.3f}",
                    'Max DD': f"{metrics['max_drawdown']:.3f}",
                    'Profit Factor': f"{metrics['profit_factor']:.2f}"
                })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Create figure for table
        fig, ax = plt.subplots(figsize=(14, len(df_summary) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df_summary.values,
                        colLabels=df_summary.columns,
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code by Sharpe ratio
        for i in range(len(df_summary)):
            sharpe_val = float(df_summary.iloc[i]['Sharpe'])
            if sharpe_val > 50:
                color = 'lightgreen'
            elif sharpe_val > 20:
                color = 'lightblue'
            elif sharpe_val > 0:
                color = 'lightyellow'
            else:
                color = 'lightcoral'
            
            for j in range(len(df_summary.columns)):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.3)
        
        # Header styling
        for j in range(len(df_summary.columns)):
            table[(0, j)].set_facecolor('darkblue')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.title('Complete Performance Summary\\n(Colored by Sharpe Ratio: Green>50, Blue>20, Yellow>0, Red<0)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/performance_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save as CSV
        df_summary.to_csv(f"{save_path}/performance_summary.csv", index=False)
        
    def generate_html_report(self, save_path):
        """Generate an HTML report with all visualizations"""
        print("10. Generating HTML report...")
        
        html_content = f"""
        <html>
        <head>
            <title>Crypto Prediction Model Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    background-color: #e0e0e0;
                    padding: 20px;
                    border-radius: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 10px;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .metadata {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric {{
                    background-color: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>Crypto Prediction Model Analysis Report</h1>
            
            <div class="metadata">
                <h3>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
                <p><strong>Models Analyzed:</strong> XGBoost, LightGBM, Ridge, MLP, Ensemble</p>
                <p><strong>Symbols:</strong> {', '.join(set(s for s in self.results.keys()))}</p>
                <p><strong>Time Horizons:</strong> 5min, 30min, 60min</p>
            </div>
            
            <div class="section">
                <h2>Key Performance Metrics</h2>
                <div class="summary">
        """
        
        # Calculate summary statistics
        all_sharpes = []
        all_accuracies = []
        for symbol_results in self.results.values():
            for metrics in symbol_results.values():
                all_sharpes.append(metrics['sharpe_ratio'])
                all_accuracies.append(metrics['direction_accuracy'])
        
        best_sharpe = max(all_sharpes)
        avg_sharpe = np.mean(all_sharpes)
        best_accuracy = max(all_accuracies)
        avg_accuracy = np.mean(all_accuracies)
        
        html_content += f"""
                    <div class="metric">
                        <div class="metric-value">{best_sharpe:.2f}</div>
                        <div class="metric-label">Best Sharpe Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{avg_sharpe:.2f}</div>
                        <div class="metric-label">Average Sharpe Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{best_accuracy:.3f}</div>
                        <div class="metric-label">Best Direction Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{avg_accuracy:.3f}</div>
                        <div class="metric-label">Average Direction Accuracy</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>1. Feature Importance Analysis</h2>
                <p>This chart shows the top 20 most important features for each model, color-coded by feature type.</p>
                <img src="feature_importances.png" alt="Feature Importances">
            </div>
            
            <div class="section">
                <h2>2. Model Performance Comparison</h2>
                <p>Comprehensive comparison of different metrics across all models.</p>
                <img src="model_performance_comparison.png" alt="Model Performance Comparison">
            </div>
            
            <div class="section">
                <h2>3. Sharpe Ratio Analysis</h2>
                <p>Detailed analysis of risk-adjusted returns across models and time horizons.</p>
                <img src="sharpe_ratio_analysis.png" alt="Sharpe Ratio Analysis">
            </div>
            
            <div class="section">
                <h2>4. Direction Accuracy Heatmap</h2>
                <p>Visual representation of prediction accuracy across all model-symbol combinations.</p>
                <img src="direction_accuracy_heatmap.png" alt="Direction Accuracy Heatmap">
            </div>
            
            <div class="section">
                <h2>5. Feature Category Analysis</h2>
                <p>Breakdown of feature importance by category (Technical, Microstructure, Whale, Social, etc.)</p>
                <img src="feature_category_analysis.png" alt="Feature Category Analysis">
            </div>
            
            <div class="section">
                <h2>6. Time Horizon Performance</h2>
                <p>How model performance varies with different prediction horizons.</p>
                <img src="horizon_performance.png" alt="Horizon Performance">
            </div>
            
            <div class="section">
                <h2>7. Risk-Return Profile</h2>
                <p>Scatter plot showing the trade-off between risk (max drawdown) and return (Sharpe ratio).</p>
                <img src="risk_return_scatter.png" alt="Risk-Return Scatter">
            </div>
            
            <div class="section">
                <h2>8. Model Leadership Analysis</h2>
                <p>Frequency of each model type achieving the best performance.</p>
                <img src="ensemble_weights.png" alt="Model Leadership">
            </div>
            
            <div class="section">
                <h2>9. Complete Performance Summary</h2>
                <p>Detailed table of all model results with color coding by performance.</p>
                <img src="performance_summary_table.png" alt="Performance Summary Table">
            </div>
            
        </body>
        </html>
        """
        
        # Save HTML report
        with open(f"{save_path}/analysis_report.html", 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {save_path}/analysis_report.html")


def run_analysis(pipeline, results):
    """Main function to run complete analysis"""
    visualizer = ModelVisualizer(pipeline, results)
    visualizer.create_full_analysis()
    
    print("\\n✅ Analysis complete! Check the analysis folder for all visualizations and report.")
    
    
# If running as a script
if __name__ == "__main__":
    print("This module should be imported and used with a trained pipeline.")
    print("Usage:")
    print("  from model_visualization import run_analysis")
    print("  run_analysis(pipeline, results)")

# """
# model_visualization.py
# Comprehensive visualization and analysis of crypto prediction model results
# """
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # Set style
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette("husl")

# class ModelVisualizer:
#     def __init__(self, pipeline, results):
#         """
#         Initialize visualizer with pipeline and results
        
#         Args:
#             pipeline: The trained EnhancedCryptoPipeline object
#             results: The results dictionary from pipeline.run_enhanced_pipeline()
#         """
#         self.pipeline = pipeline
#         self.results = results
#         self.feature_importances = pipeline.feature_importances
#         self.selected_features = pipeline.selected_features
        
#     def create_full_analysis(self, save_path="/content/drive/MyDrive/crypto_pipeline_whale/analysis"):
#         """Create all visualizations and save them"""
#         import os
#         os.makedirs(save_path, exist_ok=True)
        
#         print("\n" + "="*80)
#         print("CREATING COMPREHENSIVE MODEL ANALYSIS")
#         print("="*80)
        
#         # 1. Feature Importance Analysis
#         self.plot_feature_importances(save_path)
        
#         # 2. Model Performance Comparison
#         self.plot_model_performance_comparison(save_path)
        
#         # 3. Sharpe Ratio Analysis
#         self.plot_sharpe_ratio_analysis(save_path)
        
#         # 4. Direction Accuracy Heatmap
#         self.plot_direction_accuracy_heatmap(save_path)
        
#         # 5. Feature Category Analysis
#         self.plot_feature_category_analysis(save_path)
        
#         # 6. Time Horizon Analysis
#         self.plot_horizon_performance(save_path)
        
#         # 7. Risk-Return Scatter
#         self.plot_risk_return_scatter(save_path)
        
#         # 8. Model Ensemble Weights
#         self.plot_ensemble_weights(save_path)
        
#         # 9. Performance Summary Table
#         self.create_performance_summary_table(save_path)
        
#         # 10. Generate HTML Report
#         self.generate_html_report(save_path)
        
#         print(f"\n✅ All visualizations saved to: {save_path}")
        
#     def plot_feature_importances(self, save_path):
#         """Create feature importance visualizations"""
#         print("\n1. Creating feature importance plots...")
        
#         # Create a figure with subplots for each symbol-horizon combination
#         n_models = len(self.feature_importances)
#         fig = plt.figure(figsize=(20, 5 * ((n_models + 2) // 3)))
        
#         for idx, (key, importance_df) in enumerate(self.feature_importances.items()):
#             if importance_df is None or importance_df.empty:
#                 continue
                
#             # Parse key
#             parts = key.split('_')
#             symbol = parts[0] + '/' + parts[1]
#             horizon = int(parts[-1]) * 5
            
#             # Get top 20 features
#             top_features = importance_df.nlargest(20, 'importance')
            
#             # Create subplot
#             ax = plt.subplot((n_models + 2) // 3, 3, idx + 1)
            
#             # Color code by feature type
#             colors = []
#             for feat in top_features['feature']:
#                 if any(x in feat for x in ['whale', 'inst', 'retail']):
#                     colors.append('darkred')
#                 elif any(x in feat for x in ['mention', 'sentiment', 'social']):
#                     colors.append('darkblue')
#                 elif any(x in feat for x in ['rsi', 'macd', 'bb_']):
#                     colors.append('darkgreen')
#                 elif any(x in feat for x in ['volume', 'cvd', 'flow']):
#                     colors.append('darkorange')
#                 else:
#                     colors.append('gray')
            
#             # Create horizontal bar plot
#             y_pos = np.arange(len(top_features))
#             ax.barh(y_pos, top_features['importance'], color=colors, alpha=0.8)
#             ax.set_yticks(y_pos)
#             ax.set_yticklabels(top_features['feature'], fontsize=8)
#             ax.set_xlabel('Importance Score')
#             ax.set_title(f'{symbol} @ {horizon}min - Top 20 Features', fontsize=10, fontweight='bold')
#             ax.grid(axis='x', alpha=0.3)
            
#             # Add legend for first subplot
#             if idx == 0:
#                 from matplotlib.patches import Patch
#                 legend_elements = [
#                     Patch(facecolor='darkred', label='Whale Features'),
#                     Patch(facecolor='darkblue', label='Social Features'),
#                     Patch(facecolor='darkgreen', label='Technical'),
#                     Patch(facecolor='darkorange', label='Volume/Flow'),
#                     Patch(facecolor='gray', label='Other')
#                 ]
#                 ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
#         plt.suptitle('Feature Importances by Model', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/feature_importances.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def plot_model_performance_comparison(self, save_path):
#         """Compare performance across different models"""
#         print("2. Creating model performance comparison...")
        
#         # Collect metrics for all models
#         metrics_data = []
#         for symbol, symbol_results in self.results.items():
#             for model_key, metrics in symbol_results.items():
#                 parts = model_key.split('_')
#                 model_name = '_'.join(parts[:-1])
#                 horizon = int(parts[-1]) * 5
                
#                 metrics_data.append({
#                     'symbol': symbol,
#                     'model': model_name,
#                     'horizon': horizon,
#                     'sharpe_ratio': metrics['sharpe_ratio'],
#                     'direction_accuracy': metrics['direction_accuracy'],
#                     'win_rate': metrics['win_rate'],
#                     'max_drawdown': metrics['max_drawdown']
#                 })
        
#         df_metrics = pd.DataFrame(metrics_data)
        
#         # Create comparison plots
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
#         # Sharpe Ratio by Model
#         ax = axes[0, 0]
#         df_pivot = df_metrics.pivot_table(values='sharpe_ratio', index='model', columns='horizon', aggfunc='mean')
#         df_pivot.plot(kind='bar', ax=ax)
#         ax.set_title('Average Sharpe Ratio by Model and Horizon', fontweight='bold')
#         ax.set_xlabel('Model')
#         ax.set_ylabel('Sharpe Ratio')
#         ax.legend(title='Horizon (min)', bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.grid(axis='y', alpha=0.3)
        
#         # Direction Accuracy by Model
#         ax = axes[0, 1]
#         df_pivot = df_metrics.pivot_table(values='direction_accuracy', index='model', columns='symbol', aggfunc='mean')
#         df_pivot.plot(kind='bar', ax=ax)
#         ax.set_title('Average Direction Accuracy by Model and Symbol', fontweight='bold')
#         ax.set_xlabel('Model')
#         ax.set_ylabel('Direction Accuracy')
#         ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
#         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.grid(axis='y', alpha=0.3)
        
#         # Win Rate Distribution
#         ax = axes[1, 0]
#         for model in df_metrics['model'].unique():
#             model_data = df_metrics[df_metrics['model'] == model]['win_rate']
#             ax.hist(model_data, alpha=0.5, label=model, bins=10)
#         ax.set_title('Win Rate Distribution by Model', fontweight='bold')
#         ax.set_xlabel('Win Rate')
#         ax.set_ylabel('Frequency')
#         ax.legend()
#         ax.grid(axis='y', alpha=0.3)
        
#         # Max Drawdown Comparison
#         ax = axes[1, 1]
#         df_pivot = df_metrics.pivot_table(values='max_drawdown', index='model', columns='horizon', aggfunc='mean')
#         df_pivot.plot(kind='bar', ax=ax)
#         ax.set_title('Average Maximum Drawdown by Model and Horizon', fontweight='bold')
#         ax.set_xlabel('Model')
#         ax.set_ylabel('Max Drawdown')
#         ax.legend(title='Horizon (min)', bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.grid(axis='y', alpha=0.3)
        
#         plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/model_performance_comparison.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def plot_sharpe_ratio_analysis(self, save_path):
#         """Detailed Sharpe ratio analysis"""
#         print("3. Creating Sharpe ratio analysis...")
        
#         # Extract Sharpe ratios
#         sharpe_data = []
#         for symbol, symbol_results in self.results.items():
#             for model_key, metrics in symbol_results.items():
#                 parts = model_key.split('_')
#                 model_name = '_'.join(parts[:-1])
#                 horizon = int(parts[-1]) * 5
                
#                 sharpe_data.append({
#                     'symbol': symbol,
#                     'model': model_name,
#                     'horizon': horizon,
#                     'sharpe': metrics['sharpe_ratio']
#                 })
        
#         df_sharpe = pd.DataFrame(sharpe_data)
        
#         # Create visualization
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Sharpe by horizon
#         ax = axes[0]
#         for model in df_sharpe['model'].unique():
#             model_data = df_sharpe[df_sharpe['model'] == model]
#             avg_sharpe = model_data.groupby('horizon')['sharpe'].mean()
#             ax.plot(avg_sharpe.index, avg_sharpe.values, marker='o', label=model, linewidth=2)
        
#         ax.set_xlabel('Prediction Horizon (minutes)')
#         ax.set_ylabel('Average Sharpe Ratio')
#         ax.set_title('Sharpe Ratio vs Prediction Horizon', fontweight='bold')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         # Best models by Sharpe
#         ax = axes[1]
#         top_models = df_sharpe.nlargest(10, 'sharpe')
        
#         y_pos = np.arange(len(top_models))
#         colors = plt.cm.viridis(np.linspace(0, 1, len(top_models)))
        
#         bars = ax.barh(y_pos, top_models['sharpe'], color=colors)
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels([f"{row['symbol']} {row['model']} {row['horizon']}m" 
#                             for _, row in top_models.iterrows()], fontsize=9)
#         ax.set_xlabel('Sharpe Ratio')
#         ax.set_title('Top 10 Models by Sharpe Ratio', fontweight='bold')
#         ax.grid(axis='x', alpha=0.3)
        
#         # Add value labels on bars
#         for i, (bar, value) in enumerate(zip(bars, top_models['sharpe'])):
#             ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
#                    f'{value:.1f}', va='center', fontsize=8)
        
#         plt.suptitle('Sharpe Ratio Analysis', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/sharpe_ratio_analysis.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def plot_direction_accuracy_heatmap(self, save_path):
#         """Create heatmap of direction accuracy"""
#         print("4. Creating direction accuracy heatmap...")
        
#         # Prepare data for heatmap
#         accuracy_data = {}
#         for symbol, symbol_results in self.results.items():
#             accuracy_data[symbol] = {}
#             for model_key, metrics in symbol_results.items():
#                 parts = model_key.split('_')
#                 model_name = '_'.join(parts[:-1])
#                 horizon = int(parts[-1])
                
#                 key = f"{model_name}_{horizon*5}m"
#                 accuracy_data[symbol][key] = metrics['direction_accuracy']
        
#         df_heatmap = pd.DataFrame(accuracy_data).T
        
#         # Create heatmap
#         plt.figure(figsize=(12, 8))
        
#         # Create mask for values below 0.5 (worse than random)
#         mask = df_heatmap < 0.5
        
#         sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', 
#                    center=0.5, vmin=0.4, vmax=0.7,
#                    cbar_kws={'label': 'Direction Accuracy'},
#                    linewidths=1, linecolor='gray')
        
#         plt.title('Direction Accuracy Heatmap\n(Green = Better, Red = Worse, 0.5 = Random)', 
#                  fontsize=14, fontweight='bold')
#         plt.xlabel('Model & Horizon')
#         plt.ylabel('Symbol')
#         plt.xticks(rotation=45, ha='right')
        
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/direction_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def plot_feature_category_analysis(self, save_path):
#         """Analyze feature importance by category"""
#         print("5. Creating feature category analysis...")
        
#         # Aggregate feature importance by category
#         category_importance = {
#             'Whale': [],
#             'Social': [],
#             'Technical': [],
#             'Microstructure': [],
#             'Volume/Flow': [],
#             'Time': [],
#             'Other': []
#         }
        
#         for key, importance_df in self.feature_importances.items():
#             if importance_df is None or importance_df.empty:
#                 continue
                
#             for _, row in importance_df.iterrows():
#                 feat = row['feature']
#                 imp = row['importance']
                
#                 if any(x in feat for x in ['whale', 'inst', 'retail', 'smart_dumb']):
#                     category_importance['Whale'].append(imp)
#                 elif any(x in feat for x in ['mention', 'sentiment', 'social', 'viral']):
#                     category_importance['Social'].append(imp)
#                 elif any(x in feat for x in ['rsi', 'macd', 'bb_', 'atr']):
#                     category_importance['Technical'].append(imp)
#                 elif any(x in feat for x in ['spread', 'imbalance', 'microprice', 'book']):
#                     category_importance['Microstructure'].append(imp)
#                 elif any(x in feat for x in ['volume', 'cvd', 'flow', 'vwap']):
#                     category_importance['Volume/Flow'].append(imp)
#                 elif any(x in feat for x in ['hour', 'day', 'session']):
#                     category_importance['Time'].append(imp)
#                 else:
#                     category_importance['Other'].append(imp)
        
#         # Calculate statistics
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Total importance by category
#         ax = axes[0]
#         categories = []
#         total_imp = []
#         avg_imp = []
        
#         for cat, imps in category_importance.items():
#             if imps:  # Only include categories with features
#                 categories.append(cat)
#                 total_imp.append(sum(imps))
#                 avg_imp.append(np.mean(imps))
        
#         x = np.arange(len(categories))
#         width = 0.35
        
#         bars1 = ax.bar(x - width/2, total_imp, width, label='Total Importance', alpha=0.8)
#         ax.set_xlabel('Feature Category')
#         ax.set_ylabel('Total Importance')
#         ax.set_title('Total Feature Importance by Category', fontweight='bold')
#         ax.set_xticks(x)
#         ax.set_xticklabels(categories, rotation=45, ha='right')
#         ax.legend()
#         ax.grid(axis='y', alpha=0.3)
        
#         # Add value labels
#         for bar in bars1:
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height,
#                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
#         # Average importance by category
#         ax = axes[1]
#         bars2 = ax.bar(categories, avg_imp, alpha=0.8, color='orange')
#         ax.set_xlabel('Feature Category')
#         ax.set_ylabel('Average Importance')
#         ax.set_title('Average Feature Importance by Category', fontweight='bold')
#         ax.set_xticklabels(categories, rotation=45, ha='right')
#         ax.grid(axis='y', alpha=0.3)
        
#         # Add value labels
#         for bar in bars2:
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height,
#                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
#         plt.suptitle('Feature Category Analysis', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/feature_category_analysis.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def plot_horizon_performance(self, save_path):
#         """Analyze performance across different time horizons"""
#         print("6. Creating time horizon analysis...")
        
#         # Collect data by horizon
#         horizon_data = []
#         for symbol, symbol_results in self.results.items():
#             for model_key, metrics in symbol_results.items():
#                 parts = model_key.split('_')
#                 model_name = '_'.join(parts[:-1])
#                 horizon = int(parts[-1]) * 5
                
#                 horizon_data.append({
#                     'horizon': horizon,
#                     'model': model_name,
#                     'sharpe': metrics['sharpe_ratio'],
#                     'accuracy': metrics['direction_accuracy'],
#                     'profit_factor': metrics['profit_factor']
#                 })
        
#         df_horizon = pd.DataFrame(horizon_data)
        
#         # Create plots
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
#         # Average metrics by horizon
#         metrics_to_plot = ['sharpe', 'accuracy', 'profit_factor']
#         titles = ['Average Sharpe Ratio', 'Average Direction Accuracy', 'Average Profit Factor']
        
#         for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
#             ax = axes[idx]
            
#             horizon_avg = df_horizon.groupby('horizon')[metric].agg(['mean', 'std'])
            
#             ax.bar(horizon_avg.index, horizon_avg['mean'], 
#                   yerr=horizon_avg['std'], capsize=5, alpha=0.7)
#             ax.set_xlabel('Prediction Horizon (minutes)')
#             ax.set_ylabel(title.split()[-1] + ' ' + title.split()[-2])
#             ax.set_title(f'{title} by Time Horizon', fontweight='bold')
#             ax.grid(axis='y', alpha=0.3)
            
#             # Add value labels
#             for x, (y, err) in zip(horizon_avg.index, zip(horizon_avg['mean'], horizon_avg['std'])):
#                 ax.text(x, y + err + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
        
#         plt.suptitle('Performance Analysis by Time Horizon', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/horizon_performance.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def plot_risk_return_scatter(self, save_path):
#         """Create risk-return scatter plot"""
#         print("7. Creating risk-return scatter...")
        
#         # Collect risk-return data
#         risk_return_data = []
#         for symbol, symbol_results in self.results.items():
#             for model_key, metrics in symbol_results.items():
#                 parts = model_key.split('_')
#                 model_name = '_'.join(parts[:-1])
#                 horizon = int(parts[-1]) * 5
                
#                 # Use absolute max drawdown as risk measure
#                 risk = abs(metrics['max_drawdown'])
#                 # Use Sharpe ratio as return measure
#                 return_metric = metrics['sharpe_ratio']
                
#                 risk_return_data.append({
#                     'model': model_name,
#                     'symbol': symbol,
#                     'horizon': horizon,
#                     'risk': risk,
#                     'return': return_metric,
#                     'accuracy': metrics['direction_accuracy']
#                 })
        
#         df_rr = pd.DataFrame(risk_return_data)
        
#         # Create scatter plot
#         plt.figure(figsize=(12, 8))
        
#         # Create color map for models
#         models = df_rr['model'].unique()
#         colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
#         model_colors = dict(zip(models, colors))
        
#         # Plot each model
#         for model in models:
#             model_data = df_rr[df_rr['model'] == model]
            
#             # Size based on accuracy (better accuracy = larger point)
#             sizes = (model_data['accuracy'] - 0.4) * 1000
            
#             plt.scatter(model_data['risk'], model_data['return'], 
#                        c=[model_colors[model]], s=sizes, alpha=0.6, 
#                        label=model, edgecolors='black', linewidth=1)
        
#         plt.xlabel('Risk (Max Drawdown)', fontsize=12)
#         plt.ylabel('Return (Sharpe Ratio)', fontsize=12)
#         plt.title('Risk-Return Profile of Models\n(Size = Direction Accuracy)', 
#                  fontsize=14, fontweight='bold')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(True, alpha=0.3)
        
#         # Add quadrant lines
#         plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
#         median_risk = df_rr['risk'].median()
#         plt.axvline(x=median_risk, color='black', linestyle='--', alpha=0.3)
        
#         # Annotate quadrants
#         plt.text(0.02, 50, 'Low Risk\nHigh Return', fontsize=10, alpha=0.5, 
#                 style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.2))
        
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/risk_return_scatter.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def plot_ensemble_weights(self, save_path):
#         """Visualize meta-learner weights if available"""
#         print("8. Creating ensemble weights visualization...")
        
#         # This would require access to the meta-learner weights
#         # For now, create a placeholder showing model contribution
        
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         # Count how often each model type performs best
#         best_counts = {'xgb': 0, 'lgb': 0, 'ridge': 0, 'mlp': 0, 'ensemble': 0}
        
#         for symbol, symbol_results in self.results.items():
#             horizons = set()
#             for model_key in symbol_results.keys():
#                 horizon = model_key.split('_')[-1]
#                 horizons.add(horizon)
            
#             for h in horizons:
#                 best_sharpe = -float('inf')
#                 best_model = None
                
#                 for model in ['xgb', 'lgb', 'ridge', 'mlp', 'ensemble']:
#                     key = f"{model}_{h}"
#                     if key in symbol_results:
#                         if symbol_results[key]['sharpe_ratio'] > best_sharpe:
#                             best_sharpe = symbol_results[key]['sharpe_ratio']
#                             best_model = model
                
#                 if best_model:
#                     best_counts[best_model] += 1
        
#         # Create bar plot
#         models = list(best_counts.keys())
#         counts = list(best_counts.values())
        
#         bars = ax.bar(models, counts, alpha=0.7, edgecolor='black')
#         ax.set_xlabel('Model Type')
#         ax.set_ylabel('Times Best Performer')
#         ax.set_title('Model Performance Leadership Count\n(How often each model had highest Sharpe ratio)', 
#                     fontweight='bold')
#         ax.grid(axis='y', alpha=0.3)
        
#         # Color code bars
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#         for bar, color in zip(bars, colors):
#             bar.set_facecolor(color)
        
#         # Add value labels
#         for bar in bars:
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height,
#                    f'{int(height)}', ha='center', va='bottom')
        
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/ensemble_weights.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def create_performance_summary_table(self, save_path):
#         """Create a summary table of all results"""
#         print("9. Creating performance summary table...")
        
#         # Collect all results
#         summary_data = []
#         for symbol, symbol_results in self.results.items():
#             for model_key, metrics in symbol_results.items():
#                 parts = model_key.split('_')
#                 model_name = '_'.join(parts[:-1])
#                 horizon = int(parts[-1]) * 5
                
#                 summary_data.append({
#                     'Symbol': symbol,
#                     'Model': model_name.upper(),
#                     'Horizon': f"{horizon}min",
#                     'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
#                     'Accuracy': f"{metrics['direction_accuracy']:.3f}",
#                     'Win Rate': f"{metrics['win_rate']:.3f}",
#                     'Max DD': f"{metrics['max_drawdown']:.3f}",
#                     'Profit Factor': f"{metrics['profit_factor']:.2f}"
#                 })
        
#         df_summary = pd.DataFrame(summary_data)
        
#         # Create figure for table
#         fig, ax = plt.subplots(figsize=(14, len(df_summary) * 0.4 + 2))
#         ax.axis('tight')
#         ax.axis('off')
        
#         # Create table
#         table = ax.table(cellText=df_summary.values,
#                         colLabels=df_summary.columns,
#                         cellLoc='center',
#                         loc='center')
        
#         # Style the table
#         table.auto_set_font_size(False)
#         table.set_fontsize(9)
#         table.scale(1.2, 1.5)
        
#         # Color code by Sharpe ratio
#         for i in range(len(df_summary)):
#             sharpe_val = float(df_summary.iloc[i]['Sharpe'])
#             if sharpe_val > 50:
#                 color = 'lightgreen'
#             elif sharpe_val > 20:
#                 color = 'lightblue'
#             elif sharpe_val > 0:
#                 color = 'lightyellow'
#             else:
#                 color = 'lightcoral'
            
#             for j in range(len(df_summary.columns)):
#                 table[(i+1, j)].set_facecolor(color)
#                 table[(i+1, j)].set_alpha(0.3)
        
#         # Header styling
#         for j in range(len(df_summary.columns)):
#             table[(0, j)].set_facecolor('darkblue')
#             table[(0, j)].set_text_props(weight='bold', color='white')
        
#         plt.title('Complete Performance Summary\n(Colored by Sharpe Ratio: Green>50, Blue>20, Yellow>0, Red<0)', 
#                  fontsize=14, fontweight='bold', pad=20)
        
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/performance_summary_table.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # Also save as CSV
#         df_summary.to_csv(f"{save_path}/performance_summary.csv", index=False)
        
#     def generate_html_report(self, save_path):
#         """Generate an HTML report with all visualizations"""
#         print("10. Generating HTML report...")
        
#         html_content = f"""
#         <html>
#         <head>
#             <title>Crypto Prediction Model Analysis Report</title>
#             <style>
#                 body {{
#                     font-family: Arial, sans-serif;
#                     margin: 20px;
#                     background-color: #f5f5f5;
#                 }}
#                 h1 {{
#                     color: #333;
#                     text-align: center;
#                     background-color: #e0e0e0;
#                     padding: 20px;
#                     border-radius: 10px;
#                 }}
#                 h2 {{
#                     color: #555;
#                     margin-top: 30px;
#                     border-bottom: 2px solid #ddd;
#                     padding-bottom: 10px;
#                 }}
#                 .section {{
#                     background-color: white;
#                     padding: 20px;
#                     margin: 20px 0;
#                     border-radius: 10px;
#                     box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#                 }}
#                 img {{
#                     max-width: 100%;
#                     height: auto;
#                     display: block;
#                     margin: 20px auto;
#                     border: 1px solid #ddd;
#                     border-radius: 5px;
#                 }}
#                 .metadata {{
#                     background-color: #f9f9f9;
#                     padding: 15px;
#                     border-radius: 5px;
#                     margin: 20px 0;
#                 }}
#                 .summary {{
#                     display: grid;
#                     grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#                     gap: 15px;
#                     margin: 20px 0;
#                 }}
#                 .metric {{
#                     background-color: #f0f0f0;
#                     padding: 15px;
#                     border-radius: 5px;
#                     text-align: center;
#                 }}
#                 .metric-value {{
#                     font-size: 24px;
#                     font-weight: bold;
#                     color: #333;
#                 }}
#                 .metric-label {{
#                     font-size: 14px;
#                     color: #666;
#                     margin-top: 5px;
#                 }}
#             </style>
#         </head>
#         <body>
#             <h1>Crypto Prediction Model Analysis Report</h1>
            
#             <div class="metadata">
#                 <h3>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
#                 <p><strong>Models Analyzed:</strong> XGBoost, LightGBM, Ridge, MLP, Ensemble</p>
#                 <p><strong>Symbols:</strong> {', '.join(set(s for s in self.results.keys()))}</p>
#                 <p><strong>Time Horizons:</strong> 5min, 30min, 60min</p>
#             </div>
            
#             <div class="section">
#                 <h2>Key Performance Metrics</h2>
#                 <div class="summary">
#         """
        
#         # Calculate summary statistics
#         all_sharpes = []
#         all_accuracies = []
#         for symbol_results in self.results.values():
#             for metrics in symbol_results.values():
#                 all_sharpes.append(metrics['sharpe_ratio'])
#                 all_accuracies.append(metrics['direction_accuracy'])
        
#         best_sharpe = max(all_sharpes)
#         avg_sharpe = np.mean(all_sharpes)
#         best_accuracy = max(all_accuracies)
#         avg_accuracy = np.mean(all_accuracies)
        
#         html_content += f"""
#                     <div class="metric">
#                         <div class="metric-value">{best_sharpe:.2f}</div>
#                         <div class="metric-label">Best Sharpe Ratio</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-value">{avg_sharpe:.2f}</div>
#                         <div class="metric-label">Average Sharpe Ratio</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-value">{best_accuracy:.3f}</div>
#                         <div class="metric-label">Best Direction Accuracy</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-value">{avg_accuracy:.3f}</div>
#                         <div class="metric-label">Average Direction Accuracy</div>
#                     </div>
#                 </div>
#             </div>
            
#             <div class="section">
#                 <h2>1. Feature Importance Analysis</h2>
#                 <p>This chart shows the top 20 most important features for each model, color-coded by feature type.</p>
#                 <img src="feature_importances.png" alt="Feature Importances">
#             </div>
            
#             <div class="section">
#                 <h2>2. Model Performance Comparison</h2>
#                 <p>Comprehensive comparison of different metrics across all models.</p>
#                 <img src="model_performance_comparison.png" alt="Model Performance Comparison">
#             </div>
            
#             <div class="section">
#                 <h2>3. Sharpe Ratio Analysis</h2>
#                 <p>Detailed analysis of risk-adjusted returns across models and time horizons.</p>
#                 <img src="sharpe_ratio_analysis.png" alt="Sharpe Ratio Analysis">
#             </div>
            
#             <div class="section">
#                 <h2>4. Direction Accuracy Heatmap</h2>
#                 <p>Visual representation of prediction accuracy across all model-symbol combinations.</p>
#                 <img src="direction_accuracy_heatmap.png" alt="Direction Accuracy Heatmap">
#             </div>
            
#             <div class="section">
#                 <h2>5. Feature Category Analysis</h2>
#                 <p>Breakdown of feature importance by category (Technical, Microstructure, Whale, Social, etc.)</p>
#                 <img src="feature_category_analysis.png" alt="Feature Category Analysis">
#             </div>
            
#             <div class="section">
#                 <h2>6. Time Horizon Performance</h2>
#                 <p>How model performance varies with different prediction horizons.</p>
#                 <img src="horizon_performance.png" alt="Horizon Performance">
#             </div>
            
#             <div class="section">
#                 <h2>7. Risk-Return Profile</h2>
#                 <p>Scatter plot showing the trade-off between risk (max drawdown) and return (Sharpe ratio).</p>
#                 <img src="risk_return_scatter.png" alt="Risk-Return Scatter">
#             </div>
            
#             <div class="section">
#                 <h2>8. Model Leadership Analysis</h2>
#                 <p>Frequency of each model type achieving the best performance.</p>
#                 <img src="ensemble_weights.png" alt="Model Leadership">
#             </div>
            
#             <div class="section">
#                 <h2>9. Complete Performance Summary</h2>
#                 <p>Detailed table of all model results with color coding by performance.</p>
#                 <img src="performance_summary_table.png" alt="Performance Summary Table">
#             </div>
            
#         </body>
#         </html>
#         """
        
#         # Save HTML report
#         with open(f"{save_path}/analysis_report.html", 'w') as f:
#             f.write(html_content)
        
#         print(f"HTML report saved to: {save_path}/analysis_report.html")


# def run_analysis(pipeline, results):
#     """Main function to run complete analysis"""
#     visualizer = ModelVisualizer(pipeline, results)
#     visualizer.create_full_analysis()
    
#     print("\n✅ Analysis complete! Check the analysis folder for all visualizations and report.")
    
    
# # If running as a script
# if __name__ == "__main__":
#     print("This module should be imported and used with a trained pipeline.")
#     print("Usage:")
#     print("  from model_visualization import run_analysis")
#     print("  run_analysis(pipeline, results)")