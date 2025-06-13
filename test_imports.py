"""
Test script to verify all imports are working correctly
"""

# First install required packages
from utils import install_required_packages
print("Installing required packages...")
install_required_packages()

# Test imports
print("\nTesting imports...")

try:
    from config import GDRIVE_DIR, DIRECTION_THRESHOLD_MULTIPLIER, CRYPTO_PERIODS_PER_YEAR
    print("✓ Config imports successful")
    print(f"  - GDRIVE_DIR: {GDRIVE_DIR}")
    print(f"  - DIRECTION_THRESHOLD_MULTIPLIER: {DIRECTION_THRESHOLD_MULTIPLIER}")
    print(f"  - CRYPTO_PERIODS_PER_YEAR: {CRYPTO_PERIODS_PER_YEAR}")
except ImportError as e:
    print(f"✗ Config import failed: {e}")

try:
    from utils import optimize_memory_usage, calculate_dynamic_sharpe
    print("✓ Utils imports successful")
except ImportError as e:
    print(f"✗ Utils import failed: {e}")

try:
    from evaluation import ModelEvaluator
    print("✓ Evaluation imports successful")
    evaluator = ModelEvaluator()
    print(f"  - ModelEvaluator initialized with periods_per_year: {evaluator.periods_per_year}")
except ImportError as e:
    print(f"✗ Evaluation import failed: {e}")

# Test basic functionality
print("\nTesting basic functionality...")
import numpy as np
import pandas as pd

# Create dummy data
test_data = pd.DataFrame({
    'value': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

# Test memory optimization
print("\nTesting memory optimization...")
optimized_data = optimize_memory_usage(test_data, verbose=True)

# Test Sharpe calculation
print("\nTesting Sharpe ratio calculation...")
returns = np.random.randn(100) * 0.01
volatility = np.std(returns)
sharpe = calculate_dynamic_sharpe(returns, volatility)
print(f"Test Sharpe ratio: {sharpe:.4f}")

print("\n✓ All tests completed successfully!")
print("\nYou can now run your pipeline with:")
print("from pipeline import EnhancedCryptoPipeline")
print("pipeline = EnhancedCryptoPipeline(GDRIVE_DIR)")
print("pipeline.run()")  # Assuming the pipeline has a run method