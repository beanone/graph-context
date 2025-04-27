#!/bin/bash

# Exit on error
set -e

echo "Cleaning up test artifacts..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Remove pytest cache
rm -rf .pytest_cache/
rm -rf .coverage
rm -rf coverage.xml
rm -rf htmlcov/

# Remove mypy cache
rm -rf .mypy_cache/

# Remove ruff cache
rm -rf .ruff_cache/

# Remove any temporary test files
find . -type f -iname "*,cover" -delete  # Case-insensitive match for .cover files
find . -type f -name "*.tmp" -delete
find . -type f -name "temp_*" -delete
find . -type f -name "test_*.tmp" -delete

echo "Cleanup complete!"
