"""
This file contains shared fixtures and configuration for pytest.
"""
import pytest
import sys
import os

# Add the project root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))