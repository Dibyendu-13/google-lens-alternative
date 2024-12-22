"""
Package initialization for test scripts.
This directory contains unit tests for all the models implemented in the project.
The tests verify the functionality and performance of various image similarity models, including 
Autoencoder, CNN-based, Siamese Network, Transformer, and Triplet Network with SimCLR.
"""

# Importing test scripts for each model
from .test_autoencoder import *
from .test_cnn import *
from .test_siamese import *
from .test_transformer import *
from .test_triplet import *

# Optional: Add common setup/teardown functions or shared utilities for tests
import os

# Path to the directory where test data is stored (if applicable)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def setup_test_environment():
    """
    A utility function to set up the testing environment (optional).
    Could be used for data loading, resource allocation, etc.
    """
    pass

def teardown_test_environment():
    """
    A utility function to tear down the testing environment (optional).
    Clean up resources, clear caches, etc.
    """
    pass
