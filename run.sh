#!/bin/bash

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create new virtual environment with explicit Python 3.11
echo "Creating virtual environment..."
python3.11 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify Python version
echo "Checking Python version..."
python --version

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the application
echo "Starting the application..."
python agent.py 