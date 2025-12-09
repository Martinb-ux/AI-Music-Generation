#!/bin/bash
# Setup script for Music Generation LSTM project

echo "Setting up Music Generation LSTM project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To install FluidSynth for audio synthesis:"
echo "  brew install fluid-synth"
