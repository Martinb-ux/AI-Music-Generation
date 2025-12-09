#!/bin/bash
# Convenience script to run the Streamlit demo

echo "LSTM Music Generator - Streamlit Demo"
echo "======================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Please run ./setup.sh first."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if model exists
if [ ! -f "music_generation/checkpoints/melody_lstm_best.keras" ]; then
    echo "Warning: Trained model not found at music_generation/checkpoints/melody_lstm_best.keras"
    echo ""
    echo "Please train the model first:"
    echo "  cd music_generation/training"
    echo "  python train.py --quick_test"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run Streamlit
echo "Starting Streamlit demo..."
echo "Open browser at http://localhost:8501"
echo ""

cd music_generation/demo
streamlit run streamlit_app.py
