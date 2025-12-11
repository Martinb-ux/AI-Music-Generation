#!/bin/bash
# GPU Verification and Retraining Script
# Run this after rebooting to verify GPU and start training

echo "======================================"
echo "GPU Verification Script"
echo "======================================"
echo ""

echo "[1/4] Checking NVIDIA Driver..."
nvidia-smi
if [ $? -eq 0 ]; then
    echo "✅ NVIDIA driver loaded successfully"
else
    echo "❌ NVIDIA driver failed to load"
    exit 1
fi
echo ""

echo "[2/4] Checking TensorFlow GPU Detection..."
cd /home/rix/Documents/Github/CST435_FinalProj/music_generation/training
source ../../venv/bin/activate
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs detected: {len(gpus)}'); print(gpus); exit(0 if len(gpus) > 0 else 1)"
if [ $? -eq 0 ]; then
    echo "✅ TensorFlow can see GPU"
else
    echo "❌ TensorFlow cannot see GPU"
    exit 1
fi
echo ""

echo "[3/4] System Ready!"
echo "======================================"
echo ""

echo "[4/4] Starting Training..."
echo ""
echo "Training Options:"
echo "  1) Start fresh training (50 epochs)"
echo "  2) Resume from checkpoint (continue training)"
echo ""
echo -n "Enter choice (1 or 2): "
read choice
echo ""

if [ "$choice" = "2" ]; then
    echo "Resuming training from checkpoint..."
    echo "Command: python train.py --epochs 50 --batch_size 64 --resume"
    python train.py --epochs 50 --batch_size 64 --resume
else
    echo "Starting fresh training..."
    echo "Command: python train.py --epochs 50 --batch_size 64"
    python train.py --epochs 50 --batch_size 64
fi
