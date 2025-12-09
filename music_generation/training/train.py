"""
Training Script for Melody LSTM

This script handles the full training pipeline:
1. Download and preprocess JSB Chorales dataset
2. Build and train LSTM model
3. Evaluate and generate sample melodies
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.midi_encoder import MIDIEventEncoder
from preprocessing.data_pipeline import JSBChoralesDataset, prepare_dataset
from models.melody_lstm import MelodyLSTM, create_model
from models.generator import MusicGenerator


def train_melody_lstm(
    data_dir: str = "../data",
    checkpoint_dir: str = "../checkpoints",
    epochs: int = 10,
    batch_size: int = 64,
    seq_length: int = 64,
    force_download: bool = False
):
    """
    Complete training pipeline for melody LSTM.

    Args:
        data_dir: Directory for data storage
        checkpoint_dir: Directory for model checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        seq_length: Length of training sequences
        force_download: Whether to re-download dataset
    """
    print("="*80)
    print("MELODY LSTM TRAINING PIPELINE")
    print("="*80)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================
    # Step 1: Prepare Dataset
    # ========================================
    print("\n[STEP 1] Preparing Dataset...")
    print("-"*80)

    dataset = JSBChoralesDataset(data_dir)

    # Check if processed data exists
    try:
        X_train, y_train = dataset.load_processed_data(split="train")
        X_val, y_val = dataset.load_processed_data(split="val")
        print("Loaded preprocessed data from disk.")
    except FileNotFoundError:
        print("Preprocessed data not found. Running full preprocessing pipeline...")
        dataset.download_dataset(force_download=force_download)
        X_all, y_all = dataset.preprocess_dataset(seq_length=seq_length, augment=True)
        X_train, X_val, y_train, y_val = dataset.create_train_val_split(X_all, y_all)
        dataset.save_processed_data(X_train, y_train, split="train")
        dataset.save_processed_data(X_val, y_val, split="val")

    print(f"\nDataset Summary:")
    print(f"  Train sequences: {len(X_train)}")
    print(f"  Val sequences: {len(X_val)}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Vocabulary size: {dataset.encoder.vocab_size}")

    # ========================================
    # Step 2: Build Model
    # ========================================
    print("\n[STEP 2] Building Model...")
    print("-"*80)

    model = create_model(
        vocab_size=dataset.encoder.vocab_size,
        seq_length=seq_length
    )

    # ========================================
    # Step 3: Train Model
    # ========================================
    print("\n[STEP 3] Training Model...")
    print("-"*80)

    checkpoint_path = os.path.join(checkpoint_dir, "melody_lstm_best.keras")

    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path
    )

    # ========================================
    # Step 4: Evaluate Model
    # ========================================
    print("\n[STEP 4] Evaluating Model...")
    print("-"*80)

    # Load best model
    model.load(checkpoint_path)

    # Evaluate on validation set
    val_loss, val_acc = model.model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # ========================================
    # Step 5: Generate Sample Melodies
    # ========================================
    print("\n[STEP 5] Generating Sample Melodies...")
    print("-"*80)

    generator = MusicGenerator(model, dataset.encoder)

    # Create output directory for samples
    samples_dir = os.path.join(checkpoint_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Generate melodies with different temperatures
    temperatures = [0.5, 0.8, 1.0, 1.2]

    for temp in temperatures:
        print(f"\nGenerating melody with temperature={temp}...")

        # Use a random validation sequence as seed
        seed_idx = np.random.randint(0, len(X_val))
        seed_sequence = X_val[seed_idx]

        # Generate
        output_path = os.path.join(samples_dir, f"sample_temp_{temp}.mid")
        generator.generate_to_midi(
            seed_sequence=seed_sequence,
            output_path=output_path,
            length=128,  # 128 time steps = 16 bars at 16th notes
            temperature=temp,
            sampling_method='temperature'
        )

    print(f"\nSample melodies saved to: {samples_dir}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {checkpoint_path}")
    print(f"Samples saved to: {samples_dir}")
    print(f"\nFinal Metrics:")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print("\nNext Steps:")
    print("  1. Listen to generated samples")
    print("  2. Run Streamlit demo: streamlit run ../demo/streamlit_app.py")
    print("  3. Experiment with different temperatures and sampling methods")
    print("="*80)

    return model, history


def quick_test():
    """Quick test with minimal data for debugging."""
    print("Running quick test with minimal settings...")

    train_melody_lstm(
        data_dir="../data",
        checkpoint_dir="../checkpoints",
        epochs=2,  # Just 2 epochs for testing
        batch_size=32,
        seq_length=64,
        force_download=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Melody LSTM model")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory for data storage")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--seq_length", type=int, default=64,
                        help="Sequence length")
    parser.add_argument("--force_download", action="store_true",
                        help="Force re-download of dataset")
    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test with minimal settings")

    args = parser.parse_args()

    if args.quick_test:
        quick_test()
    else:
        train_melody_lstm(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            force_download=args.force_download
        )
