"""
Data Pipeline for JSB Chorales Dataset

Downloads and preprocesses the JSB Chorales dataset for LSTM training.
Data is downloaded from the internet and not stored in git.
"""

import os
import numpy as np
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, List
import glob
from .midi_encoder import MIDIEventEncoder


class JSBChoralesDataset:
    """
    Download and preprocess JSB Chorales dataset.

    The dataset contains 382 Bach chorales in MIDI format.
    Source: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
    """

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize dataset handler.

        Args:
            data_dir: Directory to store downloaded and processed data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "jsb_chorales"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = MIDIEventEncoder(time_step=0.125)  # 16th note resolution

    def download_dataset(self, force_download: bool = False):
        """
        Download JSB Chorales dataset from the internet.

        Args:
            force_download: Re-download even if files exist
        """
        # Alternative source: JSB Chorales from GitHub
        dataset_url = "https://github.com/czhuang/JSB-Chorales-dataset/archive/refs/heads/master.zip"
        zip_path = self.raw_dir / "jsb_chorales.zip"
        extract_dir = self.raw_dir

        # Check if already downloaded
        midi_files = list(self.raw_dir.glob("**/*.mid")) + list(self.raw_dir.glob("**/*.midi"))
        if len(midi_files) > 0 and not force_download:
            print(f"Found {len(midi_files)} MIDI files. Skipping download.")
            print(f"Use force_download=True to re-download.")
            return

        print(f"Downloading JSB Chorales dataset from {dataset_url}...")

        try:
            # Download with progress
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownload progress: {percent}%", end="")

            urllib.request.urlretrieve(dataset_url, zip_path, reporthook=progress_hook)
            print("\nDownload complete!")

            # Extract
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"Dataset extracted to {extract_dir}")

            # Clean up zip file
            zip_path.unlink()

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nAlternative: Download manually from:")
            print("https://github.com/czhuang/JSB-Chorales-dataset")
            print(f"And extract to: {self.raw_dir}")
            raise

    def load_midi_files(self) -> List[str]:
        """
        Get list of all MIDI files in the dataset.

        Returns:
            List of MIDI file paths
        """
        # Search recursively for MIDI files
        midi_files = []
        for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
            midi_files.extend(glob.glob(str(self.raw_dir / "**" / ext), recursive=True))

        print(f"Found {len(midi_files)} MIDI files")
        return sorted(midi_files)

    def preprocess_dataset(self, seq_length: int = 64, augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess all MIDI files into training sequences.

        Args:
            seq_length: Length of each training sequence
            augment: Whether to apply data augmentation (transposition)

        Returns:
            X_train, y_train arrays
        """
        midi_files = self.load_midi_files()

        if len(midi_files) == 0:
            raise ValueError("No MIDI files found. Run download_dataset() first.")

        all_X = []
        all_y = []
        failed_files = 0

        print(f"Processing {len(midi_files)} MIDI files...")

        for i, midi_path in enumerate(midi_files):
            try:
                # Convert MIDI to sequence
                sequence = self.encoder.midi_to_sequence(midi_path)

                # Skip if too short
                if len(sequence) < seq_length + 1:
                    continue

                # Create training sequences
                X, y = self.encoder.create_training_sequences(sequence, seq_length)
                all_X.append(X)
                all_y.append(y)

                # Data augmentation: transpose by ±1, ±2 semitones
                if augment:
                    for semitones in [-2, -1, 1, 2]:
                        transposed = self.encoder.augment_transpose(sequence, semitones)
                        if transposed is not None:
                            X_aug, y_aug = self.encoder.create_training_sequences(transposed, seq_length)
                            all_X.append(X_aug)
                            all_y.append(y_aug)

                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(midi_files)} files...")

            except Exception as e:
                failed_files += 1
                if failed_files < 5:  # Only show first few errors
                    print(f"Warning: Failed to process {midi_path}: {e}")

        if len(all_X) == 0:
            raise ValueError("No valid sequences extracted from MIDI files")

        # Concatenate all sequences
        X_train = np.concatenate(all_X, axis=0)
        y_train = np.concatenate(all_y, axis=0)

        print(f"\nPreprocessing complete!")
        print(f"Successfully processed: {len(midi_files) - failed_files}/{len(midi_files)} files")
        print(f"Total sequences: {len(X_train)}")
        print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")
        print(f"Vocabulary size: {self.encoder.vocab_size}")

        return X_train, y_train

    def save_processed_data(self, X: np.ndarray, y: np.ndarray, split: str = "train"):
        """
        Save preprocessed data to disk.

        Args:
            X: Input sequences
            y: Target sequences
            split: Data split name ('train', 'val', 'test')
        """
        X_path = self.processed_dir / f"X_{split}.npy"
        y_path = self.processed_dir / f"y_{split}.npy"

        np.save(X_path, X)
        np.save(y_path, y)

        print(f"Saved {split} data to:")
        print(f"  {X_path}")
        print(f"  {y_path}")

    def load_processed_data(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed data from disk.

        Args:
            split: Data split name ('train', 'val', 'test')

        Returns:
            X, y arrays
        """
        X_path = self.processed_dir / f"X_{split}.npy"
        y_path = self.processed_dir / f"y_{split}.npy"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Processed data not found. Run preprocess_dataset() first."
            )

        X = np.load(X_path)
        y = np.load(y_path)

        print(f"Loaded {split} data: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def create_train_val_split(self, X: np.ndarray, y: np.ndarray, val_split: float = 0.15) -> Tuple:
        """
        Split data into train and validation sets.

        Args:
            X: Input sequences
            y: Target sequences
            val_split: Fraction of data to use for validation

        Returns:
            X_train, X_val, y_train, y_val
        """
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Split
        val_size = int(len(X) * val_split)
        X_train = X_shuffled[val_size:]
        X_val = X_shuffled[:val_size]
        y_train = y_shuffled[val_size:]
        y_val = y_shuffled[:val_size]

        print(f"Train set: {len(X_train)} sequences")
        print(f"Val set: {len(X_val)} sequences")

        return X_train, X_val, y_train, y_val


def prepare_dataset(data_dir: str = "../data", force_download: bool = False):
    """
    Convenience function to download and preprocess the entire dataset.

    Args:
        data_dir: Directory to store data
        force_download: Re-download even if files exist

    Returns:
        Dataset object
    """
    dataset = JSBChoralesDataset(data_dir)

    # Download if needed
    dataset.download_dataset(force_download=force_download)

    # Preprocess
    X_all, y_all = dataset.preprocess_dataset(seq_length=64, augment=True)

    # Split into train/val
    X_train, X_val, y_train, y_val = dataset.create_train_val_split(X_all, y_all, val_split=0.15)

    # Save
    dataset.save_processed_data(X_train, y_train, split="train")
    dataset.save_processed_data(X_val, y_val, split="val")

    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print(f"Vocabulary size: {dataset.encoder.vocab_size}")
    print(f"Sequence length: 64")
    print(f"Train sequences: {len(X_train)}")
    print(f"Val sequences: {len(X_val)}")
    print("="*50)

    return dataset


if __name__ == "__main__":
    # Run the full pipeline
    import sys

    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"

    print("JSB Chorales Dataset Pipeline")
    print("="*50)

    dataset = prepare_dataset(data_dir=data_dir, force_download=False)
