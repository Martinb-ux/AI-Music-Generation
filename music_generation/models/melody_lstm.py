"""
Event-based LSTM Model for Monophonic Melody Generation (Week 1)

This model generates melodies one note at a time, similar to text generation.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple


class MelodyLSTM:
    """
    LSTM model for monophonic melody generation.

    Architecture:
    - Embedding layer (learn note representations)
    - 2x LSTM layers (256 units each)
    - Dense output layer (vocab_size)
    - Softmax activation (next note prediction)
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128, lstm_units: int = 256):
        """
        Initialize the model.

        Args:
            vocab_size: Size of note vocabulary (REST + note range)
            embedding_dim: Dimension of note embeddings
            lstm_units: Number of units in LSTM layers
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.history = None

    def build_model(self, seq_length: int = 64) -> keras.Model:
        """
        Build the LSTM model.

        Args:
            seq_length: Length of input sequences

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input: (batch, seq_length) - sequence of note indices
            layers.Input(shape=(seq_length,)),

            # Embedding: learn representations for each note
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                mask_zero=False  # Don't mask REST tokens
            ),

            # First LSTM layer with return sequences
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2
            ),

            # Second LSTM layer
            layers.LSTM(
                self.lstm_units,
                dropout=0.3,
                recurrent_dropout=0.2
            ),

            # Dense hidden layer
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),

            # Output layer: predict next note
            layers.Dense(self.vocab_size, activation='softmax')
        ], name='MelodyLSTM')

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        # Print model summary
        print("\nModel Architecture:")
        print("="*60)
        model.summary()
        print("="*60)

        return model

    def get_callbacks(self, checkpoint_path: str = "../checkpoints/melody_lstm_best.keras"):
        """
        Get training callbacks (checkpointing, early stopping, etc.).

        Args:
            checkpoint_path: Path to save best model

        Returns:
            List of callbacks
        """
        checkpoint_dir = os.path.dirname(checkpoint_path)

        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),

            # Early stopping (increased patience for longer training)
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),

            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(checkpoint_dir, 'logs'),
                histogram_freq=1
            ),

            # CSV logger for easy analysis
            keras.callbacks.CSVLogger(
                os.path.join(checkpoint_dir, 'training_log.csv'),
                append=True
            )
        ]

        return callbacks

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 10,
              batch_size: int = 64,
              checkpoint_path: str = "../checkpoints/melody_lstm_best.keras") -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training sequences (num_sequences, seq_length)
            y_train: Training targets (num_sequences,)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"\nStarting training...")
        print(f"Train samples: {len(X_train)}")
        if X_val is not None:
            print(f"Val samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("="*60)

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(checkpoint_path),
            verbose=1
        )

        print("\nTraining complete!")
        return self.history

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def predict_next_note(self, sequence: np.ndarray, temperature: float = 1.0) -> int:
        """
        Predict the next note given a sequence.

        Args:
            sequence: Input sequence of shape (seq_length,)
            temperature: Sampling temperature (higher = more random)

        Returns:
            Next note index
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        # Add batch dimension
        sequence = np.expand_dims(sequence, 0)

        # Get predictions
        predictions = self.model.predict(sequence, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions - np.max(predictions))
        predictions = predictions / np.sum(predictions)

        # Sample from distribution
        next_note = np.random.choice(len(predictions), p=predictions)

        return next_note

    def generate_melody(self,
                        seed_sequence: np.ndarray,
                        length: int = 128,
                        temperature: float = 0.8) -> np.ndarray:
        """
        Generate a melody by sampling from the model.

        Args:
            seed_sequence: Starting sequence of shape (seq_length,)
            length: Number of notes to generate
            temperature: Sampling temperature

        Returns:
            Generated sequence of shape (length,)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        seq_length = seed_sequence.shape[0]
        generated = list(seed_sequence)

        print(f"Generating {length} notes (temperature={temperature})...")

        for i in range(length):
            # Get last seq_length notes
            current_seq = np.array(generated[-seq_length:])

            # Predict next note
            next_note = self.predict_next_note(current_seq, temperature)
            generated.append(next_note)

            if (i + 1) % 32 == 0:
                print(f"Generated {i + 1}/{length} notes...")

        # Return only the newly generated part
        return np.array(generated[seq_length:])


def create_model(vocab_size: int, seq_length: int = 64) -> MelodyLSTM:
    """
    Convenience function to create and build a model.

    Args:
        vocab_size: Size of note vocabulary
        seq_length: Length of input sequences

    Returns:
        Built MelodyLSTM model
    """
    model = MelodyLSTM(vocab_size=vocab_size, embedding_dim=128, lstm_units=256)
    model.build_model(seq_length=seq_length)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing MelodyLSTM model...")

    # Create test model (vocab_size = 89 for 88 piano notes + REST)
    model = create_model(vocab_size=89, seq_length=64)

    # Create dummy data
    X_test = np.random.randint(0, 89, size=(100, 64))
    y_test = np.random.randint(0, 89, size=(100,))

    print("\nTest prediction...")
    seed = np.random.randint(0, 89, size=(64,))
    next_note = model.predict_next_note(seed, temperature=1.0)
    print(f"Predicted next note: {next_note}")

    print("\nTest generation...")
    generated = model.generate_melody(seed, length=32, temperature=0.8)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Sample notes: {generated[:10]}")

    print("\nModel test complete!")
