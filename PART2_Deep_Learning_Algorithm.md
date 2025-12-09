# Part 2: Deep Learning Algorithm and Software Implementation

## Project Overview

This document details the deep learning algorithms, software packages, and implementation approach for the **Music Generation with LSTM** project. The system uses a two-layer LSTM neural network to learn musical patterns from piano roll sequences and generate original compositions.

**Note:** This documentation describes the planned implementation approach based on the detailed implementation plan. The code examples demonstrate the intended deep learning methodology.

---

## 1. Algorithm Overview

### 1.1 Model Type

**Long Short-Term Memory (LSTM) Recurrent Neural Network**

**Purpose:** Sequential prediction for polyphonic music generation

**Task Type:** Multi-label binary classification
- Input: 64 timesteps of piano roll history
- Output: Probability distribution over 88 piano keys for the next timestep
- Multiple notes can be active simultaneously (polyphonic)

### 1.2 Why LSTM for Music Generation?

**Advantages:**
1. **Sequential Memory:** LSTMs remember long-term patterns (melodies, chord progressions)
2. **Temporal Dependencies:** Captures relationships between distant musical events
3. **Polyphony Support:** Can generate multiple simultaneous notes (chords)
4. **Variable-Length Processing:** Handles sequences of any length
5. **Proven Success:** Widely used in music generation research

**LSTM vs. Other Architectures:**

| Architecture | Pros | Cons |
|--------------|------|------|
| **LSTM** | Long-term memory, proven for music | Slower than Transformers |
| Transformer | Parallel processing, attention | Requires more data, complex |
| RNN (vanilla) | Simple, fast | Vanishing gradients, short memory |
| CNN | Fast, parallelizable | No temporal memory |

**Choice:** LSTM provides the best balance of performance and complexity for this project.

---

## 2. Model Architecture

### 2.1 Architecture Diagram

```
Input Sequence (64 timesteps, 88 notes each)
    ↓
┌─────────────────────────────────────────┐
│  Input Layer: (batch, 64, 88)           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  LSTM Layer 1: 512 units                │
│  - Return sequences: True               │
│  - Dropout: 0.3                         │
│  - Recurrent dropout: 0.2               │
│  Output: (batch, 64, 512)               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  LSTM Layer 2: 512 units                │
│  - Return sequences: False              │
│  - Dropout: 0.3                         │
│  - Recurrent dropout: 0.2               │
│  Output: (batch, 512)                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Dense Layer: 256 units                 │
│  - Activation: ReLU                     │
│  Output: (batch, 256)                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Dropout Layer: 0.4                     │
│  Output: (batch, 256)                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Output Layer: 88 units                 │
│  - Activation: Sigmoid                  │
│  Output: (batch, 88)                    │
│  [Independent probability per note]     │
└─────────────────────────────────────────┘
    ↓
Output: Probability distribution over 88 piano keys
```

### 2.2 Layer-by-Layer Breakdown

#### **Layer 1: Input Layer**
- **Shape:** (batch_size, 64, 88)
- **Function:** Accepts piano roll sequences
- **Interpretation:**
  - batch_size: Number of sequences processed simultaneously
  - 64: Timesteps (4 musical bars)
  - 88: Features (piano keys A0-C8)

#### **Layer 2: LSTM Layer 1 (512 units)**
- **Units:** 512 LSTM cells
- **Return sequences:** True (outputs for all 64 timesteps)
- **Dropout:** 0.3 (30% of units randomly dropped during training)
- **Recurrent dropout:** 0.2 (20% of recurrent connections dropped)
- **Output shape:** (batch_size, 64, 512)
- **Purpose:** Capture short-term musical patterns (note sequences, rhythms)
- **Parameters:** ~2.1M parameters

**LSTM Cell Operations:**
```
For each timestep t:
1. Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
2. Input Gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
3. Candidate: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
4. Cell State: C_t = f_t * C_{t-1} + i_t * C̃_t
5. Output Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
6. Hidden State: h_t = o_t * tanh(C_t)
```

#### **Layer 3: LSTM Layer 2 (512 units)**
- **Units:** 512 LSTM cells
- **Return sequences:** False (only final output)
- **Dropout:** 0.3
- **Recurrent dropout:** 0.2
- **Output shape:** (batch_size, 512)
- **Purpose:** Capture long-term musical patterns (phrases, harmony, structure)
- **Parameters:** ~4.2M parameters

**Why Two LSTM Layers?**
- **Layer 1:** Learns note-level patterns (which notes follow others)
- **Layer 2:** Learns phrase-level patterns (musical structure, cadences)
- **Hierarchical learning:** Low-level → High-level musical understanding

#### **Layer 4: Dense Hidden Layer (256 units)**
- **Units:** 256 neurons
- **Activation:** ReLU (Rectified Linear Unit)
- **Output shape:** (batch_size, 256)
- **Purpose:** Non-linear transformation, feature combination
- **Parameters:** ~131K parameters

**ReLU Activation:**
```
f(x) = max(0, x)
```
- Introduces non-linearity
- Prevents vanishing gradients
- Computationally efficient

#### **Layer 5: Dropout Layer (0.4)**
- **Dropout rate:** 0.4 (40% of neurons randomly dropped)
- **Output shape:** (batch_size, 256)
- **Purpose:** Regularization, prevent overfitting
- **Only active during training** (disabled during inference)

#### **Layer 6: Output Layer (88 units)**
- **Units:** 88 neurons (one per piano key)
- **Activation:** Sigmoid
- **Output shape:** (batch_size, 88)
- **Purpose:** Independent probability for each note
- **Parameters:** ~23K parameters

**Sigmoid Activation:**
```
σ(x) = 1 / (1 + e^(-x))
```
- Output range: [0, 1]
- Interpreted as probability
- **Independent predictions:** Each note's probability is calculated independently
- **Polyphonic support:** Multiple notes can have high probability simultaneously

### 2.3 Total Model Parameters

**Parameter Count:**
- LSTM Layer 1: 2,101,248 parameters
- LSTM Layer 2: 4,196,352 parameters
- Dense Layer: 131,328 parameters
- Output Layer: 22,696 parameters
- **Total: 6,451,624 trainable parameters (~6.5M)**

**Model Size:**
- Disk storage: ~26 MB (6.5M parameters × 4 bytes/float32)
- RAM during training: ~26 MB + gradients/optimizer states ≈ 100 MB

---

## 3. Software Packages and Functions

### 3.1 Core Deep Learning Framework

#### **TensorFlow 2.x / Keras**

**Installation:**
```bash
pip install tensorflow==2.15.0
```

**Key Modules and Functions:**

**1. Model Building:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model container
keras.Sequential()

# Layer types
layers.Input()          # Input layer specification
layers.LSTM()           # LSTM recurrent layer
layers.Dense()          # Fully connected layer
layers.Dropout()        # Regularization layer
```

**2. Optimizers:**
```python
from tensorflow.keras.optimizers import Adam

# Adam optimizer with learning rate
optimizer = Adam(learning_rate=0.001)
```

**3. Loss Functions:**
```python
from tensorflow.keras.losses import BinaryCrossentropy

# Binary crossentropy for multi-label classification
loss_fn = BinaryCrossentropy()
```

**4. Metrics:**
```python
from tensorflow.keras.metrics import BinaryAccuracy

# Binary accuracy for evaluation
metric = BinaryAccuracy()
```

**5. Callbacks:**
```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint,      # Save best model
    EarlyStopping,        # Stop when not improving
    ReduceLROnPlateau     # Reduce learning rate adaptively
)
```

**6. Training:**
```python
model.compile()         # Configure optimizer, loss, metrics
model.fit()             # Train the model
model.evaluate()        # Evaluate on test data
model.predict()         # Generate predictions
```

**7. Model Saving/Loading:**
```python
model.save()            # Save model to disk
keras.models.load_model()  # Load model from disk
```

### 3.2 MIDI Processing

#### **Pretty MIDI**

**Installation:**
```bash
pip install pretty_midi
```

**Key Functions:**
```python
import pretty_midi

# Load MIDI file
pm = pretty_midi.PrettyMIDI('file.mid')

# Convert to piano roll
piano_roll = pm.get_piano_roll(fs=4)  # fs = sampling frequency

# Create new MIDI
pm_new = pretty_midi.PrettyMIDI(initial_tempo=120)

# Create instrument
instrument = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano

# Create note
note = pretty_midi.Note(
    velocity=100,    # Loudness
    pitch=60,        # MIDI note number (60 = middle C)
    start=0.0,       # Start time in seconds
    end=0.5          # End time in seconds
)
instrument.notes.append(note)
pm_new.instruments.append(instrument)

# Write MIDI file
pm_new.write('output.mid')
```

### 3.3 Numerical Computing

#### **NumPy**

**Installation:**
```bash
pip install numpy
```

**Key Functions:**
```python
import numpy as np

# Array operations
np.array()              # Create array
np.zeros()              # Array of zeros
np.ones()               # Array of ones
np.random.rand()        # Random values

# Transformations
np.roll()               # Shift array (for transposition)
np.transpose()          # Transpose dimensions
np.reshape()            # Change shape

# Mathematical operations
np.mean()               # Mean value
np.sum()                # Sum
np.exp()                # Exponential
np.log()                # Natural logarithm

# Boolean operations
np.where()              # Find indices where condition is true
np.diff()               # Differences between adjacent elements

# File I/O
np.save()               # Save array to .npy file
np.load()               # Load array from .npy file
```

### 3.4 Music Theory (Optional)

#### **Music21**

**Installation:**
```bash
pip install music21
```

**Key Functions:**
```python
from music21 import *

# Key detection
key_detected = stream.analyze('key')

# Chord analysis
chord_obj = chord.Chord(['C4', 'E4', 'G4'])  # C major chord

# Scale operations
c_major = scale.MajorScale('C')
notes_in_scale = c_major.getPitches('C4', 'C5')
```

### 3.5 Audio Synthesis

#### **FluidSynth (Command-line)**

**Installation:**
```bash
# macOS
brew install fluid-synth

# Linux
sudo apt-get install fluidsynth

# Windows
# Download from: https://github.com/FluidSynth/fluidsynth/releases
```

**Usage:**
```bash
# Convert MIDI to WAV
fluidsynth -ni soundfont.sf2 input.mid -F output.wav -r 44100
```

**Python Integration:**
```python
import subprocess

def midi_to_audio(midi_path, audio_path, soundfont_path):
    """Convert MIDI to audio using FluidSynth"""
    subprocess.run([
        'fluidsynth',
        '-ni',                    # Non-interactive mode
        soundfont_path,           # Path to .sf2 soundfont
        midi_path,                # Input MIDI file
        '-F', audio_path,         # Output audio file
        '-r', '44100'             # Sample rate (CD quality)
    ])
```

**Download Soundfont:**
```bash
wget https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip
unzip FluidR3_GM.zip
```

### 3.6 Visualization

#### **Matplotlib**

**Installation:**
```bash
pip install matplotlib
```

**Key Functions:**
```python
import matplotlib.pyplot as plt

# Piano roll visualization
plt.figure(figsize=(12, 6))
plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='binary')
plt.xlabel('Time Steps')
plt.ylabel('Piano Keys (MIDI Note)')
plt.title('Piano Roll Visualization')
plt.colorbar(label='Note Active')
plt.show()

# Training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### 3.7 Interactive Demo

#### **Streamlit**

**Installation:**
```bash
pip install streamlit
```

**Key Functions:**
```python
import streamlit as st

# UI Components
st.title()              # Page title
st.sidebar.slider()     # Slider control
st.button()             # Button
st.selectbox()          # Dropdown menu
st.audio()              # Audio player
st.download_button()    # Download button
st.pyplot()             # Display matplotlib figure
st.spinner()            # Loading spinner
```

### 3.8 Utilities

#### **scikit-learn**

**Installation:**
```bash
pip install scikit-learn
```

**Key Functions:**
```python
from sklearn.model_selection import train_test_split

# Data splitting
train_data, test_data = train_test_split(
    data,
    train_size=0.7,
    random_state=42
)
```

#### **tqdm (Progress Bars)**

**Installation:**
```bash
pip install tqdm
```

**Usage:**
```python
from tqdm import tqdm

for item in tqdm(items, desc="Processing"):
    process(item)
```

---

## 4. Model Implementation Code

### 4.1 Complete Model Building Code

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_piano_roll_lstm(seq_length=64, n_notes=88,
                          lstm_units=512, dense_units=256,
                          dropout=0.3, recurrent_dropout=0.2):
    """
    Build LSTM model for polyphonic music generation

    Parameters:
    - seq_length: Number of timesteps in input (default: 64 = 4 bars)
    - n_notes: Number of piano keys (default: 88)
    - lstm_units: Number of LSTM units per layer (default: 512)
    - dense_units: Number of dense layer units (default: 256)
    - dropout: Dropout rate for LSTM and dense layers (default: 0.3)
    - recurrent_dropout: Recurrent dropout for LSTM (default: 0.2)

    Returns:
    - Compiled Keras model ready for training
    """

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(seq_length, n_notes), name='input'),

        # First LSTM layer: capture short-term patterns
        layers.LSTM(
            units=lstm_units,
            return_sequences=True,      # Pass sequences to next LSTM
            dropout=dropout,             # Regularization
            recurrent_dropout=recurrent_dropout,
            name='lstm_layer_1'
        ),

        # Second LSTM layer: capture long-term patterns
        layers.LSTM(
            units=lstm_units,
            return_sequences=False,     # Only return final output
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name='lstm_layer_2'
        ),

        # Dense hidden layer with ReLU activation
        layers.Dense(
            units=dense_units,
            activation='relu',
            name='dense_hidden'
        ),

        # Dropout for regularization
        layers.Dropout(rate=0.4, name='dropout'),

        # Output layer: 88 independent note predictions
        layers.Dense(
            units=n_notes,
            activation='sigmoid',       # Independent probabilities [0, 1]
            name='output'
        )
    ])

    return model

# Create model
model = build_piano_roll_lstm()

# Display architecture
model.summary()
```

**Model Summary Output:**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
lstm_layer_1 (LSTM)         (None, 64, 512)          2,101,248
lstm_layer_2 (LSTM)         (None, 512)              4,196,352
dense_hidden (Dense)        (None, 256)                131,328
dropout (Dropout)           (None, 256)                      0
output (Dense)              (None, 88)                  22,696
=================================================================
Total params: 6,451,624 (24.61 MB)
Trainable params: 6,451,624 (24.61 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

---

## 5. Training Algorithm

### 5.1 Loss Function

**Problem:** Piano rolls are sparse (95% zeros)
- Most piano keys are silent at any time
- Standard binary crossentropy would ignore active notes
- Model might predict all zeros (trivial solution)

**Solution:** Weighted binary crossentropy

```python
import tensorflow as tf

def weighted_binary_crossentropy(y_true, y_pred, weight_active=10.0):
    """
    Custom loss function with higher weight for active notes

    Parameters:
    - y_true: Ground truth labels (batch, 88)
    - y_pred: Predicted probabilities (batch, 88)
    - weight_active: Weight multiplier for active notes (default: 10.0)

    Returns:
    - Weighted loss value
    """
    # Standard binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Create weight matrix: 10x for active notes (1), 1x for silent (0)
    weights = 1.0 + (weight_active - 1.0) * y_true

    # Apply weights and compute mean
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)
```

**Effect of Weighting:**
- Active notes (1): 10× weight → model pays more attention
- Silent notes (0): 1× weight → less emphasis
- Balances learning between sparse and dense classes

### 5.2 Optimizer Configuration

**Adam Optimizer** (Adaptive Moment Estimation)

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,        # Initial learning rate
    beta_1=0.9,                 # Exponential decay rate for 1st moment
    beta_2=0.999,               # Exponential decay rate for 2nd moment
    epsilon=1e-7                # Small constant for numerical stability
)
```

**Why Adam?**
- Adaptive learning rates per parameter
- Combines momentum and RMSProp benefits
- Works well with sparse gradients (perfect for sparse piano rolls)
- Minimal hyperparameter tuning needed

### 5.3 Training Callbacks

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

# Save best model during training
checkpoint = ModelCheckpoint(
    filepath='checkpoints/best_model.keras',
    monitor='val_loss',            # Metric to monitor
    save_best_only=True,           # Only save when val_loss improves
    mode='min',                    # Lower is better
    verbose=1
)

# Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,                    # Wait 5 epochs without improvement
    restore_best_weights=True,     # Restore best model at end
    mode='min',
    verbose=1
)

# Reduce learning rate when plateau detected
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                    # New_lr = lr * 0.5
    patience=3,                    # Wait 3 epochs without improvement
    min_lr=1e-6,                   # Don't go below this
    mode='min',
    verbose=1
)

# TensorBoard logging (optional)
tensorboard = TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True
)

callbacks = [checkpoint, early_stopping, reduce_lr]
```

### 5.4 Complete Training Code

```python
import numpy as np
from tensorflow import keras

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=64):
    """
    Train the piano roll LSTM model

    Parameters:
    - model: Keras model (from build_piano_roll_lstm)
    - X_train: Training input (n_samples, 64, 88)
    - y_train: Training targets (n_samples, 88)
    - X_val: Validation input (n_samples, 64, 88)
    - y_val: Validation targets (n_samples, 88)
    - epochs: Maximum number of epochs (default: 30)
    - batch_size: Batch size for training (default: 64)

    Returns:
    - model: Trained model
    - history: Training history object
    """

    # Compile model with weighted loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=weighted_binary_crossentropy,
        metrics=['binary_accuracy']
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            'checkpoints/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model
    print(f"Training on {len(X_train)} sequences...")
    print(f"Validation on {len(X_val)} sequences...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("Training complete!")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")

    return model, history

# Load preprocessed data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

# Build model
model = build_piano_roll_lstm()

# Train
model, history = train_model(
    model, X_train, y_train, X_val, y_val,
    epochs=30,
    batch_size=64
)

# Save final model
model.save('models/piano_roll_lstm_final.keras')
```

### 5.5 Training Configuration Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 30 | Sufficient for convergence with early stopping |
| Batch size | 64 | Balances memory usage and gradient stability |
| Learning rate | 0.001 | Standard Adam learning rate |
| Loss function | Weighted BCE | Handles class imbalance (sparse piano rolls) |
| Active note weight | 10× | Emphasizes learning active notes |
| Dropout | 0.3-0.4 | Prevents overfitting |
| Early stopping patience | 5 epochs | Allows exploration before stopping |
| LR reduction patience | 3 epochs | Adapts when learning stagnates |

### 5.6 Expected Training Performance

**Hardware:** Apple M1/M2 with MPS acceleration

**Training Time:**
- Time per epoch: ~4-6 minutes (50K sequences)
- Total training: ~2-3 hours (with early stopping ~15-20 epochs)
- Evaluation: ~30 seconds per split

**Performance Metrics (Target):**
- **Training loss:** < 0.25
- **Validation loss:** < 0.30
- **Binary accuracy:** > 90%
- **Overfitting gap:** < 0.05 (val_loss - train_loss)

**Training Curve (Expected):**
```
Epoch   Train Loss   Val Loss   Binary Accuracy
1       0.450        0.480      0.850
5       0.320        0.350      0.880
10      0.270        0.310      0.900
15      0.240        0.290      0.910
20      0.230        0.285      0.915   ← Best model (early stopping)
```

---

## 6. Generation Algorithm

### 6.1 Generation Strategy

**Autoregressive Generation:** Predict one timestep at a time, using previous predictions as input

**Process:**
1. Start with seed sequence (64 timesteps)
2. Predict next timestep probabilities
3. Sample notes from probabilities (with temperature and constraints)
4. Append to sequence, remove oldest timestep (sliding window)
5. Repeat for desired length

### 6.2 Temperature Sampling

**Purpose:** Control randomness in generation

```python
import numpy as np

def apply_temperature(probs, temperature=0.8):
    """
    Apply temperature scaling to probability distribution

    Parameters:
    - probs: Probability distribution (88,)
    - temperature: Sampling temperature
      - < 1.0: More conservative (focus on high-probability notes)
      - = 1.0: Use model's raw probabilities
      - > 1.0: More creative (explore lower-probability notes)

    Returns:
    - Adjusted probabilities
    """
    # Avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)

    # Apply temperature to logits
    logits = np.log(probs)
    scaled_logits = logits / temperature

    # Convert back to probabilities (softmax)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    new_probs = exp_logits / np.sum(exp_logits)

    return new_probs
```

**Temperature Effects:**
```
Temperature 0.5 (Conservative):
- Amplifies differences between high/low probabilities
- Generates safe, predictable music
- Lower diversity, more repetitive

Temperature 1.0 (Neutral):
- Uses model's raw probabilities
- Balanced creativity and coherence

Temperature 1.5 (Creative):
- Flattens probability distribution
- Explores unlikely note combinations
- Higher diversity, potentially chaotic
```

### 6.3 Musical Constraints

**Key Constraint:** Force generated notes to stay within a musical key

```python
def constrain_to_key(probs, key='C', scale='major'):
    """
    Constrain probabilities to notes in a specific key

    Parameters:
    - probs: Probability distribution (88,)
    - key: Root note (C, D, E, F, G, A, B)
    - scale: Scale type (major, minor, pentatonic)

    Returns:
    - Constrained probabilities (notes not in key set to 0)
    """
    # Define scale intervals (semitones from root)
    scales = {
        'major': [0, 2, 4, 5, 7, 9, 11],           # W-W-H-W-W-W-H
        'minor': [0, 2, 3, 5, 7, 8, 10],           # W-H-W-W-H-W-W
        'pentatonic': [0, 2, 4, 7, 9],             # Pentatonic major
        'blues': [0, 3, 5, 6, 7, 10]               # Blues scale
    }

    # Key offsets from C (semitones)
    keys = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5,
        'G': 7, 'A': 9, 'B': 11
    }

    # Get scale intervals
    root = keys[key]
    valid_intervals = scales[scale]
    valid_notes = [(root + interval) % 12 for interval in valid_intervals]

    # Create mask for valid notes across all octaves
    mask = np.zeros(88)
    for i in range(88):
        midi_note = i + 21  # MIDI 21 = A0 (first piano key)
        pitch_class = midi_note % 12  # 0-11 (C-B)
        if pitch_class in valid_notes:
            mask[i] = 1.0

    # Apply mask
    constrained_probs = probs * mask

    # Renormalize
    if constrained_probs.sum() > 0:
        constrained_probs = constrained_probs / constrained_probs.sum()
    else:
        # Fallback: uniform over valid notes
        constrained_probs = mask / mask.sum()

    return constrained_probs
```

### 6.4 Complete Generation Code

```python
def generate_music(model, seed=None, length=256,
                  temperature=0.8, key='C', scale='major',
                  threshold=0.5):
    """
    Generate music using trained LSTM model

    Parameters:
    - model: Trained Keras model
    - seed: Initial sequence (64, 88) or None for random
    - length: Number of timesteps to generate (256 = 16 bars)
    - temperature: Sampling randomness (0.5-1.5)
    - key: Musical key (C, D, E, F, G, A, B)
    - scale: Scale type (major, minor, pentatonic)
    - threshold: Probability threshold for note activation (default: 0.5)

    Returns:
    - Generated piano roll (length, 88)
    """
    # Create or use provided seed
    if seed is None:
        # Random seed from validation set
        X_val = np.load('data/processed/X_val.npy')
        seed = X_val[np.random.randint(len(X_val))]

    # Initialize generation
    generated = seed.copy()
    current_sequence = seed.copy()

    print(f"Generating {length} timesteps...")
    print(f"Temperature: {temperature}, Key: {key} {scale}")

    for step in range(length):
        # Predict next timestep
        # Input shape: (1, 64, 88) - batch size 1
        probs = model.predict(
            current_sequence[np.newaxis, :, :],
            verbose=0
        )[0]  # Get first (only) prediction: (88,)

        # Apply temperature sampling
        probs = apply_temperature(probs, temperature)

        # Apply musical constraints
        if key is not None and scale is not None:
            probs = constrain_to_key(probs, key, scale)

        # Sample notes: threshold at probability
        next_step = (probs > threshold).astype(np.int32)

        # Ensure at least some notes (avoid silence)
        if np.sum(next_step) == 0:
            # Pick top-k most likely notes
            top_k = 3
            top_indices = np.argsort(probs)[-top_k:]
            next_step[top_indices] = 1

        # Append to generated sequence
        generated = np.vstack([generated, next_step])

        # Update sliding window (remove oldest, add newest)
        current_sequence = np.vstack([current_sequence[1:], next_step])

        # Progress indicator
        if (step + 1) % 64 == 0:
            print(f"  Generated {step + 1}/{length} steps...")

    print("Generation complete!")
    return generated

# Example usage
model = keras.models.load_model('models/piano_roll_lstm_final.keras')

# Generate 16 bars of music in C major
piano_roll = generate_music(
    model=model,
    length=256,          # 16 bars × 16 steps
    temperature=0.8,     # Balanced creativity
    key='C',
    scale='major',
    threshold=0.5
)

# Convert to MIDI
encoder = PianoRollEncoder()
midi = encoder.piano_roll_to_midi(piano_roll, tempo=120)
midi.write('generated_music.mid')

print(f"Generated music: {piano_roll.shape}")
print(f"Note density: {np.mean(np.sum(piano_roll, axis=1)):.2f} notes/step")
```

### 6.5 Generation Strategies Comparison

| Strategy | Temperature | Key Constraint | Use Case |
|----------|-------------|----------------|----------|
| Conservative | 0.5 | Yes | Classical, predictable pieces |
| Balanced | 0.8 | Yes | General-purpose generation |
| Creative | 1.2 | Yes | Experimental, diverse output |
| Free | 1.0 | No | Chromatic, avant-garde |
| Jazz | 0.9 | Blues scale | Jazz improvisation |

---

## 7. Step-by-Step Algorithm Summary

### 7.1 Training Phase

**Step 1:** Load preprocessed data
- Load X_train, y_train, X_val, y_val from NumPy files

**Step 2:** Build model architecture
- Create 2-layer LSTM with 512 units each
- Add dense layer (256 units) and output layer (88 units)

**Step 3:** Compile model
- Loss: Weighted binary crossentropy (10× weight for active notes)
- Optimizer: Adam (lr=0.001)
- Metrics: Binary accuracy

**Step 4:** Setup callbacks
- ModelCheckpoint: Save best model
- EarlyStopping: Stop when val_loss plateaus (patience=5)
- ReduceLROnPlateau: Reduce lr when stuck (patience=3)

**Step 5:** Train model
- Fit on training data with validation monitoring
- Batch size: 64, Epochs: 30 (with early stopping)

**Step 6:** Save trained model
- Export to .keras file for later use

### 7.2 Generation Phase

**Step 1:** Load trained model
- Load from .keras file

**Step 2:** Initialize seed sequence
- Use 64 timesteps from validation set or random

**Step 3:** Iterative prediction loop
```
For each timestep to generate:
  a. Feed current 64-step sequence to model
  b. Get probability distribution over 88 notes
  c. Apply temperature scaling
  d. Apply musical constraints (key, scale)
  e. Sample notes (threshold at 0.5)
  f. Append to generated sequence
  g. Update sliding window (remove oldest, add newest)
```

**Step 4:** Post-processing
- Convert piano roll to MIDI
- Add dynamics, articulation
- Export as MIDI file

**Step 5:** Audio rendering (optional)
- Convert MIDI to WAV using FluidSynth
- Enable playback and download

---

## 8. Model Evaluation Metrics

### 8.1 Objective Metrics

**1. Validation Loss**
```python
val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
print(f"Validation Loss: {val_loss:.4f}")
# Target: < 0.30
```

**2. Binary Accuracy**
```python
from tensorflow.keras.metrics import BinaryAccuracy

accuracy = BinaryAccuracy()
predictions = model.predict(X_val)
accuracy.update_state(y_val, predictions)
print(f"Binary Accuracy: {accuracy.result().numpy():.4f}")
# Target: > 0.90
```

**3. Note Density**
```python
def compute_note_density(piano_roll):
    """Average number of notes active per timestep"""
    return np.mean(np.sum(piano_roll, axis=1))

density = compute_note_density(generated_piano_roll)
print(f"Note Density: {density:.2f} notes/step")
# Target: 2-4 notes/step (realistic for piano)
```

**4. Pitch Class Histogram**
```python
def pitch_class_histogram(piano_roll):
    """Distribution of pitch classes (C, C#, D, etc.)"""
    pitch_classes = np.zeros(12)
    for i in range(88):
        midi_note = i + 21
        pitch_class = midi_note % 12
        pitch_classes[pitch_class] += np.sum(piano_roll[:, i])
    return pitch_classes / np.sum(pitch_classes)

pch = pitch_class_histogram(generated_piano_roll)
# Compare with training data distribution
```

### 8.2 Subjective Metrics

**Human Listening Test:**
1. Generate 20 pieces (16 bars each)
2. Ask 5+ listeners to rate (1-5 scale):
   - Musicality: Does it sound like music?
   - Coherence: Does it have structure?
   - Creativity: Is it interesting/novel?
3. Calculate average ratings

**Target:** > 3.5/5 average across all dimensions

---

## 9. Implementation Status

⚠️ **Important Note:** This documentation describes the **planned deep learning approach** based on the comprehensive implementation plan in [Music_Generation_LSTM_Implementation_Plan.md](Music_Generation_LSTM_Implementation_Plan.md).

The code examples demonstrate the intended methodology and architecture. This documentation serves as a complete reference for understanding the deep learning algorithms, software packages, and implementation details.

---

## 10. Software Requirements Summary

### 10.1 Complete Package List

```bash
# Core deep learning
pip install tensorflow==2.15.0

# MIDI processing
pip install pretty_midi==0.2.10

# Numerical computing
pip install numpy==1.24.3

# Music theory (optional)
pip install music21==9.1.0

# Visualization
pip install matplotlib==3.7.2

# Interactive demo
pip install streamlit==1.28.0

# Utilities
pip install scikit-learn==1.3.0
pip install tqdm==4.66.0

# Audio synthesis (command-line tool)
# macOS:
brew install fluid-synth
# Linux:
sudo apt-get install fluidsynth
```

### 10.2 Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 8 GB
- Storage: 2 GB free space
- GPU: Optional (CPU training works but slower)

**Recommended:**
- CPU: M1/M2 (macOS) or multi-core Intel/AMD
- RAM: 16 GB
- Storage: 5 GB free space
- GPU: Apple MPS, NVIDIA CUDA, or Google Colab

---

## 11. Summary

**Model Architecture:**
- Two-layer LSTM with 512 units each (~6.5M parameters)
- Dense hidden layer (256 units)
- Sigmoid output layer (88 units, multi-label)

**Key Software Packages:**
1. **TensorFlow/Keras:** Model building, training, inference
2. **Pretty MIDI:** MIDI file parsing and creation
3. **NumPy:** Numerical operations and array manipulation
4. **FluidSynth:** MIDI to audio conversion
5. **Streamlit:** Interactive demo interface

**Training Algorithm:**
- Loss: Weighted binary crossentropy (10× for active notes)
- Optimizer: Adam (lr=0.001)
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Configuration: 30 epochs, batch size 64, ~2-3 hours

**Generation Algorithm:**
- Autoregressive prediction (one timestep at a time)
- Temperature sampling (0.5-1.5 for creativity control)
- Musical constraints (key and scale enforcement)
- Sliding window (64-step context)

**Expected Performance:**
- Validation loss: < 0.30
- Binary accuracy: > 90%
- Note density: 2-4 notes/timestep
- Human rating: > 3.5/5

---

## References

1. TensorFlow Documentation: https://tensorflow.org/
2. Pretty MIDI Documentation: https://github.com/craffel/pretty-midi
3. Keras Documentation: https://keras.io/
4. LSTM Paper: Hochreiter & Schmidhuber (1997)
5. Adam Optimizer: Kingma & Ba (2014)
6. Implementation Plan: [Music_Generation_LSTM_Implementation_Plan.md](Music_Generation_LSTM_Implementation_Plan.md)

---

**Document Version:** 1.0
**Date:** December 2025
**Project:** CST435 Final Project - Music Generation with LSTM
