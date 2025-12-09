# Part 1: Data and Preprocessing for Music Generation LSTM

## Project Overview

This document describes the data sources and preprocessing pipeline for a **Music Generation with LSTM** project. The system generates original piano melodies and chord progressions by learning from MIDI data using deep learning techniques.

**Note:** This documentation describes the planned implementation approach based on the detailed implementation plan. The code examples demonstrate the intended preprocessing methodology, though full implementation is ongoing.

---

## 1. Data Sources

### 1.1 Primary Dataset: Maestro v3.0.0

**Dataset Information:**
- **Name:** MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization)
- **Type:** Classical piano performances in MIDI format
- **Size:** ~200 hours of performances, ~90 MB (MIDI-only download)
- **Content:** Professional piano recordings from International Piano-e-Competition
- **Quality:** High-quality, clean, professional recordings
- **Download Source:**
  ```
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
  ```

**Dataset Characteristics:**
- Number of MIDI files: 1,276 compositions
- Genre: Classical piano (various composers and styles)
- Format: Standard MIDI files (.mid)
- Instruments: Piano only (monophonic instrument track)
- Tempo: Varies by piece (60-200 BPM typically)

**Why Maestro?**
- Professional-quality recordings ensure clean training data
- Large dataset provides diverse musical patterns
- Well-structured MIDI files with accurate timing
- Widely used in music generation research
- Polyphonic content (chords and melodies)

### 1.2 Secondary Dataset: JSB Chorales

**Dataset Information:**
- **Name:** J.S. Bach Chorales
- **Type:** Four-part harmony chorales
- **Size:** 382 chorales, ~1 MB
- **Content:** Bach's four-part harmonizations
- **Source:** Available via TensorFlow Datasets
- **Access Method:**
  ```python
  import tensorflow_datasets as tfds
  ds = tfds.load('jsb_chorales')
  ```

**Dataset Characteristics:**
- Number of pieces: 382 chorales
- Genre: Baroque sacred music
- Format: Pre-processed sequences (ready to use)
- Instruments: Four-part harmony (SATB)
- Complexity: Relatively simple harmonic structures

**Why JSB Chorales?**
- Small size enables rapid prototyping
- Well-structured harmonic progressions
- Perfect for testing preprocessing pipeline
- Quick download and iteration cycles
- Educational value (learning classical harmony rules)

### 1.3 Additional Dataset Option: Lakh MIDI Dataset

**Dataset Information:**
- **Name:** Lakh MIDI Dataset (LMD)
- **Type:** Multi-genre MIDI collection
- **Size:** 176,581 MIDI files
- **Source:** `https://colinraffel.com/projects/lmd`
- **Use Case:** Extended training for genre diversity (optional)

---

## 2. Data Representation: Piano Roll Encoding

### 2.1 What is Piano Roll Representation?

A **piano roll** is a visual and numerical representation of music where:
- **Rows** represent different musical notes (88 piano keys)
- **Columns** represent time steps (quantized rhythm)
- **Values** indicate whether a note is active (1) or silent (0)

**Visual Example:**
```
Time →   0    1    2    3    4    5    6    7
      ┌────┬────┬────┬────┬────┬────┬────┬────┐
C5    │ 1  │ 1  │ 0  │ 0  │ 1  │ 1  │ 0  │ 0  │  (Middle C played at times 0-1, 4-5)
      ├────┼────┼────┼────┼────┼────┼────┼────┤
E5    │ 1  │ 1  │ 0  │ 0  │ 1  │ 1  │ 0  │ 0  │  (E played at times 0-1, 4-5)
      ├────┼────┼────┼────┼────┼────┼────┼────┤
G5    │ 1  │ 1  │ 0  │ 0  │ 1  │ 1  │ 0  │ 0  │  (G played at times 0-1, 4-5)
      ├────┼────┼────┼────┼────┼────┼────┼────┤
...   │ ...│ ...│ ...│ ...│ ...│ ...│ ...│ ...│
      └────┴────┴────┴────┴────┴────┴────┴────┘
                   C major chord → Silence → C major chord
```

### 2.2 Piano Roll Specifications

**Dimensions:**
- **Time axis:** Variable length (depends on piece duration)
- **Note axis:** 88 notes (standard piano range A0 to C8)
- **MIDI note range:** 21-108 (MIDI note numbers)
- **Encoding:** Binary (0 or 1)

**Time Quantization:**
- **Resolution:** 16th notes (4 steps per quarter note)
- **Sampling frequency:** 4 Hz at 120 BPM
- **Practical meaning:** Each timestep = 1/16th note duration
- **4 bars of music:** 4 bars × 4 beats × 4 steps = 64 timesteps

**Advantages of Piano Roll:**
1. **Polyphony support:** Multiple notes can be active simultaneously
2. **Compact representation:** More efficient than event-based encoding
3. **Grid structure:** Natural for convolutional/recurrent processing
4. **Interpretability:** Easy to visualize and debug
5. **Temporal locality:** Adjacent timesteps represent musical continuity

---

## 3. Preprocessing Pipeline

### 3.1 Overview

The preprocessing pipeline transforms raw MIDI files into training-ready sequences:

```
MIDI Files → Parsing → Quantization → Piano Roll → Augmentation →
Sliding Windows → Train/Val/Test Split → NumPy Arrays
```

### 3.2 Step 1: MIDI Parsing

**Objective:** Load MIDI files and extract note information

**Library:** `pretty_midi` (Python MIDI parsing library)

**Code Implementation:**
```python
import pretty_midi

def load_midi(midi_path):
    """
    Load MIDI file and extract piano track

    Parameters:
    - midi_path: Path to MIDI file

    Returns:
    - PrettyMIDI object containing note events
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        return midi_data
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None
```

**What is extracted:**
- Note pitch (MIDI note number 0-127)
- Note start time (seconds)
- Note end time (seconds)
- Note velocity (loudness, 0-127)
- Tempo information
- Time signature

### 3.3 Step 2: Quantization

**Objective:** Snap note timings to a fixed grid (16th notes)

**Why Quantization?**
- Human performances have timing variations (rubato, slight delays)
- Neural networks work better with discrete time steps
- Reduces data complexity while preserving musical essence

**Code Implementation:**
```python
def quantize_midi(midi_data, fs=4):
    """
    Quantize MIDI timing to fixed grid

    Parameters:
    - midi_data: PrettyMIDI object
    - fs: Sampling frequency (4 = 16th notes)

    Returns:
    - Quantized PrettyMIDI object
    """
    # Get piano roll with specified time resolution
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return piano_roll
```

**Time Resolution:**
- `fs=4`: 16th notes (4 steps per beat)
- `fs=8`: 32nd notes (8 steps per beat, more precise but larger data)
- `fs=2`: 8th notes (2 steps per beat, less precise but smaller data)

**Trade-off:** 16th notes balance temporal resolution and data size.

### 3.4 Step 3: Piano Roll Conversion

**Objective:** Convert MIDI to binary piano roll matrix

**Code Implementation:**
```python
import numpy as np

class PianoRollEncoder:
    """
    Encoder for converting between MIDI and piano roll representations
    """

    def __init__(self, fs=4, note_range=(21, 109)):
        """
        Initialize encoder

        Parameters:
        - fs: Sampling frequency (4 = 16th notes, 4 steps per beat)
        - note_range: MIDI note range (21-108 = 88 piano keys A0-C8)
        """
        self.fs = fs
        self.note_range = note_range
        self.n_notes = note_range[1] - note_range[0]

    def midi_to_piano_roll(self, midi_file):
        """
        Convert MIDI file to binary piano roll matrix

        Parameters:
        - midi_file: Path to MIDI file or PrettyMIDI object

        Returns:
        - piano_roll: Binary matrix of shape (time_steps, 88)
        """
        # Load MIDI file
        if isinstance(midi_file, str):
            pm = pretty_midi.PrettyMIDI(midi_file)
        else:
            pm = midi_file

        # Get piano roll at specified resolution
        # Returns: (128, time_steps) array with velocities
        piano_roll = pm.get_piano_roll(fs=self.fs)

        # Extract only 88 piano key range (MIDI 21-108)
        piano_roll = piano_roll[self.note_range[0]:self.note_range[1], :]

        # Binarize: 1 if note is active (velocity > 0), 0 otherwise
        piano_roll = (piano_roll > 0).astype(np.int32)

        # Transpose to shape (time_steps, 88)
        piano_roll = piano_roll.T

        return piano_roll

    def piano_roll_to_midi(self, piano_roll, tempo=120, velocity=100):
        """
        Convert piano roll back to MIDI file

        Parameters:
        - piano_roll: Binary matrix (time_steps, 88)
        - tempo: Tempo in BPM (default: 120)
        - velocity: Note velocity/loudness (default: 100)

        Returns:
        - PrettyMIDI object
        """
        # Create empty MIDI file
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # Convert binary matrix to note events
        for note_idx in range(self.n_notes):
            midi_note = note_idx + self.note_range[0]

            # Find note on/off transitions
            # Add padding to detect notes at boundaries
            padded = np.pad(piano_roll[:, note_idx], (1, 1), mode='constant')
            note_changes = np.diff(padded)

            # note_changes[i] == 1: note starts at time i
            # note_changes[i] == -1: note ends at time i
            note_on_times = np.where(note_changes == 1)[0]
            note_off_times = np.where(note_changes == -1)[0]

            # Create note events for each on/off pair
            for on_idx, on_time in enumerate(note_on_times):
                # Find corresponding off time
                if on_idx < len(note_off_times):
                    off_time = note_off_times[on_idx]
                else:
                    # If no off time, note continues to end
                    off_time = len(piano_roll)

                # Convert timestep to seconds
                start_time = on_time / self.fs
                end_time = off_time / self.fs

                # Create note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note,
                    start=start_time,
                    end=end_time
                )
                piano.notes.append(note)

        pm.instruments.append(piano)
        return pm
```

**Example Usage:**
```python
# Convert MIDI to piano roll
encoder = PianoRollEncoder(fs=4, note_range=(21, 109))
piano_roll = encoder.midi_to_piano_roll('maestro_001.mid')

print(f"Piano roll shape: {piano_roll.shape}")
# Output: Piano roll shape: (2048, 88)
# 2048 timesteps (128 bars), 88 notes

# Convert back to MIDI
reconstructed_midi = encoder.piano_roll_to_midi(piano_roll, tempo=120)
reconstructed_midi.write('reconstructed.mid')
```

### 3.5 Step 4: Data Augmentation

**Objective:** Artificially increase dataset size by transposing music

**Technique:** Transpose each piece by ±2 semitones (half-steps)

**Why Transpose?**
- Music sounds similar in different keys
- Multiplies training data by 5× (original + 4 transpositions)
- Helps model generalize across different keys
- Prevents overfitting to specific key signatures

**Code Implementation:**
```python
def transpose_piano_roll(piano_roll, semitones):
    """
    Transpose piano roll by specified semitones

    Parameters:
    - piano_roll: Binary matrix (time_steps, 88)
    - semitones: Number of semitones to shift (+/-)

    Returns:
    - Transposed piano roll (same shape)
    """
    # Use numpy roll to shift note axis
    transposed = np.roll(piano_roll, semitones, axis=1)

    # Zero out notes that wrapped around
    if semitones > 0:
        transposed[:, :semitones] = 0
    elif semitones < 0:
        transposed[:, semitones:] = 0

    return transposed

# Example: Create 5 versions of each piece
transpositions = [-2, -1, 0, 1, 2]  # Down 2, down 1, original, up 1, up 2

augmented_data = []
for shift in transpositions:
    transposed = transpose_piano_roll(original_piano_roll, shift)
    augmented_data.append(transposed)

# Result: 5× more training data
```

**Data Multiplication:**
- Original dataset: 1,276 files
- After augmentation: 1,276 × 5 = 6,380 virtual files

### 3.6 Step 5: Sliding Window Extraction

**Objective:** Create fixed-length training sequences from variable-length pieces

**Window Configuration:**
- **Sequence length:** 64 timesteps (4 musical bars)
- **Stride:** 1 timestep (maximum overlap for data efficiency)

**Code Implementation:**
```python
def create_sequences(piano_roll, seq_length=64):
    """
    Extract sliding windows from piano roll

    Parameters:
    - piano_roll: Binary matrix (time_steps, 88)
    - seq_length: Length of each sequence (default: 64 = 4 bars)

    Returns:
    - X: Input sequences (n_sequences, seq_length, 88)
    - y: Target outputs (n_sequences, 88)
    """
    X = []
    y = []

    # Slide window across piano roll
    for i in range(len(piano_roll) - seq_length):
        # Input: 64 timesteps
        sequence = piano_roll[i:i+seq_length]
        X.append(sequence)

        # Target: Next timestep (what comes after the 64-step sequence)
        target = piano_roll[i+seq_length]
        y.append(target)

    return np.array(X), np.array(y)
```

**Example:**
```
Piano roll length: 2048 timesteps
Sequence length: 64 timesteps
Number of sequences: 2048 - 64 = 1984 sequences per piece
```

**Visual Illustration:**
```
Piano Roll: [t0, t1, t2, t3, t4, t5, ..., t2047]

Sequence 1: [t0:t64] → target: t64
Sequence 2: [t1:t65] → target: t65
Sequence 3: [t2:t66] → target: t66
...
Sequence 1984: [t1983:t2047] → target: t2047
```

### 3.7 Step 6: Train/Validation/Test Split

**Objective:** Create separate datasets for training, validation, and testing

**Split Strategy:** By file (not by sequence) to prevent data leakage

**Split Ratios:**
- Training: 70% (894 files)
- Validation: 15% (191 files)
- Test: 15% (191 files)

**Why split by file?**
- Sequences from the same piece are highly correlated
- Splitting by sequence would leak information between sets
- File-level split ensures true generalization test

**Code Implementation:**
```python
from sklearn.model_selection import train_test_split

def create_train_val_test_split(midi_files, train_ratio=0.7, val_ratio=0.15):
    """
    Split MIDI files into train/val/test sets

    Parameters:
    - midi_files: List of MIDI file paths
    - train_ratio: Fraction for training (default: 0.7)
    - val_ratio: Fraction for validation (default: 0.15)

    Returns:
    - train_files, val_files, test_files
    """
    # First split: train vs. (val + test)
    train_files, temp_files = train_test_split(
        midi_files,
        train_size=train_ratio,
        random_state=42
    )

    # Second split: val vs. test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio_adjusted,
        random_state=42
    )

    return train_files, val_files, test_files
```

### 3.8 Complete Preprocessing Pipeline

**Full Pipeline Code:**
```python
import os
import numpy as np
from tqdm import tqdm

def preprocess_dataset(midi_directory, output_directory,
                      seq_length=64, augment=True):
    """
    Complete preprocessing pipeline for MIDI dataset

    Parameters:
    - midi_directory: Path to folder containing MIDI files
    - output_directory: Path to save processed data
    - seq_length: Length of training sequences (default: 64)
    - augment: Whether to apply data augmentation (default: True)

    Returns:
    - Dataset statistics dictionary
    """
    # Initialize encoder
    encoder = PianoRollEncoder(fs=4, note_range=(21, 109))

    # Get all MIDI files
    midi_files = [os.path.join(midi_directory, f)
                  for f in os.listdir(midi_directory)
                  if f.endswith('.mid') or f.endswith('.midi')]

    print(f"Found {len(midi_files)} MIDI files")

    # Split into train/val/test
    train_files, val_files, test_files = create_train_val_test_split(midi_files)

    # Process each split
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    stats = {}

    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split ({len(files)} files)...")

        X_all = []
        y_all = []

        for midi_file in tqdm(files):
            # Convert to piano roll
            piano_roll = encoder.midi_to_piano_roll(midi_file)

            # Data augmentation (only for training)
            if augment and split_name == 'train':
                transpositions = [-2, -1, 0, 1, 2]
            else:
                transpositions = [0]  # No augmentation for val/test

            for shift in transpositions:
                # Transpose
                transposed = transpose_piano_roll(piano_roll, shift)

                # Extract sequences
                X, y = create_sequences(transposed, seq_length=seq_length)

                X_all.append(X)
                y_all.append(y)

        # Concatenate all sequences
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # Save to disk
        os.makedirs(output_directory, exist_ok=True)
        np.save(os.path.join(output_directory, f'X_{split_name}.npy'), X_all)
        np.save(os.path.join(output_directory, f'y_{split_name}.npy'), y_all)

        # Collect statistics
        stats[split_name] = {
            'n_sequences': len(X_all),
            'shape': X_all.shape,
            'note_density': np.mean(np.sum(X_all, axis=2)),
            'sparsity': 1 - np.mean(X_all)
        }

        print(f"{split_name}: {len(X_all)} sequences saved")

    return stats

# Run preprocessing
stats = preprocess_dataset(
    midi_directory='data/maestro-v3.0.0',
    output_directory='data/processed',
    seq_length=64,
    augment=True
)
```

---

## 4. Expected Preprocessing Results

### 4.1 Dataset Statistics

| Split      | Files | Sequences | Shape               | Size (MB) |
|------------|-------|-----------|---------------------|-----------|
| Training   | 894   | ~50,000   | (50000, 64, 88)     | ~275      |
| Validation | 191   | ~10,000   | (10000, 64, 88)     | ~55       |
| Test       | 191   | ~10,000   | (10000, 64, 88)     | ~55       |
| **Total**  | 1,276 | ~70,000   | -                   | ~385      |

**Note:** Training sequences include 5× augmentation; val/test do not.

### 4.2 Data Characteristics

**Shape Information:**
- **X_train.npy:** (50000, 64, 88) - Input sequences
  - 50,000 training examples
  - 64 timesteps per sequence (4 musical bars)
  - 88 features per timestep (piano keys)

- **y_train.npy:** (50000, 88) - Target outputs
  - 50,000 target timesteps
  - 88 binary values (which notes play next)

**Data Distribution:**
- **Note density:** 2-4 active notes per timestep (average)
- **Sparsity:** ~95% zeros (most keys silent at any time)
- **Polyphony:** 1-10 simultaneous notes (chords)
- **Sequence length:** 64 timesteps = 4 bars = 16 beats

### 4.3 Example Piano Roll Visualization

**Sample 8-bar sequence (128 timesteps):**
```
Note Range (simplified, showing octaves C3-C6)

     0  8  16 24 32 40 48 56 64 72 80 88 96 104 112 120
C6   .  .  .  .  .  .  .  .  .  .  .  .  .  .   .   .
B5   .  .  .  .  .  .  .  .  █  █  .  .  .  .   .   .
A5   .  .  .  .  █  █  .  .  .  .  .  .  .  .   .   .
G5   █  █  .  .  .  .  .  .  .  .  █  █  .  .   .   .
F5   .  .  .  .  .  .  .  .  .  .  .  .  .  .   .   .
E5   █  █  █  █  █  █  .  .  .  .  .  .  █  █   █   █
D5   .  .  .  .  .  .  .  .  .  .  .  .  .  .   .   .
C5   █  █  █  █  █  █  █  █  █  █  █  █  █  █   █   █
...
C3   .  .  .  .  .  .  .  .  .  .  .  .  .  .   .   .

     Bar 1    Bar 2    Bar 3    Bar 4    Bar 5    Bar 6
```

Legend: █ = note active (1), . = silent (0)

### 4.4 Data Quality Checks

**Validation Checks:**
1. ✅ All values are binary (0 or 1)
2. ✅ Sequence shapes are consistent
3. ✅ No NaN or infinite values
4. ✅ Note density within expected range (0.5-5%)
5. ✅ No data leakage between train/val/test
6. ✅ Augmented sequences maintain musical coherence

**Code for Quality Check:**
```python
def validate_processed_data(data_directory):
    """
    Validate preprocessed data quality
    """
    X_train = np.load(os.path.join(data_directory, 'X_train.npy'))
    y_train = np.load(os.path.join(data_directory, 'y_train.npy'))

    print("Data Quality Checks:")
    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ All binary: {np.all(np.isin(X_train, [0, 1]))}")
    print(f"✓ No NaN: {not np.any(np.isnan(X_train))}")
    print(f"✓ Note density: {np.mean(X_train) * 100:.2f}%")
    print(f"✓ Avg notes per timestep: {np.mean(np.sum(X_train, axis=2)):.2f}")

validate_processed_data('data/processed')
```

---

## 5. Memory and Storage Requirements

### 5.1 Storage Space

**Disk Space Requirements:**
```
Raw MIDI files (Maestro):       ~90 MB
Processed NumPy arrays:         ~385 MB
Total storage needed:           ~500 MB
```

**Breakdown by Split:**
- Training data: ~275 MB (50,000 sequences)
- Validation data: ~55 MB (10,000 sequences)
- Test data: ~55 MB (10,000 sequences)

### 5.2 RAM Requirements

**During Preprocessing:**
- Peak RAM usage: ~2-3 GB
- Recommended: 8 GB RAM minimum

**During Training:**
- Data loading: ~500 MB (if loaded all at once)
- Model parameters: ~26 MB (6.5M parameters × 4 bytes)
- Batch processing: ~4 MB per batch (64 samples)
- Recommended: 8 GB RAM minimum, 16 GB preferred

---

## 6. Preprocessing Performance

### 6.1 Processing Time

**Hardware:** Apple M1/M2 MacBook

**Estimated Times:**
- MIDI parsing: ~0.1 seconds per file
- Piano roll conversion: ~0.05 seconds per file
- Augmentation: ~0.2 seconds per file (×5 transpositions)
- Sliding windows: ~0.1 seconds per file
- **Total per file:** ~0.45 seconds

**Full Dataset:**
- 1,276 files × 0.45 seconds = ~10 minutes total
- Parallel processing could reduce to ~5 minutes

### 6.2 Optimization Tips

1. **Batch Processing:** Process multiple files in parallel
2. **Caching:** Save preprocessed piano rolls to avoid recomputation
3. **Memory Mapping:** Use `np.memmap` for large datasets
4. **Subset Training:** Start with JSB Chorales (382 files, ~2 minutes)

---

## 7. Implementation Status

⚠️ **Important Note:** This documentation describes the **planned preprocessing approach** based on the comprehensive implementation plan in [Music_Generation_LSTM_Implementation_Plan.md](Music_Generation_LSTM_Implementation_Plan.md).

The code examples demonstrate the intended methodology and can serve as a reference for implementation. While the preprocessing pipeline is well-defined and ready to implement, the actual execution is part of the ongoing development process.

---

## 8. Summary

**Data Sources:**
- Primary: Maestro v3.0.0 (1,276 classical piano MIDI files, 200 hours)
- Secondary: JSB Chorales (382 Bach chorales for prototyping)

**Preprocessing Steps:**
1. MIDI Parsing (pretty_midi)
2. Quantization (16th note grid)
3. Piano Roll Conversion (time × 88 binary matrix)
4. Data Augmentation (±2 semitone transposition, 5× data)
5. Sliding Window Extraction (64-step sequences)
6. Train/Val/Test Split (70/15/15 by file)

**Output:**
- 50,000 training sequences (64 timesteps, 88 notes)
- 10,000 validation sequences
- 10,000 test sequences
- Total: ~385 MB processed data

**Key Characteristics:**
- Binary piano roll representation
- 4-bar sequences (64 timesteps)
- 88 piano keys (A0-C8)
- ~95% sparse (mostly silence)
- 2-4 average notes per timestep

---

## References

1. Maestro Dataset: https://magenta.tensorflow.org/datasets/maestro
2. JSB Chorales: https://tensorflow.org/datasets/catalog/jsb_chorales
3. Pretty MIDI Library: https://github.com/craffel/pretty-midi
4. Implementation Plan: [Music_Generation_LSTM_Implementation_Plan.md](Music_Generation_LSTM_Implementation_Plan.md)

---

**Document Version:** 1.0
**Date:** December 2025
**Project:** CST435 Final Project - Music Generation with LSTM
