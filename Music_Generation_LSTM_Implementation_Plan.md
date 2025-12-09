# Music Generation with LSTM - Detailed Implementation Plan

## Executive Summary

You've chosen to build a **Music Generation with LSTM** system that generates original piano melodies and chord progressions from MIDI data. This project builds directly on your WikiText-2 LSTM experience while introducing music modeling concepts.

**Timeline:** 3-4 weeks
**Framework:** TensorFlow/Keras (consistent with your text LSTM)
**Key Learning:** Music representation, polyphonic modeling, MIDI processing
**Deliverable:** Interactive Streamlit demo with audio playback

---

## Your Background (Relevant Skills)
Based on your work in `/Users/rix/Documents/School/Github/CST435/RNN`:
- ✅ **LSTM sequence modeling** - WikiText-2 text generation with 256 LSTM units
- ✅ **Temperature sampling & top-k decoding** - Already implemented in streamlit_app.py
- ✅ **Streamlit deployment** - Interactive demo with controls and visualization
- ✅ **Data preprocessing pipelines** - Tokenization, windowing, train/val splits
- ✅ **Production best practices** - Model checkpointing, early stopping, modular code

---

## Quick Overview: What You'll Build

### System Components
1. **MIDI Preprocessing Pipeline** - Convert MIDI files → piano roll representation (time × 88 keys)
2. **Piano Roll LSTM Model** - Generate polyphonic music with binary crossentropy loss
3. **Music Generator** - Temperature sampling with musical constraints (key, rhythm)
4. **Streamlit Demo** - Interactive UI with audio playback, piano roll visualization, MIDI download

### Key Differences from Text LSTM
- **Multi-hot output** instead of single token (multiple notes can play simultaneously)
- **No pre-trained embeddings** (learn music-specific representations from scratch)
- **Temporal constraints** (music has rhythm, harmony rules unlike free-form text)
- **Evaluation** (subjective listening tests + objective metrics like note density)

---

## Architecture Design

### MIDI Encoding Strategy: Piano Roll (Recommended)

**Why Piano Roll?**
- Natural polyphony support (multiple simultaneous notes)
- Compact sequences (128 steps = 8 bars vs. 5000+ tokens for event-based)
- Similar to your WikiText sliding windows concept

**Representation:**
```
Input shape: (batch, sequence_length=64, n_notes=88)
Output shape: (batch, 88)  # Multi-hot: which notes to play next

Time quantization: 16th notes (4 steps per beat)
Note range: 88 piano keys (A0 to C8, MIDI 21-108)
Encoding: Binary matrix where 1 = note active, 0 = silent
```

**Example:**
```
Time step 0: [0, 0, 1, 0, ..., 1]  # C4 and E4 playing (chord)
Time step 1: [0, 0, 1, 0, ..., 1]  # Same chord continues
Time step 2: [0, 0, 0, 0, ..., 0]  # Silence
Time step 3: [0, 0, 0, 1, ..., 0]  # New note D4
```

### Model Architecture

Building on your WikiText LSTM pattern:

```python
model = keras.Sequential([
    # Input: (batch, 64, 88) - last 64 time steps, 88 notes each
    layers.LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
    layers.LSTM(512, dropout=0.3, recurrent_dropout=0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(88, activation='sigmoid')  # Multi-hot: each note independently
])

# Loss: binary_crossentropy (treat each of 88 notes as binary classification)
# Metrics: binary_accuracy, note_density
```

**Key differences from your text model:**
- ✏️ **Larger LSTM units** (512 vs 256) - music has richer patterns
- ✏️ **Two LSTM layers** - capture short-term (notes) + long-term (phrases)
- ✏️ **Sigmoid output** (not softmax) - multiple notes can be active
- ✏️ **Binary crossentropy loss** (not categorical) - independent predictions per note

---

## Data Pipeline

### Recommended Dataset: **Maestro** (Primary) + **JSB Chorales** (Testing)

**Maestro Dataset:**
- 200 hours of classical piano performances
- Clean, professional recordings
- Download MIDI-only: ~90 MB
- Source: `wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip`

**JSB Chorales (for Week 1 prototyping):**
- 382 Bach chorales (~1 MB)
- Perfect for testing pipeline quickly
- Available in TensorFlow: `tfds.load('jsb_chorales')`

### Preprocessing Steps

**Similar to your WikiText pipeline but adapted for music:**

1. **MIDI Parsing** → Use `pretty_midi` library
2. **Quantization** → Snap notes to 16th note grid
3. **Piano Roll Conversion** → Create (time, 88) binary matrix
4. **Augmentation** → Transpose ±2 semitones (5x more data!)
5. **Sliding Windows** → Extract 64-step sequences
6. **Train/Val/Test Split** → 70/15/15 by file (prevent leakage)

**Expected output:**
- `X_train.npy`: Shape (50000, 64, 88) - 50K training sequences
- `y_train.npy`: Shape (50000, 88) - Next time step labels

---

## Training Strategy

### Loss Function & Optimizer

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Multi-label (polyphonic)
    metrics=['binary_accuracy']
)
```

**Weighted Loss for Sparse Data:**
Piano rolls are mostly silence (sparse). Weight active notes 10x higher:

```python
# Custom loss to handle class imbalance
def weighted_bce(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = 1.0 + 9.0 * y_true  # Active notes: 10x weight
    return tf.reduce_mean(bce * weights)
```

### Callbacks (Reuse Your Pattern)

```python
callbacks = [
    ModelCheckpoint('checkpoints/best_model.keras',
                   monitor='val_loss', save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3)
]
```

### Training Configuration

- **Epochs:** 20-30 (with early stopping)
- **Batch size:** 64 (adjust for Apple MPS memory)
- **Sequence length:** 64 steps (4 bars at 16th notes)
- **Training time:** ~2-3 hours on MPS for 50K examples

---

## Generation Techniques

### Temperature Sampling (Same as Your Text Model!)

```python
def sample_with_temperature(probs, temperature=0.8):
    """Higher temp = more creative, lower = conservative"""
    probs = np.log(probs + 1e-10) / temperature
    probs = np.exp(probs - np.max(probs))
    probs = probs / np.sum(probs)
    return probs
```

**Recommended temperatures:**
- 0.5: Safe, coherent (classical style)
- 0.8: Balanced (default)
- 1.2: Creative, unexpected
- 1.5+: Experimental/chaotic

### Musical Constraints (New!)

**1. Key Constraint - Keep Generated Music in Key**

```python
def constrain_to_key(probs, key='C', scale='major'):
    """Only allow notes in the specified key"""
    C_major_notes = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
    mask = np.zeros_like(probs)
    for i in range(88):
        if (i + 21) % 12 in C_major_notes:  # MIDI 21 = A0
            mask[i] = 1.0
    probs = probs * mask
    return probs / probs.sum()
```

**2. Repetition Penalty - Avoid Loops**

```python
def penalize_repetition(probs, recent_notes, penalty=0.5):
    """Reduce probability of recently played notes"""
    for note_idx in recent_notes:
        probs[note_idx] *= penalty
    return probs / probs.sum()
```

---

## Evaluation Metrics

### Objective Metrics

1. **Validation Loss** - Should be < 0.3 for good generation
2. **Note Density** - Average notes per time step (target: 2-4 for piano)
3. **Pitch Class Histogram** - Compare generated vs. real note distribution
4. **Rhythmic Regularity** - Autocorrelation at 4-beat intervals

### Subjective Metrics

**Human Listening Test:**
- Generate 20 pieces (16 bars each)
- Ask 5+ people to rate (1-5):
  - Musicality: Does it sound like music?
  - Coherence: Does it have structure?
  - Creativity: Is it interesting?
- **Target:** Average rating > 3.5/5

---

## Streamlit Demo Design

### UI Layout (Extends Your Existing Pattern)

**File:** `music_generation/demo/streamlit_app.py`

```python
import streamlit as st

st.title("🎹 LSTM Music Generator")

# Sidebar controls (like your WikiText app)
temperature = st.sidebar.slider("Temperature", 0.3, 1.5, 0.8)
length = st.sidebar.slider("Length (bars)", 4, 32, 16)
key = st.sidebar.selectbox("Key", ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
scale = st.sidebar.selectbox("Scale", ['major', 'minor', 'pentatonic'])

# Generate button
if st.button("Generate Music"):
    with st.spinner("Composing..."):
        piano_roll = generator.generate(length*16, temperature, key, scale)
        midi_bytes = convert_to_midi(piano_roll)
        audio_bytes = convert_to_audio(midi_bytes)

        # Audio player
        st.audio(audio_bytes, format='audio/wav')

        # Downloads
        st.download_button("Download MIDI", midi_bytes, "generated.mid")

        # Visualization
        st.pyplot(plot_piano_roll(piano_roll))
```

### Audio Playback

**MIDI → Audio Conversion using FluidSynth:**

```bash
# Install
brew install fluid-synth

# Download soundfont
wget https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip
```

```python
def midi_to_audio(midi_bytes):
    """Convert MIDI to WAV using FluidSynth"""
    subprocess.run([
        'fluidsynth', '-ni', 'soundfonts/FluidR3_GM.sf2',
        'temp.mid', '-F', 'output.wav', '-r', '44100'
    ])
    return open('output.wav', 'rb').read()
```

---

## Implementation Timeline (3-4 Weeks)

### Week 1: Foundation (Monophonic Prototype)
**Goals:** Working melody generator on small dataset

- **Day 1-2:** Setup + data pipeline
  - Install: `pretty_midi`, `tensorflow`, `fluidsynth`
  - Download JSB Chorales (small dataset for testing)
  - Implement `MIDIEventEncoder` for simple monophonic encoding

- **Day 3-4:** Basic LSTM model
  - Build event-based LSTM (similar to your WikiText model)
  - Train on JSB Chorales for 10 epochs
  - Generate first 16-bar melody

- **Day 5-7:** Streamlit demo v1
  - Basic UI with temperature slider
  - MIDI download + audio playback
  - **Deliverable:** Working monophonic melody generator

### Week 2: Polyphonic Generation
**Goals:** Piano roll LSTM generating chords

- **Day 8-9:** Piano roll preprocessing
  - Implement `PianoRollEncoder`
  - Download Maestro dataset (~10 GB, use subset of 100 files)
  - Process to piano roll with augmentation

- **Day 10-12:** Piano roll LSTM
  - Build multi-hot output model
  - Train for 20 epochs with early stopping
  - Generate polyphonic pieces

- **Day 13-14:** Advanced sampling
  - Implement key constraints, top-k, nucleus sampling
  - Compare sampling strategies
  - **Deliverable:** Coherent 32-bar polyphonic generations

### Week 3: Refinement
**Goals:** Quality improvements + evaluation

- **Day 15-16:** Evaluation suite
  - Implement metrics (note density, PCH, rhythmic regularity)
  - Run listening tests with friends/classmates

- **Day 17-18:** Seed-based continuation
  - Allow users to upload seed melody
  - Generate continuation from seed

- **Day 19-21:** UI polish
  - Add piano roll visualization (Matplotlib)
  - Preset examples ("Classical", "Jazz", etc.)
  - Demo video
  - **Deliverable:** Production Streamlit app

### Week 4: Extensions (Optional)
**Goals:** Advanced features for portfolio

Pick 1-2:
- **Multi-track generation** (piano + bass)
- **Style conditioning** (generate "in style of Bach")
- **Attention mechanism** for long-range structure
- **Real-time generation** (WebSocket streaming)
- **Documentation:** Technical report, README, blog post

---

## Critical Files to Create

### File Structure
```
music_generation/
├── preprocessing/
│   ├── midi_encoder.py          # ⭐ Core: MIDI ↔ piano roll conversion
│   └── data_pipeline.py         # ⭐ Preprocessing pipeline
├── models/
│   ├── piano_roll_lstm.py       # ⭐ Main model architecture
│   └── generator.py             # ⭐ Inference engine
├── demo/
│   └── streamlit_app.py         # ⭐ Interactive demo
├── training/
│   └── train.py                 # Training script
├── evaluation/
│   └── metrics.py               # Evaluation metrics
└── data/
    ├── raw/                     # Original MIDI files
    └── processed/               # Preprocessed arrays
```

### Priority Order (Start Here)

1. **preprocessing/midi_encoder.py** - Foundation for all data processing
2. **preprocessing/data_pipeline.py** - Automate dataset creation
3. **models/piano_roll_lstm.py** - Core generative model
4. **models/generator.py** - Sampling & constraints
5. **demo/streamlit_app.py** - User-facing interface

---

## Technical Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Mode collapse** (repeating loops) | Increase dropout to 0.4-0.5, add repetition penalty |
| **Chromatic chaos** (random notes) | Add key constraint, train on single-key subset first |
| **Sparse predictions** (all silence) | Use weighted loss (10x for active notes) |
| **Rhythmic drift** (loses beat) | Strict quantization, enforce grid during generation |
| **Out of memory** | Reduce batch size (64→32), shorter sequences (64→48) |

---

## Success Metrics

### Week 1
- ✅ Generate coherent 8-bar monophonic melody
- ✅ Validation loss < 1.0 (event-based model)

### Week 2
- ✅ Generate 16-bar polyphonic piece staying in key
- ✅ Validation loss < 0.3 (piano roll model)
- ✅ Note density: 2-4 notes/step

### Week 3
- ✅ Streamlit demo with audio playback
- ✅ Human rating > 3.5/5 for musicality
- ✅ 5+ preset examples working

### Week 4
- ✅ Published demo (Streamlit Cloud/Hugging Face Spaces)
- ✅ Technical report documenting approach
- ✅ GitHub repo with README

---

## Extensions for Advanced Features

### 1. Multi-Instrument (Ambitious)
Separate output heads for piano, bass, drums

### 2. Style Transfer
Add composer/genre embedding as model input

### 3. Attention Mechanism
Better long-range dependencies for 32+ bar pieces

### 4. Real-Time Generation
WebSocket streaming for live performance

---

## Learning Outcomes

By completing this project, you'll gain:
- ✅ **Music representation** (MIDI, piano roll, time series)
- ✅ **Polyphonic modeling** (multi-label classification)
- ✅ **Musical constraints** (key, rhythm, harmony)
- ✅ **Audio synthesis** (MIDI → audio playback)
- ✅ **Portfolio piece** (impressive demo for ML roles)

---

## Resources

**Libraries:**
- `pretty_midi` - MIDI parsing and creation
- `music21` - Music theory utilities
- `fluidsynth` - MIDI to audio synthesis
- `tensorflow` - Model training

**Datasets:**
- Maestro: https://magenta.tensorflow.org/datasets/maestro
- JSB Chorales: `tfds.load('jsb_chorales')`
- Lakh MIDI: https://colinraffel.com/projects/lmd

**Papers:**
- "Music Transformer" (Huang et al., 2018) - Attention for music
- "MusicVAE" (Roberts et al., 2018) - Latent representations
- "MuseNet" (OpenAI, 2019) - Large-scale music generation

---

## Next Steps

Ready to start? Here's your immediate action plan:

1. **Create project folder:**
   ```bash
   cd /Users/rix/Documents/School/Github/CST435/RNN
   mkdir music_generation
   cd music_generation
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow pretty_midi music21 streamlit matplotlib
   brew install fluid-synth
   ```

3. **Download JSB Chorales (quick test):**
   ```python
   import tensorflow_datasets as tfds
   ds = tfds.load('jsb_chorales')
   ```

4. **Create first file: `preprocessing/midi_encoder.py`**
   - Implement `PianoRollEncoder` class
   - Methods: `midi_to_piano_roll()`, `piano_roll_to_midi()`

5. **Test with single MIDI file**
   - Load → Convert to piano roll → Convert back to MIDI
   - Verify roundtrip preserves notes

---

## Additional Project Ideas

If you want to explore other options before committing to music generation, here are alternative project ideas organized by difficulty:

### Tier 1: Direct Extensions (Build on What You Know)

1. **Sentiment-Aware Story Generator** ⭐⭐ (2-3 weeks)
   - LSTM with controllable sentiment/emotion
   - Conditional generation with sentiment embeddings

2. **Conditional Fashion Designer (cGAN)** ⭐⭐ (2 weeks)
   - Generate specific clothing items on command
   - Class-conditional DCGAN

3. **Code Autocomplete Engine** ⭐⭐⭐ (3-4 weeks)
   - LSTM trained on Python code
   - Code-specific tokenization

### Tier 2: New Architectures

4. **Attention-Based Neural Machine Translation** ⭐⭐⭐⭐ (4-6 weeks)
   - Seq2Seq with Bahdanau attention
   - Interactive translator with attention visualization

5. **Variational Autoencoder (VAE)** ⭐⭐⭐ (2-3 weeks)
   - Probabilistic image generation
   - Latent space interpolation

6. **Transformer-Based Text Generator** ⭐⭐⭐⭐ (4-5 weeks)
   - Mini-GPT with self-attention
   - Direct comparison with your LSTM

### Tier 3: Real-World Applications

7. **Stock Price Forecasting** ⭐⭐⭐ (3 weeks)
   - Time series prediction with LSTM
   - Financial data handling

8. **Deepfake Detection** ⭐⭐ (2 weeks)
   - Use GAN discriminator for classification
   - Binary classification task

---

**Author:** Claude Code
**Date:** December 2025
**Project Type:** Open-ended Deep Learning Implementation
**Estimated Completion:** 3-4 weeks
