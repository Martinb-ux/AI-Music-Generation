# Music Generation with LSTM - CST435 Final Project

An LSTM-based neural network that generates original piano melodies by learning from Bach chorales. This project demonstrates sequence modeling, music representation, and creative AI applications.

## Overview

This project implements a recurrent neural network (LSTM) trained on the JSB Chorales dataset to generate novel monophonic melodies. The system learns musical patterns and structure from Bach's compositions and can generate new melodies with controllable creativity through temperature sampling.

**Key Features:**
- Event-based MIDI encoding for music representation
- Two-layer LSTM with 256 units per layer
- Temperature-controlled generation (0.3 - 1.5)
- Multiple sampling strategies (temperature, top-k, nucleus)
- Interactive Streamlit web interface
- Audio playback and MIDI export

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

```bash
# 1. Setup
./setup.sh
source venv/bin/activate

# 2. Train model (quick test)
cd music_generation/training
python train.py --quick_test

# 3. Launch demo
cd ../demo
streamlit run streamlit_app.py
```

## Project Structure

```
CST435_FinalProj/
├── music_generation/
│   ├── preprocessing/      # MIDI encoding and data pipeline
│   │   ├── midi_encoder.py       # MIDIEventEncoder class
│   │   └── data_pipeline.py      # Dataset download/preprocessing
│   ├── models/            # Neural network architectures
│   │   ├── melody_lstm.py        # LSTM model
│   │   └── generator.py          # Music generation engine
│   ├── training/          # Training scripts
│   │   └── train.py              # Main training pipeline
│   ├── demo/              # Web interface
│   │   └── streamlit_app.py      # Interactive demo
│   ├── evaluation/        # Evaluation metrics
│   │   └── metrics.py            # Music quality metrics
│   └── data/              # Datasets (downloaded, not in git)
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
└── QUICKSTART.md         # Getting started guide
```

## Technical Details

### Architecture

- **Model Type:** LSTM (Long Short-Term Memory)
- **Layers:**
  - Embedding layer (128-dim)
  - LSTM (256 units, return sequences)
  - LSTM (256 units)
  - Dense (256 units, ReLU)
  - Dropout (0.4)
  - Output (vocab_size, Softmax)
- **Loss:** Sparse categorical crossentropy
- **Optimizer:** Adam (lr=0.001)

### Dataset

- **Name:** JSB Chorales (Johann Sebastian Bach Chorales)
- **Size:** 382 MIDI files
- **Source:** Automatically downloaded from GitHub
- **Augmentation:** Transposition (±1, ±2 semitones)
- **Sequences:** ~5000-10000 training examples (64-step sequences)

### Music Representation

- **Encoding:** Event-based (monophonic)
- **Vocabulary:** 89 tokens (88 piano notes + REST)
- **Time Resolution:** 16th notes (0.125s at 120 BPM)
- **Sequence Length:** 64 time steps (4 bars)

## Features

### 1. Music Generation
- Generate melodies of variable length (4-32 bars)
- Temperature control for creativity
- Repetition penalty to avoid loops
- Multiple sampling strategies

### 2. Interactive Demo
- Real-time melody generation
- Adjustable parameters (temperature, length, tempo)
- MIDI and audio download
- Visual statistics

### 3. Evaluation Metrics
- Note density
- Pitch class histogram
- Note range
- Rhythmic regularity
- Repetition ratio

## Usage Examples

### Training

```bash
# Quick test (2 epochs)
python train.py --quick_test

# Full training (10 epochs)
python train.py --epochs 10

# Custom configuration
python train.py --epochs 20 --batch_size 32 --seq_length 64
```

### Generation (Python)

```python
from preprocessing import MIDIEventEncoder
from models import MelodyLSTM, MusicGenerator

# Load model
encoder = MIDIEventEncoder()
model = MelodyLSTM(vocab_size=encoder.vocab_size)
model.load('checkpoints/melody_lstm_best.keras')

# Create generator
generator = MusicGenerator(model, encoder)

# Generate melody
seed = np.random.randint(0, 89, size=(64,))
melody = generator.generate(seed, length=128, temperature=0.8)

# Save to MIDI
encoder.sequence_to_midi(melody, 'output.mid', tempo=120)
```

## Implementation Timeline

**Week 1: Monophonic Melody Generator** ✅
- [x] Project structure and setup
- [x] MIDI event encoder
- [x] Data pipeline (JSB Chorales)
- [x] Basic LSTM model
- [x] Training script with callbacks
- [x] Music generator with sampling
- [x] Streamlit demo
- [x] Documentation

**Week 2: Polyphonic Piano Roll** (Planned)
- [ ] Piano roll encoder (multi-hot)
- [ ] Polyphonic LSTM model
- [ ] Maestro dataset integration
- [ ] Advanced sampling constraints

**Week 3: Refinement** (Planned)
- [ ] Evaluation metrics
- [ ] Seed-based continuation
- [ ] UI improvements

**Week 4: Extensions** (Optional)
- [ ] Multi-track generation
- [ ] Style conditioning
- [ ] Attention mechanism

## Results

After training on JSB Chorales:
- **Validation Loss:** ~0.8-1.2
- **Validation Accuracy:** ~60-70%
- **Generated Melodies:** Coherent 16-bar compositions
- **Musical Quality:** Recognizable Bach-like patterns

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- pretty_midi
- Streamlit
- FluidSynth (optional, for audio)

See [requirements.txt](requirements.txt) for complete list.

## License

This project is for educational purposes as part of CST435.

## Acknowledgments

- **Dataset:** JSB Chorales from [czhuang/JSB-Chorales-dataset](https://github.com/czhuang/JSB-Chorales-dataset)
- **Inspiration:** Based on the implementation plan in `Music_Generation_LSTM_Implementation_Plan.md`
- **Course:** CST435 - Deep Learning

## References

- Eck, D., & Schmidhuber, J. (2002). Finding temporal structure in music: Blues improvisation with LSTM recurrent networks.
- Huang, C. Z. A., et al. (2018). Music transformer. arXiv preprint arXiv:1809.04281.
- Roberts, A., et al. (2018). A hierarchical latent vector model for learning long-term structure in music.

---

**Author:** Rix
**Date:** December 2025
**Course:** CST435 Final Project
