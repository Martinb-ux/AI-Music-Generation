# Music Generation with LSTM

An LSTM-based music generation system that creates original piano melodies and chord progressions from MIDI data.

## Project Structure

```
music_generation/
├── preprocessing/       # MIDI encoding and data pipeline
├── models/             # LSTM model architectures
├── demo/               # Streamlit web interface
├── training/           # Training scripts
├── evaluation/         # Metrics and evaluation
├── data/
│   ├── raw/           # Downloaded MIDI files (not in git)
│   └── processed/     # Preprocessed arrays (not in git)
└── checkpoints/       # Trained models (not in git)
```

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FluidSynth (for MIDI to audio conversion):**
   ```bash
   # macOS
   brew install fluid-synth

   # Ubuntu/Debian
   sudo apt-get install fluidsynth
   ```

3. **Download dataset:**
   The JSB Chorales dataset will be automatically downloaded when you run the data pipeline.

## Usage

### Training

```bash
python music_generation/training/train.py
```

### Generate Music

```bash
streamlit run music_generation/demo/streamlit_app.py
```

## Week 1: Monophonic Melody Generator

- Event-based MIDI encoding
- Basic LSTM model for melody generation
- Temperature sampling
- Streamlit demo with audio playback

## Features

- Temperature-controlled generation (0.3 - 1.5)
- MIDI download
- Audio playback
- Real-time generation

## Implementation Details

- **Model:** 2-layer LSTM (256 units)
- **Dataset:** JSB Chorales (Bach chorales)
- **Sequence Length:** 64 time steps
- **Training:** ~10 epochs with early stopping
