# Quick Start Guide - LSTM Music Generator

Get up and running with the LSTM Music Generator in minutes!

## Prerequisites

- Python 3.8+
- macOS (for FluidSynth audio synthesis)
- 2-3 GB free disk space

## Installation

### 1. Clone and Setup Environment

```bash
cd /Users/rix/Documents/School/Github/CST435/CST435_FinalProj

# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv and installs dependencies)
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Install FluidSynth (Optional - for audio playback)

```bash
brew install fluid-synth
```

## Usage

### Option 1: Quick Test (Recommended First Run)

Test the system with minimal training:

```bash
cd music_generation/training
python train.py --quick_test
```

This will:
- Download JSB Chorales dataset (~1 MB)
- Train for 2 epochs (~5-10 minutes)
- Generate sample melodies
- Save model to `checkpoints/`

### Option 2: Full Training

Train the model properly:

```bash
cd music_generation/training
python train.py --epochs 10 --batch_size 64
```

Expected training time: 20-30 minutes

### Option 3: Run Demo (requires trained model)

Launch the interactive Streamlit demo:

```bash
cd music_generation/demo
streamlit run streamlit_app.py
```

Then:
1. Open browser at http://localhost:8501
2. Adjust temperature and length in sidebar
3. Click "Generate Music"
4. Listen or download MIDI/audio

## Project Structure

```
music_generation/
├── preprocessing/      # MIDI encoding and data pipeline
│   ├── midi_encoder.py
│   └── data_pipeline.py
├── models/            # LSTM models
│   ├── melody_lstm.py
│   └── generator.py
├── training/          # Training scripts
│   └── train.py
├── demo/              # Streamlit demo
│   └── streamlit_app.py
├── evaluation/        # Metrics
│   └── metrics.py
└── data/              # Downloaded datasets (not in git)
```

## Command Line Options

### Training Script

```bash
python train.py [OPTIONS]

Options:
  --data_dir DIR           Data directory (default: ../data)
  --checkpoint_dir DIR     Checkpoint directory (default: ../checkpoints)
  --epochs N               Training epochs (default: 10)
  --batch_size N           Batch size (default: 64)
  --seq_length N           Sequence length (default: 64)
  --force_download         Re-download dataset
  --quick_test             Run quick test (2 epochs)
```

### Example Commands

```bash
# Quick test
python train.py --quick_test

# Custom training
python train.py --epochs 20 --batch_size 32

# Re-download dataset
python train.py --force_download
```

## Troubleshooting

### "No module named tensorflow"

Activate the virtual environment:
```bash
source venv/bin/activate
```

### "Model not found"

Train the model first:
```bash
cd music_generation/training
python train.py --quick_test
```

### "FluidSynth not installed"

Install FluidSynth:
```bash
brew install fluid-synth
```

Or download MIDI only (no audio playback needed).

### "Failed to download dataset"

Download manually:
1. Visit: https://github.com/czhuang/JSB-Chorales-dataset
2. Download and extract to `music_generation/data/raw/jsb_chorales/`

## Next Steps

1. **Week 1 Complete**: You now have a working monophonic melody generator!

2. **Experiment**:
   - Try different temperatures (0.5 - 1.5)
   - Adjust sequence length
   - Modify sampling methods

3. **Week 2 Preview**:
   - Implement polyphonic piano roll model
   - Generate chords and harmonies
   - Train on larger Maestro dataset

4. **Share**:
   - Generate melodies and share
   - Collect feedback for improvements

## Tips

- **Temperature 0.5-0.7**: Safe, coherent melodies (sounds like Bach)
- **Temperature 0.8-1.0**: Balanced creativity
- **Temperature 1.1-1.5**: Experimental, unpredictable

## Resources

- Model checkpoints: `music_generation/checkpoints/`
- Generated samples: `music_generation/checkpoints/samples/`
- Logs: `music_generation/checkpoints/logs/`

## Support

For issues:
1. Check this guide
2. Review error messages
3. Check file paths
4. Verify dependencies installed

---

**Ready to generate music?** Run `python train.py --quick_test` to get started!
