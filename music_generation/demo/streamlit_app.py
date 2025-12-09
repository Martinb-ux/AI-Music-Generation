"""
Streamlit Demo for LSTM Music Generator

Interactive web interface for generating melodies with the trained LSTM model.
"""

import streamlit as st
import sys
import os
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.midi_encoder import MIDIEventEncoder
from models.melody_lstm import MelodyLSTM
from models.generator import MusicGenerator


# Page configuration
st.set_page_config(
    page_title="LSTM Music Generator",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_and_encoder(model_path: str):
    """Load model and encoder (cached)."""
    # Initialize encoder
    encoder = MIDIEventEncoder(time_step=0.125)

    # Load model
    model = MelodyLSTM(vocab_size=encoder.vocab_size)
    model.load(model_path)

    # Create generator
    generator = MusicGenerator(model, encoder)

    return generator, encoder


def midi_to_audio(midi_bytes: bytes, soundfont_path: str = None) -> bytes:
    """
    Convert MIDI to audio using FluidSynth.

    Args:
        midi_bytes: MIDI file as bytes
        soundfont_path: Path to soundfont file (optional)

    Returns:
        Audio bytes (WAV format)
    """
    # Check if fluidsynth is available
    try:
        subprocess.run(['fluidsynth', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error("FluidSynth not installed. Install with: brew install fluid-synth")
        return None

    # Default soundfont (using FluidSynth's default)
    if soundfont_path is None:
        # Try common soundfont locations
        common_paths = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/local/share/soundfonts/FluidR3_GM.sf2",
            "soundfonts/FluidR3_GM.sf2"
        ]
        for path in common_paths:
            if os.path.exists(path):
                soundfont_path = path
                break

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as midi_file:
        midi_file.write(midi_bytes)
        midi_path = midi_file.name

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
        audio_path = audio_file.name

    try:
        # Convert MIDI to audio
        cmd = ['fluidsynth', '-ni', '-F', audio_path, '-r', '44100']

        if soundfont_path:
            cmd.extend([soundfont_path, midi_path])
        else:
            cmd.append(midi_path)

        subprocess.run(cmd, capture_output=True, check=True)

        # Read audio file
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        return audio_bytes

    except subprocess.CalledProcessError as e:
        st.error(f"FluidSynth error: {e.stderr.decode()}")
        return None

    finally:
        # Clean up temporary files
        if os.path.exists(midi_path):
            os.unlink(midi_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def create_seed_sequence(encoder, note_range: tuple = (60, 72)) -> np.ndarray:
    """
    Create a random seed sequence for generation.

    Args:
        encoder: MIDIEventEncoder instance
        note_range: (min_note, max_note) in MIDI numbers

    Returns:
        Seed sequence
    """
    seq_length = 64
    seed = []

    # Create a simple ascending/descending pattern
    pattern_type = np.random.choice(['ascending', 'descending', 'random'])

    if pattern_type == 'ascending':
        notes = list(range(note_range[0], note_range[1], 2))
    elif pattern_type == 'descending':
        notes = list(range(note_range[1], note_range[0], -2))
    else:
        notes = [np.random.randint(note_range[0], note_range[1]) for _ in range(8)]

    # Repeat pattern to fill sequence
    for i in range(seq_length):
        if i % 2 == 0 and len(notes) > 0:
            note = notes[i % len(notes)]
            seed.append(note - encoder.min_note + 1)
        else:
            seed.append(encoder.REST_TOKEN)

    return np.array(seed, dtype=np.int32)


def main():
    """Main Streamlit app."""

    # Header
    st.title("🎹 LSTM Music Generator")
    st.markdown("Generate original melodies using a trained LSTM model on Bach chorales.")

    # Sidebar - Model settings
    st.sidebar.header("⚙️ Generation Settings")

    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value="../checkpoints/melody_lstm_best.keras",
        help="Path to trained model checkpoint"
    )

    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.info("Train the model first by running: `python music_generation/training/train.py`")
        st.stop()

    # Load model
    try:
        with st.spinner("Loading model..."):
            generator, encoder = load_model_and_encoder(model_path)
        st.sidebar.success("Model loaded!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Generation parameters
    st.sidebar.subheader("Melody Parameters")

    length_bars = st.sidebar.slider(
        "Length (bars)",
        min_value=4,
        max_value=32,
        value=16,
        step=4,
        help="Number of bars to generate (4 beats per bar)"
    )

    length_steps = length_bars * 16  # 16 steps per bar at 16th notes

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.3,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="Higher = more creative/random, Lower = more conservative"
    )

    sampling_method = st.sidebar.selectbox(
        "Sampling Method",
        options=['temperature', 'top_k', 'nucleus'],
        index=0,
        help="Strategy for selecting next note"
    )

    # Additional parameters based on sampling method
    top_k = 10
    top_p = 0.9

    if sampling_method == 'top_k':
        top_k = st.sidebar.slider("Top-K", 5, 30, 10)
    elif sampling_method == 'nucleus':
        top_p = st.sidebar.slider("Top-P", 0.5, 0.99, 0.9)

    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Penalty for repeating recent notes (0=strong penalty, 1=no penalty)"
    )

    tempo = st.sidebar.slider(
        "Tempo (BPM)",
        min_value=60,
        max_value=180,
        value=120,
        step=10
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Generate New Melody")

        if st.button("🎵 Generate Music", type="primary", use_container_width=True):
            with st.spinner(f"Generating {length_bars}-bar melody..."):
                # Create seed sequence
                seed_sequence = create_seed_sequence(encoder, note_range=(60, 72))

                # Generate melody
                generated = generator.generate(
                    seed_sequence=seed_sequence,
                    length=length_steps,
                    temperature=temperature,
                    sampling_method=sampling_method,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )

                # Combine seed and generated
                full_sequence = np.concatenate([seed_sequence, generated])

                # Convert to MIDI
                with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_midi:
                    midi_path = temp_midi.name
                    encoder.sequence_to_midi(full_sequence, midi_path, tempo=tempo)

                    # Read MIDI bytes
                    with open(midi_path, 'rb') as f:
                        midi_bytes = f.read()

                # Store in session state
                st.session_state.midi_bytes = midi_bytes
                st.session_state.full_sequence = full_sequence

                st.success(f"Generated {len(full_sequence)} notes!")

    with col2:
        st.subheader("Info")
        st.info(f"""
        **Model:** Melody LSTM
        **Vocabulary:** {encoder.vocab_size} notes
        **Sequence Length:** 64 steps
        **Dataset:** JSB Chorales
        """)

    # Display generated melody
    if 'midi_bytes' in st.session_state:
        st.divider()
        st.subheader("Generated Melody")

        # Download MIDI
        st.download_button(
            label="⬇️ Download MIDI",
            data=st.session_state.midi_bytes,
            file_name=f"generated_melody_temp{temperature}.mid",
            mime="audio/midi"
        )

        # Try to convert to audio
        with st.spinner("Converting to audio..."):
            audio_bytes = midi_to_audio(st.session_state.midi_bytes)

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                label="⬇️ Download Audio (WAV)",
                data=audio_bytes,
                file_name=f"generated_melody_temp{temperature}.wav",
                mime="audio/wav"
            )
        else:
            st.warning("""
            Audio playback unavailable. FluidSynth not installed or soundfont not found.

            To enable audio:
            1. Install FluidSynth: `brew install fluid-synth`
            2. Download soundfont (optional)

            You can still download the MIDI file above.
            """)

        # Show sequence statistics
        st.subheader("Statistics")
        sequence = st.session_state.full_sequence
        non_rest = sequence[sequence != 0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Notes", len(sequence))
        col2.metric("Active Notes", len(non_rest))
        col3.metric("Rest Ratio", f"{(1 - len(non_rest)/len(sequence)):.1%}")
        col4.metric("Duration", f"{len(sequence) * encoder.time_step:.1f}s")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **How to use:**
    1. Adjust generation settings in the sidebar
    2. Click "Generate Music" to create a new melody
    3. Listen to the result or download MIDI/audio
    4. Experiment with different temperatures and sampling methods!

    **Tips:**
    - **Temperature 0.5-0.7:** Safe, coherent melodies
    - **Temperature 0.8-1.0:** Balanced creativity
    - **Temperature 1.1-1.5:** Experimental, unpredictable
    """)


if __name__ == "__main__":
    main()
