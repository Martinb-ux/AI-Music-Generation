"""
MIDI Event Encoder for Monophonic Melody Generation (Week 1)

This encoder converts MIDI files to sequences of note events for training
a basic LSTM model. For Week 1, we focus on monophonic (single note) melodies.
"""

import numpy as np
import pretty_midi
from typing import List, Tuple, Optional


class MIDIEventEncoder:
    """
    Encodes MIDI files as sequences of note events for monophonic melody generation.

    Vocabulary:
    - 0: REST (no note playing)
    - 1-128: MIDI note numbers (21-148, but typically 21-108 for piano)
    - Special tokens added automatically
    """

    def __init__(self, time_step: float = 0.125, max_note: int = 108, min_note: int = 21):
        """
        Initialize the encoder.

        Args:
            time_step: Time quantization in seconds (0.125 = 16th note at 120 BPM)
            max_note: Maximum MIDI note number (108 = C8)
            min_note: Minimum MIDI note number (21 = A0)
        """
        self.time_step = time_step
        self.max_note = max_note
        self.min_note = min_note

        # Vocabulary: REST (0) + note range
        self.vocab_size = (max_note - min_note + 1) + 1  # +1 for REST
        self.REST_TOKEN = 0

    def midi_to_sequence(self, midi_path: str, instrument_idx: int = 0) -> np.ndarray:
        """
        Convert a MIDI file to a sequence of note events.

        Args:
            midi_path: Path to MIDI file
            instrument_idx: Which instrument to extract (0 = first instrument)

        Returns:
            Array of shape (time_steps,) with note indices
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file {midi_path}: {e}")

        if len(midi_data.instruments) == 0:
            raise ValueError(f"No instruments found in {midi_path}")

        # Get the specified instrument
        instrument = midi_data.instruments[min(instrument_idx, len(midi_data.instruments) - 1)]

        # Get end time
        end_time = midi_data.get_end_time()
        num_steps = int(np.ceil(end_time / self.time_step))

        # Initialize sequence with REST tokens
        sequence = np.zeros(num_steps, dtype=np.int32)

        # Fill in notes (for monophonic, keep the highest note at each time step)
        for note in instrument.notes:
            start_step = int(note.start / self.time_step)
            end_step = int(note.end / self.time_step)

            # Clamp note to valid range
            if note.pitch < self.min_note or note.pitch > self.max_note:
                continue

            # Convert MIDI note to vocabulary index (1-indexed, 0 is REST)
            note_idx = note.pitch - self.min_note + 1

            # Fill time steps (for overlapping notes, keep the higher one)
            for step in range(start_step, min(end_step, num_steps)):
                if step < num_steps:
                    sequence[step] = max(sequence[step], note_idx)

        return sequence

    def sequence_to_midi(self, sequence: np.ndarray, output_path: str, tempo: int = 120):
        """
        Convert a sequence of note events back to MIDI.

        Args:
            sequence: Array of shape (time_steps,) with note indices
            output_path: Path to save MIDI file
            tempo: Tempo in BPM
        """
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_note = None
        note_start = None

        for step, note_idx in enumerate(sequence):
            time = step * self.time_step

            if note_idx == self.REST_TOKEN:
                # End current note if playing
                if current_note is not None:
                    note_end = time
                    midi_note = pretty_midi.Note(
                        velocity=80,
                        pitch=current_note,
                        start=note_start,
                        end=note_end
                    )
                    instrument.notes.append(midi_note)
                    current_note = None
                    note_start = None
            else:
                # Convert vocabulary index to MIDI note
                pitch = int(note_idx) + self.min_note - 1

                # If different note, end current and start new
                if current_note != pitch:
                    if current_note is not None:
                        note_end = time
                        midi_note = pretty_midi.Note(
                            velocity=80,
                            pitch=current_note,
                            start=note_start,
                            end=note_end
                        )
                        instrument.notes.append(midi_note)

                    current_note = pitch
                    note_start = time

        # Close final note if still playing
        if current_note is not None:
            note_end = len(sequence) * self.time_step
            midi_note = pretty_midi.Note(
                velocity=80,
                pitch=current_note,
                start=note_start,
                end=note_end
            )
            instrument.notes.append(midi_note)

        midi_data.instruments.append(instrument)
        midi_data.write(output_path)

    def create_training_sequences(self, sequence: np.ndarray, seq_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training sequences using a sliding window.

        Args:
            sequence: Full sequence of note events
            seq_length: Length of each training sequence

        Returns:
            X: Input sequences of shape (num_sequences, seq_length)
            y: Target notes of shape (num_sequences,)
        """
        if len(sequence) < seq_length + 1:
            raise ValueError(f"Sequence too short ({len(sequence)}) for seq_length {seq_length}")

        X = []
        y = []

        for i in range(len(sequence) - seq_length):
            X.append(sequence[i:i + seq_length])
            y.append(sequence[i + seq_length])

        return np.array(X), np.array(y)

    def augment_transpose(self, sequence: np.ndarray, semitones: int) -> Optional[np.ndarray]:
        """
        Transpose a sequence by a number of semitones.

        Args:
            sequence: Note sequence
            semitones: Number of semitones to transpose (+/-)

        Returns:
            Transposed sequence, or None if out of range
        """
        transposed = sequence.copy()

        for i, note_idx in enumerate(sequence):
            if note_idx == self.REST_TOKEN:
                continue

            # Convert to MIDI note, transpose, convert back
            pitch = note_idx + self.min_note - 1
            new_pitch = pitch + semitones

            # Check if in valid range
            if new_pitch < self.min_note or new_pitch > self.max_note:
                return None  # Transposition would go out of range

            transposed[i] = new_pitch - self.min_note + 1

        return transposed


def test_encoder():
    """Test the encoder with a simple melody."""
    encoder = MIDIEventEncoder(time_step=0.25)  # Quarter note steps for testing

    # Create a simple test sequence
    # C4 (60) -> D4 (62) -> E4 (64) -> REST -> C4 (60)
    test_sequence = np.array([
        60 - 21 + 1,  # C4
        60 - 21 + 1,  # C4 (sustained)
        62 - 21 + 1,  # D4
        62 - 21 + 1,  # D4
        64 - 21 + 1,  # E4
        64 - 21 + 1,  # E4
        0,            # REST
        0,            # REST
        60 - 21 + 1,  # C4
        60 - 21 + 1,  # C4
    ], dtype=np.int32)

    print(f"Test sequence shape: {test_sequence.shape}")
    print(f"Vocab size: {encoder.vocab_size}")
    print(f"Unique values in sequence: {np.unique(test_sequence)}")

    # Test MIDI conversion
    encoder.sequence_to_midi(test_sequence, "test_melody.mid", tempo=120)
    print("Created test_melody.mid")

    # Test training sequence creation
    X, y = encoder.create_training_sequences(test_sequence, seq_length=4)
    print(f"Training sequences - X shape: {X.shape}, y shape: {y.shape}")

    # Test transposition
    transposed = encoder.augment_transpose(test_sequence, semitones=2)
    if transposed is not None:
        print(f"Transposed sequence: {transposed[:5]}")

    return encoder


if __name__ == "__main__":
    test_encoder()
