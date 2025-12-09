"""
Evaluation Metrics for Generated Music

Provides objective metrics to evaluate music quality.
"""

import numpy as np
from typing import Tuple
from collections import Counter


class MusicMetrics:
    """Calculate objective metrics for generated music."""

    @staticmethod
    def note_density(sequence: np.ndarray, rest_token: int = 0) -> float:
        """
        Calculate average number of active notes per time step.

        Args:
            sequence: Note sequence
            rest_token: Token representing rest/silence

        Returns:
            Note density (notes per time step)
        """
        active_notes = np.sum(sequence != rest_token)
        return active_notes / len(sequence)

    @staticmethod
    def pitch_class_histogram(sequence: np.ndarray, rest_token: int = 0) -> np.ndarray:
        """
        Calculate pitch class distribution (C, C#, D, ...).

        Args:
            sequence: Note sequence
            rest_token: Token representing rest

        Returns:
            Array of shape (12,) with pitch class frequencies
        """
        # Filter out rests
        notes = sequence[sequence != rest_token]

        if len(notes) == 0:
            return np.zeros(12)

        # Convert to pitch classes (0-11)
        # Assuming notes are MIDI numbers offset from min_note
        pitch_classes = notes % 12

        # Count occurrences
        hist = np.zeros(12)
        for pc in pitch_classes:
            hist[pc] += 1

        # Normalize
        hist = hist / np.sum(hist)

        return hist

    @staticmethod
    def note_range(sequence: np.ndarray, rest_token: int = 0, min_note: int = 21) -> Tuple[int, int, int]:
        """
        Calculate the range of notes used.

        Args:
            sequence: Note sequence
            rest_token: Token representing rest
            min_note: Minimum MIDI note (for conversion)

        Returns:
            (min_pitch, max_pitch, range_semitones)
        """
        notes = sequence[sequence != rest_token]

        if len(notes) == 0:
            return 0, 0, 0

        # Convert to MIDI pitches
        pitches = notes + min_note - 1

        min_pitch = int(np.min(pitches))
        max_pitch = int(np.max(pitches))
        note_range = max_pitch - min_pitch

        return min_pitch, max_pitch, note_range

    @staticmethod
    def rhythmic_regularity(sequence: np.ndarray, rest_token: int = 0) -> float:
        """
        Measure rhythmic regularity using autocorrelation.

        Higher values indicate more regular rhythms.

        Args:
            sequence: Note sequence
            rest_token: Token representing rest

        Returns:
            Autocorrelation at 4-beat lag (regularity score)
        """
        # Convert to binary (note/rest)
        binary = (sequence != rest_token).astype(float)

        if len(binary) < 16:
            return 0.0

        # Calculate autocorrelation at 16-step lag (1 bar)
        lag = 16

        if len(binary) <= lag:
            return 0.0

        # Autocorrelation
        signal1 = binary[:-lag]
        signal2 = binary[lag:]

        # Pearson correlation
        mean1 = np.mean(signal1)
        mean2 = np.mean(signal2)

        if np.std(signal1) == 0 or np.std(signal2) == 0:
            return 0.0

        correlation = np.mean((signal1 - mean1) * (signal2 - mean2))
        correlation /= (np.std(signal1) * np.std(signal2))

        return float(correlation)

    @staticmethod
    def repetition_ratio(sequence: np.ndarray, window_size: int = 8) -> float:
        """
        Calculate ratio of repeated subsequences.

        Args:
            sequence: Note sequence
            window_size: Size of patterns to check

        Returns:
            Repetition ratio (0-1)
        """
        if len(sequence) < window_size * 2:
            return 0.0

        patterns = []
        for i in range(len(sequence) - window_size + 1):
            pattern = tuple(sequence[i:i + window_size])
            patterns.append(pattern)

        # Count unique vs total
        unique = len(set(patterns))
        total = len(patterns)

        repetition = 1.0 - (unique / total)
        return repetition

    @staticmethod
    def unique_notes_ratio(sequence: np.ndarray, rest_token: int = 0) -> float:
        """
        Calculate ratio of unique notes used.

        Args:
            sequence: Note sequence
            rest_token: Token representing rest

        Returns:
            Unique notes ratio
        """
        notes = sequence[sequence != rest_token]

        if len(notes) == 0:
            return 0.0

        unique = len(set(notes))
        total = len(notes)

        return unique / total

    @staticmethod
    def evaluate_sequence(sequence: np.ndarray, rest_token: int = 0, min_note: int = 21) -> dict:
        """
        Comprehensive evaluation of a generated sequence.

        Args:
            sequence: Note sequence
            rest_token: Token representing rest
            min_note: Minimum MIDI note

        Returns:
            Dictionary of metrics
        """
        metrics = MusicMetrics()

        min_pitch, max_pitch, note_range = metrics.note_range(sequence, rest_token, min_note)

        results = {
            'note_density': metrics.note_density(sequence, rest_token),
            'pitch_class_histogram': metrics.pitch_class_histogram(sequence, rest_token),
            'min_pitch': min_pitch,
            'max_pitch': max_pitch,
            'note_range': note_range,
            'rhythmic_regularity': metrics.rhythmic_regularity(sequence, rest_token),
            'repetition_ratio': metrics.repetition_ratio(sequence),
            'unique_notes_ratio': metrics.unique_notes_ratio(sequence, rest_token),
            'sequence_length': len(sequence)
        }

        return results

    @staticmethod
    def print_evaluation(metrics: dict):
        """
        Print evaluation metrics in a readable format.

        Args:
            metrics: Dictionary from evaluate_sequence()
        """
        print("="*60)
        print("MUSIC EVALUATION METRICS")
        print("="*60)
        print(f"Sequence Length: {metrics['sequence_length']} steps")
        print(f"Note Density: {metrics['note_density']:.2f} (notes per step)")
        print(f"Note Range: {metrics['note_range']} semitones (MIDI {metrics['min_pitch']}-{metrics['max_pitch']})")
        print(f"Unique Notes Ratio: {metrics['unique_notes_ratio']:.2%}")
        print(f"Rhythmic Regularity: {metrics['rhythmic_regularity']:.3f}")
        print(f"Repetition Ratio: {metrics['repetition_ratio']:.2%}")

        print("\nPitch Class Distribution:")
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pch = metrics['pitch_class_histogram']
        for i, note in enumerate(notes):
            bar = '█' * int(pch[i] * 50)
            print(f"  {note:3s}: {bar} {pch[i]:.1%}")

        print("="*60)


if __name__ == "__main__":
    # Test metrics
    print("Testing music metrics...")

    # Create a test sequence (C major scale)
    test_sequence = np.array([40, 42, 44, 45, 47, 49, 51, 52] * 4)  # C major scale

    metrics = MusicMetrics()
    results = metrics.evaluate_sequence(test_sequence, rest_token=0, min_note=21)

    metrics.print_evaluation(results)
