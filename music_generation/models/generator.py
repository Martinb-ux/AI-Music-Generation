"""
Music Generator with Advanced Sampling Techniques

Provides various sampling strategies for music generation.
"""

import numpy as np
from typing import Optional, Callable


class MusicGenerator:
    """
    Advanced music generation with temperature sampling, top-k, nucleus sampling, etc.
    """

    def __init__(self, model, encoder):
        """
        Initialize generator.

        Args:
            model: Trained MelodyLSTM model
            encoder: MIDIEventEncoder instance
        """
        self.model = model
        self.encoder = encoder

    def sample_with_temperature(self, probs: np.ndarray, temperature: float = 1.0) -> int:
        """
        Sample from probability distribution with temperature.

        Args:
            probs: Probability distribution
            temperature: Sampling temperature
                - < 1.0: More conservative (sharper distribution)
                - = 1.0: Original distribution
                - > 1.0: More creative (flatter distribution)

        Returns:
            Sampled index
        """
        # Apply temperature
        log_probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)

        # Sample
        return np.random.choice(len(probs), p=probs)

    def sample_top_k(self, probs: np.ndarray, k: int = 10, temperature: float = 1.0) -> int:
        """
        Sample from top-k most likely notes.

        Args:
            probs: Probability distribution
            k: Number of top candidates to consider
            temperature: Sampling temperature

        Returns:
            Sampled index
        """
        # Get top-k indices
        top_k_indices = np.argsort(probs)[-k:]
        top_k_probs = probs[top_k_indices]

        # Apply temperature to top-k
        log_probs = np.log(top_k_probs + 1e-10) / temperature
        top_k_probs = np.exp(log_probs - np.max(log_probs))
        top_k_probs = top_k_probs / np.sum(top_k_probs)

        # Sample from top-k
        sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
        return top_k_indices[sampled_idx]

    def sample_nucleus(self, probs: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
        """
        Nucleus (top-p) sampling.

        Args:
            probs: Probability distribution
            p: Cumulative probability threshold
            temperature: Sampling temperature

        Returns:
            Sampled index
        """
        # Sort probabilities
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Get nucleus
        cumsum = np.cumsum(sorted_probs)
        nucleus_size = np.searchsorted(cumsum, p) + 1

        nucleus_indices = sorted_indices[:nucleus_size]
        nucleus_probs = sorted_probs[:nucleus_size]

        # Apply temperature
        log_probs = np.log(nucleus_probs + 1e-10) / temperature
        nucleus_probs = np.exp(log_probs - np.max(log_probs))
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)

        # Sample
        sampled_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
        return nucleus_indices[sampled_idx]

    def apply_musical_constraints(self,
                                   probs: np.ndarray,
                                   recent_notes: Optional[list] = None,
                                   repetition_penalty: float = 0.5) -> np.ndarray:
        """
        Apply musical constraints to probability distribution.

        Args:
            probs: Probability distribution
            recent_notes: Recently played notes to penalize
            repetition_penalty: Penalty factor for repeated notes (0-1)

        Returns:
            Modified probability distribution
        """
        probs = probs.copy()

        # Penalize recent repetitions
        if recent_notes is not None and len(recent_notes) > 0:
            for note in recent_notes[-8:]:  # Last 8 notes
                if 0 <= note < len(probs):
                    probs[note] *= repetition_penalty

        # Ensure probabilities sum to 1
        probs = probs / (np.sum(probs) + 1e-10)

        return probs

    def generate(self,
                 seed_sequence: np.ndarray,
                 length: int = 128,
                 temperature: float = 0.8,
                 sampling_method: str = 'temperature',
                 top_k: int = 10,
                 top_p: float = 0.9,
                 repetition_penalty: float = 0.5) -> np.ndarray:
        """
        Generate music with specified sampling method.

        Args:
            seed_sequence: Starting sequence (seq_length,)
            length: Number of notes to generate
            temperature: Sampling temperature
            sampling_method: 'temperature', 'top_k', or 'nucleus'
            top_k: K value for top-k sampling
            top_p: P value for nucleus sampling
            repetition_penalty: Penalty for repeated notes

        Returns:
            Generated note sequence
        """
        seq_length = seed_sequence.shape[0]
        generated = list(seed_sequence)
        recent_notes = []

        for i in range(length):
            # Get current sequence
            current_seq = np.array(generated[-seq_length:])
            current_seq = np.expand_dims(current_seq, 0)

            # Get predictions
            probs = self.model.model.predict(current_seq, verbose=0)[0]

            # Apply musical constraints
            probs = self.apply_musical_constraints(probs, recent_notes, repetition_penalty)

            # Sample next note based on method
            if sampling_method == 'temperature':
                next_note = self.sample_with_temperature(probs, temperature)
            elif sampling_method == 'top_k':
                next_note = self.sample_top_k(probs, k=top_k, temperature=temperature)
            elif sampling_method == 'nucleus':
                next_note = self.sample_nucleus(probs, p=top_p, temperature=temperature)
            else:
                raise ValueError(f"Unknown sampling method: {sampling_method}")

            generated.append(next_note)
            recent_notes.append(next_note)

        return np.array(generated[seq_length:])

    def generate_to_midi(self,
                         seed_sequence: np.ndarray,
                         output_path: str,
                         length: int = 128,
                         temperature: float = 0.8,
                         tempo: int = 120,
                         **kwargs) -> np.ndarray:
        """
        Generate music and save directly to MIDI.

        Args:
            seed_sequence: Starting sequence
            output_path: Path to save MIDI file
            length: Number of notes to generate
            temperature: Sampling temperature
            tempo: Tempo in BPM
            **kwargs: Additional arguments for generate()

        Returns:
            Generated note sequence
        """
        # Generate
        generated = self.generate(seed_sequence, length, temperature, **kwargs)

        # Combine seed and generated
        full_sequence = np.concatenate([seed_sequence, generated])

        # Convert to MIDI
        self.encoder.sequence_to_midi(full_sequence, output_path, tempo)

        print(f"Generated melody saved to {output_path}")
        print(f"Length: {len(full_sequence)} time steps ({len(full_sequence) * self.encoder.time_step:.1f} seconds)")

        return generated


def create_generator(model_path: str, encoder):
    """
    Convenience function to create a generator from a saved model.

    Args:
        model_path: Path to saved model
        encoder: MIDIEventEncoder instance

    Returns:
        MusicGenerator
    """
    from melody_lstm import MelodyLSTM

    # Load model
    model = MelodyLSTM(vocab_size=encoder.vocab_size)
    model.load(model_path)

    # Create generator
    generator = MusicGenerator(model, encoder)

    return generator


if __name__ == "__main__":
    print("Generator module - use with trained model")
