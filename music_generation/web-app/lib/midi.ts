/**
 * MIDI Encoding/Decoding for Monophonic Melodies
 *
 * Matches the Python MIDIEventEncoder implementation
 */

import { Midi } from '@tonejs/midi';

export interface MIDIEncoderConfig {
  timeStep: number;  // Time quantization in seconds (0.125 = 16th note at 120 BPM)
  maxNote: number;   // Maximum MIDI note number (108 = C8)
  minNote: number;   // Minimum MIDI note number (21 = A0)
}

export class MIDIEncoder {
  private config: MIDIEncoderConfig;
  public readonly vocabSize: number;
  public readonly REST_TOKEN = 0;

  constructor(config: Partial<MIDIEncoderConfig> = {}) {
    this.config = {
      timeStep: config.timeStep ?? 0.125,
      maxNote: config.maxNote ?? 108,
      minNote: config.minNote ?? 21
    };

    // Vocabulary: REST (0) + note range
    this.vocabSize = (this.config.maxNote - this.config.minNote + 1) + 1;
  }

  /**
   * Convert sequence to MIDI file
   */
  sequenceToMidi(sequence: number[], tempo: number = 120): Midi {
    const midi = new Midi();
    const track = midi.addTrack();

    let currentNote: number | null = null;
    let noteStart: number | null = null;

    sequence.forEach((noteIdx, step) => {
      const time = step * this.config.timeStep;

      if (noteIdx === this.REST_TOKEN) {
        // End current note if playing
        if (currentNote !== null && noteStart !== null) {
          track.addNote({
            midi: currentNote,
            time: noteStart,
            duration: time - noteStart,
            velocity: 0.63  // 80/127 ≈ 0.63
          });
          currentNote = null;
          noteStart = null;
        }
      } else {
        // Convert vocabulary index to MIDI note
        const pitch = noteIdx + this.config.minNote - 1;

        // If different note, end current and start new
        if (currentNote !== pitch) {
          if (currentNote !== null && noteStart !== null) {
            track.addNote({
              midi: currentNote,
              time: noteStart,
              duration: time - noteStart,
              velocity: 0.63
            });
          }

          currentNote = pitch;
          noteStart = time;
        }
      }
    });

    // Close final note if still playing
    if (currentNote !== null && noteStart !== null) {
      const endTime = sequence.length * this.config.timeStep;
      track.addNote({
        midi: currentNote,
        time: noteStart,
        duration: endTime - noteStart,
        velocity: 0.63
      });
    }

    // Set tempo
    midi.header.setTempo(tempo);

    return midi;
  }

  /**
   * Convert MIDI to downloadable file
   */
  midiToArrayBuffer(midi: Midi): Uint8Array {
    return midi.toArray();
  }

  /**
   * Create a seed sequence for generation
   */
  createSeedSequence(
    seqLength: number = 64,
    noteRange: [number, number] = [60, 72],
    patternType: 'ascending' | 'descending' | 'random' = 'random'
  ): number[] {
    const seed: number[] = [];
    let notes: number[] = [];

    if (patternType === 'ascending') {
      notes = [];
      for (let n = noteRange[0]; n < noteRange[1]; n += 2) {
        notes.push(n);
      }
    } else if (patternType === 'descending') {
      notes = [];
      for (let n = noteRange[1]; n >= noteRange[0]; n -= 2) {
        notes.push(n);
      }
    } else {
      notes = Array.from(
        { length: 8 },
        () => Math.floor(Math.random() * (noteRange[1] - noteRange[0])) + noteRange[0]
      );
    }

    for (let i = 0; i < seqLength; i++) {
      if (i % 2 === 0 && notes.length > 0) {
        const note = notes[i % notes.length];
        seed.push(note - this.config.minNote + 1);
      } else {
        seed.push(this.REST_TOKEN);
      }
    }

    return seed;
  }

  /**
   * Get note name from MIDI number
   */
  getNoteNameFromMidi(midiNumber: number): string {
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(midiNumber / 12) - 1;
    const noteName = noteNames[midiNumber % 12];
    return `${noteName}${octave}`;
  }

  /**
   * Get note name from vocabulary index
   */
  getNoteNameFromIndex(idx: number): string {
    if (idx === this.REST_TOKEN) {
      return 'REST';
    }
    const midiNumber = idx + this.config.minNote - 1;
    return this.getNoteNameFromMidi(midiNumber);
  }

  /**
   * Get sequence statistics
   */
  getSequenceStats(sequence: number[]): {
    totalNotes: number;
    activeNotes: number;
    restRatio: number;
    duration: number;
    uniqueNotes: Set<number>;
  } {
    const nonRest = sequence.filter(n => n !== this.REST_TOKEN);
    const uniqueNotes = new Set(nonRest);

    return {
      totalNotes: sequence.length,
      activeNotes: nonRest.length,
      restRatio: 1 - (nonRest.length / sequence.length),
      duration: sequence.length * this.config.timeStep,
      uniqueNotes
    };
  }
}
