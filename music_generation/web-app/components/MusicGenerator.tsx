'use client';

import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Music, Download, Play, Pause, Sparkles, Loader2 } from 'lucide-react';
import * as Tone from 'tone';
import { MelodyLSTM } from '@/lib/model';
import { MIDIEncoder } from '@/lib/midi';
import WaveformVisualizer from './WaveformVisualizer';

export default function MusicGenerator() {
  const [model, setModel] = useState<MelodyLSTM | null>(null);
  const [encoder] = useState(() => new MIDIEncoder());
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedSequence, setGeneratedSequence] = useState<number[] | null>(null);
  const [midi, setMidi] = useState<any>(null);

  // Settings
  const [temperature, setTemperature] = useState(0.8);
  const [lengthBars, setLengthBars] = useState(16);
  const [tempo, setTempo] = useState(120);

  // Audio
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const partRef = useRef<Tone.Part | null>(null);

  useEffect(() => {
    // Initialize model
    const initModel = async () => {
      const lstm = new MelodyLSTM({
        vocabSize: 89,
        seqLength: 64,
        embeddingDim: 128,
        lstmUnits: 256
      });

      lstm.buildModel();
      await lstm.initializeWeights();

      setModel(lstm);
      setIsModelLoaded(true);
    };

    initModel();

    // Initialize Tone.js synth
    synthRef.current = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'triangle' },
      envelope: {
        attack: 0.005,
        decay: 0.1,
        sustain: 0.3,
        release: 1
      }
    }).toDestination();

    return () => {
      if (synthRef.current) {
        synthRef.current.dispose();
      }
      if (partRef.current) {
        partRef.current.dispose();
      }
      model?.dispose();
    };
  }, []);

  const handleGenerate = async () => {
    if (!model || !isModelLoaded) return;

    setIsGenerating(true);
    setProgress(0);

    try {
      // Create seed sequence
      const seedSequence = encoder.createSeedSequence(64, [60, 72], 'random');

      // Generate melody
      const lengthSteps = lengthBars * 16; // 16 steps per bar
      const generated = await model.generateMelody(
        seedSequence,
        lengthSteps,
        temperature,
        (current, total) => {
          setProgress((current / total) * 100);
        }
      );

      // Combine seed and generated
      const fullSequence = [...seedSequence, ...generated];
      setGeneratedSequence(fullSequence);

      // Convert to MIDI
      const midiData = encoder.sequenceToMidi(fullSequence, tempo);
      setMidi(midiData);

      setProgress(100);
    } catch (error) {
      console.error('Generation error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePlay = async () => {
    if (!midi || !synthRef.current) return;

    await Tone.start();

    if (isPlaying) {
      Tone.getTransport().stop();
      partRef.current?.stop();
      setIsPlaying(false);
    } else {
      // Clear previous part
      if (partRef.current) {
        partRef.current.dispose();
      }

      // Create new part from MIDI
      const notes = midi.tracks[0].notes.map((note: any) => ({
        time: note.time,
        note: note.name,
        duration: note.duration,
        velocity: note.velocity
      }));

      partRef.current = new Tone.Part((time, note) => {
        synthRef.current?.triggerAttackRelease(
          note.note,
          note.duration,
          time,
          note.velocity
        );
      }, notes);

      Tone.getTransport().bpm.value = tempo;
      partRef.current.start(0);
      Tone.getTransport().start();

      setIsPlaying(true);

      // Stop when done
      setTimeout(() => {
        setIsPlaying(false);
        Tone.getTransport().stop();
      }, (midi.duration + 1) * 1000);
    }
  };

  const handleDownload = () => {
    if (!midi) return;

    const midiArray = encoder.midiToArrayBuffer(midi);
    const blob = new Blob([midiArray as unknown as BlobPart], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated-melody-temp${temperature}.mid`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const stats = generatedSequence
    ? encoder.getSequenceStats(generatedSequence)
    : null;

  return (
    <div className="min-h-screen w-full p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <div className="flex items-center justify-center gap-3">
            <Music className="w-12 h-12" />
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-white via-purple-200 to-white bg-clip-text text-transparent">
              AI Music Generator
            </h1>
          </div>
          <p className="text-lg md:text-xl text-white/80">
            Generate original melodies with LSTM neural networks
          </p>
        </motion.div>

        {/* Main Controls */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="glass-strong rounded-3xl p-8 md:p-12 glow"
        >
          <div className="grid md:grid-cols-2 gap-8">
            {/* Settings */}
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold flex items-center gap-2">
                <Sparkles className="w-6 h-6" />
                Settings
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Temperature: {temperature.toFixed(1)}
                  </label>
                  <input
                    type="range"
                    min="0.3"
                    max="1.5"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-white/60 mt-1">
                    Higher = more creative, Lower = more conservative
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Length: {lengthBars} bars
                  </label>
                  <input
                    type="range"
                    min="4"
                    max="32"
                    step="4"
                    value={lengthBars}
                    onChange={(e) => setLengthBars(parseInt(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Tempo: {tempo} BPM
                  </label>
                  <input
                    type="range"
                    min="60"
                    max="180"
                    step="10"
                    value={tempo}
                    onChange={(e) => setTempo(parseInt(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>

              <button
                onClick={handleGenerate}
                disabled={!isModelLoaded || isGenerating}
                className="w-full py-4 px-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl font-semibold text-lg hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Generating... {progress.toFixed(0)}%
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Generate Music
                  </>
                )}
              </button>
            </div>

            {/* Info */}
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold">Model Info</h2>

              <div className="glass rounded-xl p-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/60">Status:</span>
                  <span className={isModelLoaded ? 'text-green-400' : 'text-yellow-400'}>
                    {isModelLoaded ? '✓ Ready' : 'Loading...'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">Architecture:</span>
                  <span>LSTM (256 units × 2)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">Vocabulary:</span>
                  <span>89 notes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">Sequence Length:</span>
                  <span>64 steps</span>
                </div>
              </div>

              {stats && (
                <div className="glass rounded-xl p-4 space-y-2 text-sm">
                  <h3 className="font-semibold mb-2">Generated Stats</h3>
                  <div className="flex justify-between">
                    <span className="text-white/60">Total Notes:</span>
                    <span>{stats.totalNotes}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Active Notes:</span>
                    <span>{stats.activeNotes}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Rest Ratio:</span>
                    <span>{(stats.restRatio * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Duration:</span>
                    <span>{stats.duration.toFixed(1)}s</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Visualizer */}
        {generatedSequence && (
          <WaveformVisualizer isPlaying={isPlaying} sequence={generatedSequence} />
        )}

        {/* Playback Controls */}
        {generatedSequence && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-strong rounded-3xl p-8 glow"
          >
            <h2 className="text-2xl font-semibold mb-6">Playback</h2>

            <div className="flex gap-4">
              <button
                onClick={handlePlay}
                className="flex-1 py-4 px-6 bg-white/20 hover:bg-white/30 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-5 h-5" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Play
                  </>
                )}
              </button>

              <button
                onClick={handleDownload}
                className="flex-1 py-4 px-6 bg-white/20 hover:bg-white/30 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors"
              >
                <Download className="w-5 h-5" />
                Download MIDI
              </button>
            </div>
          </motion.div>
        )}

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-center text-sm text-white/60"
        >
          <p>Built with TensorFlow.js, React, and Next.js</p>
          <p className="mt-1">Trained on Bach Chorales dataset</p>
        </motion.div>
      </div>
    </div>
  );
}
