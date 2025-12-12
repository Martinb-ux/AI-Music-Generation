/**
 * TensorFlow.js LSTM Model for Music Generation
 *
 * This recreates the Python Keras model in TensorFlow.js
 */

import * as tf from '@tensorflow/tfjs';

export interface ModelConfig {
  vocabSize: number;
  seqLength: number;
  embeddingDim: number;
  lstmUnits: number;
}

export class MelodyLSTM {
  private model: tf.LayersModel | null = null;
  private config: ModelConfig;

  constructor(config: ModelConfig = {
    vocabSize: 89,  // REST (0) + 88 piano notes
    seqLength: 64,
    embeddingDim: 128,
    lstmUnits: 256
  }) {
    this.config = config;
  }

  /**
   * Build the LSTM model architecture
   * Matches the Python Keras model exactly:
   * - Embedding layer (vocab_size, embedding_dim)
   * - LSTM layer 1 (256 units, return sequences)
   * - LSTM layer 2 (256 units)
   * - Dense layer (256 units, relu)
   * - Dropout (0.4)
   * - Output Dense layer (vocab_size, softmax)
   */
  buildModel(): tf.LayersModel {
    const { vocabSize, seqLength, embeddingDim, lstmUnits } = this.config;

    const model = tf.sequential();

    // Input: (batch, seqLength) - sequence of note indices
    model.add(tf.layers.inputLayer({ inputShape: [seqLength] }));

    // Embedding layer
    model.add(tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embeddingDim,
      inputLength: seqLength,
      maskZero: false
    }));

    // First LSTM layer with return sequences
    model.add(tf.layers.lstm({
      units: lstmUnits,
      returnSequences: true,
      dropout: 0.3,
      recurrentDropout: 0.2
    }));

    // Second LSTM layer
    model.add(tf.layers.lstm({
      units: lstmUnits,
      dropout: 0.3,
      recurrentDropout: 0.2
    }));

    // Dense hidden layer
    model.add(tf.layers.dense({
      units: 256,
      activation: 'relu'
    }));

    // Dropout
    model.add(tf.layers.dropout({ rate: 0.4 }));

    // Output layer
    model.add(tf.layers.dense({
      units: vocabSize,
      activation: 'softmax'
    }));

    this.model = model;
    return model;
  }

  /**
   * Initialize with random weights (for demo purposes)
   * In production, you would load pre-trained weights
   */
  async initializeWeights(): Promise<void> {
    if (!this.model) {
      this.buildModel();
    }
    // Model is already initialized with random weights by TensorFlow.js
  }

  /**
   * Predict the next note given a sequence
   */
  async predictNextNote(
    sequence: number[],
    temperature: number = 1.0
  ): Promise<number> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }

    return tf.tidy(() => {
      // Convert to tensor and add batch dimension
      const inputTensor = tf.tensor2d([sequence], [1, sequence.length], 'int32');

      // Get predictions
      const predictions = this.model!.predict(inputTensor) as tf.Tensor;
      let probs = predictions.squeeze();

      // Apply temperature
      if (temperature !== 1.0) {
        const logits = probs.log().add(1e-10);
        const scaledLogits = logits.div(temperature);
        const maxLogit = scaledLogits.max();
        const expLogits = scaledLogits.sub(maxLogit).exp();
        probs = expLogits.div(expLogits.sum());
      }

      // Sample from distribution
      const probsArray = Array.from(probs.dataSync());
      const nextNote = this.sampleFromDistribution(probsArray);

      return nextNote;
    });
  }

  /**
   * Generate a melody by sampling from the model
   */
  async generateMelody(
    seedSequence: number[],
    length: number,
    temperature: number = 0.8,
    onProgress?: (current: number, total: number) => void
  ): Promise<number[]> {
    if (!this.model) {
      throw new Error('Model not built or loaded');
    }

    const { seqLength } = this.config;
    const generated: number[] = [...seedSequence];

    for (let i = 0; i < length; i++) {
      // Get last seqLength notes
      const currentSeq = generated.slice(-seqLength);

      // Predict next note
      const nextNote = await this.predictNextNote(currentSeq, temperature);
      generated.push(nextNote);

      if (onProgress && (i + 1) % 8 === 0) {
        onProgress(i + 1, length);
      }
    }

    // Return only the newly generated part
    return generated.slice(seqLength);
  }

  /**
   * Sample from a probability distribution
   */
  private sampleFromDistribution(probs: number[]): number {
    const rand = Math.random();
    let cumSum = 0;

    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i];
      if (rand < cumSum) {
        return i;
      }
    }

    return probs.length - 1;
  }

  /**
   * Get model summary
   */
  getSummary(): string {
    if (!this.model) {
      return 'Model not built yet';
    }

    let summary = 'Model: MelodyLSTM\n';
    summary += '─'.repeat(60) + '\n';
    this.model.layers.forEach((layer, idx) => {
      const outputShape = layer.outputShape as number[];
      summary += `${idx + 1}. ${layer.name} (${layer.getClassName()})\n`;
      summary += `   Output shape: ${JSON.stringify(outputShape)}\n`;
    });
    summary += '─'.repeat(60) + '\n';
    summary += `Total params: ${this.model.countParams()}\n`;

    return summary;
  }

  /**
   * Dispose of the model and free memory
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}
