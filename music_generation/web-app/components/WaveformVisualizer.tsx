'use client';

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface WaveformVisualizerProps {
  isPlaying: boolean;
  sequence?: number[];
}

export default function WaveformVisualizer({ isPlaying, sequence }: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();
    window.addEventListener('resize', resize);

    let time = 0;
    const bars = 64;

    const draw = () => {
      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Draw bars
      const barWidth = width / bars;

      for (let i = 0; i < bars; i++) {
        // Animated height
        let barHeight;
        if (isPlaying) {
          barHeight =
            (Math.sin(time * 0.05 + i * 0.1) * 0.5 + 0.5) * height * 0.7 +
            height * 0.1;
        } else if (sequence && sequence.length > i) {
          // Show sequence data when not playing
          const noteValue = sequence[i] || 0;
          barHeight = (noteValue / 89) * height * 0.5 + height * 0.1;
        } else {
          barHeight = height * 0.2;
        }

        // Gradient color
        const hue = (i / bars) * 360;
        const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
        gradient.addColorStop(0, `hsla(${hue}, 70%, 60%, 0.8)`);
        gradient.addColorStop(1, `hsla(${hue + 60}, 70%, 70%, 0.9)`);

        ctx.fillStyle = gradient;
        ctx.fillRect(
          i * barWidth + barWidth * 0.1,
          height - barHeight,
          barWidth * 0.8,
          barHeight
        );
      }

      time++;
      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, sequence]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.2 }}
      className="glass rounded-3xl p-6 glow"
    >
      <h2 className="text-2xl font-semibold mb-4">Visualization</h2>
      <canvas
        ref={canvasRef}
        className="w-full h-48 rounded-xl"
        style={{ background: 'rgba(0, 0, 0, 0.2)' }}
      />
    </motion.div>
  );
}
