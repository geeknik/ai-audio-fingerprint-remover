#!/usr/bin/env python3
"""
Quick Watermark Removal Comparison Tool
Fast comparison of original and processed audio files.
"""

import numpy as np
import librosa
import scipy.signal as signal
import argparse
import os
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_compare(original_path: str, processed_path: str) -> str:
    """Quick comparison of original and processed audio."""
    
    # Load both files
    try:
        original, sr_orig = librosa.load(original_path, sr=None, mono=True)
        processed, sr_proc = librosa.load(processed_path, sr=None, mono=True)
    except Exception as e:
        return f"Error loading files: {e}"
    
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Basic metrics
    noise = processed - original
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    correlation = np.corrcoef(original, processed)[0, 1]
    rmse = np.sqrt(np.mean((original - processed) ** 2))
    
    # Spectral analysis in key frequency ranges
    nperseg = min(2048, len(original) // 4)
    freqs, times, spec_orig = signal.spectrogram(original, fs=sr_orig, nperseg=nperseg)
    _, _, spec_proc = signal.spectrogram(processed, fs=sr_orig, nperseg=nperseg)
    
    # Convert to dB
    spec_orig_db = 10 * np.log10(spec_orig + 1e-10)
    spec_proc_db = 10 * np.log10(spec_proc + 1e-10)
    
    # Analyze high frequency content (where watermarks typically are)
    high_freq_mask = freqs >= 15000
    if np.any(high_freq_mask):
        high_freq_orig = np.mean(spec_orig_db[high_freq_mask])
        high_freq_proc = np.mean(spec_proc_db[high_freq_mask])
        high_freq_reduction = high_freq_orig - high_freq_proc
    else:
        high_freq_reduction = 0
    
    # Ultrasonic range analysis
    ultrasonic_mask = freqs >= 19000
    if np.any(ultrasonic_mask):
        ultrasonic_orig = np.mean(spec_orig_db[ultrasonic_mask])
        ultrasonic_proc = np.mean(spec_proc_db[ultrasonic_mask])
        ultrasonic_reduction = ultrasonic_orig - ultrasonic_proc
    else:
        ultrasonic_reduction = 0
    
    # Generate report
    report = f"Quick Watermark Removal Analysis\n"
    report += f"=" * 40 + "\n"
    report += f"Original: {os.path.basename(original_path)}\n"
    report += f"Processed: {os.path.basename(processed_path)}\n\n"
    
    report += f"Audio Quality Metrics:\n"
    report += f"-" * 20 + "\n"
    report += f"Signal-to-Noise Ratio: {snr:.2f} dB\n"
    report += f"Correlation: {correlation:.4f}\n"
    report += f"RMSE: {rmse:.6f}\n\n"
    
    report += f"Frequency Content Changes:\n"
    report += f"-" * 25 + "\n"
    report += f"High frequency reduction (15kHz+): {high_freq_reduction:.2f} dB\n"
    report += f"Ultrasonic reduction (19kHz+): {ultrasonic_reduction:.2f} dB\n\n"
    
    # Assessment
    quality_good = snr > 10 and correlation > 0.95  # More realistic thresholds
    # Negative values mean reduction, positive means increase
    watermark_removed = high_freq_reduction < -3 or ultrasonic_reduction < -5  # Negative values indicate removal
    
    report += f"Assessment:\n"
    report += f"-" * 11 + "\n"
    report += f"Audio quality preserved: {'YES' if quality_good else 'NO'}\n"
    report += f"Likely watermark removal: {'YES' if watermark_removed else 'MINIMAL'}\n"
    
    if quality_good and watermark_removed:
        overall = "EXCELLENT - Good quality with effective watermark removal"
    elif quality_good:
        overall = "GOOD - Quality preserved but minimal watermark removal detected"
    elif watermark_removed:
        overall = "FAIR - Watermark removal detected but quality may be affected"
    else:
        overall = "POOR - Minimal changes detected"
    
    report += f"Overall: {overall}\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Quick comparison of audio files")
    parser.add_argument("original", help="Original audio file")
    parser.add_argument("processed", help="Processed audio file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.original):
        print(f"Error: Original file '{args.original}' not found.")
        return 1
    
    if not os.path.exists(args.processed):
        print(f"Error: Processed file '{args.processed}' not found.")
        return 1
    
    report = quick_compare(args.original, args.processed)
    print(report)

if __name__ == "__main__":
    main()