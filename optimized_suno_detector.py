#!/usr/bin/env python3
"""
Optimized Suno AI Watermark Detector
Performance-optimized version for faster processing.
"""

import numpy as np
import librosa
import scipy.signal as signal
import logging
from typing import Dict, List, Any
import hashlib

logger = logging.getLogger(__name__)

class OptimizedSunoWatermarkDetector:
    """Fast, optimized detector for Suno AI watermarking patterns."""
    
    def __init__(self):
        self.detection_cache = {}
        
        # Simplified Suno-specific frequency ranges (most important ones)
        self.suno_freq_ranges = [
            (19000, 20000, "Suno ultrasonic watermark"),
            (15000, 16000, "Suno mid-high watermark"),
            (8000, 8300, "Suno mid-range marker"),
            (17500, 18500, "Suno extended range"),
        ]
    
    def detect_suno_watermarks(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Fast Suno watermark detection."""
        detected = []
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio
        
        # For performance, limit analysis to first 30 seconds
        max_samples = min(len(audio_mono), sr * 30)
        audio_mono = audio_mono[:max_samples]
        
        # Cache key for expensive computations
        audio_hash = hashlib.md5(audio_mono.tobytes()).hexdigest()[:8]  # Shorter hash
        cache_key = f"suno_opt_{audio_hash}_{sr}"
        
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]
        
        logger.info(f"Fast analyzing {len(audio_mono)} samples for Suno watermarks")
        
        # Quick frequency domain analysis
        detected.extend(self._fast_frequency_analysis(audio_mono, sr))
        
        # Cache results
        self.detection_cache[cache_key] = detected
        
        logger.info(f"Fast detected {len(detected)} potential Suno watermarks")
        return detected
    
    def _fast_frequency_analysis(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Fast frequency-domain watermark detection."""
        detected = []
        
        # Use smaller FFT for speed
        n_fft = 2048
        hop_length = n_fft // 2
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Check each Suno frequency range
        for low_freq, high_freq, description in self.suno_freq_ranges:
            if high_freq > sr / 2:
                continue
            
            # Find frequency bins in range
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if not np.any(freq_mask):
                continue
            
            # Extract energy in this band
            band_energy = np.mean(magnitude[freq_mask], axis=0)
            
            # Quick statistical check
            if len(band_energy) > 10:
                mean_energy = np.mean(band_energy)
                std_energy = np.std(band_energy)
                
                # Look for unusually consistent energy (potential watermark)
                if std_energy / (mean_energy + 1e-10) < 0.3 and mean_energy > 1e-6:
                    # Quick autocorrelation check for periodicity
                    if len(band_energy) > 20:
                        autocorr = np.correlate(band_energy, band_energy, mode='full')
                        autocorr = autocorr[len(autocorr)//2:]
                        peaks, _ = signal.find_peaks(autocorr, height=0.5 * np.max(autocorr))
                        
                        if len(peaks) >= 3:  # Some periodicity found
                            confidence = min(len(peaks) / 10.0, 1.0)
                            detected.append({
                                'type': 'suno_frequency_pattern',
                                'freq_range': [low_freq, high_freq],
                                'description': description,
                                'confidence': confidence,
                                'method': 'frequency_analysis',
                                'energy_consistency': std_energy / (mean_energy + 1e-10),
                                'periodicity_strength': len(peaks)
                            })
        
        return detected
    
    def remove_suno_watermarks(self, audio: np.ndarray, sr: int, 
                              watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Fast removal of detected Suno watermarks."""
        result = audio.copy()
        
        for watermark in watermarks:
            if 'freq_range' in watermark:
                low_freq, high_freq = watermark['freq_range']
                confidence = watermark.get('confidence', 0.5)
                
                # Create a conservative notch filter
                nyquist = sr / 2
                if high_freq >= nyquist:
                    continue
                
                try:
                    # Design simple bandstop filter
                    low_norm = max(0.01, low_freq / nyquist)
                    high_norm = min(0.99, high_freq / nyquist)
                    
                    if low_norm < high_norm:
                        b, a = signal.butter(2, [low_norm, high_norm], btype='bandstop')
                        
                        # Check filter stability
                        if np.all(np.abs(np.roots(a)) < 1.0):
                            filtered = signal.filtfilt(b, a, result)
                            
                            # Blend based on confidence (conservative)
                            blend_factor = min(0.3, confidence * 0.5)
                            result = (1 - blend_factor) * result + blend_factor * filtered
                            
                            logger.debug(f"Applied {blend_factor:.1%} removal to {low_freq}-{high_freq}Hz")
                
                except Exception as e:
                    logger.warning(f"Failed to filter {low_freq}-{high_freq}Hz: {e}")
        
        return result


# Monkey patch the original detector with the optimized version
if __name__ == "__main__":
    # Test the optimized detector
    import soundfile as sf
    
    # Create test signal
    t = np.linspace(0, 1, 44100)
    test_audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.1 * np.sin(2 * np.pi * 19500 * t)  # High freq watermark
    )
    
    detector = OptimizedSunoWatermarkDetector()
    
    # Test detection
    watermarks = detector.detect_suno_watermarks(test_audio, 44100)
    print(f"Detected {len(watermarks)} watermarks")
    
    for wm in watermarks:
        print(f"  {wm['description']}: confidence {wm['confidence']:.2f}")
    
    # Test removal
    cleaned = detector.remove_suno_watermarks(test_audio, 44100, watermarks)
    print(f"Removal applied, RMS change: {np.sqrt(np.mean(cleaned**2)) / np.sqrt(np.mean(test_audio**2)):.3f}")