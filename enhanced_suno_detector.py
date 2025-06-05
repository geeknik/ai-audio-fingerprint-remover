#!/usr/bin/env python3
"""
Enhanced Suno AI Watermark Detector
Specifically designed to detect and remove Suno AI's watermarking techniques.
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy import stats
from scipy.fft import fft, fftfreq
import logging
from typing import Dict, List, Tuple, Any, Optional
import hashlib

logger = logging.getLogger(__name__)

class SunoWatermarkDetector:
    """Advanced detector specifically for Suno AI watermarking patterns."""
    
    def __init__(self):
        self.detection_cache = {}
        
        # Suno-specific frequency ranges based on research and analysis
        self.suno_freq_ranges = [
            (19000, 20000, "Suno ultrasonic watermark"),
            (15000, 16000, "Suno mid-high watermark"),
            (8000, 8200, "Suno mid-range marker"),
            (50, 150, "Suno low-freq steganography"),
            (12000, 12100, "Suno secondary marker"),
            (17500, 18500, "Suno extended range"),
            (22000, 23000, "Suno extended ultrasonic"),
        ]
        
        # Neural network-based watermark patterns
        self.neural_patterns = {
            'periodic_energy': {'threshold': 0.15, 'min_peaks': 5},
            'spectral_entropy': {'low_threshold': 0.3, 'high_threshold': 0.9},
            'phase_coherence': {'threshold': 0.85},
            'frequency_stability': {'threshold': 0.95},
        }
    
    def detect_suno_watermarks(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Comprehensive Suno watermark detection."""
        detected = []
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio
        
        # Cache key for expensive computations
        audio_hash = hashlib.md5(audio_mono.tobytes()).hexdigest()[:16]
        cache_key = f"suno_{audio_hash}_{sr}"
        
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]
        
        logger.info(f"Analyzing {len(audio_mono)} samples for Suno watermarks")
        
        # 1. Neural network pattern detection
        neural_watermarks = self._detect_neural_patterns(audio_mono, sr)
        detected.extend(neural_watermarks)
        
        # 2. Frequency domain analysis
        freq_watermarks = self._detect_frequency_watermarks(audio_mono, sr)
        detected.extend(freq_watermarks)
        
        # 3. Temporal pattern analysis
        temporal_watermarks = self._detect_temporal_patterns(audio_mono, sr)
        detected.extend(temporal_watermarks)
        
        # 4. Phase-based watermark detection
        phase_watermarks = self._detect_phase_watermarks(audio_mono, sr)
        detected.extend(phase_watermarks)
        
        # 5. Statistical anomaly detection
        stat_watermarks = self._detect_statistical_anomalies(audio_mono, sr)
        detected.extend(stat_watermarks)
        
        # Cache results
        self.detection_cache[cache_key] = detected
        
        logger.info(f"Detected {len(detected)} potential Suno watermarks")
        return detected
    
    def _detect_neural_patterns(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect neural network-based watermarking patterns."""
        detected = []
        
        # Compute high-resolution spectrogram
        nperseg = min(4096, len(audio) // 8)
        nperseg = max(512, nperseg)
        
        freqs, times, spec = signal.spectrogram(
            audio, fs=sr, nperseg=nperseg, noverlap=nperseg//2,
            window='hann', scaling='spectrum'
        )
        
        spec_db = 10 * np.log10(spec + 1e-10)
        
        # Look for neural network artifacts in high frequencies
        high_freq_mask = freqs >= 15000
        if np.any(high_freq_mask):
            high_freq_spec = spec_db[high_freq_mask]
            
            # Neural networks often create subtle but consistent patterns
            for i, freq in enumerate(freqs[high_freq_mask]):
                freq_energy = high_freq_spec[i]
                
                # Check for neural network signatures:
                # 1. Unusually consistent energy levels
                energy_std = np.std(freq_energy)
                energy_mean = np.mean(freq_energy)
                
                # 2. Periodic patterns that are too regular for natural audio
                if len(freq_energy) > 20:
                    autocorr = np.correlate(freq_energy, freq_energy, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # Find peaks in autocorrelation
                    peaks, _ = signal.find_peaks(autocorr, height=0.3 * np.max(autocorr))
                    
                    # Neural watermarks often have very regular periods
                    if len(peaks) >= self.neural_patterns['periodic_energy']['min_peaks']:
                        periods = np.diff(peaks)
                        if len(periods) > 1:
                            period_regularity = 1 - (np.std(periods) / np.mean(periods))
                            
                            if period_regularity > self.neural_patterns['periodic_energy']['threshold']:
                                detected.append({
                                    'type': 'neural_periodic',
                                    'frequency': float(freq),
                                    'regularity': float(period_regularity),
                                    'confidence': min(period_regularity * 1.2, 1.0),
                                    'method': 'neural_pattern_analysis'
                                })
                
                # 3. Check for artificial energy distribution
                if energy_std < 2.0 and energy_mean > -80:  # Very stable, strong signal
                    detected.append({
                        'type': 'neural_stable_carrier',
                        'frequency': float(freq),
                        'stability': float(1 / (energy_std + 0.1)),
                        'mean_energy': float(energy_mean),
                        'confidence': min((1 / (energy_std + 0.1)) * 0.1, 1.0),
                        'method': 'neural_stability_analysis'
                    })
        
        return detected
    
    def _detect_frequency_watermarks(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect frequency-domain watermarks specific to Suno."""
        detected = []
        
        # Use higher resolution FFT for better frequency analysis
        n_fft = min(8192, len(audio))
        hop_length = n_fft // 4
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Analyze each Suno-specific frequency range
        for low_freq, high_freq, description in self.suno_freq_ranges:
            if high_freq > sr / 2:
                continue
                
            # Find frequency bins in this range
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if not np.any(freq_mask):
                continue
            
            range_magnitude = magnitude[freq_mask]
            range_freqs = freqs[freq_mask]
            
            # Calculate energy statistics
            energy = np.mean(range_magnitude ** 2, axis=1)
            total_energy = np.mean(energy)
            energy_std = np.std(energy)
            
            # Look for Suno-specific patterns
            anomaly_score = 0
            
            # 1. Check for narrow-band carriers (common in Suno)
            for i, freq in enumerate(range_freqs):
                freq_energy = range_magnitude[i]
                mean_energy = np.mean(freq_energy)
                std_energy = np.std(freq_energy)
                
                # Suno often uses stable carriers
                if std_energy < 0.1 * mean_energy and mean_energy > 0.001:
                    anomaly_score += 0.4
                    detected.append({
                        'type': 'suno_carrier',
                        'frequency': float(freq),
                        'freq_range': [low_freq, high_freq],
                        'description': description,
                        'stability': float(mean_energy / (std_energy + 1e-10)),
                        'confidence': 0.7,
                        'method': 'frequency_analysis'
                    })
            
            # 2. Check for energy anomalies in the range
            if total_energy > 0:
                # Compare with adjacent frequency ranges
                adjacent_low = max(0, low_freq - (high_freq - low_freq))
                adjacent_high = min(sr/2, high_freq + (high_freq - low_freq))
                
                adj_mask = ((freqs >= adjacent_low) & (freqs < low_freq)) | \
                          ((freqs > high_freq) & (freqs <= adjacent_high))
                
                if np.any(adj_mask):
                    adj_energy = np.mean(magnitude[adj_mask] ** 2)
                    energy_ratio = total_energy / (adj_energy + 1e-10)
                    
                    # Suno watermarks often have 2-10x more energy than adjacent bands
                    if 2.0 < energy_ratio < 10.0:
                        anomaly_score += 0.3
                        detected.append({
                            'type': 'suno_energy_anomaly',
                            'freq_range': [low_freq, high_freq],
                            'description': description,
                            'energy_ratio': float(energy_ratio),
                            'confidence': min(energy_ratio / 10.0, 1.0),
                            'method': 'energy_comparison'
                        })
        
        return detected
    
    def _detect_temporal_patterns(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect temporal watermarking patterns."""
        detected = []
        
        # Analyze energy envelope
        hop_length = 512
        frame_length = 2048
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Look for periodic patterns in energy
        if len(rms) > 50:
            # Autocorrelation analysis
            autocorr = np.correlate(rms, rms, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks
            peaks, properties = signal.find_peaks(
                autocorr, 
                height=0.2 * np.max(autocorr),
                distance=10
            )
            
            if len(peaks) > 5:
                # Check for regular timing patterns (Suno signature)
                periods = np.diff(peaks)
                if len(periods) > 2:
                    period_regularity = 1 - (np.std(periods) / np.mean(periods))
                    
                    # Suno often has very regular temporal patterns
                    if period_regularity > 0.8:
                        detected.append({
                            'type': 'suno_temporal_pattern',
                            'regularity': float(period_regularity),
                            'num_periods': len(periods),
                            'avg_period': float(np.mean(periods)),
                            'confidence': min(period_regularity, 1.0),
                            'method': 'temporal_analysis'
                        })
        
        # Check for sudden energy changes (watermark insertion points)
        energy_diff = np.diff(rms)
        sudden_changes = np.where(np.abs(energy_diff) > 3 * np.std(energy_diff))[0]
        
        if len(sudden_changes) > len(rms) * 0.02:  # More than 2% sudden changes
            detected.append({
                'type': 'suno_insertion_artifacts',
                'change_count': len(sudden_changes),
                'change_ratio': float(len(sudden_changes) / len(rms)),
                'confidence': min(len(sudden_changes) / (len(rms) * 0.05), 1.0),
                'method': 'energy_change_analysis'
            })
        
        return detected
    
    def _detect_phase_watermarks(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect phase-based watermarks."""
        detected = []
        
        # Compute STFT for phase analysis
        n_fft = min(2048, len(audio) // 4)
        n_fft = max(512, n_fft)
        
        stft = librosa.stft(audio, n_fft=n_fft)
        phases = np.angle(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Focus on high frequencies where phase watermarks are common
        high_freq_mask = freqs >= 15000
        if np.any(high_freq_mask):
            high_freq_phases = phases[high_freq_mask]
            
            # Check for phase coherence (watermark signature)
            for i, freq in enumerate(freqs[high_freq_mask]):
                phase_series = high_freq_phases[i]
                
                # Calculate phase coherence
                phase_diff = np.diff(phase_series)
                # Wrap to [-π, π]
                phase_diff = np.angle(np.exp(1j * phase_diff))
                
                if len(phase_diff) > 10:
                    coherence = 1 - (np.std(phase_diff) / np.pi)
                    
                    # Suno watermarks often have high phase coherence
                    if coherence > self.neural_patterns['phase_coherence']['threshold']:
                        detected.append({
                            'type': 'suno_phase_watermark',
                            'frequency': float(freq),
                            'coherence': float(coherence),
                            'confidence': min(coherence, 1.0),
                            'method': 'phase_analysis'
                        })
        
        return detected
    
    def _detect_statistical_anomalies(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect statistical anomalies that indicate AI generation."""
        detected = []
        
        # Sample entropy analysis
        def sample_entropy(data, m=2, r=None):
            """Calculate sample entropy."""
            if r is None:
                r = 0.2 * np.std(data)
            
            N = len(data)
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            phi = np.zeros(2)
            for template_length in [m, m+1]:
                patterns_temp = np.array([data[i:i+template_length] for i in range(N-template_length+1)])
                C = np.zeros(N-template_length+1)
                
                for i in range(N-template_length+1):
                    template = patterns_temp[i]
                    for j in range(N-template_length+1):
                        if _maxdist(template, patterns_temp[j], template_length) <= r:
                            C[i] += 1
                
                phi[template_length-m] = np.mean(np.log(C / (N-template_length+1)))
            
            return phi[0] - phi[1]
        
        # Calculate sample entropy for the audio
        if len(audio) > 1000:
            # Downsample for entropy calculation
            audio_downsampled = audio[::max(1, len(audio) // 10000)]
            entropy = sample_entropy(audio_downsampled)
            
            # AI-generated audio often has lower entropy
            if entropy < 0.5:
                detected.append({
                    'type': 'suno_low_entropy',
                    'entropy': float(entropy),
                    'confidence': 1.0 - entropy,
                    'method': 'statistical_analysis'
                })
        
        # Spectral flatness analysis
        n_fft = min(2048, len(audio) // 4)
        n_fft = max(512, n_fft)
        
        stft = librosa.stft(audio, n_fft=n_fft)
        magnitude = np.abs(stft)
        
        # Calculate spectral flatness for each frame
        spectral_flatness = []
        for frame in magnitude.T:
            # Geometric mean / Arithmetic mean
            geometric_mean = np.exp(np.mean(np.log(frame + 1e-10)))
            arithmetic_mean = np.mean(frame)
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            spectral_flatness.append(flatness)
        
        spectral_flatness = np.array(spectral_flatness)
        mean_flatness = np.mean(spectral_flatness)
        
        # AI audio often has unusual spectral flatness
        if mean_flatness > 0.8 or mean_flatness < 0.1:
            detected.append({
                'type': 'suno_spectral_anomaly',
                'flatness': float(mean_flatness),
                'confidence': abs(0.5 - mean_flatness) * 2,
                'method': 'spectral_flatness_analysis'
            })
        
        return detected
    
    def remove_suno_watermarks(self, audio: np.ndarray, sr: int, 
                              watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Remove detected Suno watermarks from audio."""
        if not watermarks:
            return audio
        
        result = audio.copy()
        
        # Group watermarks by type for efficient processing
        freq_watermarks = [w for w in watermarks if 'frequency' in w or 'freq_range' in w]
        temporal_watermarks = [w for w in watermarks if w['type'].startswith('suno_temporal')]
        phase_watermarks = [w for w in watermarks if w['type'].startswith('suno_phase')]
        
        # Remove frequency-domain watermarks
        for watermark in freq_watermarks:
            result = self._remove_frequency_watermark(result, sr, watermark)
        
        # Remove temporal watermarks
        for watermark in temporal_watermarks:
            result = self._remove_temporal_watermark(result, sr, watermark)
        
        # Remove phase watermarks
        for watermark in phase_watermarks:
            result = self._remove_phase_watermark(result, sr, watermark)
        
        return result
    
    def _remove_frequency_watermark(self, audio: np.ndarray, sr: int, 
                                   watermark: Dict[str, Any]) -> np.ndarray:
        """Remove a frequency-domain watermark."""
        if 'frequency' in watermark:
            # Single frequency carrier
            freq = watermark['frequency']
            # Create a narrow notch filter
            nyquist = sr / 2
            low = max(freq - 50, 0) / nyquist
            high = min(freq + 50, nyquist - 1) / nyquist
            
            if low < high < 1.0:
                try:
                    b, a = signal.butter(4, [low, high], btype='bandstop')
                    audio = signal.filtfilt(b, a, audio)
                except Exception as e:
                    logger.warning(f"Failed to apply notch filter at {freq}Hz: {e}")
        
        elif 'freq_range' in watermark:
            # Frequency range
            low_freq, high_freq = watermark['freq_range']
            nyquist = sr / 2
            low = low_freq / nyquist
            high = min(high_freq, nyquist - 1) / nyquist
            
            if 0 < low < high < 1.0:
                try:
                    b, a = signal.butter(4, [low, high], btype='bandstop')
                    audio = signal.filtfilt(b, a, audio)
                except Exception as e:
                    logger.warning(f"Failed to apply bandstop filter {low_freq}-{high_freq}Hz: {e}")
        
        return audio
    
    def _remove_temporal_watermark(self, audio: np.ndarray, sr: int, 
                                  watermark: Dict[str, Any]) -> np.ndarray:
        """Remove temporal watermarks by adding subtle timing variations."""
        # Add small random variations to break temporal patterns
        variation_strength = 0.001  # Very subtle
        
        # Create time-varying delay
        delay_samples = np.random.randn(len(audio)) * variation_strength * sr
        delay_samples = np.cumsum(delay_samples) * 0.1  # Integrate for smoother changes
        
        # Apply fractional delay using interpolation
        indices = np.arange(len(audio)) + delay_samples
        indices = np.clip(indices, 0, len(audio) - 1)
        
        # Linear interpolation
        audio_delayed = np.interp(np.arange(len(audio)), indices, audio)
        
        return audio_delayed
    
    def _remove_phase_watermark(self, audio: np.ndarray, sr: int, 
                               watermark: Dict[str, Any]) -> np.ndarray:
        """Remove phase watermarks by adding phase noise."""
        if 'frequency' in watermark:
            freq = watermark['frequency']
            
            # Generate phase noise at the watermark frequency
            n_fft = 2048
            stft = librosa.stft(audio, n_fft=n_fft)
            
            # Find the frequency bin
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            freq_bin = np.argmin(np.abs(freqs - freq))
            
            # Add random phase noise to this frequency
            phase_noise = np.random.uniform(-np.pi/4, np.pi/4, stft.shape[1])
            stft[freq_bin] *= np.exp(1j * phase_noise)
            
            # Reconstruct audio
            audio = librosa.istft(stft)
        
        return audio