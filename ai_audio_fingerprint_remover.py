#!/usr/bin/env python3
"""
ai_audio_fingerprint_remover.py - Comprehensive tool to remove AI fingerprints from audio files

This script implements multiple techniques to remove both overt and covert AI fingerprinting 
from audio files, targeting Suno, OpenAI, Google, Anthropic, and other AI audio generation platforms.

Features:
- Complete metadata stripping (ID3, RIFF, custom chunks)
- Audio spectral watermark detection and removal
- Sample-level normalization to remove statistical patterns
- Timing pattern randomization
- Frequency distribution normalization
- Adds subtle human-like imperfections

Usage:
    python ai_audio_fingerprint_remover.py input_file [output_file] [--aggressive]
    python ai_audio_fingerprint_remover.py --directory input_dir [output_dir] [--aggressive]
"""

import os
import sys
import argparse
import shutil
import tempfile
import random
import json
import re
import hashlib
import struct
import wave
import array
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Set
from dataclasses import dataclass
import uuid
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

try:
    import psutil
except ImportError:
    psutil = None

try:
    import numpy as np
    from enhanced_suno_detector import SunoWatermarkDetector
    from aggressive_watermark_remover import AggressiveWatermarkRemover
    from sota_watermark_remover import StateOfTheArtWatermarkRemover
    from scipy import signal, stats
    from scipy.io import wavfile
    from scipy.signal import hilbert
    # Try to import windows from scipy.signal for newer versions
    try:
        from scipy.signal import windows
    except ImportError:
        # For older scipy versions, windows might not be separate
        windows = None
except ImportError:
    print("Error: Required 'numpy' and 'scipy' libraries not found.")
    print("Please install them using: pip install numpy scipy")
    sys.exit(1)

try:
    import mutagen
    from mutagen.id3 import ID3, ID3NoHeaderError, Frames
    from mutagen.wave import WAVE
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.easyid3 import EasyID3
    from mutagen.aiff import AIFF
except ImportError:
    print("Error: Required 'mutagen' library not found.")
    print("Please install it using: pip install mutagen")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Error: Additional audio processing libraries not found.")
    print("Please install them using: pip install librosa soundfile")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """High-performance audio processor with chunked processing capabilities."""
    
    def __init__(self, config: 'ProcessingConfig'):
        self.config = config
        self.chunk_size = 30  # Process in 30-second chunks for large files
        self.overlap_size = 2  # 2-second overlap between chunks
        
    def process_large_audio_chunked(self, audio: np.ndarray, sr: int, 
                                   process_func, *args, **kwargs) -> np.ndarray:
        """Process large audio files in chunks to manage memory usage."""
        if len(audio) < sr * 60:  # Less than 1 minute, process normally
            return process_func(audio, sr, *args, **kwargs)
        
        chunk_samples = int(self.chunk_size * sr)
        overlap_samples = int(self.overlap_size * sr)
        hop_size = chunk_samples - overlap_samples
        
        processed_chunks = []
        total_chunks = (len(audio) - overlap_samples) // hop_size + 1
        
        logger.info(f"Processing large audio in {total_chunks} chunks of {self.chunk_size}s each")
        
        for i in range(total_chunks):
            start_idx = i * hop_size
            end_idx = min(start_idx + chunk_samples, len(audio))
            
            chunk = audio[start_idx:end_idx]
            
            if len(chunk) < sr:  # Skip very small chunks
                continue
                
            try:
                # Process the chunk
                processed_chunk = process_func(chunk, sr, *args, **kwargs)
                
                # Apply fade in/out to avoid clicks at chunk boundaries
                if i > 0:  # Not the first chunk
                    fade_samples = min(overlap_samples // 2, len(processed_chunk))
                    fade_in = np.linspace(0, 1, fade_samples)
                    processed_chunk[:fade_samples] *= fade_in
                
                if i < total_chunks - 1:  # Not the last chunk
                    fade_samples = min(overlap_samples // 2, len(processed_chunk))
                    fade_out = np.linspace(1, 0, fade_samples)
                    processed_chunk[-fade_samples:] *= fade_out
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.warning(f"Error processing chunk {i+1}/{total_chunks}: {e}")
                # Use original chunk if processing fails
                processed_chunks.append(chunk)
        
        if not processed_chunks:
            logger.warning("No chunks were successfully processed")
            return audio
        
        # Reconstruct the full audio with proper overlap handling
        return self._reconstruct_from_chunks(processed_chunks, hop_size, overlap_samples)
    
    def _reconstruct_from_chunks(self, chunks: List[np.ndarray], 
                                hop_size: int, overlap_samples: int) -> np.ndarray:
        """Reconstruct audio from overlapping chunks using cross-fade."""
        if not chunks:
            return np.array([])
        
        total_length = (len(chunks) - 1) * hop_size + len(chunks[-1])
        result = np.zeros(total_length)
        
        for i, chunk in enumerate(chunks):
            start_idx = i * hop_size
            end_idx = start_idx + len(chunk)
            
            if i == 0:
                # First chunk - no overlap
                result[start_idx:end_idx] = chunk
            else:
                # Overlapping chunks - use cross-fade
                overlap_start = start_idx
                overlap_end = min(overlap_start + overlap_samples, end_idx, len(result))
                
                if overlap_end > overlap_start:
                    # Cross-fade in the overlap region
                    overlap_len = overlap_end - overlap_start
                    fade_out = np.linspace(1, 0, overlap_len)
                    fade_in = np.linspace(0, 1, overlap_len)
                    
                    chunk_overlap_end = min(overlap_len, len(chunk))
                    result[overlap_start:overlap_start + chunk_overlap_end] *= fade_out[:chunk_overlap_end]
                    result[overlap_start:overlap_start + chunk_overlap_end] += chunk[:chunk_overlap_end] * fade_in[:chunk_overlap_end]
                
                # Add the non-overlapping part
                if overlap_end < end_idx:
                    non_overlap_start = overlap_len
                    result[overlap_end:end_idx] = chunk[non_overlap_start:non_overlap_start + (end_idx - overlap_end)]
        
        return result
    
    def adaptive_filter_design(self, audio: np.ndarray, sr: int, 
                              freq_range: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Design adaptive filters based on audio characteristics."""
        # Analyze the audio to determine optimal filter parameters
        freq_analysis = AdvancedAudioAnalysis.analyze_frequency_distribution(audio, sr)
        
        # Adapt filter order based on spectral complexity
        if freq_analysis['spectral_spread'] > sr / 8:
            # High spectral spread - use higher order filter
            filter_order = min(8, self.config.filter_order + 2)
        else:
            # Low spectral spread - standard order is fine
            filter_order = self.config.filter_order
        
        # Adapt filter bandwidth based on spectral characteristics
        base_width = (freq_range[1] - freq_range[0]) / (sr / 2)
        
        if freq_analysis['spectral_kurtosis'] > 5:
            # High kurtosis (peaky spectrum) - narrower filter
            width = base_width * 0.8
        else:
            # Normal or flat spectrum - standard width
            width = base_width * self.config.filter_width_multiplier
        
        # Calculate adaptive bounds
        center_freq = (freq_range[0] + freq_range[1]) / 2 / (sr / 2)
        low_bound = max(0.01, center_freq - width/2)
        high_bound = min(0.99, center_freq + width/2)
        
        try:
            b, a = signal.butter(filter_order, [low_bound, high_bound], btype='bandstop')
            return b, a
        except Exception as e:
            logger.warning(f"Adaptive filter design failed: {e}")
            # Fallback to standard design
            return signal.butter(self.config.filter_order, [low_bound, high_bound], btype='bandstop')
    
    def perceptual_masking_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply perceptually-motivated processing to preserve audio quality."""
        try:
            # Calculate psychoacoustic masking threshold
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # Rough approximation of masking threshold using spectral envelope
            masking_threshold = np.percentile(magnitude, 10, axis=1, keepdims=True)
            
            # Only process frequencies where modifications won't be perceptible
            safe_modification_mask = magnitude > masking_threshold * 3
            
            # Apply gentle modifications only where they're masked
            modified_stft = stft.copy()
            noise_level = self.config.noise_level * 0.5  # Reduced noise for perceptual masking
            
            for freq_idx in range(stft.shape[0]):
                for time_idx in range(stft.shape[1]):
                    if safe_modification_mask[freq_idx, time_idx]:
                        # Add small amount of noise only where it's perceptually masked
                        noise = np.random.randn() * noise_level * magnitude[freq_idx, time_idx]
                        modified_stft[freq_idx, time_idx] += noise
            
            # Convert back to time domain
            return librosa.istft(modified_stft, hop_length=512)
        
        except Exception as e:
            logger.warning(f"Perceptual masking failed: {e}")
            return audio


# Constants
KNOWN_AI_TAG_PATTERNS = [
    r'(?i)suno',
    r'(?i)openai',
    r'(?i)anthropic',
    r'(?i)stability',
    r'(?i)midjourney',
    r'(?i)synthesia',
    r'(?i)ai[_.-]?gen',
    r'(?i)ml[_.-]?gen',
    r'(?i)model',
    r'(?i)dalle',
    r'(?i)chatgpt',
    r'(?i)gpt',
    r'(?i)elevenlabs',
    r'(?i)neural',
    r'(?i)deepfake',
    r'(?i)synthetic',
    r'(?i)generated',
    r'(?i)claude',
    r'(?i)voice\.ai',
    r'(?i)murf',
    r'(?i)descript',
    r'(?i)resemble\.ai',
    r'(?i)play\.ht',
    r'(?i)uberduck',
    r'(?i)replica',
    r'(?i)wav2lip',
    r'(?i)tortoise',
    r'(?i)bark\.ai',
    r'(?i)vall[_.-]?e',
    r'(?i)transformers'
]

KNOWN_CUSTOM_CHUNKS = [
    'sunf', 'aicm', 'ainf', 'genm', 'gens', 'modl', 'crid', 'meta', 'json', 
    'suna', 'elev', 'mlmd', 'gena', 'orig', 'prom', 'seed', 'sigf', 'uuid',
    'lmd', 'gnmd', 'aiid', 'gptm', 'opmd', 'mrkr', 'fing', 'wtrm', 'hash',
    'cgnr', 'gpmd', 'anth', 'stbl', 'midj', 'voai'
]

# Frequencies used by common watermarking techniques
POTENTIAL_WATERMARK_FREQS = [
    [19500, 20000],  # High-frequency standard
    [15000, 17000],  # ElevenLabs/similar range
    [50, 200],       # Low-frequency steganography
    [8000, 8500],    # Mid-range markers
    [12000, 12500]   # Secondary watermark range
]


class AdvancedAudioAnalysis:
    """Advanced audio analysis methods for sophisticated fingerprint detection."""
    
    @staticmethod
    def calculate_spectral_entropy(audio: np.ndarray, sr: int, frame_length: int = 2048) -> np.ndarray:
        """Calculate spectral entropy for detecting artificial patterns."""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=frame_length, hop_length=frame_length//4)
            magnitude = np.abs(stft)
            
            # Normalize each frame
            magnitude_norm = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-10)
            
            # Calculate entropy for each frame
            entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10), axis=0)
            
            return entropy
        except Exception as e:
            logger.warning(f"Spectral entropy calculation failed: {e}")
            return np.array([])
    
    @staticmethod
    def detect_periodic_patterns(audio: np.ndarray, sr: int, min_period: float = 0.1, 
                                max_period: float = 2.0) -> Dict[str, Any]:
        """Detect periodic patterns using advanced autocorrelation."""
        try:
            # Calculate autocorrelation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Convert period bounds to samples
            min_samples = int(min_period * sr)
            max_samples = int(max_period * sr)
            
            # Only consider relevant range
            if max_samples < len(autocorr):
                autocorr_slice = autocorr[min_samples:max_samples]
            else:
                autocorr_slice = autocorr[min_samples:]
            
            if len(autocorr_slice) == 0:
                return {'periodicity_strength': 0.0, 'dominant_period': None}
            
            # Find peaks
            peaks, _ = signal.find_peaks(autocorr_slice, height=0.1*np.max(autocorr_slice))
            
            if len(peaks) == 0:
                return {'periodicity_strength': 0.0, 'dominant_period': None}
            
            # Calculate periodicity strength
            max_peak_idx = np.argmax(autocorr_slice[peaks])
            dominant_period_samples = peaks[max_peak_idx] + min_samples
            dominant_period_time = dominant_period_samples / sr
            
            # Strength is the ratio of the dominant peak to the signal energy
            periodicity_strength = autocorr_slice[peaks[max_peak_idx]] / autocorr[0]
            
            return {
                'periodicity_strength': periodicity_strength,
                'dominant_period': dominant_period_time,
                'num_peaks': len(peaks)
            }
        except Exception as e:
            logger.warning(f"Periodic pattern detection failed: {e}")
            return {'periodicity_strength': 0.0, 'dominant_period': None}
    
    @staticmethod
    def analyze_frequency_distribution(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze frequency distribution for detecting artificial characteristics."""
        try:
            # Compute power spectral density
            freqs, psd = signal.welch(audio, sr, nperseg=min(2048, len(audio)//4))
            
            # Calculate various distribution metrics
            psd_norm = psd / np.sum(psd)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd_norm)
            
            # Spectral spread
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
            
            # Spectral skewness and kurtosis
            if spectral_spread > 0:
                spectral_skewness = np.sum(((freqs - spectral_centroid) ** 3) * psd_norm) / (spectral_spread ** 3)
                spectral_kurtosis = np.sum(((freqs - spectral_centroid) ** 4) * psd_norm) / (spectral_spread ** 4)
            else:
                spectral_skewness = 0.0
                spectral_kurtosis = 3.0
            
            # High frequency energy ratio
            nyquist = sr / 2
            high_freq_mask = freqs > 0.8 * nyquist
            high_freq_energy = np.sum(psd[high_freq_mask]) / np.sum(psd) if np.sum(psd) > 0 else 0.0
            
            return {
                'spectral_centroid': spectral_centroid,
                'spectral_spread': spectral_spread,
                'spectral_skewness': spectral_skewness,
                'spectral_kurtosis': spectral_kurtosis,
                'high_freq_energy_ratio': high_freq_energy
            }
        except Exception as e:
            logger.warning(f"Frequency distribution analysis failed: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0,
                'spectral_skewness': 0.0,
                'spectral_kurtosis': 3.0,
                'high_freq_energy_ratio': 0.0
            }
    
    @staticmethod
    def calculate_complexity_measures(audio: np.ndarray) -> Dict[str, float]:
        """Calculate complexity measures to detect artificial generation."""
        try:
            # Sample entropy (simplified version)
            def sample_entropy(data, m=2, r=None):
                if r is None:
                    r = 0.2 * np.std(data)
                
                N = len(data)
                if N < m + 1:
                    return 0.0
                
                patterns = [0, 0]
                
                # Sample a subset for performance
                sample_size = min(1000, N - m)
                indices = np.random.choice(N - m, sample_size, replace=False)
                
                for i in indices:
                    template = data[i:i + m]
                    matches_m = 0
                    matches_m_plus_1 = 0
                    
                    for j in range(N - m):
                        if j != i:
                            dist = np.max(np.abs(template - data[j:j + m]))
                            if dist < r:
                                matches_m += 1
                                if j < N - m and i < N - m:
                                    dist_plus = np.max(np.abs(data[i:i + m + 1] - data[j:j + m + 1]))
                                    if dist_plus < r:
                                        matches_m_plus_1 += 1
                    
                    patterns[0] += matches_m
                    patterns[1] += matches_m_plus_1
                
                if patterns[0] == 0 or patterns[1] == 0:
                    return 0.0
                
                return -np.log(patterns[1] / patterns[0])
            
            # Calculate sample entropy
            sample_ent = sample_entropy(audio[:min(5000, len(audio))])  # Limit for performance
            
            # Spectral flatness (measure of noisiness vs tonality)
            stft = np.abs(librosa.stft(audio, n_fft=1024))
            spectral_flatness = np.mean(stats.gmean(stft + 1e-10, axis=0) / 
                                      (np.mean(stft, axis=0) + 1e-10))
            
            return {
                'sample_entropy': sample_ent if np.isfinite(sample_ent) else 0.0,
                'spectral_flatness': spectral_flatness if np.isfinite(spectral_flatness) else 0.0
            }
        
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return {
                'sample_entropy': 0.0,
                'spectral_flatness': 0.0
            }


def get_hann_window(size: int) -> np.ndarray:
    """Get a Hann window in a cross-compatible way across scipy versions."""
    try:
        # Try scipy.signal.windows.hann (newer versions)
        if windows is not None:
            return windows.hann(size)
        else:
            # Fallback to numpy.hanning for older versions
            return np.hanning(size)
    except:
        # Final fallback to numpy.hanning
        return np.hanning(size)


@dataclass
class ProcessingConfig:
    """Configuration parameters for different processing intensity levels."""
    # Processing level
    processing_level: str = "moderate"  # gentle, moderate, aggressive, extreme
    
    # Watermark removal parameters
    filter_order: int = 4  # Order of bandstop filters
    filter_width_multiplier: float = 1.5  # How wide to make the filter bands
    noise_level: float = 0.001  # Level of noise added to high frequencies
    skip_low_freq_threshold: int = 100  # Skip filters below this frequency
    
    # Pattern normalization parameters
    timing_stretch_range: float = 0.02  # ±percentage for time stretching (0.02 = ±2%)
    distribution_noise_level: float = 0.001  # Noise added for distribution normalization
    harmonic_distortion_amount: float = 0.03  # Amount of soft clipping
    phase_variance: float = 0.02  # Phase adjustment amount
    micro_dynamics_amount: float = 0.005  # Micro-dynamics variation amount
    
    # Timing variations parameters
    timing_variation_range: float = 0.01  # Random variation range (0.01 = ±1%)
    segment_overlap_ratio: float = 0.5  # Overlap ratio for processing segments
    
    # General parameters
    enable_watermark_removal: bool = True
    enable_pattern_normalization: bool = True
    enable_timing_variations: bool = True
    enable_harmonic_adjustments: bool = True

    @classmethod
    def get_profile(cls, level: str) -> 'ProcessingConfig':
        """Get predefined configuration profiles."""
        profiles = {
            'gentle': cls(
                processing_level='gentle',
                filter_order=2,
                filter_width_multiplier=0.8,
                noise_level=0.0005,
                skip_low_freq_threshold=200,
                timing_stretch_range=0.005,  # ±0.5%
                distribution_noise_level=0.0005,
                harmonic_distortion_amount=0.01,
                phase_variance=0.01,
                micro_dynamics_amount=0.002,
                timing_variation_range=0.005,  # ±0.5%
                enable_harmonic_adjustments=False,  # Skip most aggressive processing
            ),
            'moderate': cls(
                processing_level='moderate',
                filter_order=3,
                filter_width_multiplier=1.2,
                noise_level=0.0008,
                skip_low_freq_threshold=150,
                timing_stretch_range=0.015,  # ±1.5%
                distribution_noise_level=0.0008,
                harmonic_distortion_amount=0.02,
                phase_variance=0.015,
                micro_dynamics_amount=0.003,
                timing_variation_range=0.008,  # ±0.8%
            ),
            'aggressive': cls(
                processing_level='aggressive',
                filter_order=4,
                filter_width_multiplier=1.5,
                noise_level=0.001,
                skip_low_freq_threshold=100,
                timing_stretch_range=0.02,  # ±2%
                distribution_noise_level=0.001,
                harmonic_distortion_amount=0.03,
                phase_variance=0.02,
                micro_dynamics_amount=0.005,
                timing_variation_range=0.01,  # ±1%
            ),
            'extreme': cls(
                processing_level='extreme',
                filter_order=6,
                filter_width_multiplier=2.0,
                noise_level=0.002,
                skip_low_freq_threshold=50,
                timing_stretch_range=0.04,  # ±4%
                distribution_noise_level=0.002,
                harmonic_distortion_amount=0.05,
                phase_variance=0.03,
                micro_dynamics_amount=0.008,
                timing_variation_range=0.02,  # ±2%
            )
        }
        
        return profiles.get(level, profiles['moderate'])

@dataclass
class ProcessingStats:
    """Track statistics about the audio processing."""
    files_processed: int = 0
    metadata_removed: Dict[str, List[str]] = None
    watermarks_detected: int = 0
    patterns_normalized: int = 0
    timing_adjustments: int = 0
    processing_level: str = "moderate"
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    chunks_processed: int = 0
    cache_hits: int = 0
    
    def __post_init__(self):
        if self.metadata_removed is None:
            self.metadata_removed = {}
    
    def add_timing(self, operation: str, duration: float):
        """Add timing information for performance monitoring."""
        if not hasattr(self, 'operation_timings'):
            self.operation_timings = {}
        self.operation_timings[operation] = duration


class AudioFingerprint:
    """Detector for known audio fingerprinting techniques."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.log_details = []
        self.analysis = AdvancedAudioAnalysis()
        self.processor = AudioProcessor(config)
        self._cache = {}  # Cache for expensive computations
        self.suno_detector = SunoWatermarkDetector()  # Enhanced Suno detection
    
    def detect_spectral_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect potential spectral watermarks in the audio using advanced analysis."""
        detected = []
        
        # Input validation
        if len(audio_data) == 0:
            logger.warning("Empty audio data provided to watermark detection")
            return detected
        
        # Check cache first
        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()[:16]
        cache_key = f"watermarks_{audio_hash}_{sample_rate}"
        if cache_key in self._cache:
            logger.debug("Using cached watermark detection results")
            return self._cache[cache_key]
        
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=1)
            else:
                audio_mono = audio_data
            
            logger.info(f"Analyzing {len(audio_mono)} samples at {sample_rate} Hz")
            
            # For very large files, downsample for initial analysis
            if len(audio_mono) > sample_rate * 300:  # > 5 minutes
                logger.info("Large file detected - using decimated analysis")
                decimation_factor = max(1, len(audio_mono) // (sample_rate * 120))  # Target ~2 minutes
                audio_analysis = audio_mono[::decimation_factor]
                analysis_sr = sample_rate // decimation_factor
            else:
                audio_analysis = audio_mono
                analysis_sr = sample_rate
            
            # Enhanced spectral analysis with adaptive parameters
            nperseg = min(4096, max(512, len(audio_analysis) // 16))  # Adaptive window size
            noverlap = nperseg // 2
            
            freqs, times, spectrogram = signal.spectrogram(
                audio_analysis, 
                fs=analysis_sr,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='spectrum'
            )
            
            # Calculate spectral entropy to detect artificial patterns
            entropy = self.analysis.calculate_spectral_entropy(audio_analysis, analysis_sr)
            
            # Advanced frequency distribution analysis
            freq_stats = self.analysis.analyze_frequency_distribution(audio_analysis, analysis_sr)
            
            # Detect anomalies in high-frequency energy (common watermark location)
            if freq_stats['high_freq_energy_ratio'] > 0.1:  # Unusually high energy in high frequencies
                detected.append({
                    'type': 'high_freq_anomaly',
                    'energy_ratio': freq_stats['high_freq_energy_ratio'],
                    'freq_range': [0.8 * analysis_sr / 2, analysis_sr / 2],
                    'confidence': min(freq_stats['high_freq_energy_ratio'] * 10, 1.0)
                })
            
                        # Check for spectral entropy anomalies
            if len(entropy) > 0:
                entropy_std = np.std(entropy)
                entropy_mean = np.mean(entropy)
                
                # Low entropy regions might indicate watermarks
                low_entropy_mask = entropy < (entropy_mean - 2 * entropy_std)
                if np.sum(low_entropy_mask) > len(entropy) * 0.1:  # >10% of frames have low entropy
                    detected.append({
                        'type': 'entropy_anomaly',
                        'low_entropy_ratio': np.sum(low_entropy_mask) / len(entropy),
                        'confidence': min(np.sum(low_entropy_mask) / len(entropy) * 5, 1.0)
                    })
            
            # Look for anomalies in frequency bands commonly used for watermarking
            for freq_range in POTENTIAL_WATERMARK_FREQS:
                # Scale frequency range for analysis sample rate
                scaled_freq_range = [f * analysis_sr / sample_rate for f in freq_range]
                if scaled_freq_range[1] > analysis_sr / 2:
                    continue  # Skip if beyond Nyquist frequency
                    
                # Find the indices for our scaled frequency range
                freq_mask = (freqs >= scaled_freq_range[0]) & (freqs <= scaled_freq_range[1])
                if not np.any(freq_mask):
                    continue
                    
                band_energy = np.mean(spectrogram[freq_mask], axis=0)
                
                # Calculate statistics
                mean_energy = np.mean(band_energy)
                std_energy = np.std(band_energy)
                
                # Look for periodic patterns in this band
                if std_energy > 0:
                    normalized = (band_energy - mean_energy) / std_energy
                    autocorr = np.correlate(normalized, normalized, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # Check for peaks in autocorrelation (suggesting periodicity)
                    if len(autocorr) > 1:
                        peaks, _ = signal.find_peaks(autocorr, height=0.5, distance=5)
                        if len(peaks) >= 3:  # At least 3 regular peaks suggests a pattern
                            # Scale back to original frequency range
                            original_freq_range = freq_range
                            detected.append({
                                'freq_range': original_freq_range,
                                'peak_count': len(peaks),
                                'regularity': np.std(np.diff(peaks)),
                                'strength': np.max(autocorr[peaks])
                            })
                            
                # Also check for constant energy in bands where human audio would vary
                if scaled_freq_range[0] > 15000 * analysis_sr / sample_rate and np.std(band_energy) / (mean_energy + 1e-10) < 0.1:
                    detected.append({
                        'freq_range': freq_range,  # Use original frequency range
                        'type': 'constant_energy',
                        'variation': np.std(band_energy) / (mean_energy + 1e-10)
                    })
                    
        except Exception as e:
            logger.error(f"Spectral watermark detection failed: {e}")
            
        # Enhanced Suno-specific detection
        try:
            suno_watermarks = self.suno_detector.detect_suno_watermarks(audio_analysis, analysis_sr)
            detected.extend(suno_watermarks)
            logger.info(f"Suno detector found {len(suno_watermarks)} additional watermarks")
        except Exception as e:
            logger.warning(f"Suno detector failed: {e}")
        
        # Cache the results for future use
        self._cache[cache_key] = detected
        
        return detected
        
    def detect_statistical_patterns(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect statistical anomalies that could indicate AI generation using advanced analysis."""
        detected = []
        
        # Input validation
        if len(audio_data) == 0:
            logger.warning("Empty audio data provided to statistical pattern detection")
            return detected
        
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=1)
            else:
                audio_mono = audio_data
            
            # Advanced complexity analysis
            complexity_measures = self.analysis.calculate_complexity_measures(audio_mono)
            
            # Low sample entropy indicates artificial/repetitive patterns
            if complexity_measures['sample_entropy'] < 0.5:
                detected.append({
                    'type': 'low_complexity',
                    'sample_entropy': complexity_measures['sample_entropy'],
                    'confidence': 1.0 - complexity_measures['sample_entropy']
                })
            
                        # Very high spectral flatness can indicate artificial noise or processing
            if complexity_measures['spectral_flatness'] > 0.8:
                detected.append({
                    'type': 'artificial_flatness',
                    'spectral_flatness': complexity_measures['spectral_flatness'],
                    'confidence': complexity_measures['spectral_flatness']
                })
            
            # Check for unusually perfect timing
            zero_crossings = np.where(np.diff(np.signbit(audio_mono)))[0]
            if len(zero_crossings) > 0:
                # Calculate intervals between zero crossings
                intervals = np.diff(zero_crossings)
                
                # Look for too-regular patterns
                if len(intervals) > 100:
                    regularity = np.std(intervals) / np.mean(intervals)
                    if regularity < 0.2:  # Human audio is rarely this regular
                        detected.append({
                            'type': 'regular_timing',
                            'regularity': regularity
                        })
            
            # Check for unnatural amplitude distribution
            hist, _ = np.histogram(audio_mono, bins=100, range=(-1, 1), density=True)
            skewness = np.sum((hist - np.mean(hist))**3) / (len(hist) * np.std(hist)**3)
            kurtosis = np.sum((hist - np.mean(hist))**4) / (len(hist) * np.std(hist)**4) - 3
            
            # Perfect gaussian is unusual in real audio
            if abs(skewness) < 0.1 and abs(kurtosis) < 0.2:
                detected.append({
                    'type': 'perfect_distribution',
                    'skewness': skewness,
                    'kurtosis': kurtosis
                })
                
            # Check for lack of harmonics in frequency domain
            fft_data = np.abs(np.fft.rfft(audio_mono))
            if len(fft_data) > 1000:
                # Real audio typically has strong harmonic relationships
                peaks, _ = signal.find_peaks(fft_data, height=np.mean(fft_data)*2, distance=5)
                if len(peaks) > 0:
                    # Check if harmonics are too perfect or missing
                    peak_freqs = peaks.astype(float)
                    ratios = []
                    
                    for i in range(len(peak_freqs)-1):
                        for j in range(i+1, min(i+5, len(peak_freqs))):
                            ratio = peak_freqs[j] / (peak_freqs[i] + 1e-10)
                            ratios.append(ratio)
                    
                    if len(ratios) > 5:
                        has_harmonics = any(0.95 < r < 1.05 or 1.95 < r < 2.05 or 2.95 < r < 3.05 for r in ratios)
                        if not has_harmonics:
                            detected.append({'type': 'missing_harmonics'})
                        
                        # Or check if they're too perfect (exact integer multiples)
                        perfect_count = sum(1 for r in ratios if abs(round(r) - r) < 0.01)
                        if perfect_count > len(ratios) / 2:
                            detected.append({'type': 'too_perfect_harmonics'})
                            
        except Exception as e:
            logger.error(f"Statistical pattern detection failed: {e}")
        
        return detected

    def detect_timing_anomalies(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect timing anomalies that might indicate AI generation."""
        detected = []
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data
            
        # Detect onset features (starts of notes/syllables)
        onset_env = librosa.onset.onset_strength(
            y=audio_mono, 
            sr=sample_rate,
            hop_length=512
        )
        
        # Check for too-regular onset timing (machine-like precision)
        if len(onset_env) > 50:
            # Find onsets
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env, 
                sr=sample_rate,
                hop_length=512,
                units='time'
            )
            
            if len(onsets) > 5:
                # Calculate intervals and their regularity
                intervals = np.diff(onsets)
                
                # Coefficient of variation (lower means more mechanical)
                cv = np.std(intervals) / np.mean(intervals)
                
                if cv < 0.1:  # Real speech/music typically has more variation
                    detected.append({
                        'type': 'mechanical_timing',
                        'cv': cv
                    })
                    
                # Check for quantization (exact multiples of a base interval)
                base_interval = np.min(intervals)
                quantized_count = sum(1 for i in intervals if abs(round(i/base_interval) - i/base_interval) < 0.05)
                
                if quantized_count > len(intervals) * 0.7:  # >70% are multiples
                    detected.append({
                        'type': 'quantized_timing',
                        'percent_quantized': quantized_count / len(intervals)
                    })
        
        return detected


def get_ai_metadata_signatures() -> Set[str]:
    """Build a set of known AI metadata signatures and patterns."""
    signatures = set()
    
    # Add standard patterns
    for pattern in KNOWN_AI_TAG_PATTERNS:
        signatures.add(pattern)
    
    # Add common field names that might contain AI information
    field_names = [
        'generator', 'created_by', 'software', 'source', 'origin',
        'model', 'ai_model', 'voice_model', 'synthesizer', 'encoder',
        'generation', 'synthesized', 'voice_id', 'voice_preset',
        'prompt', 'text_prompt', 'parameters', 'settings', 'config',
        'version', 'api_version', 'timestamp', 'uuid', 'session_id',
        'license', 'terms', 'usage_rights', 'watermark', 'fingerprint'
    ]
    
    for field in field_names:
        signatures.add(f'(?i){field}')
    
    return signatures


def clean_metadata_comprehensive(filepath: str, output_path: Optional[str] = None, 
                               aggressive: bool = False) -> Tuple[str, Dict[str, List[str]]]:
    """Remove all metadata from audio files with comprehensive approach for all formats."""
    temp_file = None
    removed_metadata = {}
    
    if not output_path:
        # Create a temporary file for processing
        temp_fd, temp_file = tempfile.mkstemp(suffix=os.path.splitext(filepath)[1])
        os.close(temp_fd)
        output_path = temp_file
    
    # Copy the file first
    shutil.copy2(filepath, output_path)
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # Get known AI patterns
    ai_signatures = get_ai_metadata_signatures()
    
    try:
        # Process based on file type
        if file_ext == '.mp3':
            # Use ID3 for MP3 files
            try:
                audio = MP3(output_path)
                removed = []
                
                # Track removed tags
                if audio.tags:
                    # First pass: identify AI-related tags
                    ai_tags = []
                    for key in list(audio.tags.keys()):
                        tag_str = str(audio.tags[key])
                        
                        # Check if any AI signature matches
                        if any(re.search(pattern, tag_str) for pattern in ai_signatures):
                            ai_tags.append(key)
                            removed.append(f"{key}: {tag_str}")
                    
                    # Delete AI-related tags first
                    for key in ai_tags:
                        del audio.tags[key]
                    
                    # If aggressive, remove all tags
                    if aggressive:
                        audio.tags = None
                        audio.save()
                    else:
                        # Delete minimal identifying metadata
                        for key in list(audio.tags.keys()):
                            if any(x in key.upper() for x in ['COMM', 'OWNE', 'PRIV', 'USER', 'UFID', 'POPM', 'GEOB']):
                                removed.append(f"{key}: {audio.tags[key]}")
                                del audio.tags[key]
                        audio.save()
                    
                    if removed:
                        removed_metadata['mp3_id3'] = removed
            except Exception as e:
                print(f"ID3 processing error: {e}")
            
            # Try again with EasyID3 for additional fields
            try:
                easy = EasyID3(output_path)
                removed = []
                
                if easy:
                    for key in list(easy.keys()):
                        removed.append(f"{key}: {easy[key]}")
                    
                    easy.delete()
                    easy.save()
                    
                    if removed:
                        removed_metadata['mp3_easyid3'] = removed
            except Exception:
                pass  # EasyID3 might not be applicable
            
        elif file_ext == '.wav':
            # Process WAV files
            try:
                audio = WAVE(output_path)
                removed = []
                
                # Check for a LIST INFO chunk which might contain metadata
                if hasattr(audio, '_tags') and audio._tags:
                    for key, value in list(audio._tags.items()):
                        removed.append(f"{key}: {value}")
                    audio._tags = {}
                
                # Remove custom chunks that might contain fingerprinting
                for key in list(audio.keys()):
                    # Check for known custom chunks
                    if any(chunk.lower() in key.lower() for chunk in KNOWN_CUSTOM_CHUNKS):
                        removed.append(f"Custom chunk: {key}")
                        del audio[key]
                    
                    # Check for any chunks with AI-related text
                    elif isinstance(audio[key], bytes):
                        chunk_text = audio[key].decode('utf-8', 'ignore')
                        if any(re.search(pattern, chunk_text) for pattern in ai_signatures):
                            removed.append(f"AI-related chunk: {key}")
                            del audio[key]
                
                audio.save()
                
                if removed:
                    removed_metadata['wav_chunks'] = removed
                
            except Exception as e:
                print(f"WAV processing error: {e}")
            
            # Also process with wave module for more thorough cleaning
            try:
                # Re-write the WAV file with only essential chunks
                with wave.open(output_path, 'rb') as wf:
                    params = wf.getparams()
                    frames = wf.readframes(wf.getnframes())
                
                with wave.open(output_path + '.clean', 'wb') as wf:
                    wf.setparams(params)
                    wf.writeframes(frames)
                
                # Replace the original with the cleaned version
                os.replace(output_path + '.clean', output_path)
                removed_metadata['wav_rewrite'] = ["Complete WAV rewrite (strips all non-standard chunks)"]
                
            except Exception as e:
                print(f"WAV rewrite error: {e}")
                
        elif file_ext == '.flac':
            try:
                audio = FLAC(output_path)
                removed = []
                
                # Track and remove FLAC metadata
                if audio.tags:
                    for key in list(audio.tags.keys()):
                        removed.append(f"{key}: {audio[key]}")
                    
                    audio.delete()
                    audio.save()
                    
                    if removed:
                        removed_metadata['flac_tags'] = removed
                        
                # Handle FLAC pictures and application blocks which might contain fingerprints
                if audio.pictures:
                    removed_metadata['flac_pictures'] = [f"Removed {len(audio.pictures)} embedded pictures"]
                    audio.clear_pictures()
                    audio.save()
                
            except Exception as e:
                print(f"FLAC processing error: {e}")
                
        elif file_ext == '.aiff' or file_ext == '.aif':
            try:
                audio = AIFF(output_path)
                removed = []
                
                if audio.tags:
                    for key in list(audio.tags.keys()):
                        removed.append(f"{key}: {audio.tags[key]}")
                    
                    audio.tags = None
                    audio.save()
                    
                    if removed:
                        removed_metadata['aiff_tags'] = removed
            except Exception as e:
                print(f"AIFF processing error: {e}")
        
        # If we're being aggressive, also check for any binary metadata or watermarks
        if aggressive:
            # Read the entire file and search for text patterns
            try:
                with open(output_path, 'rb') as f:
                    content = f.read()
                    
                text_content = content.decode('utf-8', 'ignore')
                removed = []
                
                # Search for AI-related text
                for pattern in ai_signatures:
                    matches = re.finditer(pattern, text_content)
                    for match in matches:
                        # Extract some context around the match
                        start = max(0, match.start() - 20)
                        end = min(len(text_content), match.end() + 20)
                        context = text_content[start:end].replace('\x00', '')
                        
                        if context.strip():
                            removed.append(f"Binary pattern: {context}")
                
                if removed:
                    removed_metadata['binary_metadata'] = removed
            except Exception as e:
                print(f"Binary search error: {e}")
        
        # If we created a temp file, replace the original
        if temp_file:
            shutil.move(temp_file, filepath)
            output_path = filepath
            temp_file = None
            
        return output_path, removed_metadata
        
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        # Clean up temp file if we created one
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        return filepath, removed_metadata


def remove_spectral_watermarks(audio_path: str, output_path: str, 
                              detector: AudioFingerprint) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Remove potential spectral watermarks from an audio file.
    
    This function:
    1. Detects potential watermarks
    2. Applies targeted filters to remove them
    3. Saves a clean version
    """
    # Input validation
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load the audio file
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        logger.debug(f"Loaded audio: shape={y.shape if hasattr(y, 'shape') else len(y)}, sr={sr}")
    except Exception as e:
        logger.error(f"Error loading audio for watermark removal: {e}")
        shutil.copy2(audio_path, output_path)
        return output_path, []
    
    # Check if we have a stereo file
    is_stereo = len(y.shape) > 1
    
    # Convert to mono for analysis if needed
    if is_stereo:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y
        
    # Detect watermarks
    watermarks = detector.detect_spectral_watermarks(y_mono, sr)
    detector.log_details.append(f"Found {len(watermarks)} potential watermarks")
    
    if not watermarks:
        # No watermarks detected, just copy the file
        shutil.copy2(audio_path, output_path)
        return output_path, []
    
    # Process stereo or mono based on input
    if is_stereo:
        processed = y.copy()  # Start with original audio instead of zeros
        
        # Process each channel separately
        for i in range(y.shape[0]):
            channel_result = apply_watermark_removal(y[i], sr, watermarks, detector.config)
            # Validate channel result before assignment
            if np.any(np.abs(channel_result) > 1e-10):  # Check if not completely silent
                processed[i] = channel_result
            else:
                logger.warning(f"Channel {i} processing resulted in silence, keeping original")
                processed[i] = y[i]  # Keep original if processing failed
    else:
        processed = apply_watermark_removal(y, sr, watermarks, detector.config)
        # Validate mono result
        if not np.any(np.abs(processed) > 1e-10):  # Check if not completely silent
            logger.warning("Mono processing resulted in silence, keeping original")
            processed = y  # Keep original if processing failed
    
    # Final validation before saving
    if not np.any(np.abs(processed) > 1e-10):
        logger.error("Final processed audio is silent, keeping original")
        processed = y
    
    # Normalize to prevent clipping but preserve dynamics
    max_val = np.max(np.abs(processed))
    if max_val > 0.95:
        processed = processed * (0.95 / max_val)
    
    # Save the processed audio
    sf.write(output_path, processed.T if is_stereo else processed, sr)
    
    return output_path, watermarks


def apply_watermark_removal(audio: np.ndarray, sr: int, 
                           watermarks: List[Dict[str, Any]], 
                           config: ProcessingConfig) -> np.ndarray:
    """Apply filters to remove detected watermarks."""
    if not config.enable_watermark_removal:
        return audio
        
    result = audio.copy()
    processor = AudioProcessor(config)
    suno_detector = SunoWatermarkDetector()
    aggressive_remover = AggressiveWatermarkRemover()
    
    # Initialize SOTA remover based on processing level
    quality_mode = "conservative"
    if config.processing_level == "aggressive":
        quality_mode = "balanced"
    elif config.processing_level == "extreme":
        quality_mode = "aggressive"
    
    sota_remover = StateOfTheArtWatermarkRemover(quality_preservation_mode=quality_mode)
    
    # Group watermarks by frequency range to optimize filtering
    freq_ranges = []
    for watermark in watermarks:
        freq_range = watermark.get('freq_range')
        if freq_range and freq_range not in freq_ranges:
            freq_ranges.append(freq_range)
    
    logger.debug(f"Applying filters for {len(freq_ranges)} frequency ranges")
    
    for freq_range in freq_ranges:
        low_freq, high_freq = freq_range
        
        # Convert to normalized frequency
        nyquist = sr / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Skip if outside Nyquist range
        if high_norm > 1.0:
            continue
            
        # Skip very low frequency filters that can cause numerical instability
        if low_freq < config.skip_low_freq_threshold:
            continue
            
        # Use adaptive filter design for better performance
        try:
            b, a = processor.adaptive_filter_design(result, sr, freq_range)
            
            # Apply the filter with error checking
            filtered = signal.filtfilt(b, a, result)
            
            # Check for numerical issues
            if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                logger.warning(f"Filter for range {freq_range} produced invalid values, skipping")
                continue
            
            result = filtered
            logger.debug(f"Applied filter for frequency range {freq_range}")
            
        except Exception as e:
            logger.warning(f"Failed to apply filter for range {freq_range}: {e}")
            continue
    
    # If we're removing high-frequency watermarks, add a small amount of noise
    # to defeat fingerprinting that relies on absence of higher frequencies
    high_watermarks = [w for w in watermarks if w.get('freq_range') and w['freq_range'][0] > 15000]
    
    if high_watermarks:
        # Generate a small amount of shaped noise
        noise = np.random.randn(len(result)) * config.noise_level
        
        # Shape the noise to only affect high frequencies
        nyquist = sr / 2
        cutoff = 15000 / nyquist
        b, a = signal.butter(2, cutoff, btype='highpass')
        shaped_noise = signal.filtfilt(b, a, noise)
        
        # Add the noise
        result += shaped_noise
    
    # Apply enhanced Suno-specific removal
    try:
        suno_watermarks = [w for w in watermarks if w.get('method') in [
            'neural_pattern_analysis', 'frequency_analysis', 'energy_comparison',
            'temporal_analysis', 'phase_analysis', 'statistical_analysis'
        ]]
        
        if suno_watermarks:
            logger.debug(f"Applying enhanced Suno removal for {len(suno_watermarks)} watermarks")
            result = suno_detector.remove_suno_watermarks(result, sr, suno_watermarks)
    except Exception as e:
        logger.warning(f"Enhanced Suno removal failed: {e}")
    
    # Apply SOTA removal for comprehensive watermark elimination
    try:
        logger.info(f"Applying SOTA removal for {len(watermarks)} detected watermarks")
        result = sota_remover.remove_watermarks_sota(result, sr, watermarks)
    except Exception as e:
        logger.warning(f"SOTA removal failed: {e}")
        
        # Fallback to aggressive removal if SOTA fails
        if len(watermarks) > 50:
            try:
                logger.info(f"Falling back to aggressive removal")
                result = aggressive_remover.remove_watermarks_aggressive(result, sr, watermarks)
            except Exception as e:
                logger.warning(f"Aggressive removal fallback failed: {e}")
    
    return result


def normalize_ai_patterns(audio_path: str, output_path: str, 
                         detector: AudioFingerprint) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Normalize statistical patterns in the audio that might identify it as AI-generated.
    
    This applies various techniques to make the audio more natural:
    1. Subtle timing variations
    2. Frequency distribution normalization
    3. Dynamic range adjustments
    """
    try:
        # Load the audio
        y, sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        print(f"Error loading audio for pattern normalization: {e}")
        shutil.copy2(audio_path, output_path)
        return output_path, []
    
    # Check if we have a stereo file
    is_stereo = len(y.shape) > 1
    
    # Convert to mono for analysis if needed
    if is_stereo:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y
    
    # Detect statistical patterns
    patterns = detector.detect_statistical_patterns(y_mono)
    timing_issues = detector.detect_timing_anomalies(y_mono, sr)
    
    all_issues = patterns + timing_issues
    detector.log_details.append(f"Found {len(all_issues)} statistical anomalies")
    
    if not all_issues:
        # No anomalies detected, just copy the file
        shutil.copy2(audio_path, output_path)
        return output_path, []
    
    # Process stereo or mono based on input
    if is_stereo:
        processed = np.zeros_like(y)
        
        # Process each channel separately with slightly different settings
        # to create more realistic stereo imaging
        for i in range(y.shape[0]):
            # Add small phase differences between channels
            phase_var = 0.02 if i == 0 else -0.02
            processed[i] = apply_pattern_normalization(
                y[i], sr, patterns, timing_issues, detector.config, phase_var=phase_var
            )
    else:
        processed = apply_pattern_normalization(y, sr, patterns, timing_issues, detector.config)
    
    # Save the processed audio
    sf.write(output_path, processed.T if is_stereo else processed, sr)
    
    return output_path, all_issues


def apply_pattern_normalization(audio: np.ndarray, sr: int, 
                               patterns: List[Dict[str, Any]], 
                               timing_issues: List[Dict[str, Any]],
                               config: ProcessingConfig,
                               phase_var: float = 0) -> np.ndarray:
    """Apply corrections to normalize detected AI patterns."""
    if not config.enable_pattern_normalization:
        return audio
        
    result = audio.copy()
    
    # 1. Address timing issues
    has_timing_issues = any(issue['type'] in ['mechanical_timing', 'quantized_timing'] 
                           for issue in timing_issues)
    
    if has_timing_issues:
        # Apply subtle time-domain variations
        # This stretches and compresses small segments randomly
        segment_len = sr // 10  # ~100ms segments
        hop_len = int(segment_len * config.segment_overlap_ratio)
        
        # Break into segments
        segments = []
        for i in range(0, len(result) - segment_len, hop_len):
            segments.append(result[i:i+segment_len])
        
        # Apply random time stretching to each segment
        processed_segments = []
        for segment in segments:
            # Random stretch factor based on config
            stretch_range = config.timing_stretch_range
            stretch_factor = (1.0 - stretch_range) + (2 * stretch_range * random.random())
            stretched = librosa.effects.time_stretch(segment, rate=stretch_factor)
            
            # Ensure consistent length
            if len(stretched) > segment_len:
                stretched = stretched[:segment_len]
            elif len(stretched) < segment_len:
                stretched = np.pad(stretched, (0, segment_len - len(stretched)))
                
            processed_segments.append(stretched)
        
        # Reconstruct with overlap-add
        reconstructed = np.zeros(len(result))
        for i, segment in enumerate(processed_segments):
            pos = i * hop_len
            # Apply triangular window for smooth crossfading
            window = np.bartlett(len(segment))
            end_pos = min(pos + len(segment), len(reconstructed))
            segment_len = end_pos - pos
            if segment_len > 0:
                reconstructed[pos:end_pos] += segment[:segment_len] * window[:segment_len]
        
        # Use the reconstructed result
        result = reconstructed
    
    # 2. Handle distribution anomalies
    has_distribution_issues = any(p['type'] == 'perfect_distribution' for p in patterns)
    
    if has_distribution_issues:
        # Add shaped noise to create more natural distribution
        noise = np.random.randn(len(result)) * config.distribution_noise_level
        
        # Vary the noise level based on signal amplitude
        # (more noise where signal is louder - masked by the signal)
        amplitude_envelope = np.abs(result)
        smoothed_envelope = signal.savgol_filter(amplitude_envelope, 
                                               max(5, min(101, len(result) // 1000) // 2 * 2 + 1), 2)
        shaped_noise = noise * smoothed_envelope
        
        # Add the shaped noise
        result += shaped_noise
    
    # 3. Handle harmonic issues
    has_harmonic_issues = any(p['type'] in ['missing_harmonics', 'too_perfect_harmonics'] 
                              for p in patterns)
    
    if has_harmonic_issues and config.enable_harmonic_adjustments:
        # Apply subtle harmonic distortion to create more natural harmonic relationships
        # This simulates the tiny non-linearities in analog equipment
        
        # Non-linear waveshaping function (subtle soft clipping)
        def soft_clip(x, amount=config.harmonic_distortion_amount):
            return x - amount * np.sin(2 * np.pi * x)
        
        # Apply the non-linearity
        result = soft_clip(result)
        
        # Also apply a tiny bit of phase variance if specified
        if phase_var != 0:
            # Create an all-pass filter for phase adjustment
            # This changes phase without changing amplitude
            b, a = signal.butter(2, 0.5, 'highpass')
            phase_adjustment = signal.lfilter(b, a, result) * (phase_var * config.phase_variance)
            result += phase_adjustment
    
    # 4. Add a touch of natural micro-dynamics
    # Human performances have micro-variations in dynamics that AI often lacks
    env = np.abs(signal.hilbert(result))
    smoothed = signal.savgol_filter(env, 
                                  max(5, min(101, len(result) // 500) // 2 * 2 + 1), 2)
    
    # Create subtle volume variations
    variations = np.sin(np.linspace(0, 20 * np.pi, len(result)) + random.random() * 10) * config.micro_dynamics_amount
    dynamics_adjustment = smoothed * variations
    
    # Apply the adjustment
    result += dynamics_adjustment
    
    # Final normalization to ensure we don't clip
    max_val = np.max(np.abs(result))
    if max_val > 0.99:
        result = result / max_val * 0.99
    
    return result


def process_audio(input_path: str, output_path: Optional[str] = None, 
                 aggressive: bool = False, level: str = None) -> Tuple[str, ProcessingStats]:
    """
    Process an audio file to remove all AI fingerprinting.
    
    Steps:
    1. Remove metadata
    2. Detect and remove spectral watermarks
    3. Normalize statistical patterns
    4. Add human-like variations
    """
    # Input validation
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check file size and warn for very large files
    file_size = os.path.getsize(input_path)
    if file_size > 500 * 1024 * 1024:  # 500MB
        logger.warning(f"Large file detected ({file_size // (1024*1024)}MB). Processing may take significant time and memory.")
    
    # Check file format
    supported_formats = ['.mp3', '.wav', '.flac', '.aiff', '.aif']
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}")
    
    # Determine processing level
    if level is None:
        level = "aggressive" if aggressive else "moderate"
    
    config = ProcessingConfig.get_profile(level)
    stats = ProcessingStats()
    stats.processing_level = level
    detector = AudioFingerprint(config=config)
    
    logger.info(f"Starting processing with {level} level for file: {os.path.basename(input_path)}")
    start_time = time.time()
    
    # Memory monitoring
    if psutil:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    else:
        initial_memory = 0.0
    
    # Create temporary processing directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Get file extension
        file_ext = os.path.splitext(input_path)[1].lower()
        
        # Determine final output path
        if output_path is None:
            output_path = input_path
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Stage 1: Clean metadata
        logger.info("Stage 1: Cleaning metadata...")
        stage_start = time.time()
        temp_metadata = os.path.join(temp_dir, f"stage1_metadata{file_ext}")
        result_path, removed_metadata = clean_metadata_comprehensive(
            input_path, temp_metadata, aggressive
        )
        stats.metadata_removed = removed_metadata
        stats.add_timing("metadata_cleaning", time.time() - stage_start)
        
        # Stage 2: Remove spectral watermarks
        logger.info("Stage 2: Detecting and removing spectral watermarks...")
        stage_start = time.time()
        temp_watermark = os.path.join(temp_dir, f"stage2_watermark{file_ext}")
        result_path, watermarks = remove_spectral_watermarks(
            result_path, temp_watermark, detector
        )
        stats.watermarks_detected = len(watermarks)
        stats.add_timing("watermark_removal", time.time() - stage_start)
        
        # Stage 3: Normalize statistical patterns
        logger.info("Stage 3: Normalizing statistical patterns...")
        stage_start = time.time()
        temp_patterns = os.path.join(temp_dir, f"stage3_patterns{file_ext}")
        result_path, patterns = normalize_ai_patterns(
            result_path, temp_patterns, detector
        )
        stats.patterns_normalized = len(patterns)
        stats.add_timing("pattern_normalization", time.time() - stage_start)
        
        # Stage 4: Add human-like timing variations (final stage)
        logger.info("Stage 4: Adding timing variations...")
        stage_start = time.time()
        # This step applies additional subtle timing variations to audio content
        # to further mask AI generation patterns
        try:
            y, sr = librosa.load(result_path, sr=None, mono=False)
            
            # Create processor for chunked processing
            processor = AudioProcessor(config)
            
            is_stereo = len(y.shape) > 1
            audio_length = y.shape[1] if is_stereo else len(y)
            
            # Memory optimization and chunked processing for large files
            if audio_length > sr * 60:  # >1 minute
                logger.info(f"Large file detected ({audio_length/sr:.1f}s) - using chunked processing")
                
                if is_stereo:
                    processed = np.zeros_like(y)
                    for i in range(y.shape[0]):
                        processed[i] = processor.process_large_audio_chunked(
                            y[i], sr, add_timing_variations, config
                        )
                else:
                    processed = processor.process_large_audio_chunked(
                        y, sr, add_timing_variations, config
                    )
            else:
                # Standard processing for smaller files
                if is_stereo:
                    processed = np.zeros_like(y)
                    for i in range(y.shape[0]):
                        processed[i] = add_timing_variations(y[i], sr, config)
                else:
                    processed = add_timing_variations(y, sr, config)
                
            # Save to final output location
            sf.write(output_path, processed.T if is_stereo else processed, sr)
            stats.timing_adjustments = 1
            stats.add_timing("timing_variations", time.time() - stage_start)
            
        except Exception as e:
            logger.error(f"Error in final timing adjustments: {e}")
            # If final processing fails, use previous stage result
            shutil.copy2(result_path, output_path)
            stats.add_timing("timing_variations", time.time() - stage_start)
        
        # Increment files processed count
        stats.files_processed = 1
        
        # Final performance metrics
        processing_time = time.time() - start_time
        stats.processing_time = processing_time
        if psutil:
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            stats.memory_peak_mb = max(initial_memory, current_memory)
        else:
            current_memory = 0.0
            stats.memory_peak_mb = 0.0
        stats.cache_hits = len([k for k in detector._cache.keys() if k.startswith('watermarks_')])
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        if psutil:
            logger.info(f"Memory usage: {current_memory:.1f}MB (peak: {stats.memory_peak_mb:.1f}MB)")
        if hasattr(stats, 'operation_timings'):
            for operation, duration in stats.operation_timings.items():
                logger.debug(f"  {operation}: {duration:.2f}s")
        
        return output_path, stats
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        # If something goes wrong, try to copy original file to output
        if output_path and output_path != input_path:
            try:
                shutil.copy2(input_path, output_path)
                logger.warning("Copied original file due to processing error")
            except Exception as copy_error:
                logger.error(f"Failed to copy original file: {copy_error}")
        raise e
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temporary directory: {cleanup_error}")


def add_timing_variations(audio: np.ndarray, sr: int, config: ProcessingConfig) -> np.ndarray:
    """Add subtle timing variations to make AI-generated audio sound more natural."""
    if not config.enable_timing_variations:
        return audio
        
    # Only apply if audio is long enough
    if len(audio) < sr:  # Less than 1 second
        return audio
        
    # Adaptive segment sizing based on audio characteristics
    # Shorter segments for more complex audio, longer for simpler
    base_segment_size = sr // 4  # ~250ms base
    
    # Analyze audio complexity to determine optimal segment size
    try:
        # Simple complexity measure: standard deviation of differences
        complexity = np.std(np.diff(audio[:min(sr*10, len(audio))]))  # Sample first 10 seconds
        if complexity > 0.01:  # High complexity audio
            segment_size = max(sr // 8, base_segment_size // 2)  # Smaller segments
        else:  # Lower complexity audio
            segment_size = min(sr // 2, base_segment_size * 2)  # Larger segments
    except:
        segment_size = base_segment_size
    
    hop_size = int(segment_size * (1 - config.segment_overlap_ratio))
    
    # Calculate number of segments
    num_segments = (len(audio) - segment_size) // hop_size + 1
    
    # If too few segments, just return original
    if num_segments < 3:
        return audio
    
    # Initialize output array with proper overlap handling
    result = np.zeros(len(audio))
    normalization_weights = np.zeros(len(audio))
    
    # Use a better window function for smoother transitions
    window = get_hann_window(segment_size)
    
    logger.debug(f"Processing {num_segments} segments of {segment_size/sr:.3f}s each")
    
    # Process each segment with slight random variations
    for i in range(num_segments):
        start = i * hop_size
        end = min(start + segment_size, len(audio))
        actual_segment_size = end - start
        
        if actual_segment_size < sr // 20:  # Skip very small segments (<50ms)
            continue
            
        segment = audio[start:end].copy()
        
        # Random micro-timing adjustment based on config
        variation_range = config.timing_variation_range
        random_var = 1.0 + (2 * variation_range * (random.random() - 0.5))
        
        # Apply subtle time stretching using librosa for better quality
        try:
            # Only apply if variation is significant enough
            if abs(random_var - 1.0) > 0.001:
                adjusted = librosa.effects.time_stretch(segment, rate=1.0/random_var)
            else:
                adjusted = segment
        except:
            # Fallback to simple interpolation if librosa fails
            indices = np.arange(0, len(segment), random_var)
            indices = indices[indices < len(segment)]
            if len(indices) > 1:
                adjusted = np.interp(np.arange(len(segment)), indices, segment[np.floor(indices).astype(int)])
            else:
                adjusted = segment
        
        # Ensure the adjusted segment fits
        if len(adjusted) > actual_segment_size:
            adjusted = adjusted[:actual_segment_size]
        elif len(adjusted) < actual_segment_size:
            adjusted = np.pad(adjusted, (0, actual_segment_size - len(adjusted)))
        
        # Apply appropriate window
        if len(adjusted) == segment_size:
            windowed = adjusted * window
        else:
            # Create appropriate window for actual size
            actual_window = get_hann_window(len(adjusted))
            windowed = adjusted * actual_window
        
        # Add to result with overlap handling
        result[start:start+len(windowed)] += windowed
        normalization_weights[start:start+len(windowed)] += actual_window if len(adjusted) != segment_size else window
    
    # Normalize by window overlap
    mask = normalization_weights > 0.001
    result[mask] = result[mask] / normalization_weights[mask]
    
    # Final RMS normalization to preserve energy
    input_rms = np.sqrt(np.mean(audio**2))
    output_rms = np.sqrt(np.mean(result**2))
    if output_rms > 0:
        result = result * (input_rms / output_rms)
    
    return result


def process_directory(input_dir: str, output_dir: Optional[str] = None,
                     aggressive: bool = False, level: str = None) -> Tuple[List[str], ProcessingStats]:
    """Process all audio files in a directory to remove AI fingerprinting."""
    processed_files = []
    stats = ProcessingStats()
    
    # Determine processing level
    if level is None:
        level = "aggressive" if aggressive else "moderate"
    stats.processing_level = level
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.aiff', '.aif')):
                input_path = os.path.join(root, file)
                
                # Determine output path
                if output_dir:
                    rel_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, rel_path)
                    output_subdir = os.path.dirname(output_path)
                    
                    # Create subdirectories if they don't exist
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)
                else:
                    output_path = None
                
                # Process the file
                try:
                    result, file_stats = process_audio(input_path, output_path, aggressive, level)
                    processed_files.append(result)
                    
                    # Accumulate statistics
                    stats.files_processed += file_stats.files_processed
                    stats.watermarks_detected += file_stats.watermarks_detected
                    stats.patterns_normalized += file_stats.patterns_normalized
                    stats.timing_adjustments += file_stats.timing_adjustments
                    
                    # Merge metadata dictionaries
                    for key, value in file_stats.metadata_removed.items():
                        if key not in stats.metadata_removed:
                            stats.metadata_removed[key] = []
                        stats.metadata_removed[key].extend(value)
                        
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
    
    return processed_files, stats


def get_file_hash(filepath: str) -> str:
    """Get a hash of file contents for verification."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def display_metadata(filepath: str) -> Dict[str, List[str]]:
    """Display metadata of an audio file (for debugging)."""
    metadata = {}
    
    try:
        print(f"\nMetadata for: {filepath}")
        print("-" * 40)
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.mp3':
            try:
                # Try ID3
                audio = MP3(filepath)
                id3_tags = []
                
                if audio.tags:
                    for key in audio.tags.keys():
                        id3_tags.append(f"{key}: {audio.tags[key]}")
                
                if id3_tags:
                    metadata['mp3_id3'] = id3_tags
                    
                # Try EasyID3
                try:
                    easy = EasyID3(filepath)
                    easy_tags = []
                    
                    for key in easy.keys():
                        easy_tags.append(f"{key}: {easy[key]}")
                    
                    if easy_tags:
                        metadata['mp3_easyid3'] = easy_tags
                except:
                    pass
            except Exception as e:
                print(f"Error reading MP3 metadata: {e}")
                
        elif file_ext == '.wav':
            try:
                audio = WAVE(filepath)
                wave_tags = []
                
                # Check for LIST INFO chunk
                if hasattr(audio, '_tags') and audio._tags:
                    for key, value in audio._tags.items():
                        wave_tags.append(f"{key}: {value}")
                
                # Check for other chunks
                for key in audio.keys():
                    try:
                        if isinstance(audio[key], bytes):
                            # Try to decode as text
                            text = audio[key].decode('utf-8', 'ignore')
                            if text.strip() and not text.isspace():
                                wave_tags.append(f"Chunk {key}: {text[:50]}...")
                        else:
                            wave_tags.append(f"Chunk {key}: {str(audio[key])[:50]}...")
                    except:
                        wave_tags.append(f"Chunk {key}: [binary data]")
                
                if wave_tags:
                    metadata['wav_chunks'] = wave_tags
            except Exception as e:
                print(f"Error reading WAV metadata: {e}")
                
        elif file_ext == '.flac':
            try:
                audio = FLAC(filepath)
                flac_tags = []
                
                # Get FLAC tags
                if audio.tags:
                    for key in audio.tags.keys():
                        flac_tags.append(f"{key}: {audio.tags[key]}")
                
                if flac_tags:
                    metadata['flac_tags'] = flac_tags
                
                # Check for pictures
                if audio.pictures:
                    metadata['flac_pictures'] = [f"Found {len(audio.pictures)} embedded pictures"]
            except Exception as e:
                print(f"Error reading FLAC metadata: {e}")
                
        elif file_ext in ['.aiff', '.aif']:
            try:
                audio = AIFF(filepath)
                aiff_tags = []
                
                if audio.tags:
                    for key in audio.tags.keys():
                        aiff_tags.append(f"{key}: {audio.tags[key]}")
                
                if aiff_tags:
                    metadata['aiff_tags'] = aiff_tags
            except Exception as e:
                print(f"Error reading AIFF metadata: {e}")
        
        # Print all metadata
        if metadata:
            for section, items in metadata.items():
                print(f"\n{section}:")
                for item in items:
                    print(f"  {item}")
        else:
            print("No metadata found")
            
        print("-" * 40)
        
        return metadata
        
    except Exception as e:
        print(f"Error displaying metadata: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="""
        AI Audio Fingerprint Remover - Comprehensive tool to remove AI-generated audio fingerprinting
        
        This tool implements multiple layers of protection:
        - Complete metadata stripping (ID3, RIFF, custom chunks)
        - Audio spectral watermark detection and removal
        - Sample-level normalization to remove statistical patterns
        - Timing pattern randomization
        - Frequency distribution normalization
        - Adds subtle human-like imperfections
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("input", nargs="?", help="Input audio file to process")
    group.add_argument("-d", "--directory", help="Process all audio files in directory")
    
    parser.add_argument("output", nargs="?", help="Output file or directory (optional)")
    parser.add_argument("--show", action="store_true", help="Show metadata before removal")
    
    # Processing intensity options
    intensity_group = parser.add_mutually_exclusive_group()
    intensity_group.add_argument("--level", choices=['gentle', 'moderate', 'aggressive', 'extreme'],
                                help="Processing intensity level (default: moderate)")
    intensity_group.add_argument("--aggressive", action="store_true", 
                                help="Use aggressive mode (equivalent to --level aggressive)")
    
    parser.add_argument("--verify", action="store_true", 
                      help="Verify results by comparing with original")
    parser.add_argument("--report", action="store_true", 
                      help="Generate a detailed report of changes made")
    
    args = parser.parse_args()
    
    print("\nAI Audio Fingerprint Remover")
    print("=" * 40)
    
    # Show processing level information if requested
    if args.level or args.aggressive:
        level = args.level or ("aggressive" if args.aggressive else "moderate")
        level_descriptions = {
            'gentle': 'Minimal processing - reduces artifacts but may leave some fingerprints',
            'moderate': 'Balanced processing - good compromise between effectiveness and quality',
            'aggressive': 'Thorough processing - removes most fingerprints with minimal quality impact',
            'extreme': 'Maximum processing - removes all detectable fingerprints but may affect quality'
        }
        print(f"Processing level: {level} - {level_descriptions.get(level, 'Unknown level')}")
        print()
    
    # Process single file
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
            return 1
        
        if args.show:
            before_metadata = display_metadata(args.input)
        
        print(f"\nProcessing {args.input}...")
        result, stats = process_audio(args.input, args.output, args.aggressive, args.level)
        
        print(f"\nResults:")
        print(f"  Processing level: {stats.processing_level}")
        print(f"  Files processed: {stats.files_processed}")
        print(f"  Watermarks detected and removed: {stats.watermarks_detected}")
        print(f"  Statistical patterns normalized: {stats.patterns_normalized}")
        print(f"  Timing adjustments applied: {stats.timing_adjustments}")
        
        if stats.metadata_removed:
            print("\nMetadata removed:")
            for category, items in stats.metadata_removed.items():
                print(f"  {category}: {len(items)} items")
                if args.report:
                    for item in items[:10]:  # Show first 10 items
                        print(f"    - {item}")
                    if len(items) > 10:
                        print(f"    - ... and {len(items) - 10} more")
        
        if args.show and args.output:
            print("\nAfter metadata removal:")
            after_metadata = display_metadata(args.output)
        
        if args.verify:
            print("\nVerification:")
            orig_hash = get_file_hash(args.input)
            new_hash = get_file_hash(result)
            print(f"  Original file hash: {orig_hash}")
            print(f"  Processed file hash: {new_hash}")
            print(f"  Files are {'identical' if orig_hash == new_hash else 'different'}")
            
            if orig_hash == new_hash and args.aggressive:
                print("  Warning: Files are identical after processing. This may indicate")
                print("  that the input file had no detectable AI fingerprints, or that")
                print("  processing failed.")
    
    # Process directory
    elif args.directory:
        if not os.path.exists(args.directory):
            print(f"Error: Input directory '{args.directory}' not found.")
            return 1
        
        print(f"\nProcessing all audio files in {args.directory}...")
        processed, stats = process_directory(args.directory, args.output, args.aggressive, args.level)
        
        print(f"\nResults:")
        print(f"  Processing level: {stats.processing_level}")
        print(f"  Files processed: {stats.files_processed}")
        print(f"  Watermarks detected and removed: {stats.watermarks_detected}")
        print(f"  Statistical patterns normalized: {stats.patterns_normalized}")
        print(f"  Timing adjustments applied: {stats.timing_adjustments}")
        
        if stats.metadata_removed:
            print("\nMetadata removed by category:")
            for category, items in stats.metadata_removed.items():
                print(f"  {category}: {len(items)} items")
                if args.report and items:
                    for item in items[:5]:  # Show first 5 items
                        print(f"    - {item}")
                    if len(items) > 5:
                        print(f"    - ... and {len(items) - 5} more")
    
    print("\nProcessing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
