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
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Set
from dataclasses import dataclass
import uuid

try:
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile
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
class ProcessingStats:
    """Track statistics about the audio processing."""
    files_processed: int = 0
    metadata_removed: Dict[str, List[str]] = None
    watermarks_detected: int = 0
    patterns_normalized: int = 0
    timing_adjustments: int = 0
    
    def __post_init__(self):
        if self.metadata_removed is None:
            self.metadata_removed = {}


class AudioFingerprint:
    """Detector for known audio fingerprinting techniques."""
    
    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive
        self.log_details = []
    
    def detect_spectral_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect potential spectral watermarks in the audio."""
        detected = []
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data
            
        # Perform spectral analysis
        freqs, times, spectrogram = signal.spectrogram(
            audio_mono, 
            fs=sample_rate,
            window='hann',
            nperseg=2048,
            noverlap=1024,
            scaling='spectrum'
        )
        
        # Look for anomalies in frequency bands commonly used for watermarking
        for freq_range in POTENTIAL_WATERMARK_FREQS:
            if freq_range[1] > sample_rate / 2:
                continue  # Skip if beyond Nyquist frequency
                
            # Find the indices for our frequency range
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
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
                        detected.append({
                            'freq_range': freq_range,
                            'peak_count': len(peaks),
                            'regularity': np.std(np.diff(peaks)),
                            'strength': np.max(autocorr[peaks])
                        })
                        
            # Also check for constant energy in bands where human audio would vary
            if freq_range[0] > 15000 and np.std(band_energy) / (mean_energy + 1e-10) < 0.1:
                detected.append({
                    'freq_range': freq_range,
                    'type': 'constant_energy',
                    'variation': np.std(band_energy) / (mean_energy + 1e-10)
                })
                
        return detected
        
    def detect_statistical_patterns(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect statistical anomalies that could indicate AI generation."""
        detected = []
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data
        
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
                    for key in list(audio.keys()):
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
    # Load the audio file
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        print(f"Error loading audio for watermark removal: {e}")
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
        processed = np.zeros_like(y)
        
        # Process each channel separately
        for i in range(y.shape[0]):
            processed[i] = apply_watermark_removal(y[i], sr, watermarks)
    else:
        processed = apply_watermark_removal(y, sr, watermarks)
    
    # Save the processed audio
    sf.write(output_path, processed.T if is_stereo else processed, sr)
    
    return output_path, watermarks


def apply_watermark_removal(audio: np.ndarray, sr: int, 
                           watermarks: List[Dict[str, Any]]) -> np.ndarray:
    """Apply filters to remove detected watermarks."""
    result = audio.copy()
    
    for watermark in watermarks:
        freq_range = watermark.get('freq_range')
        if not freq_range:
            continue
            
        # Design a band-reject filter for this frequency range
        low_freq, high_freq = freq_range
        
        # Convert to normalized frequency
        nyquist = sr / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Skip if outside Nyquist range
        if high_norm > 1.0:
            continue
            
        # Design filter with appropriate width
        width = min(0.1, (high_norm - low_norm) * 1.5)  # Ensure filter isn't too narrow
        
        # Create a bandstop filter
        b, a = signal.butter(4, [max(0.001, low_norm - width/2), 
                                 min(0.999, high_norm + width/2)], 
                             btype='bandstop')
        
        # Apply the filter
        result = signal.filtfilt(b, a, result)
    
    # If we're removing high-frequency watermarks, add a small amount of noise
    # to defeat fingerprinting that relies on absence of higher frequencies
    high_watermarks = [w for w in watermarks if w.get('freq_range') and w['freq_range'][0] > 15000]
    
    if high_watermarks:
        # Generate a small amount of shaped noise
        noise_level = 0.001  # Very subtle
        noise = np.random.randn(len(result)) * noise_level
        
        # Shape the noise to only affect high frequencies
        nyquist = sr / 2
        cutoff = 15000 / nyquist
        b, a = signal.butter(2, cutoff, btype='highpass')
        shaped_noise = signal.filtfilt(b, a, noise)
        
        # Add the noise
        result += shaped_noise
    
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
                y[i], sr, patterns, timing_issues, phase_var=phase_var
            )
    else:
        processed = apply_pattern_normalization(y, sr, patterns, timing_issues)
    
    # Save the processed audio
    sf.write(output_path, processed.T if is_stereo else processed, sr)
    
    return output_path, all_issues


def apply_pattern_normalization(audio: np.ndarray, sr: int, 
                               patterns: List[Dict[str, Any]], 
                               timing_issues: List[Dict[str, Any]],
                               phase_var: float = 0) -> np.ndarray:
    """Apply corrections to normalize detected AI patterns."""
    result = audio.copy()
    
    # 1. Address timing issues
    has_timing_issues = any(issue['type'] in ['mechanical_timing', 'quantized_timing'] 
                           for issue in timing_issues)
    
    if has_timing_issues:
        # Apply subtle time-domain variations
        # This stretches and compresses small segments randomly
        segment_len = sr // 10  # ~100ms segments
        hop_len = segment_len // 2
        
        # Break into segments
        segments = []
        for i in range(0, len(result) - segment_len, hop_len):
            segments.append(result[i:i+segment_len])
        
        # Apply random time stretching to each segment
        processed_segments = []
        for segment in segments:
            # Random stretch factor between 0.98 and 1.02 (±2%)
            stretch_factor = 0.98 + 0.04 * random.random()
            stretched = librosa.effects.time_stretch(segment, rate=stretch_factor)
            
            # Ensure consistent length
            if len(stretched) > segment_len:
                stretched = stretched[:segment_len]
            elif len(stretched) < segment_len:
                stretched = np.pad(stretched, (0, segment_len - len(stretched)))
                
            processed_segments.append(stretched)
        
        # Reconstruct with overlap-add
        result = np.zeros(len(audio))
        for i, segment in enumerate(processed_segments):
            pos = i * hop_len
            # Apply triangular window for smooth crossfading
            window = np.bartlett(len(segment))
            result[pos:pos+len(segment)] += segment * window
    
    # 2. Handle distribution anomalies
    has_distribution_issues = any(p['type'] == 'perfect_distribution' for p in patterns)
    
    if has_distribution_issues:
        # Add shaped noise to create more natural distribution
        noise_level = 0.001  # Very subtle
        noise = np.random.randn(len(result)) * noise_level
        
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
    
    if has_harmonic_issues:
        # Apply subtle harmonic distortion to create more natural harmonic relationships
        # This simulates the tiny non-linearities in analog equipment
        
        # Non-linear waveshaping function (subtle soft clipping)
        def soft_clip(x, amount=0.03):
            return x - amount * np.sin(2 * np.pi * x)
        
        # Apply the non-linearity
        result = soft_clip(result)
        
        # Also apply a tiny bit of phase variance if specified
        if phase_var != 0:
            # Create an all-pass filter for phase adjustment
            # This changes phase without changing amplitude
            b, a = signal.butter(2, 0.5, 'highpass')
            phase_adjustment = signal.lfilter(b, a, result) * phase_var
            result += phase_adjustment
    
    # 4. Add a touch of natural micro-dynamics
    # Human performances have micro-variations in dynamics that AI often lacks
    env = np.abs(signal.hilbert(result))
    smoothed = signal.savgol_filter(env, 
                                  max(5, min(101, len(result) // 500) // 2 * 2 + 1), 2)
    
    # Create subtle volume variations
    variations = np.sin(np.linspace(0, 20 * np.pi, len(result)) + random.random() * 10) * 0.005
    dynamics_adjustment = smoothed * variations
    
    # Apply the adjustment
    result += dynamics_adjustment
    
    # Final normalization to ensure we don't clip
    max_val = np.max(np.abs(result))
    if max_val > 0.99:
        result = result / max_val * 0.99
    
    return result


def process_audio(input_path: str, output_path: Optional[str] = None, 
                 aggressive: bool = False) -> Tuple[str, ProcessingStats]:
    """
    Process an audio file to remove all AI fingerprinting.
    
    Steps:
    1. Remove metadata
    2. Detect and remove spectral watermarks
    3. Normalize statistical patterns
    4. Add human-like variations
    """
    stats = ProcessingStats()
    detector = AudioFingerprint(aggressive=aggressive)
    
    # Create temporary processing directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Get file extension
        file_ext = os.path.splitext(input_path)[1].lower()
        
        # Determine final output path
        if output_path is None:
            output_path = input_path
            
        # Stage 1: Clean metadata
        temp_metadata = os.path.join(temp_dir, f"stage1_metadata{file_ext}")
        result_path, removed_metadata = clean_metadata_comprehensive(
            input_path, temp_metadata, aggressive
        )
        stats.metadata_removed = removed_metadata
        
        # Stage 2: Remove spectral watermarks
        temp_watermark = os.path.join(temp_dir, f"stage2_watermark{file_ext}")
        result_path, watermarks = remove_spectral_watermarks(
            result_path, temp_watermark, detector
        )
        stats.watermarks_detected = len(watermarks)
        
        # Stage 3: Normalize statistical patterns
        temp_patterns = os.path.join(temp_dir, f"stage3_patterns{file_ext}")
        result_path, patterns = normalize_ai_patterns(
            result_path, temp_patterns, detector
        )
        stats.patterns_normalized = len(patterns)
        
        # Stage 4: Add human-like timing variations (final stage)
        # This step applies additional subtle timing variations to audio content
        # to further mask AI generation patterns
        try:
            y, sr = librosa.load(result_path, sr=None, mono=False)
            is_stereo = len(y.shape) > 1
            
            if is_stereo:
                processed = np.zeros_like(y)
                for i in range(y.shape[0]):
                    processed[i] = add_timing_variations(y[i], sr)
            else:
                processed = add_timing_variations(y, sr)
                
            # Save to final output location
            sf.write(output_path, processed.T if is_stereo else processed, sr)
            stats.timing_adjustments = 1
            
        except Exception as e:
            print(f"Warning: Error in final timing adjustments: {e}")
            # If final processing fails, use previous stage result
            shutil.copy2(result_path, output_path)
        
        # Increment files processed count
        stats.files_processed = 1
        
        return output_path, stats
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        # If something goes wrong, try to copy original file to output
        if output_path and output_path != input_path:
            try:
                shutil.copy2(input_path, output_path)
            except:
                pass
        return input_path, stats
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def add_timing_variations(audio: np.ndarray, sr: int) -> np.ndarray:
    """Add subtle timing variations to make AI-generated audio sound more natural."""
    # Only apply if audio is long enough
    if len(audio) < sr:  # Less than 1 second
        return audio
        
    # Parameters for variation
    segment_size = sr // 4  # ~250ms segments
    overlap = segment_size // 2
    
    # Calculate number of segments
    num_segments = (len(audio) - segment_size) // overlap + 1
    
    # If too few segments, just return original
    if num_segments < 3:
        return audio
    
    # Initialize output array
    result = np.zeros(len(audio))
    window = get_hann_window(segment_size)  # Hann window for smooth transitions
    
    # Process each segment with slight random variations
    for i in range(num_segments):
        start = i * overlap
        segment = audio[start:start+segment_size].copy()
        
        # Random micro-timing adjustment
        # Tiny random pitch shifts create natural-sounding variations
        random_var = 0.99 + 0.02 * random.random()  # Between 0.99 and 1.01
        
        # Apply subtle pitch shift (which affects timing)
        # Using a simple resampling approach for speed
        indices = np.arange(0, len(segment), random_var)
        indices = indices[indices < len(segment)]
        adjusted = np.interp(np.arange(len(segment)), indices, segment[np.floor(indices).astype(int)])
        
        # Apply window and add to result
        result[start:start+segment_size] += adjusted * window
    
    # Normalize output level to match input
    input_rms = np.sqrt(np.mean(audio**2))
    output_rms = np.sqrt(np.mean(result**2))
    if output_rms > 0:
        result = result * (input_rms / output_rms)
    
    return result


def process_directory(input_dir: str, output_dir: Optional[str] = None,
                     aggressive: bool = False) -> Tuple[List[str], ProcessingStats]:
    """Process all audio files in a directory to remove AI fingerprinting."""
    processed_files = []
    stats = ProcessingStats()
    
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
                    result, file_stats = process_audio(input_path, output_path, aggressive)
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
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("input", nargs="?", help="Input audio file to process")
    group.add_argument("-d", "--directory", help="Process all audio files in directory")
    
    parser.add_argument("output", nargs="?", help="Output file or directory (optional)")
    parser.add_argument("--show", action="store_true", help="Show metadata before removal")
    parser.add_argument("--aggressive", action="store_true", 
                      help="Use aggressive mode (more thorough but may affect quality)")
    parser.add_argument("--verify", action="store_true", 
                      help="Verify results by comparing with original")
    parser.add_argument("--report", action="store_true", 
                      help="Generate a detailed report of changes made")
    
    args = parser.parse_args()
    
    print("\nAI Audio Fingerprint Remover")
    print("=" * 40)
    
    # Process single file
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
            return 1
        
        if args.show:
            before_metadata = display_metadata(args.input)
        
        print(f"\nProcessing {args.input}...")
        result, stats = process_audio(args.input, args.output, args.aggressive)
        
        print(f"\nResults:")
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
        processed, stats = process_directory(args.directory, args.output, args.aggressive)
        
        print(f"\nResults:")
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
