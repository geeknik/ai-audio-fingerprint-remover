#!/usr/bin/env python3
"""
Aggressive Watermark Removal System
More aggressive removal techniques specifically for AI-generated audio watermarks.
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy import stats
from scipy.fft import fft, ifft, fftfreq
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class AggressiveWatermarkRemover:
    """Aggressive watermark removal system for AI audio."""
    
    def __init__(self):
        # Balanced removal parameters that preserve audio quality
        self.removal_strength = {
            'frequency_notch': 0.30,  # Remove 30% of energy in watermark frequencies
            'spectral_gating': 0.25,  # Light spectral gating
            'temporal_smoothing': 0.2,  # Light temporal smoothing
            'phase_randomization': 0.15,  # Very light phase randomization
            'pattern_disruption': 0.4,  # Focus on pattern disruption instead
        }
        
        # Suno-specific frequency ranges for aggressive removal
        self.suno_removal_ranges = [
            (19000, 20000, 0.80),  # Ultrasonic - aggressive but safe
            (17500, 18500, 0.70),  # Extended high - moderate
            (15000, 16000, 0.60),  # Mid-high - moderate
            (12000, 12500, 0.50),  # Secondary - conservative
            (8000, 8500, 0.40),    # Mid-range - conservative
            (50, 200, 0.30),       # Low-freq - very conservative
        ]
    
    def remove_watermarks_aggressive(self, audio: np.ndarray, sr: int, 
                                   watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Apply aggressive watermark removal techniques."""
        result = audio.copy()
        
        # Ensure audio is finite and valid
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize if needed
        if np.max(np.abs(result)) > 1.0:
            result = result / np.max(np.abs(result))
        
        logger.info(f"Applying aggressive removal to {len(watermarks)} watermarks")
        
        # Store original for fallback
        original = result.copy()
        
        # 1. Aggressive frequency domain removal
        try:
            temp_result = self._aggressive_frequency_removal(result, sr, watermarks)
            if np.any(np.abs(temp_result) > 1e-10):  # Validate result isn't silent
                result = self._safe_nan_cleanup(temp_result, result)
            else:
                logger.warning("Frequency removal produced silence, skipping")
        except Exception as e:
            logger.warning(f"Frequency removal failed: {e}")
        
        # 2. Spectral subtraction for neural patterns
        try:
            temp_result = self._spectral_subtraction(result, sr, watermarks)
            if np.any(np.abs(temp_result) > 1e-10):  # Validate result isn't silent
                result = self._safe_nan_cleanup(temp_result, result)
            else:
                logger.warning("Spectral subtraction produced silence, skipping")
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
        
        # 3. Adaptive filtering
        try:
            temp_result = self._adaptive_filtering(result, sr, watermarks)
            if np.any(np.abs(temp_result) > 1e-10):  # Validate result isn't silent
                result = self._safe_nan_cleanup(temp_result, result)
            else:
                logger.warning("Adaptive filtering produced silence, skipping")
        except Exception as e:
            logger.warning(f"Adaptive filtering failed: {e}")
        
        # 4. Phase manipulation
        try:
            temp_result = self._phase_manipulation(result, sr, watermarks)
            if np.any(np.abs(temp_result) > 1e-10):  # Validate result isn't silent
                result = self._safe_nan_cleanup(temp_result, result)
            else:
                logger.warning("Phase manipulation produced silence, skipping")
        except Exception as e:
            logger.warning(f"Phase manipulation failed: {e}")
        
        # 5. Temporal pattern disruption
        try:
            temp_result = self._temporal_pattern_disruption(result, sr, watermarks)
            if np.any(np.abs(temp_result) > 1e-10):  # Validate result isn't silent
                result = self._safe_nan_cleanup(temp_result, result)
            else:
                logger.warning("Temporal disruption produced silence, skipping")
        except Exception as e:
            logger.warning(f"Temporal disruption failed: {e}")
        
        # Final validation - if all processing failed, return original
        if not np.any(np.abs(result) > 1e-10):
            logger.warning("All processing steps failed, returning original audio")
            result = original
        
        # Apply AI detection evasion techniques
        try:
            result = self.add_ai_detection_evasion(result, sr)
            logger.info("Applied AI detection evasion techniques")
        except Exception as e:
            logger.warning(f"AI detection evasion failed: {e}")
        
        return result
    
    def _safe_nan_cleanup(self, processed_audio: np.ndarray, fallback_audio: np.ndarray) -> np.ndarray:
        """Safely clean NaN/inf values without zeroing the entire signal."""
        # Check for NaN/inf values
        has_nan = np.any(np.isnan(processed_audio))
        has_inf = np.any(np.isinf(processed_audio))
        
        if not (has_nan or has_inf):
            return processed_audio  # No cleanup needed
        
        logger.warning(f"Cleaning NaN/inf values: NaN={has_nan}, Inf={has_inf}")
        
        # Create mask for valid samples
        valid_mask = np.isfinite(processed_audio)
        
        # If we have valid samples, only replace invalid ones
        if np.any(valid_mask):
            result = processed_audio.copy()
            
            # Replace NaN/inf with interpolated values from nearby valid samples
            if has_nan or has_inf:
                invalid_indices = ~valid_mask
                if np.sum(invalid_indices) < len(processed_audio) * 0.5:  # Less than 50% invalid
                    # Linear interpolation for scattered invalid samples
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices_pos = np.where(invalid_indices)[0]
                    
                    for invalid_idx in invalid_indices_pos:
                        # Find nearest valid samples
                        left_valid = valid_indices[valid_indices < invalid_idx]
                        right_valid = valid_indices[valid_indices > invalid_idx]
                        
                        if len(left_valid) > 0 and len(right_valid) > 0:
                            # Interpolate between nearest valid samples
                            left_idx = left_valid[-1]
                            right_idx = right_valid[0]
                            alpha = (invalid_idx - left_idx) / (right_idx - left_idx)
                            result[invalid_idx] = (1 - alpha) * result[left_idx] + alpha * result[right_idx]
                        elif len(left_valid) > 0:
                            # Use nearest left sample
                            result[invalid_idx] = result[left_valid[-1]]
                        elif len(right_valid) > 0:
                            # Use nearest right sample
                            result[invalid_idx] = result[right_valid[0]]
                        else:
                            # Last resort - use fallback
                            result[invalid_idx] = fallback_audio[invalid_idx] if invalid_idx < len(fallback_audio) else 0.0
                else:
                    # Too many invalid samples, return fallback
                    logger.warning("Too many invalid samples, using fallback audio")
                    return fallback_audio
            
            return result
        else:
            # All samples are invalid, return fallback
            logger.warning("All samples invalid, using fallback audio")
            return fallback_audio
    
    def _aggressive_frequency_removal(self, audio: np.ndarray, sr: int, 
                                    watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Aggressively remove frequency-domain watermarks."""
        result = audio.copy()
        
        # Apply Suno-specific frequency removal
        for low_freq, high_freq, strength in self.suno_removal_ranges:
            if high_freq > sr / 2:
                continue
                
            # Create aggressive bandstop filter
            nyquist = sr / 2
            low_norm = low_freq / nyquist
            high_norm = min(high_freq, nyquist - 1) / nyquist
            
            if 0 < low_norm < high_norm < 1.0 and len(result) > 100:  # Ensure minimum length
                try:
                    # Use lower order filter to avoid length issues
                    filter_order = min(6, len(result) // 20)  # Adaptive order based on length
                    if filter_order < 2:
                        continue  # Skip if audio too short
                    
                    b, a = signal.butter(filter_order, [low_norm, high_norm], btype='bandstop')
                    filtered = signal.filtfilt(b, a, result)
                    
                    # Blend based on strength
                    result = (1 - strength) * result + strength * filtered
                    
                    logger.debug(f"Applied {strength:.0%} removal to {low_freq}-{high_freq}Hz")
                except Exception as e:
                    logger.warning(f"Failed to filter {low_freq}-{high_freq}Hz: {e}")
        
        # Additional removal for detected watermark frequencies
        freq_watermarks = [w for w in watermarks if 'frequency' in w]
        for watermark in freq_watermarks:
            freq = watermark['frequency']
            confidence = watermark.get('confidence', 0.5)
            
            # Create narrow notch filter
            bandwidth = 100  # Hz
            low_freq = max(freq - bandwidth/2, 0)
            high_freq = min(freq + bandwidth/2, sr/2 - 1)
            
            low_norm = low_freq / (sr/2)
            high_norm = high_freq / (sr/2)
            
            if 0 < low_norm < high_norm < 1.0 and len(result) > 100:  # Ensure minimum length
                try:
                    # Use adaptive filter order
                    filter_order = min(4, len(result) // 25)  # Conservative order
                    if filter_order < 2:
                        continue  # Skip if audio too short
                    
                    b, a = signal.butter(filter_order, [low_norm, high_norm], btype='bandstop')
                    filtered = signal.filtfilt(b, a, result)
                    
                    # Strength based on confidence but capped to preserve audio
                    strength = min(confidence * 0.3, 0.5)  # Much more conservative
                    result = (1 - strength) * result + strength * filtered
                    
                except Exception as e:
                    logger.warning(f"Failed to notch filter at {freq}Hz: {e}")
        
        return result
    
    def _spectral_subtraction(self, audio: np.ndarray, sr: int, 
                            watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Apply spectral subtraction to remove neural watermark patterns."""
        
        # Compute STFT
        n_fft = 2048
        hop_length = n_fft // 4
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum from high frequencies
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        high_freq_mask = freqs >= 15000
        
        if np.any(high_freq_mask):
            # Estimate watermark spectrum
            watermark_spectrum = np.mean(magnitude[high_freq_mask], axis=0, keepdims=True)
            
            # Apply spectral subtraction
            alpha = 1.5  # Over-subtraction factor (less aggressive)
            beta = 0.1   # Spectral floor (higher to preserve quality)
            
            # Subtract estimated watermark spectrum
            for i, freq in enumerate(freqs):
                if freq >= 15000:  # Only apply to high frequencies
                    # Spectral subtraction formula
                    magnitude[i] = magnitude[i] - alpha * watermark_spectrum[0]
                    # Apply spectral floor
                    magnitude[i] = np.maximum(magnitude[i], beta * magnitude[i])
        
        # Reconstruct audio
        stft_cleaned = magnitude * np.exp(1j * phase)
        result = librosa.istft(stft_cleaned, hop_length=hop_length)
        
        # Ensure same length as input
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
        
        return result
    
    def _adaptive_filtering(self, audio: np.ndarray, sr: int, 
                          watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Apply adaptive filtering to remove watermarks."""
        
        # Wiener filtering approach
        n_fft = 2048
        hop_length = n_fft // 4
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate signal and noise power
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Create frequency-dependent Wiener filter
        wiener_filter = np.ones_like(magnitude)
        
        for i, freq in enumerate(freqs):
            # More aggressive filtering for watermark frequencies
            if freq >= 15000:  # High frequencies
                noise_power = np.var(magnitude[i]) + 1e-10  # Add small epsilon
                signal_power = np.mean(magnitude[i] ** 2) + 1e-10
                
                # Wiener filter coefficient
                wiener_coeff = signal_power / (signal_power + 1.5 * noise_power)  # Less aggressive
                wiener_filter[i] = np.clip(wiener_coeff, 0.1, 1.0)  # Prevent complete removal
            
            elif 8000 <= freq <= 12500:  # Mid-range watermark frequencies
                noise_power = np.var(magnitude[i]) + 1e-10
                signal_power = np.mean(magnitude[i] ** 2) + 1e-10
                
                wiener_coeff = signal_power / (signal_power + 1.2 * noise_power)
                wiener_filter[i] = np.clip(wiener_coeff, 0.3, 1.0)  # More conservative
        
        # Apply filter
        magnitude_filtered = magnitude * wiener_filter
        
        # Reconstruct
        stft_filtered = magnitude_filtered * np.exp(1j * phase)
        result = librosa.istft(stft_filtered, hop_length=hop_length)
        
        # Ensure same length
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
        
        return result
    
    def _phase_manipulation(self, audio: np.ndarray, sr: int, 
                          watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Manipulate phase to disrupt phase-based watermarks."""
        
        n_fft = 2048
        hop_length = n_fft // 4
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Add controlled phase noise to high frequencies
        high_freq_mask = freqs >= 15000
        if np.any(high_freq_mask):
            # Generate smooth phase noise
            phase_noise_strength = 0.3  # Moderate phase noise
            
            for i in np.where(high_freq_mask)[0]:
                # Add correlated phase noise (smoother than random)
                noise = np.random.randn(phase.shape[1]) * phase_noise_strength
                # Smooth the noise
                noise = signal.savgol_filter(noise, min(21, len(noise)//2*2+1), 3)
                phase[i] += noise
        
        # Reconstruct
        stft_modified = magnitude * np.exp(1j * phase)
        result = librosa.istft(stft_modified, hop_length=hop_length)
        
        # Ensure same length
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
        
        return result
    
    def _temporal_pattern_disruption(self, audio: np.ndarray, sr: int, 
                                   watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Disrupt temporal patterns in watermarks."""
        
        # Check for temporal watermarks
        temporal_watermarks = [w for w in watermarks if 'temporal' in w.get('type', '')]
        
        if not temporal_watermarks:
            return audio
        
        result = audio.copy()
        
        # Apply subtle time-varying processing
        frame_size = sr // 10  # 100ms frames
        
        for i in range(0, len(result) - frame_size, frame_size // 2):
            frame = result[i:i + frame_size]
            
            # Apply subtle time stretching/compression
            stretch_factor = 1.0 + np.random.randn() * 0.002  # ±0.2% variation
            
            try:
                # Time stretch using phase vocoder
                frame_stretched = librosa.effects.time_stretch(frame, rate=stretch_factor)
                
                # Trim or pad to original size
                if len(frame_stretched) > len(frame):
                    frame_stretched = frame_stretched[:len(frame)]
                elif len(frame_stretched) < len(frame):
                    frame_stretched = np.pad(frame_stretched, 
                                           (0, len(frame) - len(frame_stretched)), 
                                           mode='constant')
                
                # Blend with original
                blend_factor = 0.1  # Subtle effect
                result[i:i + frame_size] = (1 - blend_factor) * frame + blend_factor * frame_stretched
                
            except Exception as e:
                logger.debug(f"Time stretch failed for frame {i}: {e}")
                continue
        
        return result
    
    def add_ai_detection_evasion(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add subtle modifications to evade AI detection systems."""
        result = audio.copy()
        
        # 1. Add micro-variations in timing (not perceptible but breaks AI patterns)
        result = self._add_micro_timing_variations(result, sr)
        
        # 2. Add subtle harmonic content variations
        result = self._add_harmonic_variations(result, sr)
        
        # 3. Add naturalistic noise in quiet sections
        result = self._add_naturalistic_noise(result, sr)
        
        return result
    
    def _add_micro_timing_variations(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add extremely subtle timing variations to break AI detection patterns."""
        # Very small pitch shifts that create timing-like variations
        pitch_shift_cents = np.random.uniform(-3, 3)  # Tiny pitch variation
        
        try:
            # Use librosa's pitch shift which preserves length
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_cents/100)
            
            # Blend very subtly with original
            blend_factor = 0.05  # Almost imperceptible
            result = (1 - blend_factor) * audio + blend_factor * shifted
            
            return result
        except Exception as e:
            logger.debug(f"Micro timing variation failed: {e}")
            return audio
    
    def _add_harmonic_variations(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add subtle harmonic content variations to break spectral patterns."""
        try:
            # Get spectral representation
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Add very subtle variations to harmonic content
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            
            # Focus on mid frequencies where harmonics are important
            mid_freq_mask = (freqs >= 200) & (freqs <= 5000)
            
            if np.any(mid_freq_mask):
                # Add tiny random variations to magnitude
                noise_factor = 0.02  # 2% variation
                for i in np.where(mid_freq_mask)[0]:
                    magnitude[i] *= (1 + np.random.uniform(-noise_factor, noise_factor, magnitude.shape[1]))
            
            # Reconstruct
            stft_modified = magnitude * np.exp(1j * phase)
            result = librosa.istft(stft_modified, hop_length=512)
            
            # Ensure same length
            if len(result) != len(audio):
                if len(result) > len(audio):
                    result = result[:len(audio)]
                else:
                    result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
            
            return result
            
        except Exception as e:
            logger.debug(f"Harmonic variation failed: {e}")
            return audio
    
    def _add_naturalistic_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add naturalistic noise in quiet sections to break AI silence patterns."""
        # Find quiet sections
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        quiet_threshold = np.percentile(rms, 20)  # Bottom 20% of energy
        
        # Upsample RMS to match audio length
        time_frames = librosa.frames_to_samples(np.arange(len(rms)), hop_length=512)
        quiet_mask = np.interp(np.arange(len(audio)), time_frames, rms < quiet_threshold)
        quiet_mask = quiet_mask > 0.5
        
        if np.any(quiet_mask):
            # Generate very subtle background noise
            noise_level = np.std(audio) * 0.001  # Extremely quiet noise
            noise = np.random.randn(len(audio)) * noise_level
            
            # Apply noise only to quiet sections
            result = audio.copy()
            result[quiet_mask] += noise[quiet_mask]
            
            return result
        
        return audio

def main():
    """Test the aggressive watermark remover."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggressive Watermark Removal")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--strength", type=float, default=1.0, 
                       help="Removal strength (0.0-2.0, default 1.0)")
    
    args = parser.parse_args()
    
    # Load audio
    audio, sr = librosa.load(args.input, sr=None, mono=False)
    
    # Convert to mono if needed
    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=0)
    else:
        audio_mono = audio
    
    # Create dummy watermarks for testing (in real use, these come from detection)
    dummy_watermarks = [
        {'type': 'suno_carrier', 'frequency': 19500, 'confidence': 0.8},
        {'type': 'suno_carrier', 'frequency': 17800, 'confidence': 0.7},
        {'type': 'suno_carrier', 'frequency': 15200, 'confidence': 0.6},
    ]
    
    # Apply aggressive removal
    remover = AggressiveWatermarkRemover()
    cleaned_audio = remover.remove_watermarks_aggressive(audio_mono, sr, dummy_watermarks)
    
    # Save result
    import soundfile as sf
    sf.write(args.output, cleaned_audio, sr)
    
    print(f"Aggressive watermark removal complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()