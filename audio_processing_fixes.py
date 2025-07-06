#!/usr/bin/env python3
"""
Comprehensive fixes for AI Audio Fingerprint Remover
Addresses silent outputs, noise artifacts, and audio corruption issues
"""

import numpy as np
import scipy.signal as signal
import librosa
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class AudioProcessingFixes:
    """Core fixes for audio processing issues."""
    
    @staticmethod
    def validate_audio_content(audio: np.ndarray, min_amplitude: float = 1e-10, 
                              context: str = "") -> bool:
        """
        Enhanced audio validation with detailed checking.
        
        Args:
            audio: Audio array to validate
            min_amplitude: Minimum amplitude threshold
            context: Description of where validation is happening
            
        Returns:
            True if audio is valid, False otherwise
        """
        if audio is None or len(audio) == 0:
            logger.warning(f"{context}: Audio is None or empty")
            return False
        
        # Check for NaN or inf values
        nan_count = np.sum(np.isnan(audio))
        inf_count = np.sum(np.isinf(audio))
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"{context}: Found {nan_count} NaN and {inf_count} inf values")
            return False
        
        # Check maximum amplitude
        max_amp = np.max(np.abs(audio))
        if max_amp < min_amplitude:
            logger.warning(f"{context}: Max amplitude {max_amp:.2e} below threshold {min_amplitude:.2e}")
            return False
        
        # Check RMS (root mean square) - more robust than just max
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_amplitude / 10:
            logger.warning(f"{context}: RMS {rms:.2e} too low")
            return False
        
        # Check for all zeros
        non_zero_count = np.count_nonzero(audio)
        if non_zero_count == 0:
            logger.warning(f"{context}: Audio is all zeros")
            return False
        
        # Check if audio has reasonable dynamic range
        if max_amp > 0:
            dynamic_range = 20 * np.log10(max_amp / (rms + 1e-10))
            if dynamic_range > 100:  # Suspiciously high dynamic range
                logger.warning(f"{context}: Suspicious dynamic range: {dynamic_range:.1f} dB")
        
        return True
    
    @staticmethod
    def safe_filter_design(sr: int, freq_range: List[float], audio_length: int,
                          base_filter_order: int = 4) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Safely design a bandstop filter with comprehensive validation.
        
        Args:
            sr: Sample rate
            freq_range: [low_freq, high_freq] to filter
            audio_length: Length of audio to be filtered
            base_filter_order: Base filter order
            
        Returns:
            (b, a) filter coefficients or (None, None) if design fails
        """
        nyquist = sr / 2
        low_freq, high_freq = freq_range
        
        # Validate frequency range
        if low_freq <= 0 or high_freq >= nyquist:
            logger.warning(f"Invalid frequency range: {freq_range} (Nyquist: {nyquist})")
            return None, None
        
        if low_freq >= high_freq:
            logger.warning(f"Invalid frequency range: low >= high ({low_freq} >= {high_freq})")
            return None, None
        
        # Adaptive filter order based on audio length
        min_length_per_order = 50  # Need at least 50 samples per filter order
        max_possible_order = max(1, audio_length // min_length_per_order)
        filter_order = min(base_filter_order, max_possible_order)
        
        if filter_order < 2:
            logger.warning(f"Audio too short for filtering: {audio_length} samples")
            return None, None
        
        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Add safety margins to prevent numerical issues
        low_norm = max(0.01, low_norm)  # At least 1% of Nyquist
        high_norm = min(0.99, high_norm)  # At most 99% of Nyquist
        
        # Ensure reasonable filter bandwidth
        bandwidth = high_norm - low_norm
        if bandwidth < 0.01:  # Less than 1% bandwidth
            logger.warning(f"Filter bandwidth too narrow: {bandwidth:.3f}")
            return None, None
        
        try:
            # Design filter with error handling
            b, a = signal.butter(filter_order, [low_norm, high_norm], btype='bandstop')
            
            # Check filter stability
            poles = np.roots(a)
            if np.any(np.abs(poles) >= 1.0):
                logger.warning(f"Unstable filter detected (poles outside unit circle)")
                # Try lower order
                if filter_order > 2:
                    return AudioProcessingFixes.safe_filter_design(
                        sr, freq_range, audio_length, filter_order - 1
                    )
                return None, None
            
            # Check for numerical issues
            if np.any(np.isnan(b)) or np.any(np.isnan(a)) or np.any(np.isinf(b)) or np.any(np.isinf(a)):
                logger.warning("Filter coefficients contain NaN or inf")
                return None, None
            
            return b, a
            
        except Exception as e:
            logger.warning(f"Filter design failed: {e}")
            return None, None
    
    @staticmethod
    def safe_filter_apply(audio: np.ndarray, b: np.ndarray, a: np.ndarray,
                         blend_factor: float = 0.7) -> Optional[np.ndarray]:
        """
        Safely apply a filter with validation and blending.
        
        Args:
            audio: Input audio
            b, a: Filter coefficients
            blend_factor: How much of filtered signal to blend (0-1)
            
        Returns:
            Filtered audio or None if filtering fails
        """
        try:
            # Use filtfilt for zero-phase filtering
            filtered = signal.filtfilt(b, a, audio)
            
            # Validate filtered output
            if not AudioProcessingFixes.validate_audio_content(filtered, context="Post-filter"):
                logger.warning("Filtered audio failed validation")
                return None
            
            # Check if filter removed too much content
            original_rms = np.sqrt(np.mean(audio**2))
            filtered_rms = np.sqrt(np.mean(filtered**2))
            
            if original_rms > 0 and filtered_rms / original_rms < 0.1:
                logger.warning(f"Filter removed too much content: {filtered_rms/original_rms:.1%} remaining")
                return None
            
            # Blend with original to preserve quality
            result = (1 - blend_factor) * audio + blend_factor * filtered
            
            return result
            
        except Exception as e:
            logger.warning(f"Filter application failed: {e}")
            return None
    
    @staticmethod
    def safe_nan_cleanup(audio: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Safely clean NaN/inf values using interpolation instead of zeroing.
        
        Args:
            audio: Audio with potential NaN/inf values
            fallback: Fallback audio if cleanup fails
            
        Returns:
            Cleaned audio
        """
        # Check if cleanup is needed
        if not (np.any(np.isnan(audio)) or np.any(np.isinf(audio))):
            return audio
        
        result = audio.copy()
        
        # Find valid samples
        valid_mask = np.isfinite(audio)
        valid_ratio = np.sum(valid_mask) / len(audio)
        
        if valid_ratio < 0.5:
            logger.warning(f"Too many invalid samples: {1-valid_ratio:.1%}")
            return fallback if fallback is not None else np.zeros_like(audio)
        
        if valid_ratio < 1.0:
            # Interpolate invalid samples
            valid_indices = np.where(valid_mask)[0]
            invalid_indices = np.where(~valid_mask)[0]
            
            if len(valid_indices) > 1:
                # Linear interpolation for invalid samples
                result[invalid_indices] = np.interp(
                    invalid_indices, valid_indices, audio[valid_indices]
                )
            else:
                # Not enough valid samples for interpolation
                result[~valid_mask] = 0.0
        
        return result
    
    @staticmethod
    def adaptive_processing_strength(audio: np.ndarray, sr: int, 
                                   base_strength: float) -> float:
        """
        Adaptively adjust processing strength based on audio characteristics.
        
        Args:
            audio: Input audio
            sr: Sample rate
            base_strength: Base processing strength (0-1)
            
        Returns:
            Adjusted strength
        """
        # Analyze audio characteristics
        rms = np.sqrt(np.mean(audio**2))
        
        # Check spectral centroid (brightness)
        stft = np.abs(librosa.stft(audio, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        spectral_centroid = np.sum(freqs[:, np.newaxis] * stft, axis=0) / (np.sum(stft, axis=0) + 1e-10)
        avg_centroid = np.mean(spectral_centroid)
        
        # Reduce strength for quiet or dark audio
        if rms < 0.01:  # Very quiet
            strength_multiplier = 0.5
        elif avg_centroid < sr * 0.1:  # Dark/muffled audio
            strength_multiplier = 0.7
        else:
            strength_multiplier = 1.0
        
        return base_strength * strength_multiplier
    
    @staticmethod
    def safe_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Safely apply time stretching with fallback.
        
        Args:
            audio: Input audio
            rate: Stretch rate
            
        Returns:
            Stretched audio
        """
        try:
            # Limit stretch rate to reasonable bounds
            rate = np.clip(rate, 0.5, 2.0)
            
            # Apply time stretch
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            
            # Validate result
            if AudioProcessingFixes.validate_audio_content(stretched, context="Time stretch"):
                return stretched
            else:
                logger.warning("Time stretch failed validation, returning original")
                return audio
                
        except Exception as e:
            logger.warning(f"Time stretch failed: {e}")
            return audio
    
    @staticmethod
    def safe_stft_processing(audio: np.ndarray, sr: int, 
                            process_func, n_fft: int = 2048) -> np.ndarray:
        """
        Safely process audio in STFT domain with validation.
        
        Args:
            audio: Input audio
            sr: Sample rate
            process_func: Function to process STFT
            n_fft: FFT size
            
        Returns:
            Processed audio
        """
        try:
            # Ensure audio length is sufficient for STFT
            if len(audio) < n_fft:
                logger.warning(f"Audio too short for STFT: {len(audio)} < {n_fft}")
                return audio
            
            # Compute STFT
            hop_length = n_fft // 4
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            
            # Process STFT
            processed_stft = process_func(stft, sr)
            
            # Validate STFT
            if np.any(np.isnan(processed_stft)) or np.any(np.isinf(processed_stft)):
                logger.warning("Processed STFT contains NaN/inf")
                return audio
            
            # Convert back to time domain
            result = librosa.istft(processed_stft, hop_length=hop_length)
            
            # Ensure same length as input
            if len(result) != len(audio):
                if len(result) > len(audio):
                    result = result[:len(audio)]
                else:
                    result = np.pad(result, (0, len(audio) - len(result)))
            
            # Validate result
            if AudioProcessingFixes.validate_audio_content(result, context="STFT processing"):
                return result
            else:
                logger.warning("STFT processing failed validation")
                return audio
                
        except Exception as e:
            logger.warning(f"STFT processing failed: {e}")
            return audio


class WatermarkRemovalFixes:
    """Fixes specifically for watermark removal issues."""
    
    @staticmethod
    def conservative_frequency_removal(audio: np.ndarray, sr: int,
                                     freq_ranges: List[Tuple[float, float]],
                                     max_attenuation_db: float = 20) -> np.ndarray:
        """
        Remove frequency bands conservatively to avoid artifacts.
        
        Args:
            audio: Input audio
            sr: Sample rate
            freq_ranges: List of (low_freq, high_freq) to attenuate
            max_attenuation_db: Maximum attenuation in dB
            
        Returns:
            Processed audio
        """
        result = audio.copy()
        
        for low_freq, high_freq in freq_ranges:
            # Skip invalid ranges
            if low_freq >= high_freq or high_freq > sr / 2:
                continue
            
            # Design conservative filter
            filter_result = AudioProcessingFixes.safe_filter_design(
                sr, [low_freq, high_freq], len(audio), base_filter_order=2
            )
            
            if filter_result[0] is not None:
                b, a = filter_result
                
                # Apply with conservative blending
                blend_factor = min(0.5, max_attenuation_db / 40)  # Max 50% blend
                filtered_result = AudioProcessingFixes.safe_filter_apply(
                    result, b, a, blend_factor
                )
                
                if filtered_result is not None:
                    result = filtered_result
        
        return result
    
    @staticmethod
    def psychoacoustic_watermark_removal(audio: np.ndarray, sr: int,
                                       watermark_freqs: List[float]) -> np.ndarray:
        """
        Remove watermarks using psychoacoustic masking principles.
        
        Args:
            audio: Input audio
            sr: Sample rate
            watermark_freqs: List of watermark frequencies
            
        Returns:
            Processed audio
        """
        def process_stft(stft, sr):
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Calculate masking threshold
            # Simple approximation: 20dB below local average
            kernel_size = 5
            local_avg = signal.convolve2d(
                magnitude, 
                np.ones((kernel_size, 1)) / kernel_size,
                mode='same',
                boundary='wrap'
            )
            masking_threshold = local_avg * 0.1  # -20dB
            
            # Attenuate watermark frequencies only where they exceed masking
            freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
            
            for wm_freq in watermark_freqs:
                if 0 < wm_freq < sr / 2:
                    # Find closest frequency bin
                    freq_bin = np.argmin(np.abs(freqs - wm_freq))
                    
                    # Only attenuate if above masking threshold
                    mask = magnitude[freq_bin, :] > masking_threshold[freq_bin, :]
                    magnitude[freq_bin, mask] *= 0.1  # -20dB attenuation
            
            # Reconstruct STFT
            return magnitude * np.exp(1j * phase)
        
        return AudioProcessingFixes.safe_stft_processing(audio, sr, process_stft)


class AudioQualityEnhancer:
    """Post-processing to enhance audio quality after watermark removal."""
    
    @staticmethod
    def add_natural_variations(audio: np.ndarray, sr: int,
                             variation_amount: float = 0.001) -> np.ndarray:
        """
        Add subtle natural variations to combat AI detection.
        
        Args:
            audio: Input audio
            sr: Sample rate
            variation_amount: Amount of variation (0-1)
            
        Returns:
            Enhanced audio
        """
        result = audio.copy()
        
        # 1. Micro-timing variations (very subtle)
        if len(audio) > sr:
            segment_size = sr // 10  # 100ms segments
            for i in range(0, len(audio) - segment_size, segment_size // 2):
                segment = audio[i:i+segment_size]
                
                # Very small random stretch
                stretch_factor = 1.0 + (np.random.rand() - 0.5) * variation_amount
                stretched = AudioProcessingFixes.safe_time_stretch(segment, stretch_factor)
                
                # Blend with original
                if len(stretched) == len(segment):
                    result[i:i+segment_size] = stretched
        
        # 2. Dynamic micro-variations
        # Create slowly varying envelope
        envelope_samples = max(100, len(audio) // 1000)
        random_envelope = np.random.randn(envelope_samples) * variation_amount
        smooth_envelope = signal.savgol_filter(random_envelope, 
                                              min(51, envelope_samples // 2 * 2 + 1), 3)
        
        # Interpolate to full length
        envelope = np.interp(np.arange(len(audio)), 
                           np.linspace(0, len(audio), len(smooth_envelope)),
                           smooth_envelope)
        
        # Apply subtle volume variations
        result *= (1.0 + envelope)
        
        # 3. Add imperceptible noise in quiet sections
        # Detect quiet sections
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                  hop_length=hop_length)[0]
        
        # Find quiet frames (bottom 10%)
        quiet_threshold = np.percentile(rms, 10)
        quiet_frames = rms < quiet_threshold
        
        # Add very subtle noise to quiet sections
        noise_level = variation_amount * 0.1  # Even more subtle
        for i, is_quiet in enumerate(quiet_frames):
            if is_quiet:
                start = i * hop_length
                end = min(start + frame_length, len(result))
                if start < len(result):
                    noise = np.random.randn(end - start) * noise_level
                    result[start:end] += noise
        
        # Ensure no clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result * 0.99 / max_val
        
        return result
    
    @staticmethod
    def harmonic_enhancement(audio: np.ndarray, sr: int,
                           enhancement_amount: float = 0.05) -> np.ndarray:
        """
        Enhance harmonics to make audio sound more natural.
        
        Args:
            audio: Input audio
            sr: Sample rate
            enhancement_amount: Enhancement level (0-1)
            
        Returns:
            Enhanced audio
        """
        # Use soft saturation for harmonic generation
        # This creates natural-sounding harmonics
        
        # Soft clipping function (tanh-based)
        def soft_saturate(x, amount):
            return np.tanh(x * (1 + amount)) / (1 + amount)
        
        # Apply very subtle saturation
        result = soft_saturate(audio, enhancement_amount)
        
        # Ensure same RMS as input
        input_rms = np.sqrt(np.mean(audio**2))
        output_rms = np.sqrt(np.mean(result**2))
        
        if output_rms > 0:
            result = result * (input_rms / output_rms)
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Test the fixes
    import soundfile as sf
    
    # Create test signal
    duration = 1.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Complex test signal with multiple components
    test_audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.1 * np.sin(2 * np.pi * 1320 * t) + # E6
        0.05 * np.random.randn(len(t))       # Noise
    )
    
    # Add some problematic values
    test_audio[1000:1010] = np.nan
    test_audio[2000:2010] = np.inf
    
    print("Testing audio processing fixes...")
    
    # Test validation
    print(f"1. Validation test: {AudioProcessingFixes.validate_audio_content(test_audio)}")
    
    # Test NaN cleanup
    cleaned = AudioProcessingFixes.safe_nan_cleanup(test_audio)
    print(f"2. NaN cleanup test: {AudioProcessingFixes.validate_audio_content(cleaned)}")
    
    # Test filter design
    b, a = AudioProcessingFixes.safe_filter_design(sr, [18000, 20000], len(cleaned))
    print(f"3. Filter design test: {'Success' if b is not None else 'Failed'}")
    
    # Test conservative watermark removal
    watermark_cleaned = WatermarkRemovalFixes.conservative_frequency_removal(
        cleaned, sr, [(19000, 20000), (15000, 16000)]
    )
    print(f"4. Watermark removal test: {AudioProcessingFixes.validate_audio_content(watermark_cleaned)}")
    
    # Test quality enhancement
    enhanced = AudioQualityEnhancer.add_natural_variations(watermark_cleaned, sr)
    print(f"5. Quality enhancement test: {AudioProcessingFixes.validate_audio_content(enhanced)}")
    
    # Save test output
    sf.write("test_fixes_comprehensive.wav", enhanced, sr)
    print("\nTest complete! Output saved to test_fixes_comprehensive.wav")